import asyncio
import os
import sys
import time
from pathlib import Path

import modal


app = modal.App("parakeet-streaming-transcription")

model_cache = modal.Volume.from_name("parakeet-model-cache", create_if_missing=True)
parakeet_dict = modal.Dict.from_name("parakeet-dict", create_if_missing=True)

hf_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/cache",  # cache directory for Hugging Face models
            "DEBIAN_FRONTEND": "noninteractive",
            "CXX": "g++",
            "CC": "g++",
            "TORCH_HOME": "/cache",
        }
    )
    .apt_install("ffmpeg")
    .uv_pip_install(
        "fastapi[all]>=0.115.3",
        "huggingface-hub==0.34.0",
        "nemo-toolkit[asr]",
        # "onnxruntime>=1.22.0",
        "tokenizers==0.22.0",
        "transformers==4.56.1",
        "uvicorn==0.34.3",
        "omegaconf",
    )
    .entrypoint([])  # silence chatty logs by container on start
)

SAMPLE_RATE = 16000
PARAKEET_RT_STREAMING_CHUNK_SIZE = int(0.080 * SAMPLE_RATE) * 2 # 

def chunk_audio(data: bytes, chunk_size: int):
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]

with image.imports():
    import numpy as np
    import logging
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from starlette.websockets import WebSocketState
    from urllib.request import urlopen
    from fastapi import FastAPI
    from .parakeet_realtime_eou_service import NemoStreamingASRService
    from .asr_utils import preprocess_audio


@app.cls(
    volumes={"/cache": model_cache}, 
    gpu=["A100"], 
    image=image,
    # uncomment min containers for testing
    min_containers=1,
    scaledown_window=10,
    secrets=[hf_secret] if hf_secret is not None else [],
)
@modal.concurrent(max_inputs=20)
class Transcriber:

    # @modal.enter(snap=True)
    @modal.enter()
    def load(self):
        
        # silence chatty logs from nemo
        logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)

        self.model = NemoStreamingASRService(
            model="nvidia/parakeet_realtime_eou_120m-v1",
            device="cuda:0",
            decoder_type="rnnt",
        )

        # warm up gpu
        AUDIO_URL = "https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/mono_44100/156550__acclivity__a-dream-within-a-dream.wav"
        audio_bytes = urlopen(AUDIO_URL).read()
        audio_bytes = preprocess_audio(AUDIO_URL, target_sample_rate=16000)
        
        # streaming from bytes
        times = []
        audio_chunks = chunk_audio(audio_bytes, PARAKEET_RT_STREAMING_CHUNK_SIZE)
        for chunk in audio_chunks:
            start_time = time.perf_counter()
            print(f"transcript: {self.model.transcribe(chunk)}")
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        print(f"Warmup transcription quantile values ({PARAKEET_RT_STREAMING_CHUNK_SIZE} byte chunks):")
        print(f"p5: {np.percentile(times, 5)}")
        print(f"p50: {np.percentile(times, 50)}")
        print(f"p95: {np.percentile(times, 95)}")

        print("GPU warmed up")

        self.model.reset_state()

        self._chunk_size = PARAKEET_RT_STREAMING_CHUNK_SIZE

        self.web_app = FastAPI()

        @self.web_app.websocket("/ws")
        async def run_with_websocket(ws: WebSocket):

            audio_queue = asyncio.Queue()
            transcription_queue = asyncio.Queue()
            
            async def recv_loop(ws, audio_queue):
                audio_buffer = bytearray()
                while True:
                    data = await ws.receive_bytes()
                    audio_buffer.extend(data)
                    if len(audio_buffer) > self._chunk_size:

                        await audio_queue.put(audio_buffer)
                        audio_buffer = bytearray()

            async def inference_loop(audio_queue, transcription_queue):

                while True:
                    
                    audio_data = await audio_queue.get()

                    start_time = time.perf_counter()
                    transcript = self.transcribe(audio_data)
                    print(f"transcript: {transcript}")
                    if transcript:
                        await transcription_queue.put(transcript)

                    end_time = time.perf_counter()
                    print(f"time taken to transcribe audio segment: {end_time - start_time} seconds")
                                       
            async def send_loop(transcription_queue, ws):
                while True:
                    transcript = await transcription_queue.get()
                    print(f"sending transcription data: {transcript}")
                    await ws.send_text(transcript)

            await ws.accept()

            try:
                tasks = [
                    asyncio.create_task(recv_loop(ws, audio_queue)),
                    asyncio.create_task(inference_loop(audio_queue, transcription_queue)),
                    asyncio.create_task(send_loop(transcription_queue, ws)),
                ]
                await asyncio.gather(*tasks)
            except WebSocketDisconnect:
                print("WebSocket disconnected")
                ws = None
            except Exception as e:
                print("Exception:", e)
            finally:
                if ws and ws.application_state is WebSocketState.CONNECTED:
                    await ws.close(code=1011) # internal error
                    ws = None
                for task in tasks:                    
                    if not task.done():
                        try:
                            task.cancel()
                            await task
                        except asyncio.CancelledError:
                            pass


    def transcribe(self, audio_data) -> str:

        output = self.model.transcribe(audio_data)
        return output.text

    @modal.asgi_app()
    def webapp(self):
        return self.web_app


web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi")
    .add_local_dir(
        Path(__file__).parent.parent /"parakeet-frontend", "/root/frontend"
    )
)

with web_image.imports():
    from fastapi import FastAPI,  WebSocket
    from fastapi.responses import HTMLResponse, Response
    from fastapi.staticfiles import StaticFiles

@app.cls(image=web_image)
@modal.concurrent(max_inputs=20)
class WebServer:

    @modal.asgi_app()
    def web(self):
        
        web_app = FastAPI()
        web_app.mount("/static", StaticFiles(directory="frontend"))

        @web_app.get("/status")
        async def status():
            return Response(status_code=200)

        # serve frontend
        @web_app.get("/")
        async def index():
            html_content = open("frontend/index.html").read()
            
            # Get the base WebSocket URL (without transcriber parameters)
            cls_instance = modal.Cls.from_name("parakeet-streaming-transcription", "Transcriber")()
            ws_base_url = cls_instance.webapp.get_web_url().replace('http', 'ws') + "/ws"
            script_tag = f'<script>window.WS_BASE_URL = "{ws_base_url}";</script>'
            html_content = html_content.replace(
                '<script src="/static/parakeet.js"></script>', 
                f'{script_tag}\n<script src="/static/parakeet.js"></script>'
            )
            return HTMLResponse(content=html_content)

        return web_app


class NoStdStreams(object):
    def __init__(self):
        self.devnull = open(os.devnull, "w")

    def __enter__(self):
        self._stdout, self._stderr = sys.stdout, sys.stderr
        self._stdout.flush(), self._stderr.flush()
        sys.stdout, sys.stderr = self.devnull, self.devnull

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self.devnull.close()


