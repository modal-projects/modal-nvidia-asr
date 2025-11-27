import asyncio
from pathlib import Path

from .asr_utils import SHUTDOWN_SIGNAL

import modal

app = modal.App("silero-vad-segmenter")

model_cache = modal.Volume.from_name("silero-vad-model-cache", create_if_missing=True)
cache_path = "/cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": cache_path,  # cache directory for Hugging Face models
        }
    )
    .pip_install(
        "hf_transfer==0.1.9",
        "huggingface_hub[hf-xet]==0.31.2",
        "fastapi==0.115.12",
        "pipecat-ai[silero]"
    )
)

SAMPLE_RATE = 16000

with image.imports():
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams, VADState


@app.cls(
    volumes={cache_path: model_cache}, 
    image=image,
)
class SileroVADSegmenter:

    @modal.enter()
    def load(self):


        print("Loading Silero VAD...")
        self.silero_vad = SileroVADAnalyzer()
        self.silero_vad.set_sample_rate(SAMPLE_RATE)
        self.silero_vad.set_params(
            VADParams(
                stop_secs=0.2,
            )
        )
        self.transcriber = modal.Cls.from_name("parakeet-transcription", "Parakeet")()

        print("Container ready.")

    @modal.asgi_app()
    def webapp(self):
        
        web_app = FastAPI()


        @web_app.websocket("/ws")
        async def run_with_websocket(ws: WebSocket):

            streaming_audio_queue = asyncio.Queue()
            segmented_audio_queue = asyncio.Queue()
            transcription_queue = asyncio.Queue()
            
            async def recv_loop(ws, audio_queue):
                while True:
                    data = await ws.receive_bytes()
                    if data == SHUTDOWN_SIGNAL:
                        await streaming_audio_queue.put(SHUTDOWN_SIGNAL)
                        break
                    await streaming_audio_queue.put(data)

            async def vad_loop(streaming_audio_queue, segmented_audio_queue):
                audio_buffer = bytearray()
                audio_buffer_size_1s = SAMPLE_RATE * 2
                current_vad_state = VADState.QUIET
                while True:
                    streaming_audio_chunk = await streaming_audio_queue.get()
                    if streaming_audio_chunk == SHUTDOWN_SIGNAL:
                        await segmented_audio_queue.put(SHUTDOWN_SIGNAL)
                        break
                    audio_buffer += streaming_audio_chunk
                    new_vad_state = await self.silero_vad.analyze_audio(streaming_audio_chunk)
                    print(f"New VAD state: {new_vad_state}")
                    if (
                        current_vad_state == VADState.QUIET 
                        and new_vad_state == VADState.QUIET
                        and len(audio_buffer) > audio_buffer_size_1s
                    ):
                        # keep around one second buffer if quiety
                        discarded = len(audio_buffer) - audio_buffer_size_1s
                        audio_buffer = audio_buffer[discarded:]
                    elif current_vad_state in [
                        VADState.STARTING, VADState.SPEAKING, VADState.STOPPING
                    ] and new_vad_state == VADState.QUIET:
                        print(f"Speech ended, sending {len(audio_buffer)} bytes to transcription")
                        await segmented_audio_queue.put(audio_buffer)
                        audio_buffer = bytearray()
                    current_vad_state = new_vad_state


            async def trancription_loop(segmented_audio_queue, transcription_queue):
                while True:
                    audio_segment = await segmented_audio_queue.get()
                    if audio_segment == SHUTDOWN_SIGNAL:
                        await transcription_queue.put(SHUTDOWN_SIGNAL)
                        break
                    transcript = await self.transcriber.transcribe.remote.aio(audio_segment)
                    print(f"Transcript: {transcript}")
                    await transcription_queue.put(transcript)

            async def send_loop(ws, transcription_queue):
                while True:
                    transcript = await transcription_queue.get()
                    if transcript == SHUTDOWN_SIGNAL:
                        break
                    await ws.send_text(transcript)

            await ws.accept()

            try:
                tasks = [
                    asyncio.create_task(recv_loop(ws, streaming_audio_queue)),
                    asyncio.create_task(vad_loop(streaming_audio_queue, segmented_audio_queue)),
                    asyncio.create_task(trancription_loop(segmented_audio_queue, transcription_queue)),
                    asyncio.create_task(send_loop(ws, transcription_queue)),
                ]
                await asyncio.gather(*tasks)
            except WebSocketDisconnect:
                print("WebSocket disconnected")
                await ws.close(code=1000)
            except Exception as e:
                print("Exception:", e)
                await ws.close(code=1011)  # internal error
                raise e

        return web_app

        

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
@modal.concurrent(max_inputs=1000)
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
            ws_base_url = SileroVADSegmenter().webapp.get_web_url().replace('http', 'ws') + "/ws"
            script_tag = f'<script>window.WS_BASE_URL = "{ws_base_url}";</script>'
            html_content = html_content.replace(
                '<script src="/static/parakeet.js"></script>', 
                f'{script_tag}\n<script src="/static/parakeet.js"></script>'
            )
            return HTMLResponse(content=html_content)

        return web_app