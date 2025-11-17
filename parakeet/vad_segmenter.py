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
    transcriber_app: str = modal.parameter(default="parakeet-transcription")
    transcriber_cls: str = modal.parameter(default="Parakeet")

    @modal.enter(snap=True)
    def load(self):


        print("Loading Silero VAD...")
        self.silero_vad = SileroVADAnalyzer(
            params=VADParams(
                stop_secs=0.2,
                sampling_rate=SAMPLE_RATE,
            )
        ),
        self.transcriber = modal.Cls.from_name(self.transcriber_app, self.transcriber_cls)()

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
                    new_vad_state = await self.vad.process(streaming_audio_chunk)
                    if current_vad_state == VADState.QUIET and new_vad_state == VADState.QUIET:
                        # keep around one second buffer if quiety
                        discarded = len(audio_buffer) - audio_buffer_size_1s
                        audio_buffer = audio_buffer[discarded:]
                    if current_vad_state in [
                        VADState.STARTING, VADState.SPEAKING, VADState.STOPPING
                    ] and new_vad_state == VADState.QUIET:
                        await transcription_queue.put(audio_buffer)
                        audio_buffer = bytearray()


            async def trancription_loop(segmented_audio_queue, transcription_queue):
                while True:
                    audio_segment = await segmented_audio_queue.get()
                    if transcript == SHUTDOWN_SIGNAL:
                        await transcription_queue.put(SHUTDOWN_SIGNAL)
                        break
                    transcript = await self.transcriber.transcribe(audio_segment)
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
        Path(__file__).parent /"streaming-parakeet-frontend", "/root/frontend"
    )
)

with web_image.imports():
    from fastapi import FastAPI,  WebSocket
    from fastapi.responses import HTMLResponse, Response
    from fastapi.staticfiles import StaticFiles



