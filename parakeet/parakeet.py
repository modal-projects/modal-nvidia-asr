from typing import Union, Optional, Literal
import concurrent.futures
import logging

import modal

app = modal.App("parakeet-transcription")

MODEL = "nvidia/parakeet-tdt-0.6b-v3"

model_cache = modal.Volume.from_name("parakeet-model-cache", create_if_missing=True)
CACHE_DIR = "/cache"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": CACHE_DIR,  # cache directory for Hugging Face models
            "DEBIAN_FRONTEND": "noninteractive",
            "CXX": "g++",
            "CC": "g++",
            "TORCH_HOME": CACHE_DIR,
        }
    )
    .apt_install("ffmpeg")
    .uv_pip_install(
        "hf_transfer==0.1.9",
        "huggingface_hub[hf-xet]==0.31.2",
        "nemo_toolkit[asr]==2.3.0",
        "cuda-python==12.8.0",
        "numpy<2",
        "torchaudio",
        "soundfile",
        "resampy",
        "fastapi[standard]",
    )
    .apt_install("curl")
    .entrypoint([])  # silence chatty logs by container on start
)

BATCH_SIZE = 128
 
SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2
MINUTES = 60

NUM_WARMUP_BATCHES = 4

with image.imports():
    import nemo.collections.asr as nemo_asr
    import torch
    from fastapi import FastAPI, Request
    from .asr_utils import (
        preprocess_audio, 
        batch_seq, 
        NoStdStreams,
        write_wav_file,
        bytes_to_torch,
    )


@app.cls(
    volumes={CACHE_DIR: model_cache}, 
    gpu="L40S", 
    image=image,
    min_containers=1,
)
class Parakeet:

    @modal.enter()
    async def load(self):

        # silence chatty logs from nemo
        logging.getLogger("nemo_logger").setLevel(logging.CRITICAL)

        print("Loading Parakeet...")
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=MODEL
        )
        self.model.to(torch.bfloat16)
        self.model.eval()
        # Configure decoding strategy
        if self.model.cfg.decoding.strategy != "beam":
            self.model.cfg.decoding.strategy = "greedy_batch"
            self.model.change_decoding_strategy(self.model.cfg.decoding)

        await self.warm_up_gpu()

        print("Parakeet model loaded and ready.")

    async def warm_up_gpu(self):

        print("Warming up GPU...")
        # warm up gpu
        AUDIO_URL = "https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/mono_44100/156550__acclivity__a-dream-within-a-dream.wav"
        audio_bytes = preprocess_audio(AUDIO_URL, target_sample_rate=16000)
        
        # Then chunk the audio data (not the raw bytes)
        chunk_size_seconds = 5
        chunk_size = SAMPLE_RATE * chunk_size_seconds * SAMPLE_WIDTH_BYTES  # at 16kHz
        audio_chunks = batch_seq(audio_bytes, chunk_size)

        # ensure we have enough chunks to fill 4 batches
        if len(audio_chunks) < BATCH_SIZE * NUM_WARMUP_BATCHES:
            expand_factor = int(BATCH_SIZE * NUM_WARMUP_BATCHES / len(audio_chunks))
            audio_chunks = audio_chunks * expand_factor
            audio_chunks = audio_chunks[:BATCH_SIZE * NUM_WARMUP_BATCHES]

        # batch the chunks and perform transcription
        for batch in batch_seq(audio_chunks, BATCH_SIZE):
            print(await self.transcribe.local(batch))

    @modal.method()
    async def transcribe(
        self, 
        audio_data: Union[bytes, bytearray, list[Union[bytes, bytearray]]],
        timestamp_level: Optional[Literal['word', 'segment', 'char']] = None
    ) -> Union[dict, list[dict]]:

        is_batch = isinstance(audio_data, list)
        
        if is_batch:

            print(f"Received {len(audio_data)} audio segments for transcription.")

            # we need to write the audio data to temporary files for batch transcription
            # temp_files = []
            # You can set the number of threads by passing max_workers to ThreadPoolExecutor
            num_threads = len(audio_data)  # Set this to your desired number of threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                audio_data = list(executor.map(write_wav_file, enumerate(audio_data)))
            batch_size = len(audio_data)

            with NoStdStreams():
                with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16), torch.inference_mode(), torch.no_grad():
                    output = self.model.transcribe(
                        audio_data, 
                        batch_size=batch_size, 
                        num_workers=1,
                        timestamps=timestamp_level is not None
                    )
        else:
            audio_data = preprocess_audio(audio_data, return_tensor=True)
            with NoStdStreams():
                with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16), torch.inference_mode(), torch.no_grad():
                    output = self.model.transcribe(
                        audio_data,
                        timestamps=timestamp_level is not None,
                        return_hypotheses=True,
                    )

        # Format output - always return dict with transcript and timestamps
        if is_batch:
            results = []
            for result in output:
                if timestamp_level is not None:
                    timestamps_data = result.timestamp.get(timestamp_level, [])
                    formatted_timestamps = [
                        (stamp.get(timestamp_level, stamp.get('segment', '')), stamp['start'], stamp['end'])
                        for stamp in timestamps_data
                    ]
                else:
                    formatted_timestamps = []
                
                results.append({
                    "transcript": result.text,
                    "timestamps": formatted_timestamps
                })
            return results
        else:
            result = output[0]
            if timestamp_level is not None:
                timestamps_data = result.timestamp.get(timestamp_level, [])
                formatted_timestamps = [
                    (stamp.get(timestamp_level, stamp.get('segment', '')), stamp['start'], stamp['end'])
                    for stamp in timestamps_data
                ]
            else:
                formatted_timestamps = []
            
            return {
                "transcript": result.text,
                "timestamps": formatted_timestamps
            }
        

    @modal.asgi_app()
    def webapp(self):

        web_app = FastAPI()

        @web_app.post("/api")
        async def api(
            request: Request, 
            timestamp_level: Optional[Literal['word', 'segment', 'char']] = None
        ):
            audio_data = await request.body()
            result = await self.transcribe.local(audio_data, timestamp_level=timestamp_level)
            
            # Result is always in the correct format now
            return result

        return web_app

@app.function(image=image)
async def transcribe_with_api():
    from .asr_utils import preprocess_audio, batch_seq
    import subprocess
    import time
    import json

    url = Parakeet().webapp.get_web_url() + "/api?timestamp_level=word"

    # warm up gpu
    AUDIO_URL = "https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/mono_44100/156550__acclivity__a-dream-within-a-dream.wav"
    audio_bytes = preprocess_audio(AUDIO_URL, target_sample_rate=16000)
    # Then chunk the audio data (not the raw bytes)
    chunk_size_seconds = 10
    chunk_size = SAMPLE_RATE * chunk_size_seconds * SAMPLE_WIDTH_BYTES  # at 16kHz
    audio_chunks = batch_seq(audio_bytes, chunk_size)

    latencies = []
    for i, chunk in enumerate(audio_chunks):
        # write chunk to a wave file
        filename = write_wav_file((i, chunk))
        
        start_time = time.perf_counter()
        try:
            result = subprocess.run(
                [
                    "curl",
                    "-X", "POST",
                    "-H", "Content-Type: audio/wav",
                    "--data-binary", f"@{filename}",
                    url
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            resp_json = json.loads(result.stdout)
            transcript = resp_json.get("transcript", "")
            timestamps = resp_json.get("timestamps", [])
            end_time = time.perf_counter()
            
            print(f"\n--- Chunk {i} ---")
            print(f"Transcript: {transcript}")
            if timestamps:
                print("Word-level timestamps:")
                for word, start, end in timestamps:
                    print(f"  {start:.2f}s - {end:.2f}s : '{word}'")
            
            if i != 0:
                latencies.append(end_time - start_time)
        except subprocess.CalledProcessError as e:
            print(f"Curl Error: {e.returncode}")
            print(f"stderr: {e.stderr}")
            continue
        except json.JSONDecodeError as e:
            print(f"JSON Error: {e}")
            print(f"stdout: {result.stdout}")
            continue

    if latencies:
        print(f"\nAverage latency: {sum(latencies) / len(latencies):.3f} seconds")


@app.local_entrypoint()
async def main():

    await transcribe_with_api.remote.aio()

if __name__ == "__main__":
    transcribe_with_api_func = modal.Function.from_name(app.name, "transcribe_with_api")
    transcribe_with_api_func.remote()
