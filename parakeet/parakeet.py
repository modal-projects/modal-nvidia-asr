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
    .entrypoint([])  # silence chatty logs by container on start
)

BATCH_SIZE = 128
 
SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2
MINUTES = 60

NUM_WARMUP_BATCHES = 4

with image.imports():
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceConfig
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
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
    region="us-east",
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
        print(self.model.cfg.decoding)
        print(type(self.model.cfg.decoding))
        decoding_cfg = RNNTDecodingConfig(
            strategy="greedy_batch",
            durations=self.model.cfg.decoding.durations,
            model_type=self.model.cfg.decoding.model_type,
            greedy=self.model.cfg.decoding.greedy,
            beam=self.model.cfg.decoding.beam,
            preserve_alignments=True,
            confidence_cfg = ConfidenceConfig(preserve_word_confidence=True)
        )
        self.model.change_decoding_strategy(decoding_cfg)
        print(self.model.cfg.decoding)
        print(type(self.model.cfg.decoding))
        print(self.model.cfg.decoding.confidence_cfg)

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
            result = await self.transcribe.local(batch, timestamp_level='word', return_word_confidence=True)
            print(result[0])

    @modal.method()
    async def transcribe(
        self, 
        audio_data: Union[bytes, bytearray, list[Union[bytes, bytearray]]],
        timestamp_level: Optional[Literal['word', 'segment', 'char']] = None,
        return_word_confidence: bool = True,
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
                        timestamps=timestamp_level is not None,
                        return_hypotheses=True,
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
        
        # print(output)

        # Format output - always return dict with transcript and word level info
        if is_batch:
            results = []
            for result in output:
                if timestamp_level is not None:
                    timestamps_data = result.timestamp.get(timestamp_level, [])
                    word_level_info = [
                        (stamp.get(timestamp_level, stamp.get('segment', '')), stamp['start'], stamp['end'])
                        for stamp in timestamps_data
                    ]
                    if timestamp_level == 'word' and return_word_confidence:
                        word_level_info = [
                            data + (result.word_confidence[i].item(),)
                            for i, data in enumerate(word_level_info)
                    ]

                else:
                    word_level_info = []
                
                results.append({
                    "transcript": result.text,
                    "word_level_info": word_level_info
                })
            return results
        else:
            result = output[0]
            if timestamp_level is not None:
                timestamps_data = result.timestamp.get(timestamp_level, [])
                word_level_info = [
                    (stamp.get(timestamp_level, stamp.get('segment', '')), stamp['start'], stamp['end'])
                    for stamp in timestamps_data
                ]
                if timestamp_level == 'word' and return_word_confidence:
                        word_level_info = [
                            data + (result.word_confidence[i].item(),)
                            for i, data in enumerate(word_level_info)
                    ]
            else:
                word_level_info = []
            
            return {
                "transcript": result.text,
                "word_level_info": word_level_info
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

api_test_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("nemo_toolkit[asr]==2.3.0", "numpy<2")
    .apt_install("curl")
)
    
with api_test_image.imports():
    from .asr_utils import preprocess_audio, batch_seq, write_wav_file
    import subprocess
    import time
    import json

@app.function(image=api_test_image, enable_memory_snapshot=True, region="us-east")
async def transcribe_with_api():

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
            word_level_info = resp_json.get("word_level_info", [])
            end_time = time.perf_counter()
            
            print(f"\n--- Chunk {i} ---")
            print(f"Transcript: {transcript}")
            if word_level_info:
                print("Word-level info:")
                for word, start, end, confidence in word_level_info:
                    print(f"  {start:.2f}s - {end:.2f}s : '{word}', confidence: {confidence}")
            
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
