from typing import Union
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
    .pip_install(
        "hf_transfer==0.1.9",
        "huggingface_hub[hf-xet]==0.31.2",
        "nemo_toolkit[asr]==2.3.0",
        "cuda-python==12.8.0",
        "numpy<2",
        "torchaudio",
        "soundfile",
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
    import torch
    from .asr_utils import (
        preprocess_audio, 
        batch_seq, 
        NoStdStreams,
        write_wav_file,
    )


@app.cls(
    volumes={CACHE_DIR: model_cache}, 
    gpu="L40S", 
    image=image,
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
    async def transcribe(self, audio_data: Union[bytes, bytearray, list[Union[bytes, bytearray]]]) -> str:

        print(type(audio_data))
        if not isinstance(audio_data, list):
            audio_data = [audio_data]

        print(f"Received {len(audio_data)} audio segments for transcription.")

        # we need to write the audio data to temporary files for batch transcription
        # temp_files = []
        # You can set the number of threads by passing max_workers to ThreadPoolExecutor
        num_threads = len(audio_data)  # Set this to your desired number of threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            audio_data = list(executor.map(write_wav_file, enumerate(audio_data)))


        with NoStdStreams():
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16), torch.inference_mode(), torch.no_grad():
                output = self.model.transcribe(audio_data, batch_size=len(audio_data), num_workers=1)
        
        if len(audio_data) == 1:
            return output[0].text
        return [result.text for result in output]



@app.local_entrypoint()
async def main():

    from .asr_utils import preprocess_audio, batch_seq

    parakeet = Parakeet()

    # warm up gpu
    AUDIO_URL = "https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/mono_44100/156550__acclivity__a-dream-within-a-dream.wav"
    audio_bytes = preprocess_audio(AUDIO_URL, target_sample_rate=16000)
    # Then chunk the audio data (not the raw bytes)
    chunk_size_seconds = 5
    chunk_size = SAMPLE_RATE * chunk_size_seconds * SAMPLE_WIDTH_BYTES  # at 16kHz
    audio_chunks = batch_seq(audio_bytes, chunk_size)

    transcripts = await parakeet.transcribe.remote.aio(audio_chunks)
    print(f"Batch transcripts: {[transcript for transcript in transcripts]}")

    transcript = await parakeet.transcribe.remote.aio(audio_chunks[0])
    print(f"Single transcript: {transcript[0]}")