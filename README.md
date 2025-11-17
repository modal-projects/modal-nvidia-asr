# Parakeet Patterns

This repository demonstrates how to deploy NVIDIA's Parakeet ASR model on Modal for both **batch** and **streaming** transcription.

## Implementation

### Batch Transcription (`parakeet/parakeet.py`)

The core Parakeet transcriber runs on GPU and handles both single audio files and batches:

- Accepts audio as `bytes` or `list[bytes]`
- Processes batches up to `BATCH_SIZE = 128` for efficient GPU utilization
- Exposes a Modal method that can be called from anywhere

### Streaming Transcription (`parakeet/vad_segmenter.py`)

Parakeet doesn't natively support streaming—it needs complete audio segments. So we use **Voice Activity Detection (VAD)** to segment the stream:

```
Audio Stream → VAD Segmenter (CPU) → Parakeet Transcriber (GPU)
```

The VAD segmenter:
- Runs as a **separate Modal function** (CPU-only, no GPU)
- Uses Silero VAD (via Pipecat's wrapper) to detect speech start/stop in the audio stream (see [Pipecat's docs](https://docs.pipecat.ai/guides/learn/speech-input) for settings)
- Buffers audio during speech and segments it when speech ends
- Calls the Parakeet transcriber endpoint with **batch_size = 1** for each segment

**Why separate the VAD from transcription?** This architecture enables independent autoscaling and better GPU utilization. Multiple VAD segmenters (cheap CPU) can feed a smaller pool of GPU transcribers, so GPUs only run when there's actual speech to transcribe.

## Getting Started

Install dependencies:

```bash
pip install modal
```

Authenticate your Modal account:

```bash
modal setup
```

Deploy to Modal:

```bash
# Batch transcription
modal deploy -m parakeet.parakeet

# Streaming transcription  
modal deploy -m parakeet.vad_segmenter
```

## Frontend

The `streaming-parakeet-frontend/` directory contains a simple web interface for testing streaming transcription via WebSocket. When you deploy `vad_segmenter` you'll see the URL for the frontend printed in the console. It will have the following format:
```bash
https://{workspace}-{environment}--silero-vad-segmenter-webserver-web.modal.run
```

