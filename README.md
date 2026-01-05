# NVIDIA ASR Implementations on Modal

This repository demonstrates approaches to deploy NVIDIA's Nemotron-Speech ASR and Parakeet ASR models on Modal for **batch** and **streaming** transcription.

## Install and setup Modal

```bash
pip install modal
```

Authenticate your Modal account:

```bash
modal setup
```

# Nemotron-Speech ASR

Nemotron-Speech ASR is a powerful open weights model that can stream high numbers of concurrent clients.
It outputs partial and final transcripts, capitlization and punctiontuation, and has word boosting capabilities for domain-specific vocabulary.

However, `NeMO` does not currently provide an implementation for asynchronous, concurrent clients. This codebase includes extensions to 
`NeMO` that enable batched, cache-aware inference on asynchronous streaming clients.

## Deployment

```bash
modal deploy -m nemotron_asr.nemotron_asr
```

# Batch and Streaming Parakeet

## Implementation

### 1. Batch Transcription (`parakeet/parakeet.py`)

The core Parakeet transcriber runs on GPU and handles both single audio files and batches:

- Accepts audio as `bytes` or `list[bytes]`
- Processes batches up to `BATCH_SIZE = 128` for efficient GPU utilization
- Exposes a Modal method that can be called from anywhere

### 2. Streaming with VAD Segmentation (`parakeet/vad_segmenter.py`)

For Parakeet models that don't natively support streaming, we use **Voice Activity Detection (VAD)** to segment the stream:

```
Audio Stream → VAD Segmenter (CPU) → Parakeet Transcriber (GPU)
```

The VAD segmenter:
- Runs as a **separate Modal function** (CPU-only, no GPU)
- Uses Silero VAD (via Pipecat's wrapper) to detect speech start/stop in the audio stream (see [Pipecat's docs](https://docs.pipecat.ai/guides/learn/speech-input) for settings)
- Buffers audio during speech and segments it when speech ends
- Calls the Parakeet transcriber endpoint with **batch_size = 1** for each segment

**Why separate the VAD from transcription?** This architecture enables independent autoscaling and better GPU utilization. Multiple VAD segmenters (cheap CPU) can feed a smaller pool of GPU transcribers, so GPUs only run when there's actual speech to transcribe.

### 3. Native Streaming Transcription (`parakeet/parakeet_streaming.py`)

The newest approach uses NVIDIA's **Parakeet Realtime model** (`nvidia/parakeet_realtime_eou_120m-v1`) with native streaming support:

```
Audio Stream → Parakeet Realtime (GPU) → Transcription
```

Key features:
- **No VAD required** — the model processes audio chunks directly as they arrive
- Uses `NemoStreamingASRService` with built-in end-of-utterance (EOU) detection
- Processes audio in 80ms chunks for low-latency transcription
- Single GPU-based service handles both audio ingestion and transcription
- WebSocket-based streaming interface

This approach offers the lowest latency and simplest architecture since everything runs in one place, but requires GPU for the entire audio stream (not just during speech).

### 4. Multi-Speaker Native Streaming (`parakeet/parakeet_multitalker.py`)

The most advanced approach combines real-time **speaker diarization** with **multi-talker ASR** for streaming transcription with speaker labels:

```
Audio Stream → Sortformer Diarization + Multi-talker Parakeet (GPU) → Speaker-tagged Transcription
```

Key features:
- **Multi-speaker support** — automatically separates and transcribes up to 4 concurrent speakers
- Uses NVIDIA's `multitalker-parakeet-streaming-0.6b-v1` model with `diar_streaming_sortformer_4spk-v2.1` diarization
- **Cache-aware buffering** — intelligent audio buffering that aligns with model's cache requirements
- Processes audio in 80ms frames with 13-frame buffer for optimal streaming performance
- WebSocket-based streaming interface with speaker-tagged output
- Single GPU-based service handles diarization and multi-speaker transcription simultaneously

This approach is ideal for scenarios with multiple speakers (meetings, conversations, interviews) where you need to know "who said what" in real-time.

## Deploy

```bash
# 1. Batch transcription
modal deploy -m parakeet.parakeet

# 2. Streaming with VAD segmentation
modal deploy -m parakeet.vad_segmenter

# 3. Native streaming transcription (Parakeet Realtime)
modal deploy -m parakeet.parakeet_streaming

# 4. Multi-speaker native streaming transcription
modal deploy -m parakeet.parakeet_multitalker
```

## Frontend

The `parakeet-frontend/` directory contains a simple web interface for testing streaming transcription via WebSocket. 

When you deploy any streaming version:
- **VAD segmentation** (`vad_segmenter`): Frontend URL will be printed to console with format:
  ```bash
  https://{workspace}-{environment}--silero-vad-segmenter-webserver-web.modal.run
  ```
- **Native streaming** (`parakeet_streaming`): Frontend URL will be printed to console with format:
  ```bash
  https://{workspace}-{environment}--parakeet-streaming-transcription-{shorten-id}.modal.run
  ```
- **Multi-speaker streaming** (`parakeet_multitalker`): Frontend URL will be printed to console with format:
  ```bash
  https://{workspace}-{environment}--parakeet-multitalker-webserver-web.modal.run
  ```

