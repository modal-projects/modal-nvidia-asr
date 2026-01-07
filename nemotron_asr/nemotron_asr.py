# # Nemotron ASR Streaming on Modal

import time
import json
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

import modal

app = modal.App("nemotron-streaming-asr")

model_cache = modal.Volume.from_name("nemotron-speech", create_if_missing=True)
CACHE_PATH = "/model"

hf_secret = modal.Secret.from_name("huggingface-secret")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.1-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": CACHE_PATH,
            "CXX": "g++",
            "CC": "g++",
            "TORCH_HOME": CACHE_PATH,
            "PYTHONPATH": "/root/stt-services:$PYTHONPATH",
        }
    )
    .apt_install("git", "libsndfile1", "ffmpeg")
    .uv_pip_install(
        "hf_transfer==0.1.9",
        "huggingface_hub[hf-xet]==0.31.2",
        "cuda-python==13.0.1",
        "numpy<2",
        "fastapi[standard]",
        "orjson",
        "msgpack",
        "nemo_toolkit[asr]@git+https://github.com/NVIDIA/NeMo.git@main",
        "nemo_text_processing"
    )
)


with image.imports():
    import asyncio
    import logging
    import numpy as np
    import msgpack
    import torch
    from omegaconf import OmegaConf
    from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions
    from nemo.collections.asr.inference.utils.progressbar import TQDMProgressBar
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from starlette.websockets import WebSocketState
    import uvicorn
    import threading
    
    # Import local utilities
    from .asr_utils import preprocess_audio
    from .pipelines import StreamingPipelineBuilder


# Configuration dataclasses

@dataclass
class BoostingTreeConfig:
    """Boosting tree configuration for phrase boosting"""
    model_path: Optional[str] = None
    key_phrases_file: Optional[str] = None
    key_phrases_list: Optional[List[str]] = None
    source_lang: str = "en"


@dataclass
class GreedyDecodingConfig:
    """Greedy decoding configuration"""
    use_cuda_graph_decoder: bool = False
    max_symbols: int = 10
    ngram_lm_model: Optional[str] = None
    ngram_lm_alpha: float = 0.0
    boosting_tree: BoostingTreeConfig = field(default_factory=BoostingTreeConfig)
    boosting_tree_alpha: float = 0.0


@dataclass
class DecodingConfig:
    """Decoding configuration"""
    strategy: str = "greedy_batch"
    preserve_alignments: bool = False
    fused_batch_size: int = -1
    greedy: GreedyDecodingConfig = field(default_factory=GreedyDecodingConfig)


@dataclass
class ASRConfig:
    """ASR model configuration"""
    model_name: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    device: str = "cuda"
    device_id: int = 0
    compute_dtype: str = "bfloat16"
    use_amp: bool = True
    decoding: DecodingConfig = field(default_factory=DecodingConfig)


@dataclass
class ITNConfig:
    """Inverse Text Normalization configuration"""
    input_case: str = "lower_cased"
    whitelist: Optional[str] = None
    overwrite_cache: bool = False
    max_number_of_permutations_per_split: int = 729
    left_padding_size: int = 4
    batch_size: int = 32
    n_jobs: int = 16


@dataclass
class ConfidenceConfig:
    """Confidence estimation configuration"""
    exclude_blank: bool = True
    aggregation: str = "mean"
    method_cfg: dict = field(default_factory=lambda: {
        "name": "entropy",
        "entropy_type": "tsallis",
        "alpha": 0.5,
        "entropy_norm": "exp",
    })


@dataclass
class EndpointingConfig:
    """Endpointing configuration"""
    stop_history_eou: int = 800
    residue_tokens_at_end: int = 2


@dataclass
class StreamingConfig:
    """Streaming configuration"""
    sample_rate: int = 16000
    batch_size: int = 512
    word_boundary_tolerance: int = 4
    att_context_size: List[int] = field(default_factory=lambda: [70, 6])
    use_cache: bool = True
    use_feat_cache: bool = True
    chunk_size_in_secs: Optional[float] = None
    request_type: str = "frame"
    num_slots: int = 1024
    exhaustive_batching: bool = False  # Process ALL ready frames in one cycle vs single batch
    batching_delay_secs: float = 0.300

@dataclass
class MetricsASRConfig:
    """ASR metrics configuration"""
    gt_text_attr_name: str = "text"
    clean_groundtruth_text: bool = False
    langid: str = "en"
    use_cer: bool = False
    ignore_capitalization: bool = True
    ignore_punctuation: bool = True
    strip_punc_space: bool = False


@dataclass
class MetricsConfig:
    """Metrics configuration"""
    asr: MetricsASRConfig = field(default_factory=MetricsASRConfig)


@dataclass
class CacheAwarePipelineConfig:
    """Main configuration for cache-aware RNNT pipeline"""
    # ASR configuration
    asr: ASRConfig = field(default_factory=ASRConfig)
    
    # ITN configuration
    itn: ITNConfig = field(default_factory=ITNConfig)
    
    # NMT configuration (set to None to disable)
    nmt: Optional[dict] = None
    
    # Confidence configuration
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    
    # Endpointing configuration
    endpointing: EndpointingConfig = field(default_factory=EndpointingConfig)
    
    # Streaming configuration
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    
    # Pipeline settings
    matmul_precision: str = "high"
    log_level: int = 20
    pipeline_type: str = "cache_aware"
    asr_decoding_type: str = "rnnt"
    
    # Runtime arguments
    audio_file: Optional[str] = None
    output_filename: Optional[str] = None
    output_dir: Optional[str] = None
    enable_pnc: bool = False
    enable_itn: bool = True
    enable_nmt: bool = False
    asr_output_granularity: str = "segment"
    cache_dir: Optional[str] = None
    lang: Optional[str] = 'en'
    return_tail_result: bool = False
    calculate_wer: bool = True
    calculate_bleu: bool = True
    
    # Metrics
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


@app.cls(
    volumes={CACHE_PATH: model_cache},
    gpu=["H100!"],
    image=image,
    secrets=[hf_secret] if hf_secret is not None else [],
    timeout=3600,
    # min_containers=1,
    scaledown_window=3600,
)
@modal.concurrent(max_inputs=512)
class NemotronASR:
        
    @modal.enter()
    async def load(self):
        # Silence chatty logs from nemo
        logging.getLogger("nemo_logger").setLevel(logging.WARNING)
        
        self.client_queues: dict[int, asyncio.Queue] = {}
        self.client_queues_lock = None  # Will be initialized in async context
        self.inference_task = None
        self.inference_running = False
        
        # Per-stream audio timestamp tracking (cumulative duration in seconds)
        self.stream_audio_timestamps: dict[int, float] = {}
        self.stream_timestamps_lock = None  # Will be initialized in async context
        
        # Track message format per stream (True = msgpack, False = raw bytes/JSON)
        self.stream_uses_msgpack: dict[int, bool] = {}
        self.stream_format_lock = None  # Will be initialized in async context
        
        # Track streams pending cleanup (WebSocket closed but may still have buffered frames)
        self.streams_pending_cleanup: set[int] = set()
        self.streams_cleanup_lock = None  # Will be initialized in async context
        
        # Timing reference (set after warmup)
        self.start_time = None

        print("Initializing pipeline configuration...")
        
        # Create config as dataclass, then convert to OmegaConf
        config = CacheAwarePipelineConfig()
        
        # Disable ITN and NMT
        # ITN was causing 2+ second blocking in inference loop without being used
        config.enable_itn = False  # Disabled - was blocking inference
        config.enable_nmt = False
        config.nmt = None
        
        # Convert to OmegaConf for NeMo
        self.cfg = OmegaConf.structured(config)
        
        print(f"Building pipeline with config:")
        print(f"  Model: {self.cfg.asr.model_name}")
        print(f"  Pipeline type: {self.cfg.pipeline_type}")
        print(f"  ASR decoding type: {self.cfg.asr_decoding_type}")
        print(f"  Attention context size: {self.cfg.streaming.att_context_size}")
        print(f"  Batch size: {self.cfg.streaming.batch_size}")
        
        # Build the pipeline using PipelineBuilder
        self.pipeline = StreamingPipelineBuilder.build_pipeline(self.cfg)
        
        print("Pipeline loaded successfully!")
        
        # Warm up with test audio using streaming (3x for thorough warmup)
        print("Warming up GPU with streaming inference (3 iterations)...")
        AUDIO_URL = "https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/mono_44100/156550__acclivity__a-dream-within-a-dream.wav"
        audio_bytes = preprocess_audio(AUDIO_URL, target_sample_rate=16000)
        
        # Initialize streaming support
        print("Initializing streaming request generator...")
        self.pipeline.init_streaming_request_generator()
        print("Streaming initialized!")
        
        # Run 3 streaming warmup iterations
        for warmup_iter in range(1, 4):
            print(f"\n🔥 Warmup iteration {warmup_iter}/3...")
            options = ASRRequestOptions()
            options.asr_output_granularity = "word"
            stream_id = self.pipeline.open_streaming_session(options=options)
            print(f"   Opened stream {stream_id}")
            
            # Calculate chunk size based on streaming config
            # Use 80ms chunks (1280 samples at 16kHz)
            chunk_duration = 0.080  # seconds
            chunk_size = int(chunk_duration * 16000 * 2)  # 16kHz, 16-bit (2 bytes) = 2560 bytes
            
            step_num = 0
            full_streaming_transcript = ""
            
            # Stream the audio in chunks
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                
                # Append to stream
                self.pipeline.append_streaming_audio(stream_id, chunk)
                
                # Process any ready frames
                import asyncio
                requests, outputs = await self.pipeline.process_streaming_batch()
                
                if outputs: 
                    for output in outputs:
                        if output.stream_id == stream_id:
                            if output.final_transcript:
                                full_streaming_transcript += output.final_transcript
                
                step_num += 1
            
            # Close the stream and process final frames
            self.pipeline.close_streaming_session(stream_id)
            
            # Try to get any remaining outputs
            try:
                remaining_requests, remaining_outputs = await self.pipeline.process_streaming_batch()
                for output in remaining_outputs:
                    if output.stream_id == stream_id:
                        if output.final_transcript:
                            full_streaming_transcript += output.final_transcript
            except:
                pass
            
            print(f"   ✅ Iteration {warmup_iter} complete: '{full_streaming_transcript[:50]}...' ({step_num} chunks)")
            
            # Clean up this warmup iteration
            if stream_id in self.pipeline._state_pool:
                self.pipeline.delete_state(stream_id)
            if stream_id in self.pipeline._streaming_request_generator.streams:
                self.pipeline._streaming_request_generator.streams.pop(stream_id, None)
        
        # Final verification after all warmup iterations
        print(f"\n🎉 All warmup iterations complete!")
        
        # Set timing reference point (after warmup)
        self.start_time = time.perf_counter()
        
        # Store exhaustive batching config
        self.exhaustive_batching = self.cfg.streaming.exhaustive_batching
        print(f"Exhaustive batching: {'ENABLED' if self.exhaustive_batching else 'DISABLED'}")
        
        # Initialize async primitives for multi-client support
        self.client_queues_lock = asyncio.Lock()
        self.stream_timestamps_lock = asyncio.Lock()
        self.stream_format_lock = asyncio.Lock()
        self.streams_cleanup_lock = asyncio.Lock()
        
        # Setup FastAPI WebSocket server
        self.web_app = FastAPI()
        
        # Register WebSocket handler
        @self.web_app.websocket("/ws")
        async def run_with_websocket(ws: WebSocket):
            """
            Multi-client WebSocket endpoint for real-time streaming transcription.
            Uses centralized inference loop for efficient batching across all clients.
            """
            
            # Accept WebSocket connection FIRST
            await ws.accept()
            print(f"✅ WebSocket accepted")
            
            stream_id = None  # Track for cleanup
            try:
                # Start centralized inference loop if not already running
                self.start_inference_loop_if_needed()
                
                # Open streaming session for this client
                stream_id = self.pipeline.open_streaming_session()
                print(f"✅ Opened stream_id={stream_id}")
                
                transcription_queue = await self.register_client(stream_id)
                elapsed = time.perf_counter() - self.start_time
                print(f"[+{elapsed:7.3f}s] Client {stream_id} connected ({len(self.client_queues)} total)")
            except Exception as e:
                print(f"❌ Error during connection setup: {e}")
                import traceback
                traceback.print_exc()
                if stream_id is not None:
                    await self.unregister_client(stream_id)
                if ws and ws.application_state is WebSocketState.CONNECTED:
                    await ws.close()
                raise
            
            async def recv_loop(ws, stream_id, transcription_queue):
                """
                Receive audio chunks and append to stream buffer
                """

                chunk_size = self.cfg.streaming.att_context_size[1] + 1
                audio_buffer = b""  # Accumulate chunks here as bytes
                num_buffer_samples = 0
                
                elapsed = time.perf_counter() - self.start_time
                print(f"[+{elapsed:7.3f}s] Stream {stream_id}: Micro-batching enabled (batch_size={chunk_size})")
                
                try:
                    while True:
                        # EAGER MESSAGE DRAINING: Collect all available messages before processing
                        # This reduces per-message overhead by batching WebSocket receives
                        messages = []
                        
                        # [1] First message (blocking) - always wait for at least one
                        first_message = await ws.receive()
                        messages.append(first_message)
                        
                        # [2] Drain any additional messages with timeout (non-blocking)
                        # This grabs messages that arrived during processing of previous batch
                        DRAIN_TIMEOUT_MS = 1  # 1ms timeout
                        while True:
                            try:
                                # Try to get more messages with timeout
                                next_message = await asyncio.wait_for(
                                    ws.receive(), 
                                    timeout=DRAIN_TIMEOUT_MS / 1000
                                )
                                messages.append(next_message)
                            except asyncio.TimeoutError:
                                # No more messages available, proceed with what we have
                                break
                        
                        stream_ended = False
                        
                        # Process all messages - support both msgpack and raw bytes
                        for message in messages:
                            # Check for text messages (marker signal from raw bytes client)
                            if "text" in message:
                                text_msg = message["text"]
                                if text_msg == "END" or text_msg == "MARKER":
                                    # Track that this stream uses raw bytes (not msgpack)
                                    async with self.stream_format_lock:
                                        if stream_id not in self.stream_uses_msgpack:
                                            self.stream_uses_msgpack[stream_id] = False
                                            elapsed = time.perf_counter() - self.start_time
                                            print(f"[+{elapsed:7.3f}s] Stream {stream_id}: Detected format = RAW BYTES (text marker)")
                                    
                                    elapsed = time.perf_counter() - self.start_time
                                    print(f"[+{elapsed:7.3f}s] Stream {stream_id}: Received text Marker '{text_msg}'")
                                    await transcription_queue.put({
                                        'output': None,
                                        'audio_timestamp': 0.0,
                                        'is_marker': True
                                    })
                                    stream_ended = True
                                    break
                            
                            # Check for binary messages (audio data or msgpack)
                            if "bytes" in message:
                                raw_bytes = message["bytes"]
                                is_msgpack = False
                                
                                # Ensure raw_bytes is actually bytes, not str
                                if isinstance(raw_bytes, str):
                                    # Convert string to bytes if needed (shouldn't happen with binary WebSocket)
                                    raw_bytes = raw_bytes.encode('latin-1')
                                
                                # Try to decode as msgpack first (for backwards compatibility)
                                is_msgpack_data = False
                                try:
                                    # Try msgpack decode
                                    data = msgpack.unpackb(raw_bytes, raw=False)
                                    msg_type = data.get("type")
                                    
                                    # Successfully decoded as msgpack
                                    is_msgpack_data = True
                                    
                                    # Track that this stream uses msgpack (on first message)
                                    async with self.stream_format_lock:
                                        if stream_id not in self.stream_uses_msgpack:
                                            self.stream_uses_msgpack[stream_id] = True
                                            elapsed = time.perf_counter() - self.start_time
                                            print(f"[+{elapsed:7.3f}s] Stream {stream_id}: Detected format = MSGPACK")
                                    
                                    # Handle msgpack Marker
                                    if msg_type == "Marker":
                                        elapsed = time.perf_counter() - self.start_time
                                        print(f"[+{elapsed:7.3f}s] Stream {stream_id}: Received msgpack Marker")
                                        await transcription_queue.put({
                                            'output': None,
                                            'audio_timestamp': 0.0,
                                            'is_marker': True
                                        })
                                        stream_ended = True
                                        break
                                    
                                    # Handle msgpack Audio
                                    if msg_type == "Audio":
                                        pcm_bytes = data["pcm_bytes"]
                                        # Accumulate in buffer
                                        audio_buffer += pcm_bytes
                                        num_buffer_samples += len(pcm_bytes)
                                
                                except:
                                    # Silently ignore msgpack decode errors - just means it's raw bytes
                                    pass
                                
                                # If not msgpack, treat as raw PCM bytes
                                if not is_msgpack_data:
                                    # Track that this stream uses raw bytes (on first message)
                                    async with self.stream_format_lock:
                                        if stream_id not in self.stream_uses_msgpack:
                                            self.stream_uses_msgpack[stream_id] = False
                                            elapsed = time.perf_counter() - self.start_time
                                            print(f"[+{elapsed:7.3f}s] Stream {stream_id}: Detected format = RAW BYTES")
                                    
                                    # This is the simple path for frontends without msgpack
                                    # raw_bytes should already be bytes type
                                    audio_buffer += raw_bytes
                                    num_buffer_samples += len(raw_bytes)
                        
                        # After processing all messages in batch, check if buffer should be flushed
                        if num_buffer_samples >= chunk_size * 1280 * 2:  # samples per frame
                            
                            # Pipeline append (now with batched data)  
                            self.pipeline.append_streaming_audio(stream_id, audio_buffer)
                            
                            # Clear buffer and reset sample counter
                            audio_buffer = b""
                            num_buffer_samples = 0
                            
                            # Yield to event loop after batch append
                            await asyncio.sleep(0)
                        
                        # Check if stream ended and break outer loop
                        if stream_ended:
                            break
                        
                except WebSocketDisconnect:
                    pass  # Normal disconnection
                except Exception as e:
                    # Safely print error without trying to decode binary data
                    try:
                        error_msg = str(e)
                    except:
                        error_msg = repr(e)
                    print(f"❌ recv_loop error stream {stream_id}: {error_msg}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # FLUSH REMAINING BUFFERED CHUNKS
                    if audio_buffer:
                        elapsed = time.perf_counter() - self.start_time
                        print(f"[+{elapsed:7.3f}s] Stream {stream_id}: Flushing {len(audio_buffer)} remaining chunks from buffer")
                        try:
                            np_data = np.frombuffer(audio_buffer, dtype=np.int16)
                            np_data = np_data.astype(np.float32) / 32768.0
                            torch_data = torch.from_numpy(np_data)
                            self.pipeline.append_streaming_audio(stream_id, torch_data)
                            audio_buffer = b""
                        except Exception as e:
                            print(f"[+{elapsed:7.3f}s] ⚠️  Error flushing buffer for stream {stream_id}: {e}")
                    
                    # Mark stream as ended so final frames can be processed
                    try:
                        self.pipeline._streaming_request_generator.streams[stream_id].mark_end()
                    except Exception as e:
                        print(f"⚠️  Could not mark stream {stream_id} as ended: {e}")
                    
                    # Wait for any remaining frames to be processed
                    # This ensures we don't miss the last frames in the buffer
                    await asyncio.sleep(1.0)  # 1 second buffer to process remaining frames
                    
                    # Signal send_loop to finish if no marker was sent
                    # (This handles backwards compatibility with clients that don't send markers)
                    await transcription_queue.put(None)
            
            async def send_loop(ws, transcription_queue, stream_id):
                """
                Send transcription results from centralized inference loop to client
                """
                step_num = 0
                
                try:
                    while True:
                        output_dict = await transcription_queue.get()
                        
                        if output_dict is None:  # Shutdown signal
                            break
                        
                        # Check client's message format for EACH message (format may be detected after first audio)
                        async with self.stream_format_lock:
                            uses_msgpack = self.stream_uses_msgpack.get(stream_id, True)
                        
                        # Check for Marker echo
                        if output_dict.get('is_marker'):
                            elapsed = time.perf_counter() - self.start_time
                            print(f"[+{elapsed:7.3f}s] Stream {stream_id}: Echoing Marker back to client (format={'msgpack' if uses_msgpack else 'text'})")
                            
                            # Send marker in client's format
                            if uses_msgpack:
                                # Msgpack client - send msgpack Marker
                                marker_msg = {"type": "Marker", "id": -1}
                                marker_bytes = msgpack.packb(marker_msg, use_bin_type=True)
                                await ws.send_bytes(marker_bytes)
                            else:
                                # Raw bytes client - send text marker
                                await ws.send_text("END")
                            
                            # Yield to event loop
                            await asyncio.sleep(0)
                            break  # Client will close connection after receiving Marker
                        
                        # Send transcription in client's preferred format
                        if uses_msgpack:
                            # Msgpack client - send pre-encoded msgpack bytes
                            msgpack_bytes = output_dict.get('msgpack_bytes')
                            if msgpack_bytes:
                                await ws.send_bytes(msgpack_bytes)
                        else:
                            # Raw bytes client - send JSON text
                            json_str = output_dict.get('json_str')
                            if json_str:
                                if step_num == 0:
                                    elapsed = time.perf_counter() - self.start_time
                                    print(f"[+{elapsed:7.3f}s] Stream {stream_id}: Sending first JSON text message: {json_str[:100]}")
                                await ws.send_text(json_str)
                        
                        step_num += 1
                        
                except Exception as e:
                    elapsed = time.perf_counter() - self.start_time
                    print(f"[+{elapsed:7.3f}s] ❌ send_loop error stream {stream_id}: {e}")
                finally:
                    # Connection will be closed by client after Marker echo
                    # No need to send additional signals
                    pass
            
            # WebSocket already accepted at the top
            tasks = []
            try:
                tasks = [
                    asyncio.create_task(recv_loop(ws, stream_id, transcription_queue)),
                    asyncio.create_task(send_loop(ws, transcription_queue, stream_id)),
                ]

                # Send ready message as JSON text (works for all clients)
                # We don't know the client's format yet, so use JSON which is universally supported
                ready_msg = {"type": "Ready", "id": stream_id}
                await ws.send_text(json.dumps(ready_msg))
                
                # Wait for both tasks to complete
                await asyncio.gather(*tasks, return_exceptions=True)
                
            except Exception as e:
                print(f"❌ WebSocket error for stream {stream_id}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Cleanup
                await self.unregister_client(stream_id)
                if ws and ws.application_state is WebSocketState.CONNECTED:
                    await ws.close()
                # Cancel any remaining tasks
                for task in tasks:
                    if not task.done():
                        try:
                            task.cancel()
                            await task
                        except asyncio.CancelledError:
                            pass

        def start_server():
            uvicorn.run(self.web_app, host="0.0.0.0", port=8000)

        self.server_thread = threading.Thread(target=start_server, daemon=True)
        self.server_thread.start()

        self.tunnel_ctx = modal.forward(8000)
        self.tunnel = self.tunnel_ctx.__enter__()
        self.websocket_url = self.tunnel.url.replace("https://", "wss://") + "/ws"
        print(f"Websocket URL: {self.websocket_url}")
    
    async def register_client(self, stream_id: int) -> asyncio.Queue:
        """
        Register a new client and return their transcription queue
        Args:
            stream_id (int): The stream ID for this client
        Returns:
            asyncio.Queue: The transcription queue for this client
        """
        queue = asyncio.Queue()
        async with self.client_queues_lock:
            self.client_queues[stream_id] = queue
        async with self.stream_timestamps_lock:
            self.stream_audio_timestamps[stream_id] = 0.0  # Initialize audio timestamp
        return queue
    
    async def unregister_client(self, stream_id: int):
        """
        Remove client from tracking and mark stream for cleanup.
        Actual state cleanup happens in inference loop after all buffered frames are processed.
        Args:
            stream_id (int): The stream ID to remove
        """
        try:
            async with self.client_queues_lock:
                self.client_queues.pop(stream_id, None)
            async with self.stream_timestamps_lock:
                self.stream_audio_timestamps.pop(stream_id, None)  # Clean up timestamp tracking
            async with self.stream_format_lock:
                self.stream_uses_msgpack.pop(stream_id, None)  # Clean up format tracking
            
            # Mark the stream as ended (stops accepting new audio)
            if hasattr(self.pipeline, '_streaming_request_generator'):
                self.pipeline._streaming_request_generator.close_stream(stream_id)
            
            # Mark stream for cleanup (actual cleanup happens after frames are drained)
            async with self.streams_cleanup_lock:
                self.streams_pending_cleanup.add(stream_id)
            
            elapsed = time.perf_counter() - self.start_time
            print(f"[+{elapsed:7.3f}s] Client {stream_id} disconnected, marked for cleanup ({len(self.client_queues)} remaining)")
        except Exception as e:
            elapsed = time.perf_counter() - self.start_time
            print(f"[+{elapsed:7.3f}s] ❌ Unregister error stream {stream_id}: {e}")
    
    async def route_outputs(self, outputs_with_timestamps):
        """
        Route outputs to client-specific queues in parallel without holding lock during I/O
        
        Args:
            outputs_with_timestamps: List of dicts with 'output' and 'audio_timestamp' keys
        """
        # BATCH ENCODING: Encode each unique output in BOTH formats
        # Then send the appropriate format to each client
        current_time = time.time()
        import json
        
        for output_dict in outputs_with_timestamps:
            output = output_dict['output']
            audio_timestamp = output_dict['audio_timestamp']
            
            # Check if output has transcription text to send
            if output.partial_transcript:
                # Build result dict
                result = {
                    "text": output.current_step_transcript,
                    "timestamp": current_time,
                    "audio_timestamp": audio_timestamp,
                    "is_final": False,
                }
                if len(output.final_segments) > 0:
                    result["segment_text"] = output.final_segments[0].text
                    result["segment_start_time"] = output.final_segments[0].start
                    result["segment_end_time"] = output.final_segments[0].end
                
                # Encode in BOTH formats (msgpack for old clients, JSON for new)
                output_dict['msgpack_bytes'] = msgpack.packb(result, use_bin_type=True)
                output_dict['json_str'] = json.dumps(result)
            else:
                result = {
                    "text": output.final_transcript,
                    "timestamp": current_time,
                    "audio_timestamp": audio_timestamp,
                    "is_final": True,
                }
                if len(output.final_segments) > 0:
                    result["segment_text"] = output.final_segments[0].text
                    result["segment_start_time"] = output.final_segments[0].start
                    result["segment_end_time"] = output.final_segments[0].end
                
                # Encode in BOTH formats
                output_dict['msgpack_bytes'] = msgpack.packb(result, use_bin_type=True)
                output_dict['json_str'] = json.dumps(result)
        
        # Quick snapshot under lock
        async with self.client_queues_lock:
            queues_snapshot = dict(self.client_queues)
        
        # Build queue_map from snapshot WITHOUT holding lock
        queue_map = {}  # stream_id -> (list of output_dicts, queue)
        for output_dict in outputs_with_timestamps:
            output = output_dict['output']
            queue = queues_snapshot.get(output.stream_id)
            if queue:
                if output.stream_id not in queue_map:
                    queue_map[output.stream_id] = ([], queue)
                queue_map[output.stream_id][0].append(output_dict)
            else:
                elapsed = time.perf_counter() - self.start_time
                print(f"[+{elapsed:7.3f}s] ⚠️ No queue for stream {output.stream_id}")
        
        # Now route outputs in parallel WITHOUT holding the lock
        async def send_to_queue(stream_id, output_dicts, queue):
            """
            Send all outputs for this stream to its queue
            """
            for output_dict in output_dicts:
                try:
                    await queue.put(output_dict)
                except Exception as e:
                    elapsed = time.perf_counter() - self.start_time
                    print(f"[+{elapsed:7.3f}s] ❌ Route error stream {stream_id}: {e}")
        
        # Send all outputs for each stream in parallel
        if queue_map:
            await asyncio.gather(*[
                send_to_queue(stream_id, output_dicts, queue) 
                for stream_id, (output_dicts, queue) in queue_map.items()
            ], return_exceptions=True)
    
    async def centralized_inference_loop(self):
        """
        Main inference loop.
        """
        self.inference_running = True
        
        # Metrics tracking
        batch_count = 0
        batch_stats = {
            'solo_batches': 0,
            'partial_batches': 0,
            'full_batches': 0,
        }
        
        # TIMING INSTRUMENTATION for inference loop
        timing_stats = {
            'process_batch': [],
            'route_outputs': [],
            'total_cycle': [],
        }
        batch_size_stats = []
        TIMING_REPORT_INTERVAL = 50  # Report every N cycles

        
        while self.inference_running:
            try:
                active_clients = len(self.client_queues)
                
                # Idle if no clients
                if active_clients == 0:
                    await asyncio.sleep(0.01)
                    continue
                
                # [TIMING] Start of cycle
                cycle_start = time.perf_counter()
                
                await asyncio.sleep(self.cfg.streaming.batching_delay_secs)
                
                # Process streaming batch(es) - exhaustive or single batch mode
                try:
                    all_outputs_with_timestamps = []
                    batches_in_cycle = []
                    
                    # Snapshot timestamps at start of cycle
                    async with self.stream_timestamps_lock:
                        cycle_timestamps = dict(self.stream_audio_timestamps)
                    
                    if self.exhaustive_batching:
                        # EXHAUSTIVE BATCHING: Process ALL ready frames across multiple batches
                        first_batch = True
                        while True:
                            try:
                                batch_start = time.perf_counter()
                                requests, outputs = await self.pipeline.process_streaming_batch()
                                batch_end = time.perf_counter()
                                timing_stats['process_batch'].append((batch_end - batch_start) * 1000)
                                
                                # Track batch size
                                if outputs:
                                    batch_size_stats.append(len(outputs))
                                
                                await asyncio.sleep(0)
                                
                                if not outputs:
                                    break  # No more frames ready
                                
                                first_batch = False
                                
                                # Track this batch
                                batch_info = {
                                    'size': len(outputs),
                                    'streams': [out.stream_id for out in outputs]
                                }
                                batches_in_cycle.append(batch_info)

                                # Now increment cycle timestamps for this batch
                                for request in requests:
                                    if hasattr(request, 'length'):  # It's a Frame
                                        stream_id = request.stream_id
                                        duration_secs = request.length / 16000.0  # 16kHz sample rate
                                        cycle_timestamps[stream_id] = cycle_timestamps.get(stream_id, 0.0) + duration_secs
                                
                                # Each output gets the current timestamp for its stream
                                for output in outputs:
                                    current_timestamp = cycle_timestamps.get(output.stream_id, 0.0)
                                    
                                    output_dict = {
                                        'output': output,
                                        'audio_timestamp': current_timestamp,
                                    }
                                    all_outputs_with_timestamps.append(output_dict)
                                
                                
                                
                            except Exception as e:
                                if type(e).__name__ == 'NotEnoughDataException':
                                    # No data ready yet
                                    if first_batch:
                                        # First attempt in cycle had no data - brief poll delay
                                        await asyncio.sleep(0.001)  # 1ms poll
                                    break  # All streams exhausted (or none ready yet)
                                raise
                    else:
                        # SINGLE BATCH MODE: Process one batch per cycle (default)
                        try:
                            # [TIMING] Process batch with GPU operations
                            batch_start = time.perf_counter()
                            requests, outputs = await self.pipeline.process_streaming_batch()
                            batch_end = time.perf_counter()
                            timing_stats['process_batch'].append((batch_end - batch_start) * 1000)
                            
                            # Track batch size
                            if outputs:
                                batch_size_stats.append(len(outputs))
                            
                            # Yield to event loop after batch to prevent blocking
                            await asyncio.sleep(0)

                            # Increment cycle timestamps for this batch
                            for request in requests:
                                if hasattr(request, 'length'):  # It's a Frame
                                    stream_id = request.stream_id
                                    duration_secs = request.length / 16000.0  # 16kHz sample rate
                                    cycle_timestamps[stream_id] = cycle_timestamps.get(stream_id, 0.0) + duration_secs
                            
                            if outputs:
                                # Track this batch
                                batch_info = {
                                    'size': len(outputs),
                                    'streams': [out.stream_id for out in outputs]
                                }
                                batches_in_cycle.append(batch_info)
                                
                                for output in outputs:
                                    current_timestamp = cycle_timestamps.get(output.stream_id, 0.0)
                                    
                                    output_dict = {
                                        'output': output,
                                        'audio_timestamp': current_timestamp,
                                    }
                                    all_outputs_with_timestamps.append(output_dict)
                                
                                
                            
                        except Exception as e:
                            if type(e).__name__ == 'NotEnoughDataException':
                                # No data ready yet - brief poll delay
                                await asyncio.sleep(0.001)  # 1ms poll
                            else:
                                raise
                    
                    # Now update global timestamps with final cycle values
                    if all_outputs_with_timestamps:
                        async with self.stream_timestamps_lock:
                            # Update global timestamps to final values from this cycle
                            for stream_id in cycle_timestamps:
                                if stream_id in self.stream_audio_timestamps:
                                    self.stream_audio_timestamps[stream_id] = cycle_timestamps[stream_id]
                        
                        # Log the cycle (all batches processed)
                        elapsed = time.perf_counter() - self.start_time
                        total_outputs = len(all_outputs_with_timestamps)
                        unique_streams = len(set(out_dict['output'].stream_id for out_dict in all_outputs_with_timestamps))
                        num_batches = len(batches_in_cycle)

                        # Route all outputs 
                        route_start = time.perf_counter()
                        await self.route_outputs(all_outputs_with_timestamps)
                        route_end = time.perf_counter()
                        timing_stats['route_outputs'].append((route_end - route_start) * 1000)
                        
                        batch_count += 1
                        
                        # [TIMING] End of cycle
                        cycle_end = time.perf_counter()
                        timing_stats['total_cycle'].append((cycle_end - cycle_start) * 1000)
                        
                        # Categorize the cycle based on coverage
                        # Full = all active clients had at least one output
                        # Partial = some but not all
                        # Solo = only one stream
                        if unique_streams == 1:
                            batch_stats['solo_batches'] += 1
                        elif unique_streams >= active_clients:
                            batch_stats['full_batches'] += 1
                        else:
                            batch_stats['partial_batches'] += 1
                        
                        # DEFERRED CLEANUP: Check if any streams pending cleanup are fully drained
                        await self._cleanup_finished_streams()
                        
                        # [TIMING] Report detailed timing statistics periodically
                        if batch_count % TIMING_REPORT_INTERVAL == 0:
                            elapsed = time.perf_counter() - self.start_time
                            mode = "EXHAUSTIVE" if self.exhaustive_batching else "SINGLE-BATCH"
                            print(f"[+{elapsed:7.3f}s] ⏱️  Inference Loop @ {batch_count} cycles ({active_clients} clients) [{mode}]:")
                            
                            # Report timing stats
                            for name, times in timing_stats.items():
                                if times:
                                    avg = sum(times) / len(times)
                                    min_t = min(times)
                                    max_t = max(times)
                                    p95 = sorted(times)[int(len(times) * 0.95)] if len(times) > 20 else max_t
                                    total = sum(times)
                                    print(f"         {name}: avg={avg:.3f}ms min={min_t:.3f}ms max={max_t:.3f}ms p95={p95:.3f}ms total={total:.1f}ms")
                            
                            # Report batch size stats
                            if batch_size_stats:
                                avg_batch = sum(batch_size_stats) / len(batch_size_stats)
                                min_batch = min(batch_size_stats)
                                max_batch = max(batch_size_stats)
                                print(f"         batch_sizes: avg={avg_batch:.1f} min={min_batch} max={max_batch} (total_batches={len(batch_size_stats)})")
                            
                            # Report batching efficiency
                            total = sum(batch_stats.values())
                            if total > 0:
                                full_pct = (batch_stats['full_batches'] / total) * 100
                                partial_pct = (batch_stats['partial_batches'] / total) * 100
                                solo_pct = (batch_stats['solo_batches'] / total) * 100
                                print(f"         efficiency: Full={full_pct:.0f}% Partial={partial_pct:.0f}% Solo={solo_pct:.0f}%")
                            
                            # Clear stats for next interval
                            timing_stats = {key: [] for key in timing_stats.keys()}
                            batch_size_stats = []
                
                except Exception as e:
                    if type(e).__name__ != 'NotEnoughDataException':
                        print(f"❌ Error in inference: {e}")
                        import traceback
                        traceback.print_exc()
                    # No data ready yet, continue loop
                
            except Exception as e:
                print(f"❌ Error in centralized inference loop: {e}")
                print(f"   Current clients: {len(self.client_queues)}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(0.1)  # Back off on errors
        
        print("⚠️  Centralized inference loop stopped!")
    
    async def _cleanup_finished_streams(self):
        """
        Clean up streams that are marked for cleanup AND fully drained from the request generator.
        This ensures we don't delete state while frames are still being processed.
        """
        async with self.streams_cleanup_lock:
            if not self.streams_pending_cleanup:
                return  # Nothing to clean up
            
            # Get list of streams still in the request generator
            active_stream_ids = set(self.pipeline._streaming_request_generator.streams.keys())
            
            # Find streams that are pending cleanup AND no longer in request generator
            streams_to_cleanup = self.streams_pending_cleanup - active_stream_ids
            
            if streams_to_cleanup:
                elapsed = time.perf_counter() - self.start_time
                for stream_id in streams_to_cleanup:
                    try:
                        # Now safe to delete state - no more frames will arrive
                        if stream_id in self.pipeline._state_pool:
                            del self.pipeline._state_pool[stream_id]
                            print(f"[+{elapsed:7.3f}s]    🧹 Cleaned up state_pool for stream {stream_id}")
                        
                        # Reset context manager cache slot
                        if hasattr(self.pipeline, 'context_manager') and self.pipeline.context_manager is not None:
                            try:
                                self.pipeline.context_manager.reset_slots([stream_id], [True])
                                print(f"[+{elapsed:7.3f}s]    🧹 Reset context_manager cache slot for stream {stream_id}")
                            except Exception as e:
                                print(f"[+{elapsed:7.3f}s]    ⚠️  Could not reset context_manager for stream {stream_id}: {e}")
                        
                        # Remove from pending cleanup
                        self.streams_pending_cleanup.discard(stream_id)
                        
                    except Exception as e:
                        print(f"[+{elapsed:7.3f}s] ❌ Error cleaning up stream {stream_id}: {e}")
    
    def start_inference_loop_if_needed(self):
        """
        Start the centralized inference loop if it's not already running.
        Called on first client connection.
        """
        if self.inference_task is None:
            print("🔄 Starting centralized inference loop (first client)...")
            self.inference_task = asyncio.create_task(self.centralized_inference_loop())
            print("✅ Centralized inference loop started!")
        elif self.inference_task.done():
            print("⚠️  Inference task was done! Restarting...")
            # Check if it had an exception
            try:
                exc = self.inference_task.exception()
                if exc:
                    print(f"❌ Previous task failed with: {exc}")
            except:
                pass
            self.inference_task = asyncio.create_task(self.centralized_inference_loop())
            print("✅ Centralized inference loop restarted!")
        else:
            pass  # Already running

    
    @modal.asgi_app()
    def webapp(self):
        """Expose the FastAPI WebSocket app"""
        return self.web_app

    @modal.method()
    def get_config_dict(self) -> dict:
        """
        Get a serializable dictionary representation of the pipeline configuration.
        
        Returns:
            Dictionary containing all configuration values
        """
        # Convert OmegaConf to regular dict for serialization
        config_dict = OmegaConf.to_container(self.cfg, resolve=True)
        return config_dict
    
    @modal.method()
    def transcribe_file(self, audio_url: str) -> dict:
        """
        Transcribe an audio file from a URL or local path
        
        Args:
            audio_url: URL or path to audio file
            
        Returns:
            Dictionary with transcription results
        """
        print(f"Transcribing audio from: {audio_url}")
        
        # Preprocess audio to ensure it's 16kHz mono
        start_time = time.perf_counter()
        audio_bytes = preprocess_audio(audio_url, target_sample_rate=16000)
        preprocess_time = time.perf_counter() - start_time
        print(f"Audio preprocessing took {preprocess_time:.2f} seconds")
        
        # Write to temp file for pipeline
        import tempfile
        import wave
        import os
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            with wave.open(tmp, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_bytes)
            tmp_path = tmp.name
        
        # Calculate audio duration before running pipeline
        data_dur = len(audio_bytes) / (16000 * 2)  # 16kHz, 16-bit (2 bytes per sample)
        
        # Run pipeline
        start_time = time.perf_counter()
        progress_bar = TQDMProgressBar()
        output = self.pipeline.run([tmp_path], progress_bar=progress_bar)
        inference_time = time.perf_counter() - start_time
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        result = {
            "text": output[0]["text"],
            "inference_time": inference_time,
            "audio_duration": data_dur,
        }
        
        print(f"Transcription complete!")
        print(f"  Text: {result['text']}")
        print(f"  Audio duration: {data_dur:.2f}s")
        print(f"  Inference time: {inference_time:.2f}s")
        
        return result


# ## Frontend Service
#
# We serve a simple HTML/JS frontend to interact with the transcriber.
# The frontend captures microphone input and streams it to the WebSocket endpoint.

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi")
    .add_local_dir(Path(__file__).parent.parent  / "nemotron-asr-frontend", "/root/frontend")
)

with web_image.imports():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, Response
    from fastapi.staticfiles import StaticFiles


@app.cls(image=web_image)
class WebServer:
    @modal.asgi_app()
    def web(self):
        web_app = FastAPI()
        web_app.mount("/static", StaticFiles(directory="/root/frontend"), name="static")

        @web_app.get("/status")
        async def status():
            return Response(status_code=200)

        # Serve frontend
        @web_app.get("/")
        async def index():
            html_content = open("/root/frontend/index.html").read()

            # Get the WebSocket URL from the NemotronASR
            cls_instance = NemotronASR()
            ws_base_url = (
                cls_instance.webapp.web_url.replace("http", "ws") + "/ws"
            )
            script_tag = f'<script>window.WS_BASE_URL = "{ws_base_url}";</script>'
            html_content = html_content.replace(
                '<script src="/static/cache-aware-stt.js"></script>',
                f'{script_tag}\n    <script src="/static/cache-aware-stt.js"></script>',
            )
            return HTMLResponse(content=html_content)

        return web_app


@app.local_entrypoint()
def main():
    """Test the pipeline with a sample audio file"""
    print("Starting cache-aware RNNT pipeline test...")
    
    # Use the warmup audio URL for testing
    audio_url = "https://github.com/voxserv/audio_quality_testing_samples/raw/refs/heads/master/mono_44100/156550__acclivity__a-dream-within-a-dream.wav"
    
    runner = NemotronASR()
    result = runner.transcribe_file.remote(audio_url)
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Transcription: {result['text']}")
    print(f"Audio Duration: {result['audio_duration']:.2f}s")
    print(f"Inference Time: {result['inference_time']:.2f}s")
    print(f"RTFX: {result['rtfx']:.2f}x")
    print("="*80)

