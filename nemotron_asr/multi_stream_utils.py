# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Callable, Iterator

import torch
import numpy as np

from nemo.collections.asr.inference.streaming.buffering.audio_bufferer import BatchedAudioBufferer
from nemo.collections.asr.inference.streaming.framing.mono_stream import MonoStream
from nemo.collections.asr.inference.streaming.framing.request import FeatureBuffer, Frame, Request, RequestOptions
from nemo.collections.asr.inference.streaming.framing.stream import Stream
from nemo.collections.asr.inference.utils.enums import RequestType
from nemo.collections.asr.inference.utils.progressbar import ProgressBar


class NotEnoughDataException(Exception):
    """Exception raised when a stream doesn't have enough data to produce a frame"""
    pass


class StreamingMonoStream(Stream):
    """
    Streamer for real-time audio chunks arriving incrementally.
    Unlike MonoStream which loads a complete file, this accepts audio chunks
    via append_audio() and yields frames as data becomes available.
    
    OPTIMIZATION: Audio format conversion (bytes→torch) is deferred until extraction
    in __next__() (inference loop context), keeping append_audio() lightweight for
    recv_loop tasks. Converted chunks are cached to avoid re-conversion.
    """

    def __init__(self, rate: int, frame_size_in_secs: float, stream_id: int, pad_last_frame: bool = False,
                 max_buffer_seconds: float = 1.0):
        """
        Initialize the StreamingMonoStream
        Args:
            rate (int): sampling rate
            frame_size_in_secs (float): frame length in seconds
            stream_id (int): stream id
            pad_last_frame (bool): whether to pad the last frame
            max_buffer_seconds (float): max buffered audio before dropping old frames (default 1.0s)
        """
        self.rate = rate
        self.frame_size = int(frame_size_in_secs * rate)
        self.pad_last_frame = pad_last_frame
        self.max_buffer_samples = int(max_buffer_seconds * rate)

        # Audio buffer optimization: keep chunks in a list and lazily concatenate
        # This avoids O(n) torch.cat() on every append, making appends O(1)
        self.audio_chunks = []  # List of torch tensors
        # self.audio_chunks = b"" # bytes
        self.chunk_sizes = []  # Size of each chunk for efficient chunk-level tracking
        self.total_samples = 0  # Total samples across all chunks
        self.chunk_offset = 0  # How many chunks have been fully consumed
        self.position_in_chunk = 0  # Position within the first unconsumed chunk
        self.pin_memory = torch.cuda.is_available()
        self.is_ended = False
        self.frame_count = 0
        self.options = None
        # NO LOCK NEEDED: We run in single-threaded asyncio event loop
        # - append_audio() called from async tasks (event loop serializes)
        # - __next__() blocks event loop while running (no concurrent access possible)
        # - threading.Lock() was causing GIL contention for no benefit!
        self.dropped_frames = 0  # Track dropped frames for monitoring
        
        super().__init__(stream_id)

    def set_options(self, options: RequestOptions | None = None) -> None:
        """
        Set the options for this stream
        Args:
            options (RequestOptions | None): optional options for the request
        """
        self.options = options

    def append_audio(self, samples: torch.Tensor | np.ndarray) -> None:
        """
        Append audio samples to the buffer
        Args:
            samples (torch.Tensor | np.ndarray): audio samples to append
        
        Note: No locking needed - runs in single-threaded event loop
        
        OPTIMIZATION: Keeps bytes as-is (no conversion) until extraction in inference loop.
        This minimizes work in recv_loop tasks and centralizes computation.
        """
        if isinstance(samples, bytes):
            # NEW: Store raw bytes, defer conversion to inference loop (_extract_samples)
            # This keeps recv_loop lightweight (pure I/O)
            self.audio_chunks.append(samples)
            # int16 = 2 bytes per sample
            num_samples = len(samples) // 2
            self.chunk_sizes.append(num_samples)
            self.total_samples += num_samples
        elif isinstance(samples, torch.Tensor):
            # Handle torch tensors (for backwards compatibility)
            # Ensure samples are 1D
            if samples.dim() > 1:
                samples = samples.squeeze()
            chunk_size = len(samples)
            self.audio_chunks.append(samples)
            self.chunk_sizes.append(chunk_size)
            self.total_samples += chunk_size
        else:
            raise ValueError(f"samples must be torch.Tensor or bytes, got {type(samples)}")

    def mark_end(self) -> None:
        """Mark that no more audio will be appended to this stream"""
        self.is_ended = True

    def has_available_frame(self) -> bool:
        """
        Check if there's enough data for the next frame
        Returns:
            bool: True if a frame can be yielded
        """
        remaining_samples = self._get_remaining_samples()
        return remaining_samples >= self.frame_size or (self.is_ended and remaining_samples > 0)

    def get_buffer_size(self) -> int:
        """Get the current size of the audio buffer"""
        return self._get_remaining_samples()
    
    def _get_remaining_samples(self) -> int:
        """Calculate remaining unconsumed samples across all chunks"""
        if len(self.audio_chunks) == 0:
            return 0
        # Sum all unconsumed chunks
        remaining = sum(self.chunk_sizes[self.chunk_offset:])
        # Subtract the position within the first unconsumed chunk
        remaining -= self.position_in_chunk
        return remaining

    def __iter__(self):
        """Returns the frame iterator object"""
        self.chunk_offset = 0
        self.position_in_chunk = 0
        self.frame_count = 0
        return self

    def __next__(self) -> list[Frame]:
        """
        Get the next frame from the stream
        Returns:
            list[Frame]: The next frame in the stream
        Raises:
            NotEnoughDataException: If there's not enough data for a frame yet
            StopIteration: If the stream has ended and all data has been consumed
        
        Note: No locking needed - this blocks the event loop while executing,
        preventing any concurrent access from append_audio()
        """
        remaining_samples = self._get_remaining_samples()
        
        # LAG DETECTION: If buffer is too large, drop old frames to catch up to real-time
        # if remaining_samples > self.max_buffer_samples:
        #     frames_to_drop = (remaining_samples - self.max_buffer_samples) // self.frame_size
        #     if frames_to_drop > 0:
        #         samples_to_drop = frames_to_drop * self.frame_size
        #         self._advance_position(samples_to_drop)
        #         self.dropped_frames += frames_to_drop
        #         remaining_samples = self._get_remaining_samples()
        #         print(f"⚠️  Stream {self.stream_id}: LAG DETECTED! Dropped {frames_to_drop} frames ({frames_to_drop * 0.08:.2f}s) to catch up. Total dropped: {self.dropped_frames}")
        
        # Check if we have any data
        if len(self.audio_chunks) == 0 or self.chunk_offset >= len(self.audio_chunks):
            if self.is_ended:
                raise StopIteration
            else:
                raise NotEnoughDataException(f"Stream {self.stream_id} needs more data")
        
        # Determine what we need to extract
        is_final = False
        samples_to_extract = self.frame_size
        
        # Case 1: Enough data for a full frame
        if remaining_samples >= self.frame_size:
            samples_to_extract = self.frame_size
            is_final = False
            
        # Case 2: Stream ended, return remaining samples
        # elif self.is_ended and remaining_samples > 0:
        #     samples_to_extract = remaining_samples
        #     is_final = True
            
        # Case 3: Stream ended and no data left
        elif self.is_ended:
            raise StopIteration
            
        # Case 4: Not enough data yet, wait for more
        else:
            raise NotEnoughDataException(f"Stream {self.stream_id} needs more data")
        
        # Extract the samples efficiently
        chunk_samples = self._extract_samples(samples_to_extract)
        
        # Pad if needed and this is the final frame
        # if is_final and self.pad_last_frame and len(chunk_samples) < self.frame_size:
        #     padded = torch.zeros(self.frame_size)
        #     padded[:len(chunk_samples)] = chunk_samples
        #     chunk_samples = padded

        # Package the frame
        is_first = self.frame_count == 0
        frame = Frame(
            samples=chunk_samples,
            stream_id=self.stream_id,
            is_first=is_first,
            is_last=is_final,
            length=samples_to_extract,
            options=self.options if is_first else None,
        )

        self.frame_count += 1
        
        # Advance position and clean up consumed chunks immediately
        self._advance_position(samples_to_extract)
        self._cleanup_consumed_chunks()

        return [frame]
    
    def _extract_samples(self, num_samples: int) -> torch.Tensor:
        """
        Extract samples from the buffer, handling chunk boundaries efficiently
        
        OPTIMIZATION: Performs audio format conversion HERE (in inference loop context).
        Converts bytes→numpy→float32→torch only for data being extracted, not on every append.
        Caches converted tensors to avoid re-converting the same bytes.
        
        Args:
            num_samples: Number of samples to extract
        Returns:
            torch.Tensor: Extracted samples
        """
        if num_samples == 0:
            return torch.tensor([], dtype=torch.float32)
        
        # Get unconsumed chunks and convert bytes to tensors lazily (with caching)
        # This happens in inference loop context, not recv_loop!
        converted_chunks = []
        for idx in range(self.chunk_offset, len(self.audio_chunks)):
            chunk = self.audio_chunks[idx]
            if isinstance(chunk, bytes):
                # CONVERSION HAPPENS HERE (inference loop context, not recv_loop!)
                # Convert and cache back to avoid re-converting
                np_audio = np.frombuffer(chunk, dtype=np.int16)
                np_audio = np_audio.astype(np.float32) / 32768.0
                chunk_tensor = torch.from_numpy(np_audio)
                # if self.pin_memory:
                #     chunk_tensor = chunk_tensor.pin_memory()
                # Cache the converted tensor to avoid re-converting on next call
                self.audio_chunks[idx] = chunk_tensor
                converted_chunks.append(chunk_tensor)
            else:
                # Already a torch.Tensor (either from previous conversion or original input)
                converted_chunks.append(chunk)
        
        # Fast path: single chunk with all data we need
        if len(converted_chunks) == 1 and self.position_in_chunk + num_samples <= len(converted_chunks[0]):
            result = converted_chunks[0][self.position_in_chunk:self.position_in_chunk + num_samples]
            return result.to("cuda", non_blocking=True)
        
        # print(f"*****************WARNING: Slow path: need to concatenate across chunks")
        # Slow path: need to concatenate across chunks
        # Only concatenate the unconsumed portion of the first chunk
        if self.position_in_chunk > 0:
            first_chunk = converted_chunks[0][self.position_in_chunk:]
            chunks_to_concat = [first_chunk] + converted_chunks[1:]
        else:
            chunks_to_concat = converted_chunks
        
        # Concatenate and extract
        audio_buffer = torch.cat(chunks_to_concat).to("cuda", non_blocking=True)
        return audio_buffer[:num_samples]
    
    def _advance_position(self, num_samples: int) -> None:
        """
        Advance the read position by num_samples, updating chunk tracking
        Args:
            num_samples: Number of samples to advance
        """
        remaining_to_advance = num_samples
        
        while remaining_to_advance > 0 and self.chunk_offset < len(self.audio_chunks):
            current_chunk_size = self.chunk_sizes[self.chunk_offset]
            remaining_in_chunk = current_chunk_size - self.position_in_chunk
            
            if remaining_to_advance >= remaining_in_chunk:
                # Consume the rest of this chunk and move to next
                remaining_to_advance -= remaining_in_chunk
                self.chunk_offset += 1
                self.position_in_chunk = 0
            else:
                # Stay in current chunk
                self.position_in_chunk += remaining_to_advance
                remaining_to_advance = 0
    
    def _cleanup_consumed_chunks(self) -> None:
        """
        Remove fully consumed chunks from memory immediately
        This keeps memory usage minimal by freeing data as soon as it's used
        """
        if self.chunk_offset > 0:
            # Remove consumed chunks
            self.audio_chunks = self.audio_chunks[self.chunk_offset:]
            self.chunk_sizes = self.chunk_sizes[self.chunk_offset:]
            self.chunk_offset = 0
            # total_samples stays the same as it tracks all samples ever added


class MultiStream:
    """MultiStreamer for multiple streams"""

    def __init__(self, n_frames_per_stream: int):
        """
        Args:
            n_frames_per_stream (int): Number of frames per stream
        """
        self.n_frames_per_stream = n_frames_per_stream
        self.streams = {}

    def add_stream(self, stream: Stream, stream_id: int) -> None:
        """
        Add a stream to the streamer
        Args:
            stream (Stream): The stream to add
            stream_id (int): The id of the stream
        """
        self.streams[stream_id] = iter(stream)

    def rm_stream(self, stream_id: int) -> None:
        """
        Remove a stream from the streamer
        Args:
            stream_id (int): The id of the stream
        """
        self.streams.pop(stream_id, None)

    def __len__(self) -> int:
        """Number of running streams"""
        return len(self.streams)

    def __iter__(self) -> Iterator:
        """Returns the iterator object"""
        return self

    def __next__(self) -> list[Frame]:
        """
        Get the next batch of frames
        Returns:
            list[Frame]: The next batch of frames
        """
        frame_batch = []
        ids_to_remove = []
        for stream_id, stream_iter in self.streams.items():
            # Get n_frames_per_stream frames from each stream
            for _ in range(self.n_frames_per_stream):
                frame = next(stream_iter)[0]
                frame_batch.append(frame)
                if frame.is_last:
                    ids_to_remove.append(stream_id)

        # Remove streams that have ended
        for stream_id in ids_to_remove:
            self.rm_stream(stream_id)

        # If no frames are generated, raise StopIteration
        if len(frame_batch) == 0:
            raise StopIteration

        return frame_batch


class ContinuousBatchedFrameStreamer:
    """
    A class that manages continuous streaming of audio frames from multiple audio files, providing
    frame generation in batches. The class supports dynamically adding audio streams, updating
    a progress bar, and yielding batches of frames for further processing.
    """

    def __init__(
        self,
        sample_rate: int,
        frame_size_in_secs: float,
        batch_size: int,
        n_frames_per_stream: int,
        pad_last_frame: bool = False,
    ):
        """
        Args:
            sample_rate (int): The sample rate of the audio
            frame_size_in_secs (float): The size of the frame in seconds
            batch_size (int): The batch size
            n_frames_per_stream (int): The number of frames per stream
            pad_last_frame (bool): Whether to pad the last frame
        """

        self.sample_rate = sample_rate
        self.frame_size_in_secs = frame_size_in_secs
        self.batch_size = batch_size
        self.pad_last_frame = pad_last_frame

        self.multi_streamer = MultiStream(n_frames_per_stream=n_frames_per_stream)
        self.stream_id = 0

        self._progress_bar = None
        self.processed_streams = set()

    def set_audio_filepaths(self, audio_filepaths: list[str], options: list[RequestOptions]) -> None:
        """
        Set the audio filepaths
        Args:
            audio_filepaths (list[str]): The list of audio filepaths
            options (list[RequestOptions]): The list of options
        """
        if len(audio_filepaths) != len(options):
            raise ValueError("audio_filepaths and options must have the same length")

        self.audio_filepaths = audio_filepaths
        self.options = options
        self.n_audio_files = len(audio_filepaths)
        self.total_progress_steps = self.n_audio_files * 2  # One step for adding, one for processing
        self.sid2filepath = {}
        self.elapsed_durations = {}

    def set_progress_bar(self, progress_bar: ProgressBar) -> None:
        """
        Set the progress bar
        Args:
            progress_bar (ProgressBar): The progress bar to set
        """
        self._progress_bar = progress_bar
        self.restart_progress_bar()

    def restart_progress_bar(self) -> None:
        """Restart the progress bar"""
        if self._progress_bar:
            self._progress_bar.restart()

    def update_progress_bar(self) -> None:
        """Update the progress bar"""
        if self._progress_bar:
            self._progress_bar.update_bar(1 / self.total_progress_steps)

    def finish_progress_bar(self) -> None:
        """Finish the progress bar"""
        if self._progress_bar:
            self._progress_bar.finish()

    def __iter__(self) -> Iterator:
        """Returns the iterator object"""
        return self

    def add_stream(self) -> None:
        """Create a new stream and add it to the streamer"""
        if self.stream_id >= self.n_audio_files:
            return  # No more files to add

        # Create a new stream
        stream = MonoStream(
            self.sample_rate, self.frame_size_in_secs, stream_id=self.stream_id, pad_last_frame=self.pad_last_frame
        )
        # Load the next audio file
        audio_filepath = self.audio_filepaths[self.stream_id]
        options = self.options[self.stream_id]
        self.sid2filepath[self.stream_id] = audio_filepath
        self.elapsed_durations[self.stream_id] = 0.0
        stream.load_audio(audio_filepath, options)

        # Add the stream to the multi streamer
        self.multi_streamer.add_stream(stream, stream_id=self.stream_id)
        self.stream_id += 1

        # Update the progress bar
        self.update_progress_bar()

    def __next__(self) -> list[Frame]:
        """
        Get the next batch of frames, continuously adding streams
        Returns:
            list[Frame]: The next batch of frames
        """
        # If there are fewer streams than batch size, add more streams
        while len(self.multi_streamer) < self.batch_size and self.stream_id < self.n_audio_files:
            self.add_stream()

        try:
            frames = next(self.multi_streamer)
            # Update progress when a stream is fully processed
            for frame in frames:
                sid = frame.stream_id
                self.elapsed_durations[sid] += frame.valid_size / self.sample_rate
                if sid not in self.processed_streams and frame.is_last:
                    self.processed_streams.add(sid)
                    self.update_progress_bar()
            return frames
        except StopIteration:
            # if there are remaining streams, add them
            if self.stream_id < self.n_audio_files:
                return self.__next__()

        if self.stream_id == self.n_audio_files:
            self.finish_progress_bar()
            raise StopIteration

        raise ValueError("stream_id > self.n_audio_files unexpected")


class ContinuousBatchedRequestStreamer:
    """
    A class that manages continuous streaming of requests from multiple audio files, providing
    request generation in batches. Requests can be frames or feature buffers.
    The class supports dynamically adding audio streams, updating a progress bar,
    and yielding batches of requests for further processing.
    """

    def __init__(
        self,
        sample_rate: int,
        frame_size_in_secs: float,
        batch_size: int,
        n_frames_per_stream: int,
        request_type: RequestType = RequestType.FRAME,
        preprocessor: Callable = None,
        buffer_size_in_secs: float = None,
        device: torch.device = None,
        pad_last_frame: bool = False,
        right_pad_features: bool = False,
        tail_padding_in_samples: int = 0,
    ):
        """
        Args:
            sample_rate (int): The sample rate of the audio
            frame_size_in_secs (float): The size of the frame in seconds
            batch_size (int): The batch size
            n_frames_per_stream (int): The number of frames per stream
            request_type (RequestType): The type of request
            preprocessor (Callable): Preprocessor object, required for request type FEATURE_BUFFER
            buffer_size_in_secs (float): The size of the buffer in seconds, required for request type FEATURE_BUFFER
            device (torch.device): The device to use, required for request type FEATURE_BUFFER
            pad_last_frame (bool): Whether to pad the last frame
            right_pad_features (bool): Whether to right pad the features, optional for request type FEATURE_BUFFER
            tail_padding_in_samples (int): The tail padding in samples, optional for request type FEATURE_BUFFER
        """

        if request_type is RequestType.FEATURE_BUFFER:
            if buffer_size_in_secs is None:
                raise ValueError("buffer_size_in_secs must be provided for request type FEATURE_BUFFER")
            if preprocessor is None:
                raise ValueError("preprocessor must be provided for request type FEATURE_BUFFER")
            if device is None:
                raise ValueError("device must be provided for request type FEATURE_BUFFER")

        self.request_type = request_type
        self.multi_streamer = ContinuousBatchedFrameStreamer(
            sample_rate=sample_rate,
            frame_size_in_secs=frame_size_in_secs,
            batch_size=batch_size,
            n_frames_per_stream=n_frames_per_stream,
            pad_last_frame=pad_last_frame,
        )

        if self.request_type is RequestType.FEATURE_BUFFER:
            self.preprocessor = preprocessor
            self.device = device
            self.audio_bufferer = BatchedAudioBufferer(
                sample_rate=sample_rate, buffer_size_in_secs=buffer_size_in_secs
            )
            self.right_pad_features = right_pad_features
            self.tail_padding_in_samples = tail_padding_in_samples

    def set_audio_filepaths(self, audio_filepaths: list[str], options: list[RequestOptions]) -> None:
        """
        Set the audio filepaths
        Args:
            audio_filepaths (list[str]): The list of audio filepaths
            options (list[RequestOptions]): The list of options
        """
        self.multi_streamer.set_audio_filepaths(audio_filepaths, options)

    def set_progress_bar(self, progress_bar: ProgressBar) -> None:
        """
        Set the progress bar
        Args:
            progress_bar (ProgressBar): The progress bar to set
        """
        self.multi_streamer.set_progress_bar(progress_bar)

    def get_audio_filepath(self, stream_id: int) -> str:
        """
        Get the audio filepath for a given stream id
        Args:
            stream_id (int): The id of the stream
        Returns:
            str: The audio filepath for the given stream id
        """
        return self.multi_streamer.sid2filepath[stream_id]

    def get_elapsed_duration(self, stream_id: int) -> float:
        """
        Get the elapsed audio duration for a given stream id
        Args:
            stream_id (int): The id of the stream
        Returns:
            float: The elapsed audio duration for the given stream id
        """
        return self.multi_streamer.elapsed_durations[stream_id]

    def to_feature_buffers(self, frames: list[Frame]) -> list[FeatureBuffer]:
        """
        Convert frames to feature buffers
        Args:
            frames (list[Frame]): The list of frames
        Returns:
            list[FeatureBuffer]: The list of feature buffers
        """

        # Buffer input frames
        buffered_frames, left_paddings = self.audio_bufferer.update(frames)
        buffers = []

        # If right padding is enabled, convert left paddings to tensor
        if self.right_pad_features:
            left_paddings = torch.tensor(left_paddings, dtype=torch.int64, device=self.device)

        # If right padding is enabled, roll the frames to the left
        for i in range(len(buffered_frames)):
            if self.right_pad_features:
                lpad = left_paddings[i].item()
                if lpad > 0:
                    buffered_frames[i] = buffered_frames[i].roll(shifts=-lpad)
            buffers.append(buffered_frames[i].unsqueeze_(0))

        buffer_lens = torch.tensor([buffers[0].size(1)] * len(buffers), device=self.device)

        # Calculate right paddings and subtract from buffer lens
        # tail_padding_in_samples is used to keep some amount of padding at the end of the buffer
        # some models perform better with this padding
        right_paddings = torch.tensor(
            [frame.size - frame.valid_size - self.tail_padding_in_samples for frame in frames], device=self.device
        ).clamp(min=0)

        # Subtract right paddings from buffer lens
        buffer_lens = buffer_lens - right_paddings

        # If right padding is enabled, subtract left paddings from buffer lens
        # Becouse we rolled the frames to the left
        if self.right_pad_features:
            buffer_lens = buffer_lens - left_paddings

        # Apply preprocessor to get mel spectrograms
        # Use non_blocking=True for async transfer (works with pinned memory)
        feature_buffers, feature_buffer_lens = self.preprocessor(
            input_signal=torch.cat(buffers).to(self.device, non_blocking=True), length=buffer_lens
        )

        # Adjust left paddings after preprocessor
        if self.right_pad_features:
            left_paddings = left_paddings / self.preprocessor.featurizer.hop_length
            left_paddings = left_paddings.to(torch.int64)

        return [
            FeatureBuffer(
                features=feature_buffers[i],
                is_first=frame.is_first,
                is_last=frame.is_last,
                stream_id=frame.stream_id,
                right_pad_features=self.right_pad_features,
                length=feature_buffer_lens[i].item(),
                left_padding_length=left_paddings[i].item() if self.right_pad_features else 0,
                options=frame.options,
            )
            for i, frame in enumerate(frames)
        ]

    def __iter__(self) -> Iterator:
        """Returns the iterator object"""
        return self

    def __next__(self) -> list[Request]:
        """Get the next batch of requests.
        Returns:
            list of frames or feature buffers.
        """
        if self.request_type is RequestType.FRAME:
            return next(self.multi_streamer)
        return self.to_feature_buffers(next(self.multi_streamer))

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from nemo.collections.asr.inference.streaming.framing.request import Frame, RequestOptions
from nemo.collections.asr.inference.streaming.framing.stream import Stream
from nemo.collections.asr.inference.utils.audio_io import read_audio


class MonoStream(Stream):
    """
    Streamer for mono wav files.
    Iterates over the frames of the audio file
    """

    def __init__(self, rate: int, frame_size_in_secs: float, stream_id: int, pad_last_frame: bool = False):
        """
        Initialize the MonoStream
        Args:
            rate (int): sampling rate
            frame_size_in_secs (int): frame length in seconds
            stream_id (int): stream id
        """

        self.rate = rate
        self.frame_size = int(frame_size_in_secs * rate)
        self.pad_last_frame = pad_last_frame

        self.samples = None
        self.n_samples = None
        self.options = None
        super().__init__(stream_id)

    def load_audio(self, audio: str | torch.Tensor, options: RequestOptions | None = None) -> None:
        """
        Load the audio file either from a file or from a torch tensor
        Args:
            audio (str | torch.Tensor): audio file path or torch tensor of audio samples
            options (RequestOptions | None): optional options for the request
        """
        if isinstance(audio, str):
            # Read the audio file and convert to mono
            self.samples = read_audio(audio, target_sr=self.rate, mono=True)
        else:
            self.samples = audio
        self.n_samples = len(self.samples)
        self.frame_count = 0  # Reset frame count
        self.options = options

    def __iter__(self):
        """Returns the frame iterator object"""
        self.start = 0
        self.frame_count = 0
        return self

    def __next__(self) -> list[Frame]:
        """
        Get the next frame in the stream
        Returns:
            list[Frame]: The next frame in the stream
        """
        if self.samples is None:
            raise RuntimeError("No audio samples loaded. Please call load_audio() first.")

        if self.start < self.n_samples:

            end = min(self.start + self.frame_size, self.n_samples)

            # Check if this is the last frame
            is_end = False
            chunk_length = end - self.start
            if (end - self.start < self.frame_size) or (end == self.n_samples):
                is_end = True

            # Pad the last frame if needed
            if not is_end:
                chunk_samples = self.samples[self.start : end]
            else:
                if self.pad_last_frame:
                    chunk_samples = torch.zeros(self.frame_size)
                    chunk_samples[:chunk_length] = self.samples[self.start : end]
                else:
                    chunk_samples = self.samples[self.start : end]

            # Package the frame
            is_first = self.frame_count == 0
            frame = Frame(
                samples=chunk_samples,
                stream_id=self.stream_id,
                is_first=is_first,
                is_last=is_end,
                length=chunk_length,
                options=self.options if is_first else None,
            )

            self.frame_count += 1
            self.start += frame.size

            return [frame]

        # End of stream
        raise StopIteration


class StreamingBatchedRequestStreamer:
    """
    A class that manages continuous streaming of requests from real-time audio streams.
    Unlike ContinuousBatchedRequestStreamer which works with files, this works with
    StreamingMonoStream instances that receive audio chunks incrementally.
    """

    def __init__(
        self,
        sample_rate: int,
        frame_size_in_secs: float,
        batch_size: int,
        n_frames_per_stream: int = 1,
        request_type: RequestType = RequestType.FRAME,
        preprocessor: Callable = None,
        buffer_size_in_secs: float = None,
        device: torch.device = None,
        pad_last_frame: bool = False,
        right_pad_features: bool = False,
        tail_padding_in_samples: int = 0,
        max_buffer_seconds: float = 1.0,
    ):
        """
        Args:
            sample_rate (int): The sample rate of the audio
            frame_size_in_secs (float): The size of the frame in seconds
            batch_size (int): The batch size
            n_frames_per_stream (int): The number of frames per stream
            request_type (RequestType): The type of request (FRAME or FEATURE_BUFFER)
            preprocessor (Callable): Preprocessor object, required for FEATURE_BUFFER
            buffer_size_in_secs (float): Buffer size in seconds, required for FEATURE_BUFFER
            device (torch.device): Device to use, required for FEATURE_BUFFER
            pad_last_frame (bool): Whether to pad the last frame
            right_pad_features (bool): Whether to right pad features (for FEATURE_BUFFER)
            tail_padding_in_samples (int): Tail padding in samples (for FEATURE_BUFFER)
            max_buffer_seconds (float): Max buffered audio before dropping old frames (default 1.0s)
        """
        if request_type is RequestType.FEATURE_BUFFER:
            if buffer_size_in_secs is None:
                raise ValueError("buffer_size_in_secs must be provided for request type FEATURE_BUFFER")
            if preprocessor is None:
                raise ValueError("preprocessor must be provided for request type FEATURE_BUFFER")
            if device is None:
                raise ValueError("device must be provided for request type FEATURE_BUFFER")

        self.sample_rate = sample_rate
        self.frame_size_in_secs = frame_size_in_secs
        self.batch_size = batch_size
        self.n_frames_per_stream = n_frames_per_stream
        self.pad_last_frame = pad_last_frame
        self.request_type = request_type
        self.max_buffer_seconds = max_buffer_seconds

        # Manage multiple streaming sessions
        self.streams: dict[int, StreamingMonoStream] = {}
        self.next_stream_id = 0

        if self.request_type is RequestType.FEATURE_BUFFER:
            self.preprocessor = preprocessor
            self.device = device
            self.audio_bufferer = BatchedAudioBufferer(
                sample_rate=sample_rate, buffer_size_in_secs=buffer_size_in_secs
            )
            self.right_pad_features = right_pad_features
            self.tail_padding_in_samples = tail_padding_in_samples

    def open_stream(self, options: RequestOptions | None = None) -> int:
        """
        Open a new stream for real-time transcription
        Args:
            options (RequestOptions | None): Optional options for the stream
        Returns:
            int: The stream ID for this stream
        """
        stream_id = self.next_stream_id
        self.next_stream_id += 1
        
        stream = StreamingMonoStream(
            rate=self.sample_rate,
            frame_size_in_secs=self.frame_size_in_secs,
            stream_id=stream_id,
            pad_last_frame=self.pad_last_frame,
            max_buffer_seconds=self.max_buffer_seconds,
        )
        stream.set_options(options)
        self.streams[stream_id] = stream
        
        return stream_id

    def close_stream(self, stream_id: int) -> None:
        """
        Close a stream and remove it from active streams
        Args:
            stream_id (int): The ID of the stream to close
        """
        if stream_id in self.streams:
            self.streams[stream_id].mark_end()
            # Don't remove immediately - let the stream drain its remaining frames

    def append_audio(self, stream_id: int, samples: torch.Tensor | np.ndarray) -> None:
        """
        Append audio samples to a specific stream
        Args:
            stream_id (int): The ID of the stream
            samples (torch.Tensor | np.ndarray): Audio samples to append
        """
        if stream_id not in self.streams:
            raise ValueError(f"Stream {stream_id} not found")
        self.streams[stream_id].append_audio(samples)

    def get_ready_stream_ids(self) -> list[int]:
        """
        Get list of stream IDs that have frames ready to process
        Returns:
            list[int]: List of stream IDs with available frames
        """
        ready = []
        for stream_id, stream in self.streams.items():
            if stream.has_available_frame():
                ready.append(stream_id)
        return ready

    def to_feature_buffers(self, frames: list[Frame]) -> list[FeatureBuffer]:
        """
        Convert frames to feature buffers
        Args:
            frames (list[Frame]): The list of frames
        Returns:
            list[FeatureBuffer]: The list of feature buffers
        """
        # Buffer input frames
        buffered_frames, left_paddings = self.audio_bufferer.update(frames)
        buffers = []

        # If right padding is enabled, convert left paddings to tensor
        if self.right_pad_features:
            left_paddings = torch.tensor(left_paddings, dtype=torch.int64, device=self.device)

        # If right padding is enabled, roll the frames to the left
        for i in range(len(buffered_frames)):
            if self.right_pad_features:
                lpad = left_paddings[i].item()
                if lpad > 0:
                    buffered_frames[i] = buffered_frames[i].roll(shifts=-lpad)
            buffers.append(buffered_frames[i].unsqueeze_(0))

        buffer_lens = torch.tensor([buffers[0].size(1)] * len(buffers), device=self.device)

        # Calculate right paddings
        right_paddings = torch.tensor(
            [frame.size - frame.valid_size - self.tail_padding_in_samples for frame in frames], device=self.device
        ).clamp(min=0)

        # Subtract right paddings from buffer lens
        buffer_lens = buffer_lens - right_paddings

        # If right padding is enabled, subtract left paddings from buffer lens
        if self.right_pad_features:
            buffer_lens = buffer_lens - left_paddings

        # Apply preprocessor to get mel spectrograms
        # Use non_blocking=True for async transfer (works with pinned memory)
        feature_buffers, feature_buffer_lens = self.preprocessor(
            input_signal=torch.cat(buffers).to(self.device, non_blocking=True), length=buffer_lens
        )

        # Adjust left paddings after preprocessor
        if self.right_pad_features:
            left_paddings = left_paddings / self.preprocessor.featurizer.hop_length
            left_paddings = left_paddings.to(torch.int64)

        return [
            FeatureBuffer(
                features=feature_buffers[i],
                is_first=frame.is_first,
                is_last=frame.is_last,
                stream_id=frame.stream_id,
                right_pad_features=self.right_pad_features,
                length=feature_buffer_lens[i].item(),
                left_padding_length=left_paddings[i].item() if self.right_pad_features else 0,
                options=frame.options,
            )
            for i, frame in enumerate(frames)
        ]

    def __iter__(self):
        """Returns the iterator object"""
        return self

    def __next__(self) -> list[Request]:
        """
        Get the next batch of requests from ready streams
        Returns:
            list[Request]: The next batch of frames or feature buffers
        Raises:
            NotEnoughDataException: If no streams have enough data
            StopIteration: If all streams have ended and been consumed
        """
        frames = []
        streams_to_remove = []

        # Try to get frames from ready streams up to batch_size
        for stream_id in list(self.streams.keys()):
            if len(frames) >= self.batch_size:
                break

            stream = self.streams[stream_id]
            
            # Try to get n_frames_per_stream from this stream
            for _ in range(self.n_frames_per_stream):
                try:
                    frame_list = next(stream)
                    frames.extend(frame_list)
                    
                    # Check if stream ended
                    if frame_list[0].is_last:
                        streams_to_remove.append(stream_id)
                        break
                        
                except NotEnoughDataException:
                    # This stream doesn't have data yet, try next stream
                    break
                except StopIteration:
                    # Stream is fully consumed
                    streams_to_remove.append(stream_id)
                    break

        # Remove finished streams
        for stream_id in streams_to_remove:
            del self.streams[stream_id]

        # If no frames collected, raise exception
        if len(frames) == 0:
            if len(self.streams) == 0:
                raise StopIteration
            else:
                raise NotEnoughDataException("No streams have enough data yet")

        # Convert to feature buffers if needed
        if self.request_type is RequestType.FRAME:
            return frames
        else:
            return self.to_feature_buffers(frames)