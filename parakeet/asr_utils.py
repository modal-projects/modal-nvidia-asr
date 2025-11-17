from typing import Sequence, Callable
from urllib.request import urlopen
import tempfile
import wave
import os
import sys

def preprocess_audio(audio: bytes | str, target_sample_rate: int = 16000) -> bytes:
    import array
    import io
    import wave

    if isinstance(audio, str):
        audio = get_bytes_from_wav(audio)

    with wave.open(io.BytesIO(audio), "rb") as wav_in:
        n_channels = wav_in.getnchannels()
        sample_width = wav_in.getsampwidth()
        frame_rate = wav_in.getframerate()
        n_frames = wav_in.getnframes()
        frames = wav_in.readframes(n_frames)

    # Convert frames to array based on sample width
    if sample_width == 1:
        audio_data = array.array("B", frames)  # unsigned char
    elif sample_width == 2:
        audio_data = array.array("h", frames)  # signed short
    elif sample_width == 3:
        audio_data = array.array("b", frames)  # signed byte
    elif sample_width == 4:
        audio_data = array.array("i", frames)  # signed int
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")
        
    # Downmix to mono if needed
    if n_channels > 1:
        mono_data = array.array(audio_data.typecode)
        for i in range(0, len(audio_data), n_channels):
            chunk = audio_data[i : i + n_channels]
            mono_data.append(sum(chunk) // n_channels)
        audio_data = mono_data

    # Resample to 16kHz if needed
    if frame_rate != target_sample_rate:
        ratio = target_sample_rate / frame_rate
        new_length = int(len(audio_data) * ratio)
        resampled_data = array.array(audio_data.typecode)

        for i in range(new_length):
            # Linear interpolation
            pos = i / ratio
            pos_int = int(pos)
            pos_frac = pos - pos_int

            if pos_int >= len(audio_data) - 1:
                sample = audio_data[-1]
            else:
                sample1 = audio_data[pos_int]
                sample2 = audio_data[pos_int + 1]
                sample = int(sample1 + (sample2 - sample1) * pos_frac)

            resampled_data.append(sample)

        audio_data = resampled_data

    return audio_data.tobytes()


def get_bytes_from_wav(location: str) -> bytes:

    if location.startswith("http"):
        bytes = urlopen(location).read()
    else:
        bytes = open(location, "rb").read()

    return bytes


def identity(data):
    return data

def batch_seq(data: Sequence, chunk_size: int, transform: Callable = None) -> list[bytes]:
    if transform is None:
        transform = identity
    return [transform(data[i:i+chunk_size]) for i in range(0, len(data), chunk_size)]

SHUTDOWN_SIGNAL = (
    b"END_OF_STREAM_8f13d09"  # byte sequence indicating a stream is finished
)

def int2float(audio_data):
    import numpy as np
    abs_max = np.abs(audio_data).max()
    audio_data = audio_data.astype('float32')
    if abs_max > 0:
        audio_data *= 1/32768
    audio_data = audio_data.squeeze()  # depends on the use case
    return audio_data

def bytes_to_torch(data, device = "cuda"):
    import numpy as np
    import torch
    data = np.frombuffer(data, dtype=np.int16)
    data = torch.from_numpy(int2float(data)).to(device)
    return data

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

def write_wav_file(args):
    idx, data = args
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_file, "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)  # 16-bit
        wav_out.setframerate(16000)
        wav_out.writeframes(data)
    temp_file.close()
    return temp_file.name
