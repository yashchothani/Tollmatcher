"""
src/pipeline/streaming_feeder.py
=================================
Simulates streaming microphone audio input by chunking a 16kHz mono WAV file
into discrete temporal frames (e.g. 100ms or 200ms) with accurate timestamping.
"""

import os
import struct
import sys
import wave
from typing import Iterator, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class StreamingAudioFeeder:
    """Reads 16kHz WAV file and feeds fixed-size audio frames sequentially."""

    def __init__(self, wav_path: str, chunk_duration_sec: float = 0.100):
        if not os.path.isabs(wav_path) and not os.path.exists(wav_path):
            alt_path = os.path.join(PROJECT_ROOT, wav_path)
            if os.path.exists(alt_path):
                wav_path = alt_path

        self.wav_path = wav_path
        self.chunk_duration_sec = chunk_duration_sec

        with wave.open(wav_path, "rb") as wf:
            self.sample_rate = wf.getframerate()
            self.channels = wf.getnchannels()
            self.sample_width = wf.getsampwidth()
            self.total_frames = wf.getnframes()
            self.total_duration_sec = self.total_frames / self.sample_rate
            self.raw_bytes = wf.readframes(self.total_frames)

        assert self.sample_rate == 16000, f"Expected 16kHz audio, got {self.sample_rate}"
        assert self.channels == 1, f"Expected mono audio, got {self.channels}"
        assert self.sample_width == 2, f"Expected 16-bit PCM, got {self.sample_width}"

        self.chunk_size_frames = int(self.chunk_duration_sec * self.sample_rate)
        self.chunk_size_bytes = self.chunk_size_frames * self.sample_width

    def stream_chunks(self) -> Iterator[Dict[str, Any]]:
        """Yields sequential frames with start_sec, end_sec, and PCM samples."""
        offset = 0
        current_time = 0.0

        while offset < len(self.raw_bytes):
            chunk_bytes = self.raw_bytes[offset:offset + self.chunk_size_bytes]
            frames_in_chunk = len(chunk_bytes) // self.sample_width
            duration = frames_in_chunk / self.sample_rate

            # Unpack signed 16-bit integers
            samples = struct.unpack(f"<{frames_in_chunk}h", chunk_bytes)

            yield {
                "chunk_index": offset // self.chunk_size_bytes,
                "start_time_sec": round(current_time, 4),
                "end_time_sec": round(current_time + duration, 4),
                "duration_sec": round(duration, 4),
                "samples": samples,
                "raw_bytes": chunk_bytes,
                "is_last": (offset + self.chunk_size_bytes) >= len(self.raw_bytes)
            }

            offset += self.chunk_size_bytes
            current_time += duration
