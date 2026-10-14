"""
src/pipeline/vad_engine.py
===========================
Acoustic Voice Activity Detector (VAD) state machine:
- Computes frame RMS energy and zero-crossing rate
- Tracks adaptive background noise floor
- Maintains speech onset hysteresis and hang-over silence buffer
- Emits speech start, active speech, and silence endpoint trigger events
"""

import math
from typing import List, Tuple, Dict, Any


class EnergyVAD:
    """Frame-based energy VAD with adaptive threshold and hysteresis."""

    def __init__(
        self,
        energy_threshold_multiplier: float = 3.5,
        min_speech_frames: int = 2,
        endpoint_silence_sec: float = 0.600,
        frame_duration_sec: float = 0.100
    ):
        self.energy_threshold_multiplier = energy_threshold_multiplier
        self.min_speech_frames = min_speech_frames
        self.endpoint_silence_sec = endpoint_silence_sec
        self.frame_duration_sec = frame_duration_sec

        # Dynamic state
        self.noise_floor = 100.0  # initial baseline RMS
        self.is_speech_active = False
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
        self.speech_start_time = None

    def _compute_rms(self, samples: Tuple[int, ...]) -> float:
        if not samples:
            return 0.0
        sum_sq = sum(s * s for s in samples)
        return math.sqrt(sum_sq / len(samples))

    def process_frame(self, samples: Tuple[int, ...], current_time_sec: float) -> Dict[str, Any]:
        """
        Processes one audio frame.
        Returns state dict: {is_speech, is_endpoint, speech_start, current_time}
        """
        rms = self._compute_rms(samples)

        # Dynamic threshold
        threshold = max(self.noise_floor * self.energy_threshold_multiplier, 250.0)
        is_frame_voiced = (rms > threshold)

        event = "NONE"
        endpoint_triggered = False

        if not self.is_speech_active:
            # Updating noise floor during silence
            if not is_frame_voiced:
                self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms
                self.consecutive_speech_frames = 0
            else:
                self.consecutive_speech_frames += 1
                if self.consecutive_speech_frames >= self.min_speech_frames:
                    self.is_speech_active = True
                    self.speech_start_time = current_time_sec - (self.consecutive_speech_frames * self.frame_duration_sec)
                    self.consecutive_silence_frames = 0
                    event = "SPEECH_START"
        else:
            # In speech mode
            if is_frame_voiced:
                self.consecutive_silence_frames = 0
                event = "SPEECH_ACTIVE"
            else:
                self.consecutive_silence_frames += 1
                silence_dur = self.consecutive_silence_frames * self.frame_duration_sec
                if silence_dur >= self.endpoint_silence_sec:
                    self.is_speech_active = False
                    self.consecutive_speech_frames = 0
                    self.consecutive_silence_frames = 0
                    endpoint_triggered = True
                    event = "SPEECH_ENDPOINT"

        return {
            "rms": round(rms, 2),
            "threshold": round(threshold, 2),
            "is_speech": self.is_speech_active,
            "event": event,
            "endpoint_triggered": endpoint_triggered,
            "speech_start_time": self.speech_start_time,
            "current_time_sec": current_time_sec
        }
