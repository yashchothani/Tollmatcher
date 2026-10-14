"""
src/runners/base.py
===================
Base runner interface and data containers for TollMatcher streaming ST algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class StreamingRunResult:
    variant_name: str
    audio_example_id: str
    target_language: str
    predicted_boundaries_sec: List[float]
    final_transcript: str
    final_translation: str
    first_audio_out_sec: float
    last_audio_complete_sec: float
    total_processing_time_sec: float
    false_cuts_in_pause: int
    wer: float = 0.0
    cer: float = 0.0
    bleu: float = 0.0
    chrf: float = 0.0
    ttfa_ms: float = 0.0
    ttl_ms: float = 0.0
    rtf: float = 0.0
    f1_200ms: float = 0.0
    f1_500ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseStreamingRunner(ABC):
    """Abstract base class for streaming speech translation runners."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run_example(
        self,
        audio_record: Dict[str, Any],
        gold_text_record: Dict[str, Any],
        target_lang: str
    ) -> StreamingRunResult:
        """Executes the streaming pipeline on an audio example."""
        pass
