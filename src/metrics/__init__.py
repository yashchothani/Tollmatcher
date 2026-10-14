"""
src/metrics
============
Evaluation metrics for ASR, MT, segmentation, latency, and runtime profiling.
"""

from src.metrics.asr import compute_wer, compute_cer, normalize_text
from src.metrics.translation import compute_bleu, compute_chrf
from src.metrics.segmentation import evaluate_segmentation_boundaries, compute_false_cuts_in_pauses
from src.metrics.latency import compute_latency_distribution, compute_ttfa, compute_ttl
from src.metrics.runtime import compute_rtf, get_peak_memory_mb, get_hardware_info
