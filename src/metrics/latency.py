"""
src/metrics/latency.py
=======================
Latency metrics for streaming speech-to-speech translation:
- Time-to-First-Audio (TTFA): Delay from speech start or first segment trigger to first synthesized speech packet
- Total Turnaround Latency (TTL): Delay from speaker completion (gold speech end) to full audio playback completion
- Statistical percentiles (mean, median, p95, max)
"""

import math
from typing import List, Dict, Any


def compute_latency_distribution(latencies_ms: List[float]) -> Dict[str, float]:
    """Computes summary statistics for a list of latency measurements (ms)."""
    if not latencies_ms:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "p95": 0.0, "max": 0.0}

    sorted_l = sorted(latencies_ms)
    n = len(sorted_l)

    mean_val = sum(sorted_l) / n
    median_val = sorted_l[n // 2] if n % 2 != 0 else (sorted_l[n // 2 - 1] + sorted_l[n // 2]) / 2.0

    p95_idx = min(int(math.ceil(0.95 * n)) - 1, n - 1)
    p95_val = sorted_l[p95_idx]

    variance = sum((x - mean_val) ** 2 for x in sorted_l) / n
    std_val = math.sqrt(variance)

    return {
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "std": round(std_val, 2),
        "p95": round(p95_val, 2),
        "max": round(sorted_l[-1], 2)
    }


def compute_ttfa(first_audio_out_sec: float, speech_start_sec: float) -> float:
    """Computes Time-to-First-Audio (TTFA) in milliseconds."""
    return round((first_audio_out_sec - speech_start_sec) * 1000.0, 2)


def compute_ttl(last_audio_complete_sec: float, speech_end_sec: float) -> float:
    """Computes Total Turnaround Latency (TTL) in milliseconds relative to speaker end."""
    return round((last_audio_complete_sec - speech_end_sec) * 1000.0, 2)
