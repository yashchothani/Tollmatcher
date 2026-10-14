"""
src/metrics/runtime.py
=======================
System profiling and runtime efficiency metrics:
- Real-Time Factor (RTF = processing_time / audio_duration)
- Memory RSS tracking via resource module
- Raspberry Pi hardware capability indicators
"""

import os
import platform
import resource
import time
from typing import Dict, Any


def get_peak_memory_mb() -> float:
    """Returns peak resident set size (RSS) in megabytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On macOS, ru_maxrss is in bytes; on Linux, it is in kilobytes.
    if platform.system() == "Darwin":
        return round(usage / (1024.0 * 1024.0), 2)
    else:
        return round(usage / 1024.0, 2)


def compute_rtf(processing_time_sec: float, audio_duration_sec: float) -> float:
    """Computes Real-Time Factor (RTF). Lower is faster; RTF < 1.0 means faster than real-time."""
    if audio_duration_sec <= 0.0:
        return 0.0
    return round(processing_time_sec / audio_duration_sec, 4)


def get_hardware_info() -> Dict[str, Any]:
    """Collects host machine and target edge specifications."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "target_device": "Raspberry Pi 5 (8GB LPDDR4X, Broadcom BCM2712 Quad Cortex-A76 @ 2.4GHz)"
    }
