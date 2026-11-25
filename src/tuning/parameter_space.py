"""
src/tuning/parameter_space.py
==============================
Defines streaming algorithm hyperparameters, preset configurations, and parameter grids
for tuning TollMatcher Variant C on Raspberry Pi 5 / host edge simulation.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass
class StreamingHyperparameters:
    """Hyperparameters governing the streaming speech translation algorithm."""
    name: str
    k_agreement: int = 2
    decode_interval_sec: float = 0.700
    short_endpoint_sec: float = 0.400
    safe_endpoint_sec: float = 0.750
    max_pause_bridge_sec: float = 1.000
    forced_endpoint_sec: float = 6.000
    energy_threshold_multiplier: float = 3.5
    preroll_sec: float = 0.200
    overlap_context_sec: float = 0.600
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Standard Presets for Edge Deployment Scenarios
PRESETS: Dict[str, StreamingHyperparameters] = {
    "fast": StreamingHyperparameters(
        name="Fast Response (Low Latency)",
        k_agreement=1,
        decode_interval_sec=0.400,
        short_endpoint_sec=0.300,
        safe_endpoint_sec=0.600,
        max_pause_bridge_sec=0.850,
        forced_endpoint_sec=5.000,
        energy_threshold_multiplier=3.0,
        preroll_sec=0.150,
        overlap_context_sec=0.500,
        description="Optimized for rapid dialogue turns; earliest TTFA but higher sensitivity to ASR revisions."
    ),
    "balanced": StreamingHyperparameters(
        name="Balanced Streaming (Recommended)",
        k_agreement=2,
        decode_interval_sec=0.700,
        short_endpoint_sec=0.400,
        safe_endpoint_sec=0.750,
        max_pause_bridge_sec=1.000,
        forced_endpoint_sec=6.000,
        energy_threshold_multiplier=3.5,
        preroll_sec=0.200,
        overlap_context_sec=0.600,
        description="Pareto-optimal standard from research blueprint: bridges 800ms thinking pauses with zero false cuts."
    ),
    "accurate": StreamingHyperparameters(
        name="High Precision (Stable)",
        k_agreement=3,
        decode_interval_sec=0.700,
        short_endpoint_sec=0.500,
        safe_endpoint_sec=0.850,
        max_pause_bridge_sec=1.100,
        forced_endpoint_sec=7.000,
        energy_threshold_multiplier=3.8,
        preroll_sec=0.250,
        overlap_context_sec=0.700,
        description="Requires 3-step agreement before commitment; maximum stability in reverberant or noisy audio."
    ),
    "edge_pi5": StreamingHyperparameters(
        name="Raspberry Pi 5 Constrained",
        k_agreement=2,
        decode_interval_sec=0.800,
        short_endpoint_sec=0.450,
        safe_endpoint_sec=0.750,
        max_pause_bridge_sec=1.000,
        forced_endpoint_sec=6.000,
        energy_threshold_multiplier=3.5,
        preroll_sec=0.200,
        overlap_context_sec=0.600,
        description="Slightly relaxed decode frequency (800ms) to conserve CPU cycles and maintain < 0.12 RTF on Cortex-A76."
    )
}


def get_preset(name: str) -> StreamingHyperparameters:
    key = name.lower().strip()
    if key in PRESETS:
        return PRESETS[key]
    raise ValueError(f"Unknown preset '{name}'. Choose from: {list(PRESETS.keys())}")


def get_k_sweep_grid() -> List[StreamingHyperparameters]:
    """Generates hyperparameter configs sweeping agreement depth k in [1, 2, 3]."""
    configs = []
    for k in [1, 2, 3]:
        configs.append(StreamingHyperparameters(
            name=f"Sweep-k={k}",
            k_agreement=k,
            decode_interval_sec=0.700,
            short_endpoint_sec=0.400,
            safe_endpoint_sec=0.750,
            max_pause_bridge_sec=1.000,
            description=f"Local Agreement depth k={k}"
        ))
    return configs


def get_decode_interval_sweep_grid() -> List[StreamingHyperparameters]:
    """Generates hyperparameter configs sweeping decode interval (400ms to 1000ms)."""
    configs = []
    for interval in [0.400, 0.550, 0.700, 0.850, 1.000]:
        configs.append(StreamingHyperparameters(
            name=f"Sweep-Decode={int(interval*1000)}ms",
            k_agreement=2,
            decode_interval_sec=interval,
            short_endpoint_sec=0.400,
            safe_endpoint_sec=0.750,
            max_pause_bridge_sec=1.000,
            description=f"Periodic ASR decode interval = {int(interval*1000)} ms"
        ))
    return configs


def get_pause_bridge_sweep_grid() -> List[StreamingHyperparameters]:
    """Generates hyperparameter configs sweeping pause bridging threshold (500ms to 1200ms)."""
    configs = []
    for bridge in [0.500, 0.700, 0.850, 1.000, 1.200]:
        configs.append(StreamingHyperparameters(
            name=f"Sweep-Bridge={int(bridge*1000)}ms",
            k_agreement=2,
            decode_interval_sec=0.700,
            short_endpoint_sec=0.400,
            safe_endpoint_sec=0.750,
            max_pause_bridge_sec=bridge,
            description=f"Thinking pause bridge threshold = {int(bridge*1000)} ms"
        ))
    return configs


def get_endpoint_silence_sweep_grid() -> List[StreamingHyperparameters]:
    """Generates hyperparameter configs sweeping short and safe silence endpoints."""
    configs = []
    pairs = [
        (0.300, 0.600),
        (0.400, 0.750),
        (0.500, 0.900),
    ]
    for short_ep, safe_ep in pairs:
        configs.append(StreamingHyperparameters(
            name=f"Sweep-Endpoints-Short{int(short_ep*1000)}ms-Safe{int(safe_ep*1000)}ms",
            k_agreement=2,
            decode_interval_sec=0.700,
            short_endpoint_sec=short_ep,
            safe_endpoint_sec=safe_ep,
            max_pause_bridge_sec=1.000,
            description=f"Short endpoint: {int(short_ep*1000)}ms, Safe endpoint: {int(safe_ep*1000)}ms"
        ))
    return configs


def get_curated_grid() -> List[StreamingHyperparameters]:
    """Returns curated representative tuning grid combining key algorithmic variants."""
    grid = list(PRESETS.values())
    grid.extend(get_k_sweep_grid())
    grid.extend(get_pause_bridge_sweep_grid())
    # Deduplicate by unique tuple of key parameters
    seen = set()
    unique_grid = []
    for c in grid:
        key = (c.k_agreement, round(c.decode_interval_sec, 3), round(c.short_endpoint_sec, 3), round(c.max_pause_bridge_sec, 3))
        if key not in seen:
            seen.add(key)
            unique_grid.append(c)
    return unique_grid
