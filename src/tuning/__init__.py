"""
src/tuning/__init__.py
======================
TollMatcher streaming parameter space and tuning engine.
"""

from src.tuning.parameter_space import (
    StreamingHyperparameters,
    PRESETS,
    get_preset,
    get_curated_grid,
    get_k_sweep_grid,
    get_decode_interval_sweep_grid,
    get_pause_bridge_sweep_grid,
    get_endpoint_silence_sweep_grid
)
from src.tuning.grid_search import StreamingParameterTuner

__all__ = [
    "StreamingHyperparameters",
    "PRESETS",
    "get_preset",
    "get_curated_grid",
    "get_k_sweep_grid",
    "get_decode_interval_sweep_grid",
    "get_pause_bridge_sweep_grid",
    "get_endpoint_silence_sweep_grid",
    "StreamingParameterTuner"
]
