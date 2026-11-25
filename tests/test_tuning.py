"""
tests/test_tuning.py
====================
Unit tests for streaming parameter tuning, sweep grids, and Pareto frontier analysis.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tuning.parameter_space import (
    StreamingHyperparameters,
    PRESETS,
    get_preset,
    get_k_sweep_grid,
    get_decode_interval_sweep_grid,
    get_pause_bridge_sweep_grid,
    get_endpoint_silence_sweep_grid,
    get_curated_grid
)
from src.tuning.grid_search import StreamingParameterTuner


class TestParameterTuning(unittest.TestCase):

    def test_presets_validity(self):
        for name in ["fast", "balanced", "accurate", "edge_pi5"]:
            preset = get_preset(name)
            self.assertIsInstance(preset, StreamingHyperparameters)
            self.assertGreater(preset.k_agreement, 0)
            self.assertGreater(preset.decode_interval_sec, 0.0)
            self.assertGreater(preset.short_endpoint_sec, 0.0)
            self.assertGreater(preset.safe_endpoint_sec, preset.short_endpoint_sec)
            self.assertGreaterEqual(preset.max_pause_bridge_sec, preset.safe_endpoint_sec)

        with self.assertRaises(ValueError):
            get_preset("non_existent_preset")

    def test_sweep_grid_generators(self):
        k_grid = get_k_sweep_grid()
        self.assertEqual(len(k_grid), 3)
        self.assertEqual([c.k_agreement for c in k_grid], [1, 2, 3])

        decode_grid = get_decode_interval_sweep_grid()
        self.assertEqual(len(decode_grid), 5)
        self.assertEqual(decode_grid[0].decode_interval_sec, 0.400)
        self.assertEqual(decode_grid[-1].decode_interval_sec, 1.000)

        bridge_grid = get_pause_bridge_sweep_grid()
        self.assertEqual(len(bridge_grid), 5)

        curated = get_curated_grid()
        self.assertGreater(len(curated), 5)

    def test_pareto_dominance_logic(self):
        # Synthetic summaries to verify Pareto tagging
        # Config A: Low latency (TTFA=600), 0 false cuts, chrF=100, RTF=0.10
        # Config B: Higher latency (TTFA=1200), 0 false cuts, chrF=100, RTF=0.10 -> Dominated by A!
        # Config C: Extremely low latency (TTFA=400), 2 false cuts, chrF=98, RTF=0.12 -> Non-dominated (trade-off)
        summaries = [
            {
                "config_name": "A",
                "total_false_cuts_in_pause": 0,
                "ttfa_ms_median": 600.0,
                "ttl_ms_median": 600.0,
                "avg_chrf": 100.0,
                "avg_rtf": 0.10
            },
            {
                "config_name": "B",
                "total_false_cuts_in_pause": 0,
                "ttfa_ms_median": 1200.0,
                "ttl_ms_median": 650.0,
                "avg_chrf": 100.0,
                "avg_rtf": 0.10
            },
            {
                "config_name": "C",
                "total_false_cuts_in_pause": 2,
                "ttfa_ms_median": 400.0,
                "ttl_ms_median": 580.0,
                "avg_chrf": 98.0,
                "avg_rtf": 0.12
            }
        ]

        StreamingParameterTuner._tag_pareto_front(summaries)
        self.assertTrue(summaries[0]["is_pareto_optimal"])   # Config A is optimal
        self.assertFalse(summaries[1]["is_pareto_optimal"])  # Config B is dominated by A
        self.assertTrue(summaries[2]["is_pareto_optimal"])   # Config C is optimal on latency frontier

    def test_single_config_evaluation(self):
        tuner = StreamingParameterTuner()
        cfg = get_preset("fast")
        summary = tuner.evaluate_config(cfg)

        self.assertIn("avg_chrf", summary)
        self.assertIn("ttfa_ms_median", summary)
        self.assertIn("ttl_ms_median", summary)
        self.assertIn("total_false_cuts_in_pause", summary)
        self.assertEqual(summary["hyperparameters"]["k_agreement"], 1)


if __name__ == "__main__":
    unittest.main()
