"""
tests/test_metrics.py
======================
Unit tests for metric modules: ASR (WER, CER), Translation (BLEU, chrF++),
Segmentation (F1, False Cuts), and Latency.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.metrics.asr import compute_wer, compute_cer, normalize_text
from src.metrics.translation import compute_bleu, compute_chrf
from src.metrics.segmentation import evaluate_segmentation_boundaries, compute_false_cuts_in_pauses
from src.metrics.latency import compute_latency_distribution, compute_ttfa, compute_ttl


class TestMetrics(unittest.TestCase):

    def test_asr_normalization(self):
        self.assertEqual(normalize_text("Hello, World!"), "hello world")
        self.assertEqual(normalize_text("  Über 100%   "), "über 100")

    def test_wer_cer_identical(self):
        ref = "The quick brown fox jumps over the lazy dog."
        self.assertEqual(compute_wer(ref, ref), 0.0)
        self.assertEqual(compute_cer(ref, ref), 0.0)

    def test_wer_substitution(self):
        ref = "the flight departs at ten"
        hyp = "the flight leaves at ten"
        wer = compute_wer(ref, hyp)
        self.assertAlmostEqual(wer, 0.2, places=2)  # 1 substitution out of 5 words = 0.20

    def test_bleu_identical(self):
        ref = "On Monday, scientists announced a new tool."
        self.assertAlmostEqual(compute_bleu(ref, ref), 100.0, places=1)

    def test_bleu_short_turn(self):
        # Short turns (<4 words) must not collapse to 0.0
        ref = "Yes, exactly."
        hyp = "Yes, exactly."
        self.assertGreater(compute_bleu(hyp, ref), 90.0)

    def test_chrf_identical(self):
        ref = "Kaltes Wetter ist die einzige Gefahr."
        self.assertAlmostEqual(compute_chrf(ref, ref), 100.0, places=1)

    def test_segmentation_f1_perfect(self):
        gold = [2.5, 5.0]
        pred = [2.55, 4.95]
        res = evaluate_segmentation_boundaries(pred, gold, tau_sec=0.200)
        self.assertEqual(res["f1"], 1.0)
        self.assertLess(res["mae_ms"], 60.0)

    def test_false_cuts_in_pause(self):
        pauses = [{"pause_start_sec": 1.5, "pause_end_sec": 2.3, "duration_sec": 0.8, "pause_type": "thinking_pause"}]
        # Cut at 1.8s falls inside pause
        cuts = [1.0, 1.8, 3.0]
        self.assertEqual(compute_false_cuts_in_pauses(cuts, pauses), 1)
        # Clean cuts
        clean_cuts = [1.2, 2.5, 3.5]
        self.assertEqual(compute_false_cuts_in_pauses(clean_cuts, pauses), 0)

    def test_latency_stats(self):
        latencies = [100.0, 200.0, 300.0, 400.0, 500.0]
        dist = compute_latency_distribution(latencies)
        self.assertEqual(dist["median"], 300.0)
        self.assertEqual(dist["max"], 500.0)


if __name__ == "__main__":
    unittest.main()
