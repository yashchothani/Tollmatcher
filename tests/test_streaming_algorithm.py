"""
tests/test_streaming_algorithm.py
==================================
Comprehensive test suite verifying streaming algorithm parameters:
1. Local Agreement depth k (k=1, 2, 3) and prefix commitment
2. Punctuation and token normalization in hypothesis stabilization
3. Voice Activity Detection (VAD) circular pre-roll buffer
4. Three-Tier Adaptive Endpointing:
   - Tier 1: Fast Short-Silence Endpoint (short_endpoint_sec)
   - Tier 2: Safe Acoustic Silence & Thinking Pause Bridging (safe_endpoint_sec, max_pause_bridge_sec)
   - Tier 3: Forced Duration Cap (forced_endpoint_sec)
5. Periodic Incremental ASR Decode Interval (decode_interval_sec)
6. Parameter tuning presets and Pareto front properties
"""

import json
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.vad_engine import EnergyVAD
from src.pipeline.local_agreement import LocalAgreementStabilizer
from src.runners.variant_c import VariantCRunner
from src.tuning.parameter_space import (
    StreamingHyperparameters,
    get_preset,
    get_k_sweep_grid,
    get_pause_bridge_sweep_grid
)

SPEECH_JSONL = os.path.join(PROJECT_ROOT, "benchmark", "data", "speech_dataset.jsonl")
PILOT_JSONL = os.path.join(PROJECT_ROOT, "benchmark", "data", "pilot_dataset.jsonl")


class TestStreamingAlgorithmParameters(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(SPEECH_JSONL, "r", encoding="utf-8") as f:
            cls.speech_records = [json.loads(line) for line in f if line.strip()]
        with open(PILOT_JSONL, "r", encoding="utf-8") as f:
            cls.pilot_groups = {json.loads(line)["sentence_group_id"]: json.loads(line) for line in f if line.strip()}
        cls.sample_spk = cls.speech_records[0]  # TM_A01 (EN with 800ms thinking pause)
        cls.ref_group = cls.pilot_groups[cls.sample_spk["reference_translation_ids"][0]]

    # -------------------------------------------------------------------------
    # 1. Local Agreement Depth (k) Tests
    # -------------------------------------------------------------------------
    def test_local_agreement_k1_immediate_commit(self):
        """k=1 commits tokens on the very first hypothesis update."""
        la = LocalAgreementStabilizer(k_agreement=1)
        r1 = la.update("hello world")
        self.assertEqual(r1["newly_committed"], ["hello", "world"])
        self.assertEqual(r1["all_committed_text"], "hello world")
        self.assertEqual(r1["uncommitted_text"], "")
        self.assertTrue(r1["is_stable"])

    def test_local_agreement_k2_two_step_commit(self):
        """k=2 requires identical tokens across two consecutive hypotheses."""
        la = LocalAgreementStabilizer(k_agreement=2)
        r1 = la.update("good morning")
        self.assertEqual(r1["newly_committed"], [])
        self.assertEqual(r1["all_committed_text"], "")

        r2 = la.update("good morning everyone")
        self.assertEqual(r2["newly_committed"], ["good", "morning"])
        self.assertEqual(r2["all_committed_text"], "good morning")
        self.assertEqual(r2["uncommitted_text"], "everyone")

    def test_local_agreement_k3_three_step_commit(self):
        """k=3 requires identical tokens across three consecutive hypotheses."""
        la = LocalAgreementStabilizer(k_agreement=3)
        la.update("train to")
        la.update("train to Berlin")
        # After 2 steps with k=3, nothing committed yet
        self.assertEqual(la.committed_text, "")

        # 3rd step: "train to" is present across all 3 steps
        r3 = la.update("train to Berlin today")
        self.assertEqual(r3["newly_committed"], ["train", "to"])
        self.assertEqual(la.committed_text, "train to")

        # 4th step: "Berlin" has now appeared in 3 consecutive steps (steps 2, 3, 4)
        r4 = la.update("train to Berlin today soon")
        self.assertEqual(r4["newly_committed"], ["Berlin"])
        self.assertEqual(la.committed_text, "train to Berlin")

    def test_local_agreement_punctuation_normalization(self):
        """Punctuation differences shouldn't break Longest Common Prefix matching."""
        la = LocalAgreementStabilizer(k_agreement=2, normalize_punctuation=True)
        la.update("Munich, Germany")
        r2 = la.update("Munich Germany is")
        # "Munich" matches "Munich," under punctuation normalization
        self.assertIn("Munich", r2["all_committed_text"])

    # -------------------------------------------------------------------------
    # 2. VAD Pre-Roll Buffer Tests
    # -------------------------------------------------------------------------
    def test_vad_preroll_buffer_capture(self):
        """VAD maintains pre-roll frames during silence and emits them at onset."""
        frame_dur = 0.100
        preroll_dur = 0.200  # 2 frames
        vad = EnergyVAD(
            min_speech_frames=2,
            endpoint_silence_sec=0.400,
            frame_duration_sec=frame_dur,
            preroll_duration_sec=preroll_dur
        )

        silence = tuple([0] * 1600)
        loud_voice = tuple([2500] * 1600)

        # Feed 3 silence frames
        for i in range(3):
            vad.process_frame(silence, i * frame_dur)

        # Hysteresis onset: frame 1
        r1 = vad.process_frame(loud_voice, 0.3)
        self.assertFalse(r1["is_speech"])

        # Frame 2 triggers SPEECH_START and emits pre-roll frames
        r2 = vad.process_frame(loud_voice, 0.4)
        self.assertTrue(r2["is_speech"])
        self.assertEqual(r2["event"], "SPEECH_START")
        self.assertGreater(len(r2["preroll_frames"]), 0)
        self.assertLessEqual(len(r2["preroll_frames"]), vad.preroll_max_frames)

    # -------------------------------------------------------------------------
    # 3. Three-Tier Adaptive Endpointing Tests
    # -------------------------------------------------------------------------
    def test_thinking_pause_bridge_prevents_false_cuts(self):
        """
        TM_A01 has an 800ms thinking pause (pause_start=1.85, pause_end=2.65).
        - max_pause_bridge_sec = 0.500s -> Premature cut (false cut = 1)
        - max_pause_bridge_sec = 1.000s -> Bridges pause (false cut = 0)
        """
        # Test premature cut when bridge threshold is too short
        runner_short_bridge = VariantCRunner(
            k_agreement=2,
            short_endpoint_sec=0.400,
            max_pause_bridge_sec=0.500  # < 800ms pause!
        )
        res_cut = runner_short_bridge.run_example(self.sample_spk, self.ref_group, "de")
        self.assertGreater(res_cut.false_cuts_in_pause, 0, "Expected false cut with 500ms bridge")
        self.assertLess(res_cut.chrf, 100.0, "Expected quality degradation due to premature cut")

        # Test successful bridge with recommended 1.000s bridge
        runner_bridged = VariantCRunner(
            k_agreement=2,
            short_endpoint_sec=0.400,
            max_pause_bridge_sec=1.000  # >= 800ms pause!
        )
        res_ok = runner_bridged.run_example(self.sample_spk, self.ref_group, "de")
        self.assertEqual(res_ok.false_cuts_in_pause, 0, "Expected zero false cuts with 1000ms bridge")
        self.assertEqual(res_ok.chrf, 100.0, "Expected 100% chrF++ with bridged pause")

    def test_forced_duration_cap_tier3(self):
        """
        When speech segment duration exceeds forced_endpoint_sec,
        Variant C forces a segmentation cut to bound latency.
        """
        # TM_A01 total speech is ~4.5s. If forced cap is 2.5s, it must segment mid-turn.
        runner_forced = VariantCRunner(
            k_agreement=2,
            decode_interval_sec=0.500,
            short_endpoint_sec=0.400,
            max_pause_bridge_sec=1.000,
            forced_endpoint_sec=2.500  # Forces cut at 2.5s
        )
        res = runner_forced.run_example(self.sample_spk, self.ref_group, "de")
        self.assertGreater(len(res.predicted_boundaries_sec), 1, "Expected forced mid-turn cut")

    # -------------------------------------------------------------------------
    # 4. Periodic Decode Interval & TTFA Scaling
    # -------------------------------------------------------------------------
    def test_decode_interval_ttfa_scaling(self):
        """
        Time-to-First-Audio scales directly with k * decode_interval_sec:
        TTFA(decode=0.4s) < TTFA(decode=0.7s) < TTFA(decode=1.0s)
        """
        runner_fast = VariantCRunner(k_agreement=1, decode_interval_sec=0.400)
        runner_med = VariantCRunner(k_agreement=2, decode_interval_sec=0.700)
        runner_slow = VariantCRunner(k_agreement=3, decode_interval_sec=0.900)

        res_fast = runner_fast.run_example(self.sample_spk, self.ref_group, "de")
        res_med = runner_med.run_example(self.sample_spk, self.ref_group, "de")
        res_slow = runner_slow.run_example(self.sample_spk, self.ref_group, "de")

        self.assertLess(res_fast.ttfa_ms, res_med.ttfa_ms)
        self.assertLess(res_med.ttfa_ms, res_slow.ttfa_ms)

    # -------------------------------------------------------------------------
    # 5. Metadata Schema & Parameter Preservation
    # -------------------------------------------------------------------------
    def test_metadata_parameter_schema(self):
        """Validates that all configured parameters are preserved in result metadata."""
        runner = VariantCRunner(
            k_agreement=2,
            decode_interval_sec=0.650,
            short_endpoint_sec=0.350,
            safe_endpoint_sec=0.700,
            max_pause_bridge_sec=0.950,
            forced_endpoint_sec=5.500,
            energy_threshold_multiplier=3.2,
            preroll_sec=0.250,
            overlap_context_sec=0.550
        )
        res = runner.run_example(self.sample_spk, self.ref_group, "fr")
        m = res.metadata

        self.assertEqual(m["k_agreement"], 2)
        self.assertEqual(m["decode_interval_sec"], 0.650)
        self.assertEqual(m["short_endpoint_sec"], 0.350)
        self.assertEqual(m["safe_endpoint_sec"], 0.700)
        self.assertEqual(m["max_pause_bridge_sec"], 0.950)
        self.assertEqual(m["forced_endpoint_sec"], 5.500)
        self.assertEqual(m["energy_threshold_multiplier"], 3.2)
        self.assertEqual(m["preroll_sec"], 0.250)
        self.assertEqual(m["overlap_context_sec"], 0.550)


if __name__ == "__main__":
    unittest.main()
