"""
tests/test_runners.py
======================
Unit tests for algorithm runners: VariantARunner, VariantBRunner, VariantCRunner.
"""

import json
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.runners.variant_a import VariantARunner
from src.runners.variant_b import VariantBRunner
from src.runners.variant_c import VariantCRunner

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEECH_JSONL = os.path.join(WORKSPACE_ROOT, "benchmark", "data", "speech_dataset.jsonl")
PILOT_JSONL = os.path.join(WORKSPACE_ROOT, "benchmark", "data", "pilot_dataset.jsonl")


class TestRunners(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(SPEECH_JSONL, "r", encoding="utf-8") as f:
            cls.speech_records = [json.loads(line) for line in f if line.strip()]

        with open(PILOT_JSONL, "r", encoding="utf-8") as f:
            cls.pilot_groups = {json.loads(line)["sentence_group_id"]: json.loads(line) for line in f if line.strip()}

    def test_variant_a_execution(self):
        runner = VariantARunner(chunk_sec=1.0)
        spk = self.speech_records[0]  # TM_A01 (EN)
        ref = self.pilot_groups[spk["reference_translation_ids"][0]]
        res = runner.run_example(spk, ref, target_lang="de")

        self.assertEqual(res.variant_name, "Variant A (Fixed 1.0s Chunks)")
        self.assertGreater(len(res.predicted_boundaries_sec), 1)
        self.assertGreater(res.rtf, 0.0)

    def test_variant_b_false_cut(self):
        runner = VariantBRunner(endpoint_silence_sec=0.600)
        spk = self.speech_records[0]  # TM_A01 has an 800ms thinking pause
        ref = self.pilot_groups[spk["reference_translation_ids"][0]]
        res = runner.run_example(spk, ref, target_lang="de")

        self.assertEqual(res.variant_name, "Variant B (VAD Pause Endpointing)")
        # Because TM_A01 has an 800ms pause and threshold is 600ms, Variant B triggers premature cut!
        self.assertEqual(res.false_cuts_in_pause, 1)

    def test_variant_c_bridges_pause(self):
        runner = VariantCRunner(k_agreement=2, base_silence_sec=0.400, max_pause_bridge_sec=1.000)
        spk = self.speech_records[0]  # TM_A01 has an 800ms thinking pause
        ref = self.pilot_groups[spk["reference_translation_ids"][0]]
        res = runner.run_example(spk, ref, target_lang="de")

        self.assertEqual(res.variant_name, "Variant C (Streaming Adaptive)")
        # Variant C adaptively bridges the 800ms pause: zero false cuts!
        self.assertEqual(res.false_cuts_in_pause, 0)
        self.assertEqual(res.bleu, 100.0)


if __name__ == "__main__":
    unittest.main()
