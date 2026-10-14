"""
tests/test_pipeline.py
=======================
Unit tests for pipeline components: EnergyVAD and LocalAgreementStabilizer.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.vad_engine import EnergyVAD
from src.pipeline.local_agreement import LocalAgreementStabilizer


class TestPipeline(unittest.TestCase):

    def test_local_agreement_k2(self):
        la = LocalAgreementStabilizer(k_agreement=2)

        # Step 1: initial hypothesis
        res1 = la.update("the flight")
        self.assertEqual(res1["newly_committed"], [])
        self.assertEqual(res1["all_committed_text"], "")
        self.assertEqual(res1["uncommitted_text"], "the flight")

        # Step 2: second hypothesis agrees on "the flight" and adds "to"
        res2 = la.update("the flight to")
        self.assertEqual(res2["newly_committed"], ["the", "flight"])
        self.assertEqual(res2["all_committed_text"], "the flight")
        self.assertEqual(res2["uncommitted_text"], "to")

        # Step 3: third hypothesis agrees on "to" and adds "Berlin"
        res3 = la.update("the flight to Berlin")
        self.assertEqual(res3["newly_committed"], ["to"])
        self.assertEqual(res3["all_committed_text"], "the flight to")
        self.assertEqual(res3["uncommitted_text"], "Berlin")

        # Flush
        final = la.flush()
        self.assertEqual(final, "the flight to Berlin")

    def test_vad_silence_detection(self):
        vad = EnergyVAD(endpoint_silence_sec=0.300, frame_duration_sec=0.100)
        silence_samples = tuple([0] * 1600)  # 100ms silence at 16kHz

        # Feed silence frames
        for i in range(5):
            res = vad.process_frame(silence_samples, current_time_sec=i * 0.1)
            self.assertFalse(res["is_speech"])
            self.assertFalse(res["endpoint_triggered"])

    def test_vad_speech_and_endpoint(self):
        vad = EnergyVAD(endpoint_silence_sec=0.200, frame_duration_sec=0.100)
        speech_samples = tuple([2000] * 1600)  # loud voice frames
        silence_samples = tuple([0] * 1600)

        # Onset hysteresis: needs min_speech_frames=2
        r1 = vad.process_frame(speech_samples, 0.1)
        self.assertFalse(r1["is_speech"])
        r2 = vad.process_frame(speech_samples, 0.2)
        self.assertTrue(r2["is_speech"])

        # Continuous speech
        r3 = vad.process_frame(speech_samples, 0.3)
        self.assertTrue(r3["is_speech"])

        # Silence frame 1 (100ms) - should still be in speech
        r4 = vad.process_frame(silence_samples, 0.4)
        self.assertTrue(r4["is_speech"])
        self.assertFalse(r4["endpoint_triggered"])

        # Silence frame 2 (200ms) - reaches 0.200s endpoint threshold!
        r5 = vad.process_frame(silence_samples, 0.5)
        self.assertTrue(r5["endpoint_triggered"])
        self.assertFalse(r5["is_speech"])


if __name__ == "__main__":
    unittest.main()
