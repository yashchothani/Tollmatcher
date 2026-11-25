"""
tests/test_web_lab.py
======================
Automated test suite for TollMatcher Speech Translation Lab.

Validates:
1. Backend health check and supported languages.
2. Directional routing for all six language directions:
   - EN -> DE & FR
   - DE -> EN & FR
   - FR -> EN & DE
3. Prevention of translation into own language.
4. Manual language override vs Auto-detection.
5. Live audio translation with benchmark WAV fixtures.
6. Handling empty audio, corrupted bytes, and invalid file sizes.
7. Benchmark FLORES-200 sample evaluation (WER, CER, BLEU, chrF).
8. Result export schema validation.
"""

import os
import sys
import unittest
import json
import io

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from web_lab.backend_adapter import TollMatcherAdapter, LANGUAGE_METADATA, TARGET_ROUTING


class TestWebLabAdapter(unittest.TestCase):
    """Unit tests for the TollMatcher backend adapter."""

    def setUp(self):
        self.adapter = TollMatcherAdapter()

    def test_health_and_capabilities(self):
        """Validates adapter initialization, execution mode, and metadata."""
        self.assertIsNotNone(self.adapter.execution_mode)
        self.assertIn("Offline", self.adapter.execution_mode)
        self.assertGreater(len(self.adapter.pilot_groups), 0)
        self.assertGreater(len(self.adapter.speech_examples), 0)

    def test_supported_languages(self):
        """Verifies the three required languages (EN, DE, FR) are present."""
        lang_data = self.adapter.get_supported_languages()
        langs = lang_data["languages"]
        codes = [l["code"] for l in langs]
        self.assertIn("en", codes)
        self.assertIn("de", codes)
        self.assertIn("fr", codes)
        self.assertEqual(len(codes), 3)

    def test_all_six_translation_directions_no_self_translation(self):
        """
        Verifies that every supported source maps to the other TWO languages,
        and never translates to itself.
        """
        # Direction 1 & 2: EN -> DE & FR
        en_targets = [t[0] for t in self.adapter.get_targets_for_source("en")]
        self.assertEqual(en_targets, ["de", "fr"])
        self.assertNotIn("en", en_targets)

        # Direction 3 & 4: DE -> EN & FR
        de_targets = [t[0] for t in self.adapter.get_targets_for_source("de")]
        self.assertEqual(de_targets, ["en", "fr"])
        self.assertNotIn("de", de_targets)

        # Direction 5 & 6: FR -> EN & DE
        fr_targets = [t[0] for t in self.adapter.get_targets_for_source("fr")]
        self.assertEqual(fr_targets, ["en", "de"])
        self.assertNotIn("fr", fr_targets)

    def test_prohibit_unsupported_language(self):
        """Checks error handling when an unsupported source language is supplied."""
        with self.assertRaises(ValueError):
            self.adapter.get_targets_for_source("es")
        with self.assertRaises(ValueError):
            self.adapter.get_targets_for_source("zh")

    def test_prohibit_self_translation(self):
        """Checks error when self-translation is explicitly requested."""
        with self.assertRaises(ValueError):
            self.adapter.translate_single_direction("en", "en", "Hello world")
        with self.assertRaises(ValueError):
            self.adapter.translate_single_direction("de", "de", "Guten Tag")

    def test_text_translation_all_directions(self):
        """Tests dual text translations across all 3 source languages using FLORES examples."""
        # EN source (Example TM_G01)
        res_en = self.adapter.translate_text("The pilot was identified as Squadron Leader Dilokrit Pattavee.", "en")
        self.assertEqual(res_en["source_language"], "en")
        self.assertEqual(len(res_en["translations"]), 2)
        target_codes_en = [t["language_code"] for t in res_en["translations"]]
        self.assertEqual(target_codes_en, ["de", "fr"])
        self.assertTrue(any(term in res_en["translations"][0]["text"] for term in ["Pilot", "Staffelführer", "Dilokrit"]))

        # DE source (Example TM_G01)
        res_de = self.adapter.translate_text("Der Pilot wurde als Staffelführer Dilokrit Pattavee identifiziert.", "de")
        self.assertEqual(res_de["source_language"], "de")
        self.assertEqual(len(res_de["translations"]), 2)
        target_codes_de = [t["language_code"] for t in res_de["translations"]]
        self.assertEqual(target_codes_de, ["en", "fr"])
        self.assertTrue(any(term in res_de["translations"][0]["text"] for term in ["pilot", "Pilot", "Dilokrit"]))

        # FR source (Example TM_G01)
        res_fr = self.adapter.translate_text("Le pilote a été identifié comme étant le chef d’escadron Dilokrit Pattavee.", "fr")
        self.assertEqual(res_fr["source_language"], "fr")
        self.assertEqual(len(res_fr["translations"]), 2)
        target_codes_fr = [t["language_code"] for t in res_fr["translations"]]
        self.assertEqual(target_codes_fr, ["en", "de"])
        self.assertTrue(any(term in res_fr["translations"][0]["text"] for term in ["pilot", "Pilot", "Dilokrit"]))

    def test_audio_translation_with_benchmark_fixture(self):
        """Tests end-to-end audio translation with an actual benchmark WAV fixture."""
        wav_path = os.path.join(PROJECT_ROOT, "benchmark", "audio", "en", "TM_A01.wav")
        self.assertTrue(os.path.exists(wav_path), f"Audio fixture missing: {wav_path}")

        with open(wav_path, "rb") as f:
            audio_bytes = f.read()

        result = self.adapter.translate_audio(audio_bytes, filename="TM_A01.wav", manual_language="en")
        self.assertEqual(result["source_language"], "en")
        self.assertIsNotNone(result["transcript"])
        self.assertEqual(len(result["translations"]), 2)
        self.assertGreater(result["timing"]["audio_duration_sec"], 0)
        self.assertGreater(result["timing"]["total_ms"], 0)
        self.assertIsNotNone(result["timing"]["rtf"])

    def test_empty_audio_rejection(self):
        """Rejects empty or corrupted audio bytes."""
        with self.assertRaises(ValueError):
            self.adapter.translate_audio(b"", filename="empty.wav")
        with self.assertRaises(ValueError):
            self.adapter.translate_audio(b"12345", filename="corrupt.wav")

    def test_audio_size_limit(self):
        """Rejects audio exceeding 25 MB limit."""
        huge_bytes = b"0" * (26 * 1024 * 1024)
        with self.assertRaises(ValueError):
            self.adapter.translate_audio(huge_bytes, filename="huge.wav")

    def test_benchmark_evaluation(self):
        """Tests calculation of WER, CER, BLEU, and chrF on benchmark records."""
        eval_res = self.adapter.evaluate_benchmark_sample(
            group_id="TM_G01",
            source_lang="en"
        )
        self.assertEqual(eval_res["sentence_group_id"], "TM_G01")
        self.assertIn("wer", eval_res["asr_metrics"])
        self.assertIn("cer", eval_res["asr_metrics"])
        self.assertEqual(len(eval_res["translation_eval"]), 2)
        self.assertIn("bleu", eval_res["translation_eval"][0])
        self.assertIn("chrf", eval_res["translation_eval"][0])

    def test_export_record_schema(self):
        """Verifies evaluation records contain all required export attributes."""
        eval_res = self.adapter.evaluate_benchmark_sample(
            group_id="TM_G01",
            source_lang="de"
        )
        required_keys = ["sentence_group_id", "source_language", "gold_transcript", "hypothesis_transcript", "asr_metrics", "translation_eval"]
        for key in required_keys:
            self.assertIn(key, eval_res)


if __name__ == "__main__":
    unittest.main()
