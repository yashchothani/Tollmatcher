"""
tests/test_hailo_integration.py
================================
Unit tests for Option C: AI HAT+ (Hailo-8 NPU) Whisper engine and edge runner.
Verifies:
1. HailoWhisperEngine initialization, device metadata, and fallback handling.
2. End-to-end transcription accuracy on benchmark audio fixtures.
3. TollMatcherHailoRunner language pair routing and 3-tier timing telemetry.
"""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import wave
import numpy as np

from pi_edge.hailo_whisper import HailoWhisperEngine
from pi_edge.edge_hailo_runner import (
    TollMatcherHailoRunner,
    _score_input_device,
    suppress_native_stderr,
    PIN_BUTTON1,
    PIN_BUTTON2,
    PIN_BUTTON3,
    PIN_BUTTON4,
    PIN_BUTTON5
)
from pi_edge.grammar_corrector import clean_human_transcript
from src.pipeline.vad_engine import EnergyVAD
from src.pipeline.local_agreement import LocalAgreementStabilizer


class TestHailoIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_wav = os.path.join(PROJECT_ROOT, "benchmark", "audio", "en", "TM_A01.wav")
        cls.assertTrue(os.path.isfile(cls.test_wav), f"Missing test audio fixture: {cls.test_wav}")

    def test_hailo_engine_metadata(self):
        engine = HailoWhisperEngine(model_size="tiny")
        info = engine.get_device_info()

        self.assertIn("hardware_accelerated", info)
        self.assertIn("device_name", info)
        self.assertIn("model_size", info)
        self.assertIn("hailort_available", info)
        self.assertEqual(info["model_size"], "tiny")

    def test_hailo_engine_transcription(self):
        engine = HailoWhisperEngine(model_size="tiny")
        text, det_lang, conf, latency = engine.transcribe(self.test_wav)

        self.assertEqual(det_lang, "en")
        self.assertGreater(conf, 0.8)
        self.assertIn("pilot", text.lower())
        self.assertGreater(latency, 0.0)

    def test_hailo_runner_pipeline(self):
        runner = TollMatcherHailoRunner(user_lang="de", guest_lang="en")
        res = runner.process_speech_file(self.test_wav)

        self.assertEqual(res["detected_language"], "en")
        self.assertIn("de", res["translations"])
        self.assertIn("fr", res["translations"])
        self.assertNotIn("en", res["translations"], "Should not self-translate into source language")

        # Telemetry
        self.assertIn("timing", res)
        self.assertGreater(res["timing"]["asr_sec"], 0.0)
        self.assertGreaterEqual(res["timing"]["translation_sec"], 0.0)
        self.assertGreater(res["timing"]["total_sec"], 0.0)

    def test_fast_in_ram_streaming_transcription(self):
        engine = HailoWhisperEngine(model_size="tiny")
        with wave.open(self.test_wav, "rb") as wf:
            pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        audio_f32 = pcm.astype(np.float32) / 32768.0
        text, det_lang, conf, latency = engine.transcribe(audio_f32, language="en", fast_mode=True)

        self.assertEqual(det_lang, "en")
        self.assertIn("pilot", text.lower())
        self.assertGreater(latency, 0.0)

    def test_microphone_priority_scoring(self):
        self.assertGreater(
            _score_input_device("SF-558: USB Audio (hw:3,0)"),
            _score_input_device("USB Audio Device: - (hw:2,0)"),
            "Dedicated USB microphone must rank above generic USB speaker dongle"
        )
        self.assertGreater(
            _score_input_device("EPOS USB Headset (hw:2,0)"),
            _score_input_device("default")
        )
        with suppress_native_stderr():
            pass

    def test_vad_ambient_calibration_and_preservation(self):
        vad = EnergyVAD(min_rms_threshold=120.0)
        vad.calibrate_noise_floor(200.0, headroom_multiplier=2.6)
        self.assertEqual(vad.noise_floor, 200.0)
        self.assertGreaterEqual(vad.min_rms_threshold, 520.0)

        # Ambient lab noise at 220 RMS must NOT trigger speech after calibration
        lab_noise = tuple([220] * 1600)
        r1 = vad.process_frame(lab_noise, 0.1)
        r2 = vad.process_frame(lab_noise, 0.2)
        self.assertFalse(r1["is_speech"])
        self.assertFalse(r2["is_speech"])

        # Preserving adapted noise floor across turns
        adapted_floor = vad.noise_floor
        self.assertGreater(adapted_floor, 150.0)
        vad.reset(preserve_noise_floor=True)
        self.assertEqual(vad.noise_floor, adapted_floor)
        vad.reset(preserve_noise_floor=False)
        self.assertEqual(vad.noise_floor, 100.0)

    def test_simultaneous_local_agreement_emission_no_duplicates(self):
        la = LocalAgreementStabilizer(k_agreement=2)
        emitted = []
        emitted_idx = 0
        hypotheses = [
            "Hello! Good morning!",
            "Hello! Good morning! How are you",
            "Hello! Good morning! How are you doing? I am in the lab right now."
        ]
        for hyp in hypotheses:
            la.update(hyp)
            pending = la.committed_words[emitted_idx:]
            if pending and (len(pending) >= 4 or pending[-1][-1] in ".!?,;:"):
                emitted_idx = len(la.committed_words)
                emitted.extend(pending)
        full_text = la.flush()
        emitted.extend(full_text.split()[emitted_idx:])
        self.assertEqual(" ".join(emitted), full_text)

    def test_grammar_anti_hallucination_guard(self):
        # 1. Prompt recitation rejection
        self.assertEqual(
            clean_human_transcript("Toll, highway, vignette, ticket, cash, contactless, payment.", "en"),
            ""
        )
        self.assertEqual(
            clean_human_transcript("TollMatcher speech translation for human conversation.", "en"),
            ""
        )
        # 2. Chinese / CJK character hallucination rejection (Turn #2 fix)
        self.assertEqual(
            clean_human_transcript("微微微微微微微微微微微微微微微微微微微微", "en"),
            ""
        )
        # 3. Character-level repetition loop rejection
        self.assertEqual(
            clean_human_transcript("weieieieieiei", "en"),
            ""
        )
        # 4. Human conversational cleaning
        self.assertEqual(
            clean_human_transcript("um how much is the actual?", "en"),
            "How much is the toll?"
        )

    def test_whisper_noise_hallucination_guard(self):
        """Pure noise must decode to empty string with zero foreign glyphs."""
        engine = HailoWhisperEngine(model_size="tiny")
        noise = np.random.randn(16000).astype(np.float32) * 0.005
        text, det_lang, conf, lat = engine.transcribe(noise, language=None, fast_mode=True)
        self.assertEqual(text, "", "Pure noise must not produce hallucinated words")
        self.assertIn(det_lang, {"en", "de", "fr"}, "Language must be constrained to {en, de, fr}")

    def test_five_button_pin_mappings_and_specification(self):
        """Verifies exact GPIO pins matching hardware layout diagram."""
        self.assertEqual(PIN_BUTTON1, 27, "Button 1 (Guest Mic) must be GPIO 27")
        self.assertEqual(PIN_BUTTON2, 22, "Button 2 (User Mic) must be GPIO 22")
        self.assertEqual(PIN_BUTTON3, 20, "Button 3 (Source Language) must be GPIO 20")
        self.assertEqual(PIN_BUTTON4, 21, "Button 4 (Target Language) must be GPIO 21")
        self.assertEqual(PIN_BUTTON5, 16, "Button 5 (Language Detection) must be GPIO 16")

    def test_five_gpio_button_actions_and_language_routing(self):
        """Verifies language cycling and push-to-talk routing for all 5 buttons."""
        runner = TollMatcherHailoRunner(user_lang="de", guest_lang="en", model_size="tiny")

        # 1. Test Button 3 (Source Language Cycle)
        initial_source = runner.user_lang  # "de"
        runner._cycle_source_lang()
        self.assertEqual(runner.user_lang, "fr", "Button 3 must cycle DE -> FR")
        runner._cycle_source_lang()
        self.assertEqual(runner.user_lang, "en", "Button 3 must cycle FR -> EN")
        runner._cycle_source_lang()
        self.assertEqual(runner.user_lang, "de", "Button 3 must cycle EN -> DE")

        # 2. Test Button 4 (Target Language Cycle)
        initial_target = runner.guest_lang  # "en"
        runner._cycle_target_lang()
        self.assertEqual(runner.guest_lang, "de", "Button 4 must cycle EN -> DE")
        runner._cycle_target_lang()
        self.assertEqual(runner.guest_lang, "fr", "Button 4 must cycle DE -> FR")
        runner._cycle_target_lang()
        self.assertEqual(runner.guest_lang, "en", "Button 4 must cycle FR -> EN")

        # 3. Test Button 1 (Guest Mic: English audio fixture -> translates to User German)
        runner.user_lang = "de"
        runner.guest_lang = "en"
        runner.recorder.is_recording = True
        runner.recorder.stop_recording = lambda: self.test_wav

        # Mock speak_out_loud to capture output
        spoken = []
        runner.speak_out_loud = lambda txt, lng: spoken.append((txt, lng))

        runner._on_button_release("guest")
        self.assertTrue(len(spoken) > 0, "Button 1 must speak translated output")
        self.assertEqual(spoken[-1][1], "de", "Button 1 Guest audio must translate to User language [DE]")

        # 4. Test Button 5 (Auto-Detect: auto-detects English -> translates to DE and FR)
        spoken.clear()
        runner.recorder.is_recording = True
        runner._on_button_release("auto")
        self.assertTrue(len(spoken) > 0, "Button 5 must speak translation")
        self.assertIn(spoken[-1][1], ["de", "fr"], "Button 5 must output to non-detected target language")


if __name__ == "__main__":
    unittest.main()
