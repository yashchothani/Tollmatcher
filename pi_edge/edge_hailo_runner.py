"""
pi_edge/edge_hailo_runner.py
=============================
Main application runner for TollMatcher with Raspberry Pi AI HAT+ 2 (Hailo-8 NPU).

Combines:
1. Hailo-8 NPU Accelerated Whisper (via HailoWhisperEngine).
2. TollMatcher 3-Tier Streaming Pipeline:
   - EnergyVAD with 200ms pre-roll buffer (pure NumPy, zero librosa).
   - Local Agreement (k=2) hypothesis stabilization.
   - Three-Tier Adaptive Endpointing (400ms short / 1000ms thinking pause bridge / 6.0s forced cap).
3. CTranslate2 INT8 Argos translation (< 50ms per sentence).
4. Piper TTS for audio output on the USB speaker.
5. GPIO Buttons (Buttons 1, 2, 3, 4, 5) or Interactive CLI mode.
"""

import argparse
import os
import sys
import time
import wave
import numpy as np
from typing import Dict, Any, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pi_edge.hailo_whisper import HailoWhisperEngine
from src.pipeline.vad_engine import EnergyVAD
from src.pipeline.local_agreement import LocalAgreementStabilizer

# GPIO Button imports
try:
    from gpiozero import Button
    HAS_GPIO = True
except (ImportError, Exception):
    HAS_GPIO = False

import threading
try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

# Physical Pin Definitions from CAD Specification & 5-Button Hardware Layout
PIN_BUTTON1 = 27  # Button 1 (GPIO 27): Guest Mic (Hold to speak as Guest -> User Lang)
PIN_BUTTON2 = 22  # Button 2 (GPIO 22): User Mic (Hold to speak as User -> Guest Lang)
PIN_BUTTON3 = 20  # Button 3 (GPIO 20): Source Language (Tap to cycle EN -> DE -> FR)
PIN_BUTTON4 = 21  # Button 4 (GPIO 21): Target Language (Tap to cycle EN -> DE -> FR)
PIN_BUTTON5 = 16  # Button 5 (GPIO 16): Language Detection (Hold to speak with Auto-Detection EN/DE/FR)


import contextlib

# ALSA error handler suppression for clean terminal output
from ctypes import CFUNCTYPE, c_char_p, c_int, cdll
try:
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    def py_error_handler(filename, line, function, err, fmt):
        pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except Exception:
    pass


@contextlib.contextmanager
def suppress_native_stderr():
    """
    Temporarily redirects C-level stderr (fd 2) to /dev/null during PortAudio/JACK
    device probing so harmless 'jack server is not running' messages don't clutter output.
    """
    try:
        stderr_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(stderr_fd)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, stderr_fd)
        os.close(devnull_fd)
        try:
            yield
        finally:
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stderr_fd)
    except Exception:
        yield


def _score_input_device(name: str) -> int:
    """
    Ranks audio input devices so dedicated USB microphones (e.g. SF-558) are
    selected ahead of generic USB speaker/soundcard dongles ('USB Audio Device')
    that expose an unused 3.5mm mic jack.
    """
    low = name.lower()
    # Tier 4: Dedicated USB microphones
    dedicated_mics = [
        "sf-558", "sf558", "microphone", "mic", "music-boost",
        "snowball", "yeti", "fifine", "boya", "tonor", "maono",
        "rode", "hyperx", "samson", "pnp sound"
    ]
    if any(k in low for k in dedicated_mics):
        return 4
    # Tier 3: Headsets, UAC audio, or webcams with integrated mics
    if any(k in low for k in ["headset", "epos", "logitech", "uac", "webcam", "camera"]):
        return 3
    # Tier 1: Generic 'USB Audio Device' (often a GeneralPlus USB speaker adapter)
    if "usb audio device" in low:
        return 1
    # Tier 2: Other USB audio cards
    if any(k in low for k in ["usb", "card"]):
        return 2
    # Tier 0: Virtual / system defaults (pulse, pipewire, default)
    return 0


def list_input_devices(pa):
    """Returns a list of (index, name, max_channels, priority, default_sr) for all input devices."""
    devices = []
    for i in range(pa.get_device_count()):
        try:
            info = pa.get_device_info_by_index(i)
            max_in = int(info.get("maxInputChannels", 0))
            if max_in > 0:
                name = info.get("name", f"Device {i}")
                priority = _score_input_device(name)
                default_sr = int(info.get("defaultSampleRate", 48000))
                devices.append((i, name, max_in, priority, default_sr))
        except Exception:
            continue
    return devices


def find_best_input_device(pa, preferred_idx: Optional[int] = None) -> Tuple[Optional[int], str, int, int]:
    """
    Scans all audio devices to locate the connected USB microphone.
    Probes hardware-supported sample rates and channel counts.
    Returns: (device_index, device_name, native_sample_rate, channels)
    """
    candidate_devices = list_input_devices(pa)

    if not candidate_devices:
        return None, "Default Device", 48000, 1

    if preferred_idx is not None:
        matched = [d for d in candidate_devices if d[0] == preferred_idx]
        if matched:
            best_idx, best_name, max_ch, _, default_sr = matched[0]
        else:
            print(f"[Audio Warning] Requested --mic-index {preferred_idx} not found. Auto-selecting best microphone.")
            candidate_devices.sort(key=lambda x: (x[3], x[2]), reverse=True)
            best_idx, best_name, max_ch, _, default_sr = candidate_devices[0]
    else:
        # Sort by dedicated microphone priority first, then channel count
        candidate_devices.sort(key=lambda x: (x[3], x[2]), reverse=True)
        best_idx, best_name, max_ch, _, default_sr = candidate_devices[0]

    channels = 1 if max_ch >= 1 else 2
    supported_rate = None

    # Test candidate sample rates in order of preference
    with suppress_native_stderr():
        for sr in [default_sr, 48000, 44100, 16000, 32000, 22050, 8000]:
            try:
                if pa.is_format_supported(
                    rate=sr,
                    input_device=best_idx,
                    input_channels=channels,
                    input_format=pyaudio.paInt16
                ):
                    supported_rate = sr
                    break
            except Exception:
                continue

    if supported_rate is None:
        supported_rate = 48000

    return best_idx, best_name, supported_rate, channels


class EdgeAudioRecorder:
    """Captures audio from USB microphone with persistent PyAudio handle and AGC normalization."""

    def __init__(self, target_sample_rate: int = 16000, chunk: int = 1024):
        self.target_sample_rate = target_sample_rate
        self.chunk = chunk
        self.is_recording = False
        self.frames = []
        self._thread = None
        self._audio = None
        self._stream = None
        self.native_sr = 48000
        self.channels = 1
        if HAS_PYAUDIO:
            try:
                with suppress_native_stderr():
                    self._audio = pyaudio.PyAudio()
            except Exception:
                self._audio = None

    def start_recording(self):
        if not HAS_PYAUDIO or self._audio is None:
            print("[Recorder] PyAudio not available, audio recording simulated.")
            self.is_recording = True
            return
        if self.is_recording:
            return
        try:
            with suppress_native_stderr():
                dev_idx, dev_name, self.native_sr, self.channels = find_best_input_device(self._audio)
                chunk_size = int(self.native_sr * 0.100)
                self._stream = self._audio.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.native_sr,
                    input=True,
                    input_device_index=dev_idx,
                    frames_per_buffer=chunk_size
                )
            self.frames = []
            self.is_recording = True
            self._thread = threading.Thread(target=self._record_loop, args=(chunk_size,), daemon=True)
            self._thread.start()
        except Exception as e:
            print(f"[Recorder] Error opening microphone: {e}")
            self.is_recording = False

    def _record_loop(self, chunk_size):
        while self.is_recording and self._stream:
            try:
                data = self._stream.read(chunk_size, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.int16)
                if self.channels > 1:
                    samples = samples.reshape(-1, self.channels).mean(axis=1).astype(np.int16)

                # Resample to 16kHz
                if self.native_sr == 16000:
                    samples_16k = samples
                elif self.native_sr % 16000 == 0:
                    samples_16k = samples[::(self.native_sr // 16000)]
                else:
                    dur = len(samples) / self.native_sr
                    t_len = int(dur * 16000)
                    x_old = np.linspace(0, dur, len(samples), endpoint=False)
                    x_new = np.linspace(0, dur, t_len, endpoint=False)
                    samples_16k = np.interp(x_new, x_old, samples).astype(np.int16)

                self.frames.append(samples_16k.tobytes())
            except Exception:
                break

    def stop_recording(self) -> Optional[str]:
        if not self.is_recording:
            return None
        # Allow 0.25s for ALSA hardware capture buffer to drain any trailing words (e.g. 'planning')
        time.sleep(0.25)
        self.is_recording = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._stream:
            try:
                with suppress_native_stderr():
                    if self._stream.is_active():
                        self._stream.stop_stream()
                    self._stream.close()
            except Exception:
                pass
            self._stream = None

        if not self.frames:
            return None

        # Concatenate raw recorded int16 PCM
        raw_bytes = b"".join(self.frames)
        pcm_array = np.frombuffer(raw_bytes, dtype=np.int16)

        # Apply Automatic Gain Control (AGC) and trim dead silence from leading edge only (preserve trailing words)
        try:
            from pi_edge.audio_preprocessor import normalize_human_speech
            normalized_pcm = normalize_human_speech(pcm_array, target_peak=0.88, max_gain_factor=150.0, trim_edges=True, trim_tail=False)
        except Exception:
            normalized_pcm = pcm_array

        if len(normalized_pcm) == 0:
            return None

        out_wav = f"/tmp/tollmatcher_rec_{int(time.time())}.wav"
        with wave.open(out_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(normalized_pcm.tobytes())
        return out_wav


class TollMatcherHailoRunner:
    """TollMatcher Speech Translation system running on Raspberry Pi AI HAT+ 2."""

    def __init__(
        self,
        hef_path: Optional[str] = None,
        k_agreement: int = 2,
        decode_interval_sec: float = 0.700,
        short_endpoint_sec: float = 0.400,
        safe_endpoint_sec: float = 0.750,
        max_pause_bridge_sec: float = 1.000,
        forced_endpoint_sec: float = 6.000,
        preroll_sec: float = 0.200,
        overlap_context_sec: float = 0.600,
        user_lang: str = "de",
        guest_lang: str = "en",
        model_size: str = "tiny"
    ):
        self.k_agreement = k_agreement
        self.decode_interval_sec = decode_interval_sec
        self.short_endpoint_sec = short_endpoint_sec
        self.safe_endpoint_sec = safe_endpoint_sec
        self.max_pause_bridge_sec = max_pause_bridge_sec
        self.forced_endpoint_sec = forced_endpoint_sec
        self.preroll_sec = preroll_sec
        self.overlap_context_sec = overlap_context_sec
        self.model_size = model_size

        self.user_lang = user_lang
        self.guest_lang = guest_lang
        self.supported_langs = ["en", "de", "fr"]

        print("=" * 75)
        print("TOLLMATCHER — AI HAT+ 2 (HAILO-8 NPU) STREAMING ENGINE")
        print("=" * 75)

        # 1. Initialize Hailo NPU Whisper
        self.whisper = HailoWhisperEngine(hef_path=hef_path, model_size=model_size, compute_type="int8")
        info = self.whisper.get_device_info()
        print(f"[Engine] ASR Backend: {info['device_name']}")
        print(f"[Engine] Hardware NPU Active: {info['hardware_accelerated']}")

        # 2. Initialize Argos Translation
        print("[Engine] Loading CTranslate2 INT8 Translation Engine...")
        import argostranslate.translate
        self.argos = argostranslate.translate
        self._installed = {l.code: l for l in self.argos.get_installed_languages()}
        if len(self._installed) < 2:
            print("[Engine] Notice: Translation models not found. Auto-installing EN, DE, FR packages...")
            try:
                try:
                    from pi_edge.install_argos_models import main as install_argos
                except ImportError:
                    from install_argos_models import main as install_argos
                install_argos()
                self._installed = {l.code: l for l in self.argos.get_installed_languages()}
            except Exception as e:
                print(f"[Engine] Warning: Argos package install failed: {e}")
        print(f"[Engine] Active Languages: User=[{self.user_lang.upper()}], Guest=[{self.guest_lang.upper()}]")

        # 3. Audio Recording State
        self.recorder = EdgeAudioRecorder()
        self.active_speaker = None

        # 4. Setup GPIO if available
        self.buttons = {}
        if HAS_GPIO:
            print("[Engine] Initializing physical GPIO buttons (Pins 27, 22, 20, 21, 16)...")
            self._setup_buttons()
        else:
            print("[Engine] Running in CLI / Virtual Hardware Mode (GPIO unavailable).")

        # 5. Pre-warm translation models once at startup (ensures sub-30ms MT latency on first turn)
        self._warmup_translation_models()

    def speak_out_loud(self, text: str, lang: str):
        """Plays translated text through the USB speaker using Piper TTS or espeak-ng."""
        if not text.strip():
            return

        print(f"[Speaker Output] Speaking ({lang.upper()}): '{text}'")

        # 1. Try Piper TTS if models exist
        piper_bin = os.path.join(PROJECT_ROOT, "piper_rebuild", "piper", "piper")
        piper_model_map = {
            "en": os.path.join(PROJECT_ROOT, "piper_rebuild", "piper", "en_US-john-medium.onnx"),
            "de": os.path.join(PROJECT_ROOT, "text2audio", "piper", "de_DE-eva_k-x_low.onnx"),
            "fr": os.path.join(PROJECT_ROOT, "piper_rebuild", "piper", "fr_FR-tom-medium.onnx")
        }

        model_path = piper_model_map.get(lang)
        if os.path.isfile(piper_bin) and model_path and os.path.isfile(model_path):
            try:
                import subprocess
                cmd = f'echo "{text}" | {piper_bin} --model {model_path} --output-raw | aplay -r 22050 -f S16_LE -t raw'
                subprocess.run(cmd, shell=True, check=False)
                return
            except Exception as e:
                pass

        # 2. Try system espeak-ng / espeak
        import subprocess
        try:
            res = subprocess.run(["espeak-ng", "-v", lang, text], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if res.returncode == 0:
                return
        except Exception:
            pass

        try:
            subprocess.run(["espeak", "-v", lang, text], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def _warmup_translation_models(self):
        """
        Pre-initializes Argos / Stanza sentence tokenizers once before live streaming begins.
        Eliminates the ~2.5s cold-start penalty on Turn #1 so every live translation runs in <50ms.
        """
        if len(self._installed) < 2:
            return
        print("[Engine] Warming up INT8 translation models (one-time startup)...")
        t0 = time.perf_counter()
        warmup_samples = [
            ("en", "de", "Hello."),
            ("en", "fr", "Hello."),
            ("de", "en", "Hallo."),
            ("fr", "en", "Bonjour.")
        ]
        for src, tgt, sample in warmup_samples:
            try:
                if src in self._installed and tgt in self._installed:
                    self.translate_text(sample, src, tgt)
            except Exception:
                pass
        print(f"[Engine] Translation models warmed up in {(time.perf_counter() - t0)*1000:.0f} ms.")

    def _normalize_pcm_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        """Normalizes raw 16 kHz int16 PCM bytes in RAM to float32 [-1.0, 1.0] for Whisper."""
        pcm_array = np.frombuffer(pcm_bytes, dtype=np.int16)
        try:
            from pi_edge.audio_preprocessor import normalize_human_speech
        except ImportError:
            from audio_preprocessor import normalize_human_speech
        normalized_pcm = normalize_human_speech(pcm_array, target_peak=0.88, max_gain_factor=100.0)
        return normalized_pcm.astype(np.float32) / 32768.0

    def run_hands_free_streaming(
        self,
        source_lang: str = "auto",
        mic_index: Optional[int] = None,
        vad_min_rms: float = 120.0,
        speak_live: bool = False
    ):
        """
        True Simultaneous Hands-Free Streaming Mode (Zero Buttons Required).
        - Calibrates ambient room noise floor at startup so lab/fan hum never triggers false turns.
        - Background worker runs incremental Whisper ASR + Local Agreement (k=2) + Argos MT
          SIMULTANEOUSLY while the speaker is still talking.
        - Prints live translated clauses in real time (< 50 ms MT latency after warmup).
        - Uses 3-tier adaptive endpointing to finalize and speak the remaining tail with minimal wait.
        """
        import queue
        import struct
        import pyaudio

        # Warm up Argos/Stanza once so live simultaneous translations take ~30-50ms instead of 2500ms
        self._warmup_translation_models()

        lang_label = "Auto-Detected (EN/DE/FR)" if source_lang == "auto" else f"Fixed Source: {source_lang.upper()}"
        audio_mode = "Simultaneous Live TTS (Headset Mode)" if speak_live else "Live Simultaneous Text + Instant Post-Clause TTS (Echo-Safe)"
        print("\n" + "=" * 74)
        print(">>> TOLLMATCHER NOVEL STREAMING ENGINE (PDF NOVEL ALGORITHM SPEC) <<<")
        print("=" * 74)
        print("  - Pipeline     : Ring Buffer + Incremental ASR (~700ms) + Local Agreement (k=2)")
        print(f"  - Decode Step  : Periodic ~{self.decode_interval_sec*1000:.0f} ms background decode while talking")
        print(f"  - Endpoints    : Rule A: {self.short_endpoint_sec*1000:.0f} ms (stable/punct) | Rule B: {self.safe_endpoint_sec*1000:.0f} ms (silence)")
        print(f"  - Segment Cap  : Rule C: {self.forced_endpoint_sec:.1f} s max speech boundary")
        print(f"  - Overlap      : {self.overlap_context_sec*1000:.0f} ms boundary context preservation")
        print(f"  - Languages    : {lang_label} (Constrained Triad)")
        print(f"  - Output Mode  : {audio_mode}")
        print("=" * 74)

        with suppress_native_stderr():
            pa = pyaudio.PyAudio()

        all_mics = list_input_devices(pa)
        if all_mics:
            print("[Audio In] Available Input Devices:")
            for idx, name, ch, prio, sr in all_mics:
                print(f"   [{idx}] {name} ({ch} ch, {sr} Hz, priority={prio})")

        dev_idx, dev_name, native_sr, channels = find_best_input_device(pa, preferred_idx=mic_index)
        print(f"[Audio In] Selected Microphone: [{dev_idx}] {dev_name}")
        print(f"[Audio In] Hardware Rate: {native_sr} Hz | Channels: {channels}")

        chunk_native = int(native_sr * 0.100)  # 100ms at native rate
        try:
            with suppress_native_stderr():
                stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=channels,
                    rate=native_sr,
                    input=True,
                    input_device_index=dev_idx,
                    frames_per_buffer=chunk_native
                )
        except Exception as e:
            print(f"[Error] Could not open microphone device [{dev_idx}]: {e}")
            print("Please ensure your USB microphone is plugged in or pass '--mic-index <N>'.")
            pa.terminate()
            return

        def _read_16k_frame():
            try:
                raw_chunk = stream.read(chunk_native, exception_on_overflow=False)
            except Exception:
                return np.zeros(1600, dtype=np.int16)
            if not raw_chunk or len(raw_chunk) == 0:
                return np.zeros(1600, dtype=np.int16)
            samples_native = np.frombuffer(raw_chunk, dtype=np.int16)
            if channels > 1:
                samples_native = samples_native.reshape(-1, channels).mean(axis=1).astype(np.int16)
            if native_sr == 16000:
                samples_16k = samples_native
            elif native_sr % 16000 == 0:
                samples_16k = samples_native[::(native_sr // 16000)]
            else:
                duration = len(samples_native) / native_sr
                target_len = int(duration * 16000)
                x_old = np.linspace(0, duration, len(samples_native), endpoint=False)
                x_new = np.linspace(0, duration, target_len, endpoint=False)
                samples_16k = np.interp(x_new, x_old, samples_native).astype(np.int16)
            if len(samples_16k) != 1600:
                if len(samples_16k) < 1600:
                    samples_16k = np.pad(samples_16k, (0, 1600 - len(samples_16k)))
                else:
                    samples_16k = samples_16k[:1600]
            return samples_16k

        # 1.0-second startup ambient noise calibration (prevents lab fan/HVAC false triggers!)
        print("[Audio In] Calibrating room background noise (stay quiet for 1 second)...")
        calib_rms = []
        for _ in range(10):
            s16 = _read_16k_frame().astype(np.float32)
            s16 -= np.mean(s16)
            calib_rms.append(float(np.sqrt(np.mean(s16 ** 2))))
        ambient_rms = float(np.percentile(calib_rms, 75)) if calib_rms else 80.0

        vad = EnergyVAD(
            energy_threshold_multiplier=3.2,
            min_speech_frames=3,
            endpoint_silence_sec=self.short_endpoint_sec,
            frame_duration_sec=0.100,
            preroll_duration_sec=self.preroll_sec,
            min_rms_threshold=vad_min_rms
        )
        vad.calibrate_noise_floor(ambient_rms, headroom_multiplier=3.2)
        print(
            f"[Audio In] Room Noise Floor: RMS={vad.noise_floor:.1f} | "
            f"Speech Trigger Threshold: RMS={vad.min_rms_threshold:.1f}"
        )
        print("\nReady! Just speak into your microphone — translations stream live as you talk.")
        print("(Press Ctrl+C to exit)\n")

        # Optional non-blocking TTS thread for --speak-live (headset/no-echo mode)
        tts_queue: "queue.Queue[Optional[Tuple[str, str]]]" = queue.Queue()
        tts_thread = None
        if speak_live:
            def _tts_loop():
                while True:
                    item = tts_queue.get()
                    if item is None:
                        tts_queue.task_done()
                        break
                    txt, lng = item
                    self.speak_out_loud(txt, lng)
                    tts_queue.task_done()
            tts_thread = threading.Thread(target=_tts_loop, daemon=True)
            tts_thread.start()

        # Thread-safe simultaneous turn state
        state_lock = threading.Lock()
        turn_state: Dict[str, Any] = {
            "active": False,
            "frames": [],
            "decoded_frame_count": 0,
            "stabilizer": LocalAgreementStabilizer(k_agreement=self.k_agreement),
            "emitted_word_count": 0,
            "locked_lang": None if source_lang == "auto" else source_lang,
            "last_conf": 0.0,
            "last_hypothesis": "",
            "translations": {},
            "worker_busy": False,
            "worker_thread": None,
            "spoken_live_chunks": 0,
            "turn_id": 0
        }

        def _emit_live_phrase(words_slice, lang_code, target_dict, is_final=False):
            """Translates a newly committed clause immediately (<50ms) and prints it live."""
            phrase = " ".join(words_slice).strip()
            if not phrase:
                return
            primary_lang = lang_code if lang_code in self.supported_langs else self.guest_lang
            targets = [l for l in self.supported_langs if l != primary_lang][:2]
            t_mt0 = time.perf_counter()
            tr_outputs = {}
            for tgt in targets:
                tr_outputs[tgt] = self.translate_text(phrase, primary_lang, tgt)
            mt_ms = (time.perf_counter() - t_mt0) * 1000.0
            with state_lock:
                for tgt, tr_txt in tr_outputs.items():
                    target_dict.setdefault(tgt, []).append(tr_txt)
                if speak_live and targets:
                    turn_state["spoken_live_chunks"] += 1
            tag = "FINAL" if is_final else "LIVE"
            print(f"\n  [{tag} {primary_lang.upper()}]    {phrase}")
            for idx_t, (tgt, tr_txt) in enumerate(tr_outputs.items()):
                suffix = f"  ({mt_ms:.0f} ms MT)" if idx_t == 0 else ""
                print(f"  [{tag} -> {tgt.upper()}] {tr_txt}{suffix}")
            sys.stdout.flush()
            if speak_live and targets:
                primary_tgt = targets[0]
                tts_queue.put((tr_outputs[primary_tgt], primary_tgt))

        def _incremental_worker(expected_turn_id: int):
            """Runs fast in-RAM Whisper ASR + Local Agreement (~700ms) while speaker keeps talking."""
            with state_lock:
                if not turn_state["active"] or turn_state["turn_id"] != expected_turn_id:
                    turn_state["worker_busy"] = False
                    return
                pcm_bytes = b"".join(turn_state["frames"])
                snap_frames = len(turn_state["frames"])
                lang_hint = turn_state["locked_lang"]
            try:
                audio_f32 = self._normalize_pcm_to_float32(pcm_bytes)
                partial_text, det_lang, conf, _ = self.whisper.transcribe(
                    audio_f32,
                    language=lang_hint,
                    fast_mode=True
                )
                with state_lock:
                    if not turn_state["active"] or turn_state["turn_id"] != expected_turn_id:
                        return
                    turn_state["decoded_frame_count"] = snap_frames
                    if partial_text and not any(ord(c) > 0x2E80 for c in partial_text):
                        turn_state["last_hypothesis"] = partial_text
                        turn_state["last_conf"] = conf
                        if turn_state["locked_lang"] is None and conf >= 0.40:
                            turn_state["locked_lang"] = det_lang
                        turn_state["stabilizer"].update(partial_text)
                        committed = " ".join(turn_state["stabilizer"].committed_words).strip()
                        disp_text = committed if committed else partial_text
                        sys.stdout.write(f"\r  [Live ASR ~700ms] {disp_text[:50]}...   ")
                        sys.stdout.flush()
            except Exception:
                pass
            finally:
                with state_lock:
                    turn_state["worker_busy"] = False

        def _finalize_simultaneous_turn(reason_label: str, silence_sec: float = 0.0):
            """Flushes remaining tail with minimal post-speech wait and speaks with echo isolation."""
            t_post0 = time.perf_counter()
            sys.stdout.write(f" [{reason_label}]\n")
            sys.stdout.flush()

            # Wait for any in-flight incremental pass (already running in background)
            w_thread = turn_state.get("worker_thread")
            if w_thread is not None and w_thread.is_alive():
                w_thread.join(timeout=8.0)

            with state_lock:
                frames_snapshot = list(turn_state["frames"])
                decoded_so_far = turn_state["decoded_frame_count"]
                lang_hint = turn_state["locked_lang"]
                last_hyp = turn_state["last_hypothesis"]
                last_conf = turn_state["last_conf"]
                stabilizer = turn_state["stabilizer"]
                cur_turn_id = turn_state["turn_id"]
                turn_state["active"] = False

            if not frames_snapshot:
                return

            # Compute how many actual speech frames arrived after the last background decode
            # (subtracting silence frames added while waiting for the endpoint)
            silence_frames = int(round(silence_sec * 10))
            speech_frames_at_end = max(0, len(frames_snapshot) - silence_frames)
            new_speech_frames = speech_frames_at_end - decoded_so_far

            final_asr_sec = 0.0
            det_lang = lang_hint or self.guest_lang
            conf = last_conf

            # Determine whether a final ASR pass is strictly needed:
            # 1. If no hypothesis was produced yet (e.g. short utterance < 1s), ASR is required.
            # 2. If the background hypothesis already ended with terminal punctuation (e.g. . ! ?),
            #    the sentence is already complete — skip redundant re-transcription for instant (<50ms) response!
            # 3. If there is substantial new speech (>= 8 frames / 800ms) that arrived after the worker snapshot
            #    and the hypothesis was incomplete, do a final decode.
            needs_final_asr = False
            if not last_hyp:
                needs_final_asr = True
            elif new_speech_frames >= 8 and not (last_hyp.rstrip().endswith((".", "!", "?"))):
                needs_final_asr = True

            if needs_final_asr:
                pcm_bytes = b"".join(frames_snapshot)
                audio_f32 = self._normalize_pcm_to_float32(pcm_bytes)
                final_text, det_lang_new, conf_new, final_asr_sec = self.whisper.transcribe(
                    audio_f32,
                    language=lang_hint,
                    fast_mode=True
                )
                if final_text:
                    det_lang = lang_hint or det_lang_new
                    conf = conf_new
                    stabilizer.update(final_text)

            full_transcript = stabilizer.flush().strip()
            if not full_transcript and last_hyp:
                full_transcript = last_hyp.strip()
            if not full_transcript or len(full_transcript) < 2 or any(ord(c) > 0x2E80 for c in full_transcript):
                print(" [Turn ignored: Ambient noise / Inaudible speech]")
                return

            primary_lang = det_lang if det_lang in self.supported_langs else self.guest_lang
            targets = [l for l in self.supported_langs if l != primary_lang][:2]

            # High-fluency full-sentence translation using warmed-up Argos INT8 (< 30ms)
            t_mt0 = time.perf_counter()
            full_translations = {
                tgt: self.translate_text(full_transcript, primary_lang, tgt)
                for tgt in targets
            }
            mt_time_ms = (time.perf_counter() - t_mt0) * 1000.0
            post_speech_ms = (time.perf_counter() - t_post0) * 1000.0

            print("\n" + "=" * 68)
            print(f"SPEECH TRANSLATION TURN #{cur_turn_id + 1} COMPLETE")
            print(f"Post-Speech Wait : {post_speech_ms:.0f} ms (MT: {mt_time_ms:.0f} ms | Final ASR: {final_asr_sec*1000:.0f} ms)")
            print(f"Detected Spoken  : [{primary_lang.upper()}] (conf={conf:.2f})")
            print(f"Original Speech  : {full_transcript}")
            for tgt, tr_txt in full_translations.items():
                print(f" -> Translation [{tgt.upper()}]: {tr_txt}")
            print("=" * 68)

            # Speak translation on USB speaker with Echo Isolation (ALSA stream stays continuously OPEN)
            if targets:
                primary_tgt = targets[0]
                tr_text = full_translations[primary_tgt]
                self.speak_out_loud(tr_text, primary_tgt)
                # Settle room reverberation and drain/discard any frames captured during speaker playback
                time.sleep(0.30)
                try:
                    avail = stream.get_read_available()
                    if avail > 0:
                        stream.read(avail, exception_on_overflow=False)
                except Exception:
                    pass

        is_utterance_active = False
        consecutive_silence_sec = 0.0
        utterance_start_time = 0.0
        last_decode_dispatch = 0.0
        turn_counter = 0
        flatline_frames = 0
        flatline_warned = False

        try:
            while True:
                samples_16k = _read_16k_frame()
                chunk_16k_bytes = samples_16k.tobytes()
                samples_tuple = tuple(samples_16k.tolist())

                current_time = time.time()
                vad_state = vad.process_frame(samples_tuple, current_time)

                # Detect dead/unplugged mic jack (e.g., speaker USB dongle with empty 3.5mm mic port)
                if not flatline_warned:
                    if vad_state["rms"] < 1.0:
                        flatline_frames += 1
                        if flatline_frames >= 50:
                            flatline_warned = True
                            print(
                                f"\n[Audio Warning] Device [{dev_idx}] '{dev_name}' is receiving zero audio (RMS=0.0)."
                                "\n  -> Check the device list above and re-run with '--mic-index <N>'."
                            )
                    else:
                        flatline_frames = 0

                if vad_state["is_speech"]:
                    if not is_utterance_active:
                        is_utterance_active = True
                        utterance_start_time = current_time
                        last_decode_dispatch = current_time
                        consecutive_silence_sec = 0.0
                        initial_frames = []
                        for p_frame in vad_state.get("preroll_frames", []):
                            initial_frames.append(struct.pack(f"<{len(p_frame)}h", *p_frame))
                        initial_frames.append(chunk_16k_bytes)
                        with state_lock:
                            turn_state.update({
                                "active": True,
                                "frames": initial_frames,
                                "decoded_frame_count": 0,
                                "stabilizer": LocalAgreementStabilizer(k_agreement=self.k_agreement),
                                "emitted_word_count": 0,
                                "locked_lang": None if source_lang == "auto" else source_lang,
                                "last_conf": 0.0,
                                "last_hypothesis": "",
                                "translations": {},
                                "worker_busy": False,
                                "worker_thread": None,
                                "spoken_live_chunks": 0,
                                "turn_id": turn_counter
                            })
                        sys.stdout.write(f"\n[Turn #{turn_counter + 1} - Listening & Translating Live] ")
                        sys.stdout.flush()
                    else:
                        consecutive_silence_sec = 0.0
                        with state_lock:
                            turn_state["frames"].append(chunk_16k_bytes)

                    # Periodically launch incremental background ASR + MT while speaker keeps talking!
                    if (current_time - last_decode_dispatch) >= self.decode_interval_sec:
                        with state_lock:
                            enough_new = (len(turn_state["frames"]) - turn_state["decoded_frame_count"]) >= 6
                            can_launch = (not turn_state["worker_busy"]) and enough_new
                            if can_launch:
                                turn_state["worker_busy"] = True
                                last_decode_dispatch = current_time
                                t_worker = threading.Thread(
                                    target=_incremental_worker,
                                    args=(turn_counter,),
                                    daemon=True
                                )
                                turn_state["worker_thread"] = t_worker
                                t_worker.start()

                    # Tier 3: Forced Segment Cap (default 6.0s) keeps latency bounded on long monologues
                    if (current_time - utterance_start_time) >= self.forced_endpoint_sec:
                        _finalize_simultaneous_turn("Segment Cap (Rule C) -> Finalizing", consecutive_silence_sec)
                        turn_counter += 1
                        vad.reset(preserve_noise_floor=True)
                        is_utterance_active = False
                        consecutive_silence_sec = 0.0

                else:
                    if is_utterance_active:
                        consecutive_silence_sec += 0.100
                        with state_lock:
                            turn_state["frames"].append(chunk_16k_bytes)
                            cur_hyp = turn_state["last_hypothesis"].strip()

                        # Adaptive 3-Tier Endpointing (PDF Section 3 & 4):
                        # Rule A: 350-450 ms silence + terminal punctuation / stable -> finalize
                        # Rule B: 650-800 ms silence -> finalize even without terminal punctuation
                        has_terminal_punct = bool(cur_hyp) and cur_hyp[-1] in ".!?,;:"
                        endpoint_limit = self.short_endpoint_sec if has_terminal_punct else self.safe_endpoint_sec

                        if consecutive_silence_sec >= endpoint_limit:
                            _finalize_simultaneous_turn("Pause Ended -> Finalizing", consecutive_silence_sec)
                            turn_counter += 1
                            vad.reset(preserve_noise_floor=True)
                            is_utterance_active = False
                            consecutive_silence_sec = 0.0

        except KeyboardInterrupt:
            print("\nSimultaneous streaming stopped by user.")
        finally:
            if speak_live and tts_thread is not None:
                tts_queue.put(None)
                tts_thread.join(timeout=2.0)
            try:
                if stream.is_active():
                    stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
            try:
                pa.terminate()
            except Exception:
                pass

    def _process_accumulated_turn(self, accumulated_frames, turn_id, stream=None, source_lang="auto"):
        if not accumulated_frames:
            return
        raw_bytes = b"".join(accumulated_frames)
        pcm_array = np.frombuffer(raw_bytes, dtype=np.int16)

        # Apply Automatic Gain Control (AGC) & human voice peak normalization
        try:
            from pi_edge.audio_preprocessor import normalize_human_speech
        except ImportError:
            from audio_preprocessor import normalize_human_speech
        normalized_pcm = normalize_human_speech(pcm_array, target_peak=0.88, max_gain_factor=15.0)

        tmp_wav = f"/tmp/turn_{turn_id}_{int(time.time())}.wav"
        with wave.open(tmp_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(normalized_pcm.tobytes())

        res = self.process_speech_file(tmp_wav, source_lang=source_lang)
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)

        # Anti-noise guard: reject empty speech, silence, or noise hallucinations
        transcript = res.get("transcript", "").strip()
        if not transcript or len(transcript) < 2:
            print(" [Turn ignored: Background noise / Inaudible speech]")
            return

        print("\n" + "=" * 65)
        print(f"SPEECH TRANSLATION TURN #{turn_id + 1}")
        print("=" * 65)
        print(f"Detected Spoken Language: {res['detected_language'].upper()} (Confidence: {res['confidence']})")
        print(f"Original Transcript     : {res['transcript']}")
        for lang, text in res["translations"].items():
            print(f" -> Translation [{lang.upper()}]: {text}")
        print(f"Latency: ASR={res['timing']['asr_sec']*1000:.1f}ms | Total={res['timing']['total_sec']*1000:.1f}ms")
        print("=" * 65)

        # Speak the primary translation through the USB speaker with Echo Isolation
        if res["translations"]:
            primary_tgt, tr_text = next(iter(res["translations"].items()))
            # 1. Stop mic capture while speaker is active to prevent acoustic echo feedback
            if stream is not None:
                try:
                    stream.stop_stream()
                except Exception:
                    pass

            self.speak_out_loud(tr_text, primary_tgt)

            # 2. Re-enable mic capture and flush room reverberations
            if stream is not None:
                time.sleep(0.35)  # Allow room echo to decay
                try:
                    stream.start_stream()
                    # Drain any stale frames buffered during playback
                    while stream.get_read_available() > 0:
                        stream.read(stream.get_read_available(), exception_on_overflow=False)
                except Exception:
                    pass

    def _setup_buttons(self):
        try:
            self.buttons["b1"] = Button(PIN_BUTTON1, pull_up=True, bounce_time=0.1)
            self.buttons["b2"] = Button(PIN_BUTTON2, pull_up=True, bounce_time=0.1)
            self.buttons["b3"] = Button(PIN_BUTTON3, pull_up=True, bounce_time=0.2)
            self.buttons["b4"] = Button(PIN_BUTTON4, pull_up=True, bounce_time=0.2)
            self.buttons["b5"] = Button(PIN_BUTTON5, pull_up=True, bounce_time=0.1)

            # Button 1 (GPIO 27): Guest Mic (Hold to speak in Guest Lang -> User Lang)
            self.buttons["b1"].when_pressed = lambda: self._on_button_press("guest")
            self.buttons["b1"].when_released = lambda: self._on_button_release("guest")

            # Button 2 (GPIO 22): User Mic (Hold to speak in User Lang -> Guest Lang)
            self.buttons["b2"].when_pressed = lambda: self._on_button_press("user")
            self.buttons["b2"].when_released = lambda: self._on_button_release("user")

            # Button 3 (GPIO 20): Source Language Cycle (EN -> DE -> FR)
            self.buttons["b3"].when_pressed = self._cycle_source_lang

            # Button 4 (GPIO 21): Target Language Cycle (EN -> DE -> FR)
            self.buttons["b4"].when_pressed = self._cycle_target_lang

            # Button 5 (GPIO 16): Language Detection (Hold to speak with Auto-Detection EN/DE/FR)
            self.buttons["b5"].when_pressed = lambda: self._on_button_press("auto")
            self.buttons["b5"].when_released = lambda: self._on_button_release("auto")

            print("[Engine] All 5 GPIO Buttons attached successfully (Pins 27, 22, 20, 21, 16).")
        except Exception as e:
            print(f"[Engine] Warning: Could not bind GPIO pins: {e}")

    def _cycle_source_lang(self):
        """Button 3 (GPIO 20): Cycles Source (User) Language through [EN, DE, FR]."""
        idx = (self.supported_langs.index(self.user_lang) + 1) % len(self.supported_langs)
        self.user_lang = self.supported_langs[idx]
        print(f"\n[Button 3: Source Language (GPIO 20)] Set to: [{self.user_lang.upper()}] (Route: User [{self.user_lang.upper()}] <-> Guest [{self.guest_lang.upper()}])")

    def _cycle_target_lang(self):
        """Button 4 (GPIO 21): Cycles Target (Guest) Language through [EN, DE, FR]."""
        idx = (self.supported_langs.index(self.guest_lang) + 1) % len(self.supported_langs)
        self.guest_lang = self.supported_langs[idx]
        print(f"\n[Button 4: Target Language (GPIO 21)] Set to: [{self.guest_lang.upper()}] (Route: User [{self.user_lang.upper()}] <-> Guest [{self.guest_lang.upper()}])")

    # Backward compatibility aliases
    _cycle_user_lang = _cycle_source_lang
    _cycle_guest_lang = _cycle_target_lang

    def _on_button_press(self, speaker: str):
        if self.recorder.is_recording:
            return
        self.active_speaker = speaker
        if speaker == "guest":
            print(f"\n>>> [BUTTON 1: GUEST MIC (GPIO 27)] Recording speech in [{self.guest_lang.upper()}]... (Release button when done)")
        elif speaker == "user":
            print(f"\n>>> [BUTTON 2: USER MIC (GPIO 22)] Recording speech in [{self.user_lang.upper()}]... (Release button when done)")
        elif speaker == "auto":
            print(f"\n>>> [BUTTON 5: AUTO-DETECT MIC (GPIO 16)] Recording speech in any language (EN/DE/FR)... (Release button when done)")
        else:
            print(f"\n[Microphone ON] Recording for speaker: {speaker.upper()}... (Release button when done)")
        self.recorder.start_recording()

    def _on_button_release(self, speaker: str):
        if not self.recorder.is_recording:
            return
        t_speech_end = time.perf_counter()
        print(f"[Microphone OFF] Speaker finished. Processing complete audio...")
        wav_path = self.recorder.stop_recording()
        if not wav_path or not os.path.exists(wav_path):
            print("[Warning] No audio recorded.")
            return

        # Determine language routing according to the 5-button hardware specification:
        # Button 1 (Guest Mic): Guest speaks -> Translates to User Language
        # Button 2 (User Mic): User speaks -> Translates to Guest Language
        # Button 5 (Auto-Detect): Auto-identifies spoken language -> Translates to both other languages
        if speaker == "guest":
            src = self.guest_lang
            tgt = self.user_lang
        elif speaker == "user":
            src = self.user_lang
            tgt = self.guest_lang
        else:
            src = "auto"
            tgt = None

        res = self.process_speech_file(wav_path, source_lang=src)
        t_trans_ready = time.perf_counter()

        transcript = res.get('transcript', '')
        detected_lang = res.get('detected_language', src)
        translations = res.get('translations', {})

        asr_ms = res['timing']['asr_sec'] * 1000.0
        mt_ms = res['timing']['translation_sec'] * 1000.0
        post_speech_turnaround_ms = (t_trans_ready - t_speech_end) * 1000.0

        print("\n" + "=" * 68)
        first_tgt = tgt
        tr_text = ""
        if speaker == "guest":
            print(">>> BUTTON 1: GUEST TRANSLATION COMPLETE <<<")
            print(f"Guest Spoke [{src.upper()}] : {transcript}")
            tr_text = translations.get(tgt, "")
            print(f" -> User [{tgt.upper()}]     : {tr_text}")
        elif speaker == "user":
            print(">>> BUTTON 2: USER TRANSLATION COMPLETE <<<")
            print(f"User Spoke [{src.upper()}]  : {transcript}")
            tr_text = translations.get(tgt, "")
            print(f" -> Guest [{tgt.upper()}]    : {tr_text}")
        else:
            print(">>> BUTTON 5: AUTO-DETECT TRANSLATION COMPLETE <<<")
            print(f"Detected Spoken [{detected_lang.upper()}] (conf={res.get('confidence', 1.0):.2f}) : {transcript}")
            for l_code, t_txt in translations.items():
                print(f" -> [{l_code.upper()}]: {t_txt}")
            first_tgt = next(iter(translations.keys())) if translations else tgt
            tr_text = translations.get(first_tgt, "")

        print("\n--- TIMEFRAME & LATENCY (SPEAKER FINISHED -> TRANSLATION OUTPUT) ---")
        print(f"  * 1. Speaker Ended at       : 0.0 ms  (Button Released / Enter Pressed)")
        print(f"  * 2. ASR Transcription Time : {asr_ms:.1f} ms")
        print(f"  * 3. Machine Translation    : {mt_ms:.1f} ms")
        print(f"  * 4. Post-Speech Turnaround : {post_speech_turnaround_ms:.1f} ms  (Speech End -> Translation Ready)")
        print("=" * 68 + "\n")

        if tr_text:
            out_lang = tgt if speaker in ("guest", "user") else first_tgt
            self.speak_out_loud(tr_text, out_lang)

    def translate_text(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if not text.strip() or src_lang == tgt_lang:
            return text
        src = self._installed.get(src_lang)
        tgt = self._installed.get(tgt_lang)
        if not src or not tgt:
            return f"[{src_lang}->{tgt_lang} not loaded] {text}"
        tr = src.get_translation(tgt)
        return tr.translate(text) if tr else text

    def process_speech_file(self, wav_path: str, source_lang: str = "auto") -> Dict[str, Any]:
        """
        Processes an audio recording through the full Hailo NPU streaming pipeline:
        1. VAD onset detection with pre-roll buffer.
        2. Hailo-8 NPU Whisper transcription.
        3. Dual translation to the other two languages.
        """
        t0 = time.perf_counter()

        # Transcribe with NPU (fast_mode=True preserves complete sentences without Silero VAD cutting words)
        lang_hint = None if source_lang == "auto" else source_lang
        transcript, det_lang, conf, asr_time = self.whisper.transcribe(wav_path, language=lang_hint, fast_mode=True)

        primary_lang = det_lang if det_lang in self.supported_langs else self.guest_lang
        targets = [l for l in self.supported_langs if l != primary_lang][:2]

        t_tr0 = time.perf_counter()
        translations = {}
        for tgt in targets:
            translations[tgt] = self.translate_text(transcript, primary_lang, tgt)
        tr_time = time.perf_counter() - t_tr0

        total_time = time.perf_counter() - t0

        return {
            "transcript": transcript,
            "detected_language": primary_lang,
            "confidence": conf,
            "translations": translations,
            "hardware_accelerated": self.whisper.is_hardware_accelerated,
            "device_name": self.whisper.device_name,
            "timing": {
                "asr_sec": asr_time,
                "translation_sec": round(tr_time, 4),
                "total_sec": round(total_time, 4)
            }
        }


def main():
    parser = argparse.ArgumentParser(description="TollMatcher AI HAT+ (Hailo-8 / Hailo-10H NPU) Runner")
    parser.add_argument("--wav", default=None, help="Process a test WAV audio file")
    parser.add_argument("--hef", default=None, help="Path to Hailo Whisper HEF file")
    parser.add_argument("--user-lang", default="de", choices=["en", "de", "fr"], help="Default user language")
    parser.add_argument("--guest-lang", default="en", choices=["en", "de", "fr"], help="Default guest language")
    parser.add_argument("--source-lang", default="auto", choices=["auto", "en", "de", "fr"], help="Spoken input language: 'auto' for 3-way auto-detection, or 'de', 'en', 'fr' to lock speech language")
    parser.add_argument("--model", default="tiny", choices=["tiny", "base"], help="Whisper model size ('tiny' recommended for fastest simultaneous CPU streaming, 'base' for heavier accent tolerance)")
    parser.add_argument("--decode-interval", type=float, default=None, help="Simultaneous live ASR interval in seconds (PDF default: 0.7s for tiny)")
    parser.add_argument("--pause-bridge", type=float, default=1.0, help="Adaptive pause bridge duration for thinking pauses (default: 1.0s)")
    parser.add_argument("--max-turn-sec", type=float, default=6.0, help="Maximum continuous speech segment duration before auto-commit (PDF Rule C: 6.0s)")
    parser.add_argument("--speak-live", action="store_true", help="Speak each committed clause aloud simultaneously while talking (best with headphones/headset to avoid speaker-to-mic echo)")
    parser.add_argument("--mic-index", type=int, default=None, help="Manually select PyAudio input device index (see --list-mics)")
    parser.add_argument("--list-mics", action="store_true", help="List all detected microphone input devices and exit")
    parser.add_argument("--vad-threshold", type=float, default=120.0, help="Minimum VAD RMS energy threshold (lower = more sensitive for quiet mics)")
    parser.add_argument("--buttons", action="store_true", help="Enable physical GPIO buttons mode (Pins 27, 22, 20, 21, 16)")
    parser.add_argument("--hands-free", action="store_true", help="Continuous hands-free VAD streaming mode (mic + speaker, no buttons required)")
    args = parser.parse_args()

    if args.list_mics:
        if not HAS_PYAUDIO:
            print("PyAudio is not installed.")
            return
        with suppress_native_stderr():
            pa = pyaudio.PyAudio()
        mics = list_input_devices(pa)
        print("Detected Input Devices:")
        for idx, name, ch, prio, sr in mics:
            print(f"  [{idx}] {name} | channels={ch} | default_rate={sr} Hz | priority={prio}")
        pa.terminate()
        return

    decode_interval = args.decode_interval if args.decode_interval is not None else (0.700 if args.model == "tiny" else 1.000)

    runner = TollMatcherHailoRunner(
        hef_path=args.hef,
        decode_interval_sec=decode_interval,
        max_pause_bridge_sec=args.pause_bridge,
        forced_endpoint_sec=args.max_turn_sec,
        user_lang=args.user_lang,
        guest_lang=args.guest_lang,
        model_size=args.model
    )

    # Mode 1: Test WAV file
    if args.wav and os.path.exists(args.wav):
        print(f"\nProcessing test audio: {args.wav}")
        res = runner.process_speech_file(args.wav)
        print("\n" + "=" * 60)
        print("TRANSLATION RESULT (HAILO NPU PIPELINE)")
        print("=" * 60)
        print(f"Device        : {res['device_name']}")
        print(f"Detected Lang : {res['detected_language'].upper()} (Confidence: {res['confidence']})")
        print(f"Original Text : {res['transcript']}")
        for tgt, text in res['translations'].items():
            print(f" -> [{tgt.upper()}]      : {text}")
        print(f"ASR Latency   : {res['timing']['asr_sec'] * 1000:.1f} ms")
        print(f"Total Latency : {res['timing']['total_sec'] * 1000:.1f} ms")
        print("=" * 60)

    # Mode 2: 5 Physical GPIO Buttons & Interactive Simulation
    elif args.buttons:
        print("\n" + "=" * 74)
        print(">>> TOLLMATCHER 5-BUTTON HARDWARE CONTROL ACTIVE <<<")
        print("=" * 74)
        print("  - Button 1 (GPIO 27) : Guest Mic       (Hold to speak in Guest Lang -> User Lang)")
        print("  - Button 2 (GPIO 22) : User Mic        (Hold to speak in User Lang -> Guest Lang)")
        print("  - Button 3 (GPIO 20) : Source Language (Tap to cycle Source Language: EN/DE/FR)")
        print("  - Button 4 (GPIO 21) : Target Language (Tap to cycle Target Language: EN/DE/FR)")
        print("  - Button 5 (GPIO 16) : Language Detect (Hold to speak with Auto-Detection EN/DE/FR)")
        print("-" * 74)
        print("  [Interactive Terminal Debugger]:")
        print("    Type '1' + Enter : Simulate Button 1 (Guest Mic)")
        print("    Type '2' + Enter : Simulate Button 2 (User Mic)")
        print("    Type '3' + Enter : Simulate Button 3 (Cycle Source Language)")
        print("    Type '4' + Enter : Simulate Button 4 (Cycle Target Language)")
        print("    Type '5' + Enter : Simulate Button 5 (Auto-Detect Language)")
        print("    Type 'q' + Enter : Exit")
        print("=" * 74)
        print(f"Current Config: User (Source) = [{runner.user_lang.upper()}] | Guest (Target) = [{runner.guest_lang.upper()}]")
        print("Physical GPIO pins are active. Ready!\n")

        import select
        try:
            while True:
                # Poll terminal stdin for debug simulation commands while GPIO event loop runs in background
                rlist, _, _ = select.select([sys.stdin], [], [], 0.3)
                if rlist:
                    cmd = sys.stdin.readline().strip()
                    if cmd == "1":
                        runner._on_button_press("guest")
                        print("  [MIC RECORDING...] Speak your full sentence now!")
                        print("  >>> Press [ENTER] when you are finished speaking... <<<")
                        sys.stdin.readline()
                        runner._on_button_release("guest")
                    elif cmd == "2":
                        runner._on_button_press("user")
                        print("  [MIC RECORDING...] Speak your full sentence now!")
                        print("  >>> Press [ENTER] when you are finished speaking... <<<")
                        sys.stdin.readline()
                        runner._on_button_release("user")
                    elif cmd == "3":
                        runner._cycle_source_lang()
                    elif cmd == "4":
                        runner._cycle_target_lang()
                    elif cmd == "5":
                        runner._on_button_press("auto")
                        print("  [MIC RECORDING (Auto-Detect)...] Speak your full sentence in any language!")
                        print("  >>> Press [ENTER] when you are finished speaking... <<<")
                        sys.stdin.readline()
                        runner._on_button_release("auto")
                    elif cmd.lower() in ("q", "quit", "exit"):
                        break
        except KeyboardInterrupt:
            print("\nShutting down TollMatcher edge engine.")

    # Mode 3 (Default): Simultaneous Hands-Free Streaming Mode (Speaker + Microphone only, NO buttons required!)
    else:
        runner.run_hands_free_streaming(
            source_lang=args.source_lang,
            mic_index=args.mic_index,
            vad_min_rms=args.vad_threshold,
            speak_live=args.speak_live
        )


if __name__ == "__main__":
    main()
