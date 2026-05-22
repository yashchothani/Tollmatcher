# Button3 + Button4 decide source and target languages (source = guest's native language, target = user's native language)
# And then Button1 function or Button2 function using gpiozero to record sounds
# will use 2 sets of specific hardwares to record and replay the translation
# Button5 = AUTO language detection (hold to record, release to detect + translate)
# How to test? 1. activate virtual env 2. connect 5 buttons rightly 3. run test

# USB Audio devices (check with: cat /proc/asound/cards)
# "USB"        -> EPOS PC 7 USB headset (headphone for User - confirmations & guest's translation)
# "Device"     -> GeneralPlus USB Audio Device (speaker for Guest - user's translation)
# "microphone" -> MUSIC-BOOST usb microphone (recording input)

# ─────────────────────────────────────────────────────────────────────────────
# BUTTON 5 — AUTO LANGUAGE DETECTION FLOW:
#
#  Press & hold B5 → speaker talks → release B5
#  Whisper detects language automatically (no hint given)
#
#  Case A: detected lang == target (user's language)
#    → USER is speaking
#    → Translate to guest's pre-selected source language
#    → Play on USER_TTS_DEVICE (guest hears it)
#    → Plays confirmation: "t_<target>" audio so guest knows detected lang
#
#  Case B: detected lang == source (guest's language) OR any other lang
#    → GUEST is speaking
#    → If detected lang is new (not current source), auto-update source lang
#    → Translate to target (user's) language
#    → Play on GUEST_TTS_DEVICE (user hears it)
#    → Plays confirmation: "<target>_<detected>" audio so user knows the pair
#
#  Note: Button3 must be used first to pre-configure guest's language.
#        Button4 must be used first to pre-configure user's language.
#        Button5 uses those as fallbacks when detection is ambiguous.
# ─────────────────────────────────────────────────────────────────────────────

from gpiozero import Button
import threading
import time
import os
import glob
import subprocess
import pyaudio
import wave
from faster_whisper import WhisperModel
import warnings
import argostranslate.package
import argostranslate.translate
import gc
import psutil
import sys
import atexit

# Single-instance lock: prevent multiple instances from running
LOCK = "/tmp/tollmatcher.lock"

if os.path.exists(LOCK):
    print("Another instance is already running. Exiting.")
    sys.exit(1)

with open(LOCK, "w") as f:
    f.write(str(os.getpid()))

atexit.register(lambda: os.remove(LOCK) if os.path.exists(LOCK) else None)

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Guard: Wait for USB audio device before any audio is played
while True:
    out = subprocess.getoutput("aplay -L")
    if "sysdefault:CARD=USB" in out:
        break
    time.sleep(0.5)

# Preload Whisper models once at startup to avoid per-call load latency
WHISPER_MODEL = WhisperModel("tiny", device="cpu", compute_type="int8", num_workers=1)
WHISPER_MODEL_EN = WhisperModel("tiny.en", device="cpu", compute_type="int8", num_workers=1)

print("Warming up Whisper models...")
try:
    dummy_segments, _ = WHISPER_MODEL.transcribe("dummy", language="en")
    list(dummy_segments)
    print("Whisper models warmed up successfully")
except Exception as e:
    print(f"Warning: Model warmup failed: {e}")


def optimize_audio_file(input_file):
    """Optimize audio file for faster transcription by trimming and normalizing"""
    try:
        import librosa
        import soundfile as sf

        y, sr = librosa.load(input_file, sr=16000)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        optimized_file = input_file.replace('.wav', '_optimized.wav')
        sf.write(optimized_file, y_trimmed, sr)
        print(f"Audio optimized: {input_file} -> {optimized_file}")
        return optimized_file

    except ImportError as e:
        print(f"CRITICAL: librosa/soundfile not available: {e}")
        try:
            subprocess.run(["pip", "install", "librosa", "soundfile"], check=True)
            print("Packages installed. Please restart the application.")
            return input_file
        except Exception as install_error:
            print(f"Failed to install packages: {install_error}")
            return fallback_audio_optimization(input_file)
    except Exception as e:
        print(f"Warning: Audio optimization failed: {e}, using fallback")
        return fallback_audio_optimization(input_file)


def fallback_audio_optimization(input_file):
    """Fallback audio optimization using basic Python libraries"""
    try:
        import numpy as np

        with wave.open(input_file, 'rb') as wav_file:
            frames = wav_file.readframes(-1)
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()

        if sample_width == 1:
            dtype = np.uint8
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            print("Unsupported sample width, using original file")
            return input_file

        audio_data = np.frombuffer(frames, dtype=dtype)
        threshold = np.max(np.abs(audio_data)) * 0.01
        non_silent = np.abs(audio_data) > threshold

        if np.any(non_silent):
            first_non_silent = np.argmax(non_silent)
            last_non_silent = len(non_silent) - np.argmax(non_silent[::-1])
            audio_data = audio_data[first_non_silent:last_non_silent]

        if sample_rate > 16000:
            downsample_factor = sample_rate // 16000
            audio_data = audio_data[::downsample_factor]
            sample_rate = 16000

        optimized_file = input_file.replace('.wav', '_optimized.wav')
        with wave.open(optimized_file, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        print(f"Fallback audio optimization completed: {optimized_file}")
        return optimized_file

    except Exception as e:
        print(f"Fallback optimization failed: {e}, using original file")
        return input_file


def cleanup_memory():
    gc.collect()
    memory_usage = psutil.virtual_memory().percent
    if memory_usage > 80:
        print(f"Warning: High memory usage ({memory_usage:.1f}%), forcing cleanup")
        gc.collect()


def log_performance_metrics(stt_time, translation_time, tts_time, total_time, audio_file):
    try:
        file_size = os.path.getsize(audio_file) if os.path.exists(audio_file) else 0
        file_size_mb = file_size / (1024 * 1024)
        print(f"[Performance] STT: {stt_time:.2f}s, Translation: {translation_time:.2f}s, "
              f"TTS: {tts_time:.2f}s, Total: {total_time:.2f}s, File: {file_size_mb:.2f}MB")
        if stt_time > 10:
            print(f"⚠️  WARNING: STT time ({stt_time:.2f}s) is unusually high!")
        if total_time > 30:
            print(f"⚠️  WARNING: Total processing time ({total_time:.2f}s) is very high!")
    except Exception as e:
        print(f"Performance logging error: {e}")


def transcribe_with_optimization(model, audio_file, language):
    """Transcribe with a known language — used by Button1 and Button2"""
    cleanup_memory()
    optimized_file = optimize_audio_file(audio_file)
    try:
        segments, info = model.transcribe(
            optimized_file,
            language=language,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            initial_prompt=None,
            word_timestamps=False,
            prepend_punctuations="\"'([{-",
            append_punctuations="\"'.!?:;,)",
        )
        recognized = " ".join([segment.text for segment in segments]).strip()
        if optimized_file != audio_file and os.path.exists(optimized_file):
            os.remove(optimized_file)
        return recognized
    except Exception as e:
        print(f"Transcription error: {e}")
        if optimized_file != audio_file and os.path.exists(optimized_file):
            os.remove(optimized_file)
        raise e


# ─── NEW: Auto-detect transcription for Button 5 ─────────────────────────────
def transcribe_auto_detect(audio_file):
    """
    Transcribe WITHOUT specifying a language — Whisper detects it automatically.
    Returns: (recognized_text, detected_lang_code, lang_probability)
    Uses WHISPER_MODEL (multilingual) since WHISPER_MODEL_EN can only detect English.
    """
    cleanup_memory()
    optimized_file = optimize_audio_file(audio_file)
    try:
        segments, info = WHISPER_MODEL.transcribe(
            optimized_file,
            language=None,          # KEY: no hint → Whisper auto-detects
            beam_size=1,
            best_of=1,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
            initial_prompt=None,
            word_timestamps=False,
            prepend_punctuations="\"'([{-",
            append_punctuations="\"'.!?:;,)",
        )
        recognized = " ".join([segment.text for segment in segments]).strip()
        detected_lang = info.language          # e.g. "en", "de", "fr" …
        lang_prob = info.language_probability  # confidence 0.0–1.0

        if optimized_file != audio_file and os.path.exists(optimized_file):
            os.remove(optimized_file)

        return recognized, detected_lang, lang_prob

    except Exception as e:
        print(f"Auto-detect transcription error: {e}")
        if optimized_file != audio_file and os.path.exists(optimized_file):
            os.remove(optimized_file)
        raise e
# ─────────────────────────────────────────────────────────────────────────────


# Audio device configuration
CONFIRM_AUDIO_DEVICE = os.getenv("CONFIRM_AUDIO_DEVICE", "sysdefault:CARD=USB")
GUEST_TTS_DEVICE = os.getenv("GUEST_TTS_DEVICE", "sysdefault:CARD=USB")
USER_TTS_DEVICE = os.getenv("USER_TTS_DEVICE", "sysdefault:CARD=Device")

PIPER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../piper_rebuild/piper/piper'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

piper_models = {
    "en": ("en_US-john-medium.onnx",       "en_US-john-medium.onnx.json"),
    "de": ("de_DE-thorsten-medium.onnx",    "de_DE-thorsten-medium.onnx.json"),
    "es": ("es_ES-carlfm-x_low.onnx",       "es_ES-carlfm-x_low.onnx.json"),
    "fr": ("fr_FR-tom-medium.onnx",         "fr_FR-tom-medium.onnx.json"),
    "uk": ("uk_UA-lada-x_low.onnx",         "uk_UA-lada-x_low.onnx.json"),
    "it": ("it_IT-paola-medium.onnx",       "it_IT-paola-medium.onnx.json"),
    "ca": ("ca_ES-upc_ona-x_low.onnx",      "ca_ES-upc_ona-x_low.onnx.json"),
    "ar": ("ar_JO-kareem-low.onnx",         "ar_JO-kareem-low.onnx.json"),
    "nl": ("nl_NL-pim-medium.onnx",         "nl_NL-pim-medium.onnx.json"),
    "zh": ("zh_CN-huayan-medium.onnx",      "zh_CN-huayan-medium.onnx.json"),
}

piper_args_base = {}
for lang, (model, config) in piper_models.items():
    piper_args_base[lang] = [
        PIPER_PATH,
        "--model",  os.path.abspath(os.path.join(os.path.dirname(__file__), f"../piper_rebuild/piper/{model}")),
        "--config", os.path.abspath(os.path.join(os.path.dirname(__file__), f"../piper_rebuild/piper/{config}")),
    ]

# Button pin definitions
BUTTON1_PIN = 27   # button1 - guest mic (manual)
BUTTON2_PIN = 22   # button2 - user mic  (manual)
BUTTON3_PIN = 20   # button3 - select source language (guest)
BUTTON4_PIN = 21   # button4 - select target language (user)
BUTTON5_PIN = 16   # button5 - AUTO language detection  ← NEW

button1 = Button(BUTTON1_PIN, pull_up=True, bounce_time=0.2)
button2 = Button(BUTTON2_PIN, pull_up=True, bounce_time=0.2)
button3 = Button(BUTTON3_PIN, pull_up=True, bounce_time=0.2)
button4 = Button(BUTTON4_PIN, pull_up=True, bounce_time=0.2)
button5 = Button(BUTTON5_PIN, pull_up=True, bounce_time=0.2)   # ← NEW

SYSTEM_BUTTON_PIN = 4
system_button = Button(SYSTEM_BUTTON_PIN, hold_time=3, pull_up=True)


def handle_system_press():
    print("System button held down")
    press_time = time.time()
    while system_button.is_held:
        time.sleep(0.1)
    duration = time.time() - press_time
    print(f"Held for {duration:.2f} seconds")
    if duration >= 5:
        print("Rebooting system...")
        os.system("sudo /sbin/reboot")
    elif duration >= 3:
        print("Shutting down system...")
        os.system("sudo /sbin/shutdown -h now")
    else:
        print("Press too short, ignored.")


def system_button_monitor():
    while True:
        system_button.wait_for_press()
        if system_button.is_held:
            handle_system_press()


threading.Thread(target=system_button_monitor, daemon=True).start()

system_locked_by = None

guest_button_confirmed = False
guest_record_start_time = None
user_button_confirmed = False
user_record_start_time = None
auto_record_start_time = None    # ← NEW: for Button5


# Audio confirmation files
audio_map = {
    'zh_en': os.path.join(os.path.dirname(__file__), '../languages/zh_en.wav'),
    'zh_de': os.path.join(os.path.dirname(__file__), '../languages/zh_de.wav'),
    'zh_es': os.path.join(os.path.dirname(__file__), '../languages/zh_es.wav'),
    'zh_fr': os.path.join(os.path.dirname(__file__), '../languages/zh_fr.wav'),
    'zh_uk': os.path.join(os.path.dirname(__file__), '../languages/zh_uk.wav'),
    'zh_it': os.path.join(os.path.dirname(__file__), '../languages/zh_it.wav'),
    'zh_ca': os.path.join(os.path.dirname(__file__), '../languages/zh_ca.wav'),
    'zh_ar': os.path.join(os.path.dirname(__file__), '../languages/zh_ar.wav'),
    'zh_nl': os.path.join(os.path.dirname(__file__), '../languages/zh_nl.wav'),

    'en_zh': os.path.join(os.path.dirname(__file__), '../languages/en_zh.wav'),
    'en_de': os.path.join(os.path.dirname(__file__), '../languages/en_de.wav'),
    'en_es': os.path.join(os.path.dirname(__file__), '../languages/en_es.wav'),
    'en_fr': os.path.join(os.path.dirname(__file__), '../languages/en_fr.wav'),
    'en_uk': os.path.join(os.path.dirname(__file__), '../languages/en_uk.wav'),
    'en_it': os.path.join(os.path.dirname(__file__), '../languages/en_it.wav'),
    'en_ca': os.path.join(os.path.dirname(__file__), '../languages/en_ca.wav'),
    'en_nl': os.path.join(os.path.dirname(__file__), '../languages/en_nl.wav'),

    'de_zh': os.path.join(os.path.dirname(__file__), '../languages/de_zh.wav'),
    'de_en': os.path.join(os.path.dirname(__file__), '../languages/de_en.wav'),
    'de_es': os.path.join(os.path.dirname(__file__), '../languages/de_es.wav'),
    'de_fr': os.path.join(os.path.dirname(__file__), '../languages/de_fr.wav'),
    'de_uk': os.path.join(os.path.dirname(__file__), '../languages/de_uk.wav'),
    'de_it': os.path.join(os.path.dirname(__file__), '../languages/de_it.wav'),
    'de_ca': os.path.join(os.path.dirname(__file__), '../languages/de_ca.wav'),
    'de_ar': os.path.join(os.path.dirname(__file__), '../languages/de_ar.wav'),
    'de_nl': os.path.join(os.path.dirname(__file__), '../languages/de_nl.wav'),

    'es_zh': os.path.join(os.path.dirname(__file__), '../languages/es_zh.wav'),
    'es_en': os.path.join(os.path.dirname(__file__), '../languages/es_en.wav'),
    'es_de': os.path.join(os.path.dirname(__file__), '../languages/es_de.wav'),
    'es_fr': os.path.join(os.path.dirname(__file__), '../languages/es_fr.wav'),
    'es_uk': os.path.join(os.path.dirname(__file__), '../languages/es_uk.wav'),
    'es_it': os.path.join(os.path.dirname(__file__), '../languages/es_it.wav'),
    'es_ca': os.path.join(os.path.dirname(__file__), '../languages/es_ca.wav'),
    'es_ar': os.path.join(os.path.dirname(__file__), '../languages/es_ar.wav'),
    'es_nl': os.path.join(os.path.dirname(__file__), '../languages/es_nl.wav'),

    'fr_zh': os.path.join(os.path.dirname(__file__), '../languages/fr_zh.wav'),
    'fr_en': os.path.join(os.path.dirname(__file__), '../languages/fr_en.wav'),
    'fr_es': os.path.join(os.path.dirname(__file__), '../languages/fr_es.wav'),
    'fr_de': os.path.join(os.path.dirname(__file__), '../languages/fr_de.wav'),
    'fr_uk': os.path.join(os.path.dirname(__file__), '../languages/fr_uk.wav'),
    'fr_it': os.path.join(os.path.dirname(__file__), '../languages/fr_it.wav'),
    'fr_ca': os.path.join(os.path.dirname(__file__), '../languages/fr_ca.wav'),
    'fr_ar': os.path.join(os.path.dirname(__file__), '../languages/fr_ar.wav'),
    'fr_nl': os.path.join(os.path.dirname(__file__), '../languages/fr_nl.wav'),

    'uk_zh': os.path.join(os.path.dirname(__file__), '../languages/uk_zh.wav'),
    'uk_en': os.path.join(os.path.dirname(__file__), '../languages/uk_en.wav'),
    'uk_de': os.path.join(os.path.dirname(__file__), '../languages/uk_de.wav'),
    'uk_es': os.path.join(os.path.dirname(__file__), '../languages/uk_es.wav'),
    'uk_fr': os.path.join(os.path.dirname(__file__), '../languages/uk_fr.wav'),
    'uk_it': os.path.join(os.path.dirname(__file__), '../languages/uk_it.wav'),
    'uk_ca': os.path.join(os.path.dirname(__file__), '../languages/uk_ca.wav'),
    'uk_ar': os.path.join(os.path.dirname(__file__), '../languages/uk_ar.wav'),
    'uk_nl': os.path.join(os.path.dirname(__file__), '../languages/uk_nl.wav'),

    'it_zh': os.path.join(os.path.dirname(__file__), '../languages/it_zh.wav'),
    'it_en': os.path.join(os.path.dirname(__file__), '../languages/it_en.wav'),
    'it_de': os.path.join(os.path.dirname(__file__), '../languages/it_de.wav'),
    'it_es': os.path.join(os.path.dirname(__file__), '../languages/it_es.wav'),
    'it_fr': os.path.join(os.path.dirname(__file__), '../languages/it_fr.wav'),
    'it_uk': os.path.join(os.path.dirname(__file__), '../languages/it_uk.wav'),
    'it_ca': os.path.join(os.path.dirname(__file__), '../languages/it_ca.wav'),
    'it_ar': os.path.join(os.path.dirname(__file__), '../languages/it_ar.wav'),
    'it_nl': os.path.join(os.path.dirname(__file__), '../languages/it_nl.wav'),

    'ca_zh': os.path.join(os.path.dirname(__file__), '../languages/ca_zh.wav'),
    'ca_en': os.path.join(os.path.dirname(__file__), '../languages/ca_en.wav'),
    'ca_de': os.path.join(os.path.dirname(__file__), '../languages/ca_de.wav'),
    'ca_es': os.path.join(os.path.dirname(__file__), '../languages/ca_es.wav'),
    'ca_fr': os.path.join(os.path.dirname(__file__), '../languages/ca_fr.wav'),
    'ca_uk': os.path.join(os.path.dirname(__file__), '../languages/ca_uk.wav'),
    'ca_it': os.path.join(os.path.dirname(__file__), '../languages/ca_it.wav'),
    'ca_ar': os.path.join(os.path.dirname(__file__), '../languages/ca_ar.wav'),
    'ca_nl': os.path.join(os.path.dirname(__file__), '../languages/ca_nl.wav'),

    'ar_zh': os.path.join(os.path.dirname(__file__), '../languages/ar_zh.wav'),
    'ar_en': os.path.join(os.path.dirname(__file__), '../languages/ar_en.wav'),
    'ar_de': os.path.join(os.path.dirname(__file__), '../languages/ar_de.wav'),
    'ar_es': os.path.join(os.path.dirname(__file__), '../languages/ar_es.wav'),
    'ar_fr': os.path.join(os.path.dirname(__file__), '../languages/ar_fr.wav'),
    'ar_uk': os.path.join(os.path.dirname(__file__), '../languages/ar_uk.wav'),
    'ar_it': os.path.join(os.path.dirname(__file__), '../languages/ar_it.wav'),
    'ar_ca': os.path.join(os.path.dirname(__file__), '../languages/ar_ca.wav'),
    'ar_nl': os.path.join(os.path.dirname(__file__), '../languages/ar_nl.wav'),

    'nl_zh': os.path.join(os.path.dirname(__file__), '../languages/nl_zh.wav'),
    'nl_en': os.path.join(os.path.dirname(__file__), '../languages/nl_en.wav'),
    'nl_de': os.path.join(os.path.dirname(__file__), '../languages/nl_de.wav'),
    'nl_es': os.path.join(os.path.dirname(__file__), '../languages/nl_es.wav'),
    'nl_fr': os.path.join(os.path.dirname(__file__), '../languages/nl_fr.wav'),
    'nl_uk': os.path.join(os.path.dirname(__file__), '../languages/nl_uk.wav'),
    'nl_it': os.path.join(os.path.dirname(__file__), '../languages/nl_it.wav'),
    'nl_ca': os.path.join(os.path.dirname(__file__), '../languages/nl_ca.wav'),
    'nl_ar': os.path.join(os.path.dirname(__file__), '../languages/nl_ar.wav'),

    't_en': os.path.join(os.path.dirname(__file__), '../languages/en_user.wav'),
    't_de': os.path.join(os.path.dirname(__file__), '../languages/de_user.wav'),
    't_zh': os.path.join(os.path.dirname(__file__), '../languages/zh_user.wav'),
    't_es': os.path.join(os.path.dirname(__file__), '../languages/es_user.wav'),
    't_fr': os.path.join(os.path.dirname(__file__), '../languages/fr_user.wav'),
    't_uk': os.path.join(os.path.dirname(__file__), '../languages/uk_user.wav'),
    't_it': os.path.join(os.path.dirname(__file__), '../languages/it_user.wav'),
    't_ca': os.path.join(os.path.dirname(__file__), '../languages/ca_user.wav'),
    't_ar': os.path.join(os.path.dirname(__file__), '../languages/ar_user.wav'),
    't_nl': os.path.join(os.path.dirname(__file__), '../languages/nl_user.wav'),

    'default': os.path.join(os.path.dirname(__file__), '../languages/default.wav'),
    'error':   os.path.join(os.path.dirname(__file__), '../languages/error.wav'),
}

current_audio_process = None
current_audio_owner = None
interrupt_whitelist = set()


def play_audio(lang_code, owner=None, allow_interrupt_from=None):
    global current_audio_process, current_audio_owner, interrupt_whitelist

    path = audio_map.get(lang_code)
    if path and os.path.exists(path):
        print(f"Playing audio: {path}")
        if current_audio_process and current_audio_process.poll() is None:
            current_audio_process.terminate()
            print(f"Previous audio interrupted. Owner was {current_audio_owner}")
        current_audio_owner = owner
        interrupt_whitelist = set(allow_interrupt_from or [])
        current_audio_process = subprocess.Popen(['aplay', '-D', CONFIRM_AUDIO_DEVICE, path])
        threading.Thread(target=monitor_audio_completion, args=(owner,), daemon=True).start()
    else:
        print(f"Audio file not found: {path}")


def wait_for_confirmation_audio():
    """
    Block until the current confirmation audio (play_audio) finishes.
    Used by Button5 so the confirmation plays fully before TTS starts.
    """
    global current_audio_process
    if current_audio_process and current_audio_process.poll() is None:
        current_audio_process.wait()


def monitor_audio_completion(owner_to_clear):
    global current_audio_process, current_audio_owner
    if current_audio_process:
        current_audio_process.wait()
        print(f"Audio playback by {owner_to_clear} completed.")
        if current_audio_owner == owner_to_clear:
            current_audio_owner = None
            print("Audio owner cleared. Buttons can now record.")


def interrupt_audio_playback(caller):
    global current_audio_process, current_audio_owner, interrupt_whitelist
    if current_audio_process and current_audio_process.poll() is None:
        if caller in interrupt_whitelist:
            current_audio_process.terminate()
            print(f"Audio interrupted by {caller}. Previous owner was {current_audio_owner}")
            return True
        else:
            print(f"Caller {caller} not allowed to interrupt current playback from {current_audio_owner}")
            return False
    return True


# Language selection logic
language_codes = ['ar', 'ca', 'de', 'en', 'es', 'fr', 'it', 'nl', 'uk', 'zh']
source_index = language_codes.index('en')  # guest's default lang
target_index = language_codes.index('de')  # user's default lang
selected_langs = {'source': language_codes[source_index], 'target': language_codes[target_index]}
print("Default setting: user's language is German (target), guest's language is English (source).")
# Button5 added to the interrupt whitelist so it can always start recording
play_audio("default", owner=None,
           allow_interrupt_from=["button1", "button2", "button3", "button4", "button5"])


# Button4 — select target (user's language)
def select_target():
    global system_locked_by
    if system_locked_by and system_locked_by != "button4":
        print(f"System is locked by {system_locked_by}. Button4 cannot operate.")
        return
    if not interrupt_audio_playback("button4"):
        return
    global target_index
    target_index = (target_index + 1) % len(language_codes)
    selected_langs['target'] = language_codes[target_index]
    print(f"Selected target language: {selected_langs['target']}")
    play_audio(f"t_{selected_langs['target']}", owner="button4", allow_interrupt_from=["button4"])


# Button3 — select source (guest's language)
def select_source():
    global system_locked_by
    if system_locked_by and system_locked_by != "button3":
        print(f"System is locked by {system_locked_by}. Button3 cannot operate.")
        return
    if not interrupt_audio_playback("button3"):
        return
    global source_index
    target_lang = selected_langs['target']
    available_sources = [lang for lang in language_codes if lang != target_lang]
    if selected_langs['source'] not in available_sources:
        selected_langs['source'] = available_sources[0]
        source_index = language_codes.index(selected_langs['source'])
    current_index = available_sources.index(selected_langs['source'])
    new_index = (current_index + 1) % len(available_sources)
    selected_langs['source'] = available_sources[new_index]
    source_index = language_codes.index(selected_langs['source'])
    print(f"Selected source language: {selected_langs['target']}_{selected_langs['source']}")
    play_audio(f"{selected_langs['target']}_{selected_langs['source']}",
               owner="button3", allow_interrupt_from=["button3"])


# Argos translation setup
installed_languages = argostranslate.translate.get_installed_languages()
language_by_code = {lang.code: lang for lang in installed_languages}
translation_cache = {}


def translate_text(source_lang_code, target_lang_code, text):
    key = (source_lang_code, target_lang_code)
    cached = translation_cache.get(key)

    if cached is None:
        from_lang = language_by_code.get(source_lang_code)
        to_lang   = language_by_code.get(target_lang_code)
        en_lang   = language_by_code.get('en')

        if not from_lang or not to_lang:
            translation_cache[key] = None
            return f"Error: '{source_lang_code}' or '{target_lang_code}' not installed."

        try:
            direct = from_lang.get_translation(to_lang)
            translation_cache[key] = ("direct", direct)
        except Exception:
            if en_lang:
                try:
                    to_en   = from_lang.get_translation(en_lang)
                    from_en = en_lang.get_translation(to_lang)
                    translation_cache[key] = ("via_en", (to_en, from_en))
                except Exception:
                    translation_cache[key] = None
            else:
                translation_cache[key] = None

        cached = translation_cache.get(key)

    if cached is None:
        return f"Error: No translation path from '{source_lang_code}' to '{target_lang_code}'."

    kind, obj = cached
    if kind == "direct":
        return obj.translate(text)
    else:
        to_en, from_en = obj
        return from_en.translate(to_en.translate(text))


class AudioRecorder:
    def __init__(self, sample_rate=48000, channels=1, chunk=1024,
                 folder_name=None, mic_name=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.device_index = self.get_input_device_index(mic_name)
        self.device_name = self.get_device_name(self.device_index)
        self.folder_name = folder_name or (
            "guest_mic_recordings" if "usb microphone" in (mic_name or "").lower()
            else "user_mic_recordings"
        )
        self.output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", self.folder_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def get_input_device_index(self, target_name=None):
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                if target_name is None or target_name.lower() in info["name"].lower():
                    return i
        raise ValueError(f"No input device found matching: {target_name}")

    def get_device_name(self, index):
        return self.audio.get_device_info_by_index(index)["name"].replace(" ", "_")

    def start_recording(self):
        if self.is_recording:
            return
        self.frames = []
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk,
        )
        self.is_recording = True

    def stop_recording(self):
        if not self.is_recording:
            return
        self.stream.stop_stream()
        self.stream.close()
        self.is_recording = False
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.device_name}_{timestamp}.wav"
        filepath = os.path.join(self.output_dir, filename)
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.frames))
        self.last_filepath = filepath

    def record_chunk(self):
        if self.is_recording:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
            except Exception as e:
                print(f"Audio read error: {e}")

    def close(self):
        if self.stream:
            self.stream.close()
        self.audio.terminate()

    def get_last_filepath(self):
        return getattr(self, "last_filepath", None)


guest_recorder = AudioRecorder(mic_name="usb microphone")
user_recorder   = AudioRecorder(mic_name="EPOS PC 7 USB")

guest_thread = None
user_thread  = None
auto_thread  = None    # ← NEW: thread for Button5 recording

stop_guest = False
stop_user  = False
stop_auto  = False     # ← NEW: stop flag for Button5


def get_latest_recording(folder_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(base_dir, "..", folder_name)
    files = glob.glob(os.path.join(folder, "*.wav"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def record_loop(recorder, stop_flag_name):
    while not globals()[stop_flag_name]:
        recorder.record_chunk()
        time.sleep(0.01)


def speak_with_piper(text, lang_code, device):
    timestamp = int(time.time())
    output_file = os.path.join(OUTPUT_DIR, f"{lang_code}_{timestamp}.wav")
    base = piper_args_base.get(lang_code)
    if not base:
        raise ValueError(f"No Piper model for language: {lang_code}")
    command = base + ["--output_file", output_file]
    print("Synthesizing speech with Piper...")
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    process.communicate(input=text.encode("utf-8"))
    print(f"Synthesized speech saved to: {output_file}")
    subprocess.run(["aplay", "-D", device, output_file])


# ─────────────────────────────────────────────────────────────────────────────
# Button 1 — Guest mic (manual, source→target)
# ─────────────────────────────────────────────────────────────────────────────
def button1_pressed():
    global system_locked_by
    if system_locked_by and system_locked_by != "button1":
        print(f"System locked by {system_locked_by}. Button1 blocked.")
        return
    system_locked_by = "button1"
    if not interrupt_audio_playback("button1"):
        system_locked_by = None
        return

    global guest_thread, stop_guest, guest_button_confirmed, guest_record_start_time

    if current_audio_owner in ["button3", "button4"]:
        print("Playback from Button3/4 active. Button1 blocked.")
        system_locked_by = None
        return

    source = selected_langs['source']
    target = selected_langs['target']
    if source == target:
        print("Source and target must differ.")
        play_audio("error", owner="button1", allow_interrupt_from=["button1"])
        system_locked_by = None
        return

    if guest_recorder.is_recording:
        system_locked_by = None
        return

    print("Button1 pressed — recording guest")
    guest_button_confirmed = False
    guest_record_start_time = time.time()
    guest_recorder.start_recording()
    stop_guest = False
    guest_thread = threading.Thread(target=record_loop, args=(guest_recorder, 'stop_guest'))
    guest_thread.start()


def button1_released():
    global guest_thread, stop_guest, guest_button_confirmed, guest_record_start_time
    global system_locked_by

    if not guest_recorder.is_recording:
        return

    duration = time.time() - guest_record_start_time
    stop_guest = True
    guest_thread.join()
    guest_recorder.stop_recording()

    latest_file = guest_recorder.get_last_filepath()
    if duration < 1.0:
        print("Button1 too short — discarding")
        if latest_file and os.path.exists(latest_file):
            os.remove(latest_file)
        system_locked_by = None
        return

    print("Button1 released — STT + translation + TTS")
    if latest_file:
        try:
            source = selected_langs['source']
            target = selected_langs['target']

            t0 = time.time()
            model_to_use = WHISPER_MODEL_EN if source == "en" else WHISPER_MODEL
            recognized = transcribe_with_optimization(model_to_use, latest_file, source)
            t1 = time.time()
            print(f"Recognized ({source}): {recognized}  [{t1-t0:.2f}s]")

            translation = translate_text(source, target, recognized)
            t2 = time.time()
            print(f"Translated  ({target}): {translation}  [{t2-t1:.2f}s]")

            speak_with_piper(translation, target, device=GUEST_TTS_DEVICE)
            t3 = time.time()
            print(f"[Total] {t3-t0:.2f}s")
            log_performance_metrics(t1-t0, t2-t1, t3-t2, t3-t0, latest_file)

        except Exception as e:
            print(f"Button1 error: {e}")
        finally:
            system_locked_by = None
            print("Lock released from button1")


# ─────────────────────────────────────────────────────────────────────────────
# Button 2 — User mic (manual, target→source)
# ─────────────────────────────────────────────────────────────────────────────
def button2_pressed():
    global system_locked_by
    if system_locked_by and system_locked_by != "button2":
        print(f"System locked by {system_locked_by}. Button2 blocked.")
        return
    system_locked_by = "button2"
    if not interrupt_audio_playback("button2"):
        system_locked_by = None
        return

    global user_thread, stop_user, user_button_confirmed, user_record_start_time

    if current_audio_owner in ["button3", "button4"]:
        print("Playback from Button3/4 active. Button2 blocked.")
        system_locked_by = None
        return

    source = selected_langs['source']
    target = selected_langs['target']
    if source == target:
        print("Source and target must differ.")
        play_audio("error", owner="button2", allow_interrupt_from=["button2"])
        system_locked_by = None
        return

    if user_recorder.is_recording:
        system_locked_by = None
        return

    print("Button2 pressed — recording user")
    user_button_confirmed = False
    user_record_start_time = time.time()
    user_recorder.start_recording()
    stop_user = False
    user_thread = threading.Thread(target=record_loop, args=(user_recorder, 'stop_user'))
    user_thread.start()


def button2_released():
    global user_thread, stop_user, user_button_confirmed, user_record_start_time
    global system_locked_by

    if not user_recorder.is_recording:
        return

    duration = time.time() - user_record_start_time
    stop_user = True
    user_thread.join()
    user_recorder.stop_recording()

    latest_file = user_recorder.get_last_filepath()
    if duration < 1.0:
        print("Button2 too short — discarding")
        if latest_file and os.path.exists(latest_file):
            os.remove(latest_file)
        system_locked_by = None
        return

    print("Button2 released — STT + translation + TTS")
    if latest_file:
        try:
            # reversed: user speaks target → guest hears source
            source = selected_langs['target']
            target = selected_langs['source']

            t0 = time.time()
            model_to_use = WHISPER_MODEL_EN if source == "en" else WHISPER_MODEL
            recognized = transcribe_with_optimization(model_to_use, latest_file, source)
            t1 = time.time()
            print(f"Recognized ({source}): {recognized}  [{t1-t0:.2f}s]")

            translation = translate_text(source, target, recognized)
            t2 = time.time()
            print(f"Translated  ({target}): {translation}  [{t2-t1:.2f}s]")

            speak_with_piper(translation, target, device=USER_TTS_DEVICE)
            t3 = time.time()
            print(f"[Total] {t3-t0:.2f}s")
            log_performance_metrics(t1-t0, t2-t1, t3-t2, t3-t0, latest_file)

        except Exception as e:
            print(f"Button2 error: {e}")
        finally:
            system_locked_by = None
            print("Lock released from button2")


# ─────────────────────────────────────────────────────────────────────────────
# Button 5 — AUTO language detection                                    ← NEW
# ─────────────────────────────────────────────────────────────────────────────
def button5_pressed():
    """
    Press & hold Button5 while someone speaks (guest OR user).
    Whisper will auto-detect the language on release.
    Uses the guest mic (shared microphone) for recording.
    """
    global system_locked_by, auto_thread, stop_auto, auto_record_start_time

    if system_locked_by and system_locked_by != "button5":
        print(f"System locked by {system_locked_by}. Button5 blocked.")
        return

    system_locked_by = "button5"
    if not interrupt_audio_playback("button5"):
        system_locked_by = None
        return

    if guest_recorder.is_recording:
        print("Mic already in use. Button5 blocked.")
        system_locked_by = None
        return

    print("Button5 pressed — AUTO-DETECT mode: recording…")
    auto_record_start_time = time.time()
    guest_recorder.start_recording()
    stop_auto = False
    auto_thread = threading.Thread(target=record_loop, args=(guest_recorder, 'stop_auto'))
    auto_thread.start()


def button5_released():
    """
    Release Button5 to stop recording and run auto language detection.

    Decision logic:
      detected == target (user's lang)  →  User spoke  → translate to source → guest hears it
      detected == source (guest's lang) →  Guest spoke → translate to target → user hears it
      detected == unknown supported lang → treat as new guest language, auto-update source,
                                           translate to target → user hears it
      detected == unsupported lang       → fall back to configured source, treat as guest
    """
    global auto_thread, stop_auto, auto_record_start_time, system_locked_by, source_index

    if not guest_recorder.is_recording:
        return

    duration = time.time() - auto_record_start_time
    stop_auto = True
    auto_thread.join()
    guest_recorder.stop_recording()

    latest_file = guest_recorder.get_last_filepath()

    if duration < 1.0:
        print("Button5 too short — discarding recording")
        if latest_file and os.path.exists(latest_file):
            os.remove(latest_file)
        system_locked_by = None
        return

    print("Button5 released — running auto language detection…")

    if not latest_file:
        system_locked_by = None
        return

    try:
        t0 = time.time()
        recognized, detected_lang, lang_prob = transcribe_auto_detect(latest_file)
        t1 = time.time()

        print(f"[Auto-detect] Detected: '{detected_lang}' "
              f"(confidence {lang_prob:.0%}), text: '{recognized}'  [{t1-t0:.2f}s]")

        # ── Guard: nothing was said ──────────────────────────────────────────
        if not recognized.strip():
            print("No speech detected.")
            play_audio("error", owner="button5", allow_interrupt_from=["button5"])
            system_locked_by = None
            return

        target_lang = selected_langs['target']   # user's language  (set by Button4)
        source_lang = selected_langs['source']   # guest's language (set by Button3)
        supported    = set(language_codes)

        # ── Decide who spoke ────────────────────────────────────────────────
        if detected_lang == target_lang:
            # ── Case A: USER is speaking ────────────────────────────────────
            # Translate to guest's pre-selected source language
            # Output device: USER_TTS_DEVICE (the speaker that faces the guest)
            actual_source  = target_lang
            actual_target  = source_lang
            output_device  = USER_TTS_DEVICE
            speaker_label  = "USER"

            # Confirm detected language to guest (plays guest's lang audio)
            print(f"[Auto-detect] → {speaker_label} speaking {actual_source} "
                  f"→ translate to guest lang ({actual_target})")
            play_audio(f"t_{actual_source}",
                       owner="button5", allow_interrupt_from=["button5"])

        else:
            # ── Case B: GUEST is speaking (or unrecognised speaker) ─────────
            # Output device: GUEST_TTS_DEVICE (the speaker that faces the user)
            output_device = GUEST_TTS_DEVICE
            speaker_label = "GUEST"

            if detected_lang in supported:
                # Auto-update guest's source language if it changed
                if detected_lang != source_lang:
                    print(f"[Auto-detect] Guest language changed: "
                          f"{source_lang} → {detected_lang}  (auto-update)")
                    selected_langs['source'] = detected_lang
                    source_index = language_codes.index(detected_lang)
                    source_lang  = detected_lang

                actual_source = detected_lang
            else:
                # Language detected but not in our supported set — fall back
                print(f"[Auto-detect] '{detected_lang}' not in supported list. "
                      f"Falling back to configured source '{source_lang}'.")
                actual_source = source_lang

            actual_target = target_lang

            # Confirm detected language pair to user (plays "target_source" audio)
            confirm_key = f"{actual_target}_{actual_source}"
            print(f"[Auto-detect] → {speaker_label} speaking {actual_source} "
                  f"→ translate to user lang ({actual_target})")
            play_audio(confirm_key, owner="button5", allow_interrupt_from=["button5"])

        # ── Wait for confirmation audio to finish before TTS ─────────────────
        # This prevents the two aplay calls from colliding on the same device.
        wait_for_confirmation_audio()

        # ── Guard: same language → no translation needed ─────────────────────
        if actual_source == actual_target:
            print("[Auto-detect] Source == target, skipping translation.")
            play_audio("error", owner="button5", allow_interrupt_from=["button5"])
            system_locked_by = None
            return

        # ── Translate ────────────────────────────────────────────────────────
        print(f"Translating {actual_source} → {actual_target}…")
        translation = translate_text(actual_source, actual_target, recognized)
        t2 = time.time()
        print(f"Translated: {translation}  [{t2-t1:.2f}s]")

        # ── TTS output ───────────────────────────────────────────────────────
        print(f"Speaking with Piper on {output_device}…")
        speak_with_piper(translation, actual_target, device=output_device)
        t3 = time.time()
        print(f"[Total Button5] {t3-t0:.2f}s")

        log_performance_metrics(t1-t0, t2-t1, t3-t2, t3-t0, latest_file)

    except Exception as e:
        print(f"Button5 auto-detect error: {e}")
        play_audio("error", owner="button5", allow_interrupt_from=["button5"])
    finally:
        system_locked_by = None
        print("Lock released from button5")
# ─────────────────────────────────────────────────────────────────────────────


# Bind all buttons
button1.when_pressed  = button1_pressed
button1.when_released = button1_released
button2.when_pressed  = button2_pressed
button2.when_released = button2_released
button3.when_pressed  = select_source
button4.when_pressed  = select_target
button5.when_pressed  = button5_pressed    # ← NEW
button5.when_released = button5_released   # ← NEW

try:
    print("Waiting for button presses…")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    guest_recorder.close()
    user_recorder.close()
    print("Cleaned up and exited.")
