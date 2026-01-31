# Button3 + Button4 decide source and target languages (source = guest's native language, target = user's native language)
# And then Button1 function or Button2 function using gpiozero to record sounds
# will use 2 sets of specific hardwares to record and replay the translation
# How to test? 1. activate virtual env 2. connect 4 buttons rightly 3. run test
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

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Preload Whisper models once at startup to avoid per-call load latency
# Use faster-whisper for better performance with same accuracy
WHISPER_MODEL = WhisperModel("tiny", device="cpu", compute_type="int8", num_workers=1)
WHISPER_MODEL_EN = WhisperModel("tiny.en", device="cpu", compute_type="int8", num_workers=1)

# Model warmup to ensure consistent performance
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
        print("Installing required packages...")
        import subprocess
        try:
            subprocess.run(["pip", "install", "librosa", "soundfile"], check=True)
            print("Packages installed successfully. Please restart the application.")
            return input_file
        except Exception as install_error:
            print(f"Failed to install packages: {install_error}")
            print("Using fallback optimization...")
            return fallback_audio_optimization(input_file)
    except Exception as e:
        print(f"Warning: Audio optimization failed: {e}, using fallback")
        return fallback_audio_optimization(input_file)

def fallback_audio_optimization(input_file):
    """Fallback audio optimization using basic Python libraries"""
    try:
        import wave as wav_mod
        import numpy as np

        with wav_mod.open(input_file, 'rb') as wav_file:
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
        with wav_mod.open(optimized_file, 'wb') as wav_file:
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
    """Clean up memory to prevent accumulation"""
    gc.collect()
    memory_usage = psutil.virtual_memory().percent
    if memory_usage > 80:
        print(f"Warning: High memory usage ({memory_usage:.1f}%), forcing cleanup")
        gc.collect()

def log_performance_metrics(stt_time, translation_time, tts_time, total_time, audio_file):
    """Log performance metrics for monitoring"""
    try:
        file_size = os.path.getsize(audio_file) if os.path.exists(audio_file) else 0
        file_size_mb = file_size / (1024 * 1024)
        print(f"[Performance] STT: {stt_time:.2f}s, Translation: {translation_time:.2f}s, TTS: {tts_time:.2f}s, Total: {total_time:.2f}s, File: {file_size_mb:.2f}MB")
        if stt_time > 10:
            print(f"WARNING: STT time ({stt_time:.2f}s) is unusually high!")
        if total_time > 30:
            print(f"WARNING: Total processing time ({total_time:.2f}s) is very high!")
    except Exception as e:
        print(f"Performance logging error: {e}")

def transcribe_with_optimization(model, audio_file, language):
    """Transcribe with performance optimizations"""
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
            append_punctuations="\"'.!?:;)",
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

# Button pin definitions
BUTTON1_PIN = 27  # button1 - guest mic
BUTTON2_PIN = 22  # button2 - user mic
BUTTON3_PIN = 20  # button3 - select source language
BUTTON4_PIN = 21  # button4 - select target language

button1 = Button(BUTTON1_PIN, pull_up=True, bounce_time=0.2)
button2 = Button(BUTTON2_PIN, pull_up=True, bounce_time=0.2)
button3 = Button(BUTTON3_PIN)
button4 = Button(BUTTON4_PIN)

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
# For Button1 (Guest Mic) long-press confirmation
guest_button_confirmed = False
guest_record_start_time = None
# For Button2 (User Mic) long-press confirmation
user_button_confirmed = False
user_record_start_time = None


# Audio confirmation files 
audio_map = {
    'zh_en': os.path.join(os.path.dirname(__file__), '../languages/zh_en.wav'), # means user_guest lang
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
    'it_ca': os.path.join(os.path.dirname(__file__), '../languages/it_ar.wav'), 
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
        
    't_en': os.path.join(os.path.dirname(__file__), '../languages/en_user.wav'), # target = user lang
    't_de': os.path.join(os.path.dirname(__file__), '../languages/de_user.wav'),
    't_zh': os.path.join(os.path.dirname(__file__), '../languages/zh_user.wav'),
    't_es': os.path.join(os.path.dirname(__file__), '../languages/es_user.wav'), # new
    't_fr': os.path.join(os.path.dirname(__file__), '../languages/fr_user.wav'), # new
    't_uk': os.path.join(os.path.dirname(__file__), '../languages/uk_user.wav'), # new
    't_it': os.path.join(os.path.dirname(__file__), '../languages/it_user.wav'), # new
    't_ca': os.path.join(os.path.dirname(__file__), '../languages/ca_user.wav'), # new
    't_ar': os.path.join(os.path.dirname(__file__), '../languages/ar_user.wav'), # new
    't_nl': os.path.join(os.path.dirname(__file__), '../languages/nl_user.wav'), # new
    
        
    'default': os.path.join(os.path.dirname(__file__), '../languages/default.wav'),
    'error': os.path.join(os.path.dirname(__file__), '../languages/error.wav')
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
        current_audio_process = subprocess.Popen(['aplay', '-D', 'plughw:CARD=USB,DEV=0', path])

        # Start monitor thread to clear owner when playback finishes
        threading.Thread(target=monitor_audio_completion, args=(owner,), daemon=True).start()

    else:
        print(f"Audio file not found: {path}")

def monitor_audio_completion(owner_to_clear):
    global current_audio_process, current_audio_owner

    if current_audio_process:
        current_audio_process.wait()  # block until playback ends
        print(f"Audio playback by {owner_to_clear} completed.")
        if current_audio_owner == owner_to_clear:
            current_audio_owner = None
            print("Audio owner cleared. Button1/2 can now record.")


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
    return True  # nothing is playing, so allow



# Language selection logic
language_codes = ['ar', 'ca', 'de', 'en', 'es', 'fr', 'it', 'nl', 'uk', 'zh']
source_index = language_codes.index('en') # guest's default lang
target_index = language_codes.index('de') # user's default lang
selected_langs = {'source': language_codes[source_index], 'target': language_codes[target_index]}
print("Default setting: user's language is German (target), guest's language is English (source).")
play_audio("default", owner=None, allow_interrupt_from=["button1", "button2", "button3", "button4"]) 

# Button4
def select_target():
    global system_locked_by  

    if system_locked_by and system_locked_by != "button4":
        print(f"System is locked by {system_locked_by}. Button4 cannot operate.")
        return

    if not interrupt_audio_playback("button4"):
        return  # do not proceed

    global target_index
    target_index = (target_index + 1) % len(language_codes)
    selected_langs['target'] = language_codes[target_index]
    print(f"Selected target language: {selected_langs['target']}")
    play_audio(f"t_{selected_langs['target']}", owner="button4", allow_interrupt_from=["button4"])
#play target languages' audioes, couldn't be interrupted by any other button expect itself
    
# Button3
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
    play_audio(f"{selected_langs['target']}_{selected_langs['source']}", owner="button3", allow_interrupt_from=["button3"])
# couldn't be interrupted by any other button expect itself


#Argos
def translate_text(source_lang_code, target_lang_code, text):
    installed_languages = argostranslate.translate.get_installed_languages()
    
    from_lang = next((lang for lang in installed_languages if lang.code == source_lang_code), None)
    to_lang = next((lang for lang in installed_languages if lang.code == target_lang_code), None)
    en_lang = next((lang for lang in installed_languages if lang.code == 'en'), None)

    if not from_lang or not to_lang:
        return f"Error: Either '{source_lang_code}' or '{target_lang_code}' language is not installed."

    # Try direct translation
    try:
        direct_translation = from_lang.get_translation(to_lang)
        return direct_translation.translate(text)
    except Exception:
        pass  # direct translation not available

    # Try using English as intermediate
    if en_lang:
        try:
            to_en = from_lang.get_translation(en_lang)
            intermediate = to_en.translate(text)
            from_en = en_lang.get_translation(to_lang)
            return from_en.translate(intermediate)
        except Exception:
            return f"Error: Cannot find translation path from '{source_lang_code}' to '{target_lang_code}' (even with English as bridge)."
    else:
        return "Error: English language model is not installed for bridging translation."


class AudioRecorder:
    def __init__(self, sample_rate=48000, channels=1, chunk=1024, folder_name=None, mic_name=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.is_recording = False
        self.device_index = self.get_input_device_index(mic_name)
        self.device_name = self.get_device_name(self.device_index)
        self.folder_name = folder_name or ("guest_mic_recordings" if "SF-558" in mic_name else "user_mic_recordings") #Button1's mic
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", self.folder_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def get_input_device_index(self, target_name=None):
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                if target_name is None or target_name.lower() in info["name"].lower():
                    return i
        raise ValueError(f"No input device found matching name: {target_name}")

    def get_device_name(self, index):
        return self.audio.get_device_info_by_index(index)["name"].replace(" ", "_")

    def start_recording(self):
        if self.is_recording:
            return
        self.frames = []
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=self.channels,
                                      rate=self.sample_rate,
                                      input=True,
                                      input_device_index=self.device_index,
                                      frames_per_buffer=self.chunk)
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

guest_recorder = AudioRecorder(mic_name="SF-558")
user_recorder = AudioRecorder(mic_name="EPOS PC 7 USB")

guest_thread = None
user_thread = None
stop_guest = False
stop_user = False

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
    piper_models = {
        "en": {
            "model": "en_US-john-medium.onnx",
            "config": "en_US-john-medium.onnx.json"
        },
        "de": {
            "model": "de_DE-thorsten-medium.onnx",
            "config": "de_DE-thorsten-medium.onnx.json"
        },
	    "es": {
            "model": "es_ES-carlfm-x_low.onnx",
            "config": "es_ES-carlfm-x_low.onnx.json"
        },
        "fr": {
            "model": "fr_FR-tom-medium.onnx",
            "config": "fr_FR-tom-medium.onnx.json"
        },
        "uk": {
            "model": "uk_UA-lada-x_low.onnx",
            "config": "uk_UA-lada-x_low.onnx.json"
        },    
        "it": {
            "model": "it_IT-paola-medium.onnx",
            "config": "it_IT-paola-medium.onnx.json"
        },    
        "ca": {
            "model": "ca_ES-upc_ona-x_low.onnx",
            "config": "ca_ES-upc_ona-x_low.onnx.json"
        },    
        "ar": {
            "model": "ar_JO-kareem-low.onnx",
            "config": "ar_JO-kareem-low.onnx.json"
        },    
        "nl": {
            "model": "nl_NL-pim-medium.onnx",
            "config": "nl_NL-pim-medium.onnx.json"
        },    
        "zh": {
            "model": "zh_CN-huayan-medium.onnx",
             "config": "zh_CN-huayan-medium.onnx.json"
        }
    }

    PIPER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../piper_rebuild/piper/piper'))
    OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output'))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    voice = piper_models[lang_code]
    timestamp = int(time.time())
    output_file = os.path.join(OUTPUT_DIR, f"{lang_code}_{timestamp}.wav")

    command = [
        PIPER_PATH,
        "--model", os.path.abspath(os.path.join(os.path.dirname(__file__), f"../piper_rebuild/piper/{voice['model']}")),
        "--config", os.path.abspath(os.path.join(os.path.dirname(__file__), f"../piper_rebuild/piper/{voice['config']}")),
        "--output_file", output_file,
    ]

    print("Synthesizing speech with Piper...")
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    process.communicate(input=text.encode("utf-8"))
    print(f"Synthesized speech saved to: {output_file}")

    # Play using the specified speaker device
    subprocess.run(["aplay", "-D", device, output_file])

# Button 1 Logic (Guest)
def button1_pressed():
    global system_locked_by  
    if system_locked_by and system_locked_by != "button1":
        print(f"System is locked by {system_locked_by}. Button1 cannot operate.")
        return

    system_locked_by = "button1"  
    interrupt_audio_playback("button1")

    global guest_thread, stop_guest, guest_button_confirmed, guest_record_start_time

    if current_audio_owner in ["button3", "button4"]:
        print("Playback from Button3/4 is active. Button1 recording blocked.")
        return
    
    source = selected_langs['source']
    target = selected_langs['target']
    if source == target:
        print("Source and target languages must be different.")
        play_audio("error", owner="button1", allow_interrupt_from=["button1"])
        system_locked_by = None
        return

    if guest_recorder.is_recording:
        return

    print("Button1 pressed ΓÇö start recording immediately")
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
    
    latest_file = get_latest_recording("guest_mic_recordings")
    if duration < 1.0:  # Have to hold button1 more than 1s
        print("Button1 released too early ΓÇö delete latest recording")
        if latest_file and os.path.exists(latest_file):
            os.remove(latest_file)
        system_locked_by = None
        return

    print("Button1 released after 1s ΓÇö continue with STT + translation + TTS")

    latest_file = get_latest_recording("guest_mic_recordings")
    if latest_file:
        print(f"Latest guest recording: {latest_file}")
        try:
            source = selected_langs['source']
            target = selected_langs['target']

            t0 = time.time()
            print(f"Transcribing ({source})...")
            model_to_use = WHISPER_MODEL_EN if source == "en" else WHISPER_MODEL
            recognized = transcribe_with_optimization(model_to_use, latest_file, source)
            t1 = time.time()
            print("Recognized text:", recognized)
            print(f"[Time] STT took {t1 - t0:.2f} seconds")

            print(f"Translating to {target}...")
            translation = translate_text(source, target, recognized)
            t2 = time.time()
            print("Translated text:", translation)
            print(f"[Time] Translation took {t2 - t1:.2f} seconds")

            print("Speaking with Piper...")
            speak_with_piper(translation, target, device="plughw:CARD=USB,DEV=0")
            t3 = time.time()
            print(f"[Time] TTS took {t3 - t2:.2f} seconds")
            print(f"[Total Time] {t3 - t0:.2f} seconds")
            log_performance_metrics(t1-t0, t2-t1, t3-t2, t3-t0, latest_file)

        except Exception as e:
            print(f"Error in transcription, translation, or TTS: {e}")
        finally:
            system_locked_by = None
            print("System lock released from button1")



# Button 2 Logic (User)
def button2_pressed():
    global system_locked_by
    if system_locked_by and system_locked_by != "button2":
        print(f"System is locked by {system_locked_by}. Button2 cannot operate.")
        return
    system_locked_by = "button2" 
    interrupt_audio_playback("button2")

    global user_thread, stop_user, user_button_confirmed, user_record_start_time

    if current_audio_owner in ["button3", "button4"]:
        print("Playback from Button3/4 is active. Button2 recording blocked.")
        return

    source = selected_langs['source']
    target = selected_langs['target']
    if source == target:
        print("Source and target languages must be different.")
        play_audio("error", owner="button2", allow_interrupt_from=["button2"])
        system_locked_by = None
        return

    if user_recorder.is_recording:
        return

    print("Button2 pressed ΓÇö start recording immediately")
    user_button_confirmed = False
    user_record_start_time = time.time()

    user_recorder.start_recording()
    stop_user = False
    user_thread = threading.Thread(target=record_loop, args=(user_recorder, 'stop_user'))
    user_thread.start()




# button2s target language and source language are opposite with button1's
def button2_released():
    global user_thread, stop_user, user_button_confirmed, user_record_start_time
    global system_locked_by 
    
    if not user_recorder.is_recording:
        return

    duration = time.time() - user_record_start_time
    stop_user = True
    user_thread.join()
    user_recorder.stop_recording()

    latest_file = get_latest_recording("user_mic_recordings")
    if duration < 1.0: # Have to hold button2 more than 1s
        print("Button2 released too early ΓÇö delete latest recording")
        if latest_file and os.path.exists(latest_file):
            os.remove(latest_file)
        system_locked_by = None
        return
        
    print("Button2 released after 1s ΓÇö continue with STT + translation + TTS")

    latest_file = get_latest_recording("user_mic_recordings")
    if latest_file:
        print(f"Latest user recording: {latest_file}")
        try:
            # reverse
            source = selected_langs['target']
            target = selected_langs['source']

            t0 = time.time()
            print(f"Transcribing ({source})...")
            model_to_use = WHISPER_MODEL_EN if source == "en" else WHISPER_MODEL
            recognized = transcribe_with_optimization(model_to_use, latest_file, source)
            t1 = time.time()
            print("Recognized text:", recognized)
            print(f"[Time] STT took {t1 - t0:.2f} seconds")

            print(f"Translating to {target}...")
            translation = translate_text(source, target, recognized)
            t2 = time.time()
            print("Translated text:", translation)
            print(f"[Time] Translation took {t2 - t1:.2f} seconds")

            print("Speaking with Piper...")
            speak_with_piper(translation, target, device="plughw:CARD=UACDemoV10,DEV=0")
            t3 = time.time()
            print(f"[Time] TTS took {t3 - t2:.2f} seconds")
            print(f"[Total Time] {t3 - t0:.2f} seconds")
            log_performance_metrics(t1-t0, t2-t1, t3-t2, t3-t0, latest_file)

        except Exception as e:
            print(f"Error in transcription, translation, or TTS: {e}")
        finally:
            system_locked_by = None
            print("System lock released from button2")

# gpiozero button binding
button1.when_pressed = button1_pressed
button1.when_released = button1_released
button2.when_pressed = button2_pressed
button2.when_released = button2_released
button3.when_pressed = select_source
button4.when_pressed = select_target

try:
    print("Waiting for button presses...")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    guest_recorder.close()
    user_recorder.close()
    print("Cleaned up and exited.")
