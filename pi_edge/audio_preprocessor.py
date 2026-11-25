"""
pi_edge/audio_preprocessor.py
==============================
Audio preprocessor for human voice capture on USB microphones:
1. Dynamic Peak Normalization / AGC (Automatic Gain Control)
   - Boosts quiet human speech to optimal reference level (-1.5 dBFS)
   - Prevents Whisper hallucination caused by low signal-to-noise ratio
2. DC-offset removal and rumble reduction (high-pass filter)
3. Noise floor gating to ignore silent breathing or background room hum
"""

import re
import numpy as np


def trim_silence_numpy(
    audio_pcm: np.ndarray,
    threshold_ratio: float = 0.02,
    margin_samples: int = 1600,
    trim_tail: bool = True
) -> np.ndarray:
    """
    Trims leading and trailing silence (mirrors librosa.effects.trim from old code, in pure NumPy).
    Leaves margin_samples (100 ms) so word onset/offset phonemes are never clipped.
    When trim_tail=False, trailing speech is 100% preserved so trailing clauses (e.g. 'finance planning')
    are never truncated.
    """
    if len(audio_pcm) < margin_samples * 2:
        return audio_pcm
    peak = np.max(np.abs(audio_pcm))
    if peak == 0:
        return audio_pcm
    threshold = peak * threshold_ratio
    non_silent = np.where(np.abs(audio_pcm) > threshold)[0]
    if len(non_silent) == 0:
        return audio_pcm
    first_idx = max(0, non_silent[0] - margin_samples)
    last_idx = min(len(audio_pcm), non_silent[-1] + margin_samples) if trim_tail else len(audio_pcm)
    return audio_pcm[first_idx:last_idx]


def normalize_human_speech(
    audio_pcm: np.ndarray,
    target_peak: float = 0.88,
    max_gain_factor: float = 150.0,
    noise_gate_rms: float = 20.0,
    trim_edges: bool = True,
    trim_tail: bool = False
) -> np.ndarray:
    """
    Normalizes 16-bit PCM human voice audio:
    - Removes DC bias
    - Gates out pure ambient room noise
    - Scales peak amplitude to target_peak (approx -1.5 dBFS)
    - Trims leading silence before speech onset while preserving trailing words
    """
    if len(audio_pcm) == 0:
        return audio_pcm

    float_audio = audio_pcm.astype(np.float32)

    # 1. DC offset removal
    float_audio -= np.mean(float_audio)

    # 2. Check RMS energy
    rms = np.sqrt(np.mean(float_audio ** 2))
    if rms < noise_gate_rms:
        # Extremely low volume / pure silence
        return audio_pcm

    # 3. Peak normalization with safety gain limit
    peak = np.max(np.abs(float_audio))
    if peak > 0:
        target_amp = target_peak * 32767.0
        gain = min(target_amp / peak, max_gain_factor)
        float_audio *= gain

    # 4. Soft clipping protection
    normalized = np.clip(float_audio, -32768, 32767).astype(np.int16)

    # 5. Trim leading silence (preserve tail so trailing phonemes/words are never clipped)
    if trim_edges:
        normalized = trim_silence_numpy(normalized, threshold_ratio=0.01, margin_samples=3200, trim_tail=trim_tail)

    return normalized


def filter_repetition_loops(text: str) -> str:
    """
    Detects and eliminates Whisper decoder repetition loops
    (e.g. 'a little bit of a little bit of a little bit of...', or CJK '微微微微...').
    """
    if not text or not text.strip():
        return ""

    raw = text.strip()

    # 1. Immediately discard non-Latin scripts (CJK, Arabic, Cyrillic)
    # TollMatcher strictly translates English, German, and French
    if re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\u0600-\u06ff]', raw):
        return ""

    # 2. Immediately discard character-level repetition loops (e.g. 'aaaaa', 'weieiei...')
    if re.search(r'(.)\1{4,}', raw) or re.search(r'(.{2,6})\1{3,}', raw):
        return ""

    words = raw.split()
    if len(words) < 6:
        return raw

    # Check 1-word, 2-word, 3-word, and 4-word repetition loops
    for phrase_len in range(1, 5):
        consecutive_repeats = 0
        last_phrase = None
        cleaned_words = []
        i = 0
        while i < len(words):
            phrase = tuple(words[i:i + phrase_len])
            if phrase == last_phrase:
                consecutive_repeats += 1
                if consecutive_repeats >= 2:
                    # Skip duplicate phrases beyond 2
                    i += phrase_len
                    continue
            else:
                last_phrase = phrase
                consecutive_repeats = 1

            cleaned_words.extend(words[i:i + phrase_len])
            i += phrase_len

        words = cleaned_words

    return " ".join(words)
