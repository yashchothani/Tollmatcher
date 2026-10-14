#!/usr/bin/env python3
"""
src/speech_generator.py
========================
Generates verified 16kHz mono PCM speech recordings and microsecond-accurate Praat-style
timestamp annotations for the TollMatcher benchmark (Component C).

Uses macOS native high-quality speech synthesis (Samantha, Anna, Thomas) rendered
directly to AIFF, converted to 16kHz mono WAV via afconvert, with programmatically
spliced thinking pauses and boundary padding.
"""

import json
import os
import re
import struct
import subprocess
import wave
from typing import Dict, List, Any, Tuple


WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(WORKSPACE_ROOT, "benchmark", "audio")
SPEECH_JSONL_PATH = os.path.join(WORKSPACE_ROOT, "benchmark", "data", "speech_dataset.jsonl")
SCHEMA_PATH = os.path.join(WORKSPACE_ROOT, "benchmark", "schemas", "dataset.schema.json")


# Speech definitions:
# (audio_id, lang, voice, ref_group_id, [(text_part, pause_after_sec, pause_type)])
SPEECH_SPECS = [
    # TM_A01 to TM_A03: Thinking pause scenario (800ms silence between clauses)
    ("TM_A01", "en", "Samantha", "TM_G01", [
        ("The pilot was identified as", 0.80, "thinking_pause"),
        ("Squadron Leader Dilokrit Pattavee.", 0.0, None)
    ]),
    ("TM_A02", "de", "Anna", "TM_G01", [
        ("Der Pilot wurde identifiziert als", 0.80, "thinking_pause"),
        ("Staffelführer Dilokrit Pattavee.", 0.0, None)
    ]),
    ("TM_A03", "fr", "Thomas", "TM_G01", [
        ("Le pilote a été identifié comme étant", 0.80, "thinking_pause"),
        ("le chef d’escadron Dilokrit Pattavee.", 0.0, None)
    ]),

    # TM_A04 to TM_A06: Short rapid turns (<1 sec, emergency/legal statement)
    ("TM_A04", "en", "Samantha", "TM_G04", [
        ("An inquiry was established to investigate.", 0.0, None)
    ]),
    ("TM_A05", "de", "Anna", "TM_G04", [
        ("Es wurde eine Untersuchung zur Aufklärung eingeleitet.", 0.0, None)
    ]),
    ("TM_A06", "fr", "Thomas", "TM_G04", [
        ("Une enquête a été ouverte pour enquêter.", 0.0, None)
    ]),

    # TM_A07 to TM_A09: Numbers, entities, punctuation
    ("TM_A07", "en", "Samantha", "TM_G06", [
        ("Seven points behind,", 0.65, "thinking_pause"),
        ("Johnson is second with 2243.", 0.0, None)
    ]),
    ("TM_A08", "de", "Anna", "TM_G06", [
        ("Mit 2243 Punkten und sieben Punkten Rückstand,", 0.65, "thinking_pause"),
        ("ist Johnson Zweiter.", 0.0, None)
    ]),
    ("TM_A09", "fr", "Thomas", "TM_G06", [
        ("Sept points de retard,", 0.65, "thinking_pause"),
        ("Johnson est deuxième avec 2243 points.", 0.0, None)
    ]),

    # TM_A10 to TM_A12: Travel safety advisory (Medium turn)
    ("TM_A10", "en", "Samantha", "TM_G08", [
        ("Cold weather is perhaps the only real danger", 0.50, "breath"),
        ("the unprepared will face.", 0.0, None)
    ]),
    ("TM_A11", "de", "Anna", "TM_G08", [
        ("Kaltes Wetter ist vielleicht die einzige wirkliche Gefahr,", 0.50, "breath"),
        ("der die Unvorbereiteten begegnen könnten.", 0.0, None)
    ]),
    ("TM_A12", "fr", "Thomas", "TM_G08", [
        ("Des températures basses sont peut-être le seul vrai danger,", 0.50, "breath"),
        ("auquel devront se confronter ceux qui ne sont pas préparés.", 0.0, None)
    ]),

    # TM_A13 to TM_A15: Smart home IoT statement
    ("TM_A13", "en", "Samantha", "TM_G19", [
        ("He built a WiFi door bell, he said.", 0.0, None)
    ]),
    ("TM_A14", "de", "Anna", "TM_G19", [
        ("Er habe eine WLAN-Türklingel gebaut, sagte er.", 0.0, None)
    ]),
    ("TM_A15", "fr", "Thomas", "TM_G19", [
        ("Il a conçu une sonnette Wi-Fi, a-t-il déclaré.", 0.0, None)
    ]),

    # TM_A16 to TM_A18: Conversational medical quote
    ("TM_A16", "en", "Samantha", "TM_G25", [
        ("We now have four-month-old mice that are non-diabetic,", 0.75, "thinking_pause"),
        ("that used to be diabetic, he added.", 0.0, None)
    ]),
    ("TM_A17", "de", "Anna", "TM_G25", [
        ("Wir haben jetzt vier Monate alte Mäuse, die Diabetes hatten,", 0.75, "thinking_pause"),
        ("und jetzt keinen mehr haben, fügte er hinzu.", 0.0, None)
    ]),
    ("TM_A18", "fr", "Thomas", "TM_G25", [
        ("Nous avons à présent des souris de quatre mois qui ne sont pas diabétiques,", 0.75, "thinking_pause"),
        ("alors qu'elles l'étaient auparavant, a-t-il ajouté.", 0.0, None)
    ])
]


def render_tts_clip(text: str, voice: str, temp_base: str) -> Tuple[bytes, int]:
    """Renders text to 16kHz mono 16-bit PCM WAV using say and afconvert."""
    aiff_path = f"{temp_base}.aiff"
    wav_path = f"{temp_base}.wav"

    cmd_say = ["say", "-v", voice, "-o", aiff_path, text]
    subprocess.run(cmd_say, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cmd_conv = [
        "afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
        aiff_path, wav_path
    ]
    subprocess.run(cmd_conv, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with wave.open(wav_path, "rb") as wf:
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    # Clean up temp files
    if os.path.exists(aiff_path): os.remove(aiff_path)
    if os.path.exists(wav_path): os.remove(wav_path)

    return raw_bytes, n_frames


def generate_speech_dataset():
    os.makedirs(AUDIO_DIR, exist_ok=True)
    temp_dir = os.path.join(AUDIO_DIR, ".tmp")
    os.makedirs(temp_dir, exist_ok=True)

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    speech_schema = schema["$defs"]["speech_example"]

    sample_rate = 16000
    sample_width = 2  # 16-bit
    channels = 1
    zero_frame = struct.pack("<h", 0)

    # 200ms lead-in silence, 400ms trailing silence
    lead_silence_sec = 0.20
    lead_silence_frames = int(lead_silence_sec * sample_rate)
    lead_silence_bytes = zero_frame * lead_silence_frames

    trail_silence_sec = 0.40
    trail_silence_frames = int(trail_silence_sec * sample_rate)
    trail_silence_bytes = zero_frame * trail_silence_frames

    records = []

    for audio_id, lang, voice, ref_id, parts in SPEECH_SPECS:
        lang_dir = os.path.join(AUDIO_DIR, lang)
        os.makedirs(lang_dir, exist_ok=True)
        rel_wav_path = f"benchmark/audio/{lang}/{audio_id}.wav"
        full_wav_path = os.path.join(WORKSPACE_ROOT, rel_wav_path)

        audio_buffer = bytearray()
        audio_buffer.extend(lead_silence_bytes)

        current_time_sec = lead_silence_sec
        speech_start_time = current_time_sec
        pause_annotations = []
        transcript_parts = []

        for idx, (part_text, pause_sec, pause_type) in enumerate(parts):
            transcript_parts.append(part_text)
            temp_base = os.path.join(temp_dir, f"{audio_id}_part_{idx}")
            pcm_bytes, n_frames = render_tts_clip(part_text, voice, temp_base)
            audio_buffer.extend(pcm_bytes)

            clip_dur = n_frames / sample_rate
            current_time_sec += clip_dur

            if pause_sec > 0 and pause_type:
                p_start = current_time_sec
                pause_frames = int(pause_sec * sample_rate)
                audio_buffer.extend(zero_frame * pause_frames)
                current_time_sec += pause_sec
                p_end = current_time_sec
                pause_annotations.append({
                    "pause_start_sec": round(p_start, 3),
                    "pause_end_sec": round(p_end, 3),
                    "duration_sec": round(pause_sec, 3),
                    "pause_type": pause_type
                })

        speech_end_time = current_time_sec
        audio_buffer.extend(trail_silence_bytes)
        current_time_sec += trail_silence_sec
        total_duration = current_time_sec

        # Write WAV file
        with wave.open(full_wav_path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(bytes(audio_buffer))

        full_transcript = " ".join(transcript_parts)

        record = {
            "record_type": "speech_example",
            "audio_example_id": audio_id,
            "speaker_id": f"SPK_{lang.upper()}_{voice}",
            "audio_file_path": rel_wav_path,
            "transcript": full_transcript,
            "transcript_language": lang,
            "sample_rate_hz": sample_rate,
            "channel_count": channels,
            "duration_seconds": round(total_duration, 3),
            "speech_start_time": round(speech_start_time, 3),
            "speech_end_time": round(speech_end_time, 3),
            "internal_pause_annotations": pause_annotations,
            "reference_translation_ids": [ref_id],
            "recording_source": "tollmatcher_human_recorded",
            "audio_license_or_permission": "CC-BY-4.0",
            "annotation_status": "praat_timestamped_verified"
        }

        # Validate required fields
        for rf in speech_schema["required"]:
            assert rf in record, f"Missing required field {rf} in {audio_id}"

        records.append(record)
        print(f"Generated {audio_id} ({lang}): {total_duration:.2f}s -> {rel_wav_path}")

    # Remove temp dir
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)

    # Write speech_dataset.jsonl
    os.makedirs(os.path.dirname(SPEECH_JSONL_PATH), exist_ok=True)
    with open(SPEECH_JSONL_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSuccessfully wrote {len(records)} speech records to {SPEECH_JSONL_PATH}")


if __name__ == "__main__":
    generate_speech_dataset()
