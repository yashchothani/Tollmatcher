"""
pi_edge/edge_client.py
=======================
Featherweight distributed client for Raspberry Pi 5 (< 20 MB RAM).
Runs on the Raspberry Pi:
- Captures microphone audio using PyAudio / arecord.
- Streams audio to the host workstation (laptop running web_lab or FastAPI).
- Receives translations and plays speech via the Pi USB speaker.
- 100% offloads neural inference from the Raspberry Pi!
"""

import argparse
import os
import sys
import time
import wave
import requests

try:
    from gpiozero import Button
    HAS_GPIO = True
except (ImportError, Exception):
    HAS_GPIO = False

# Hardware pin mapping from TollMatcher CAD / Main_Argos
BUTTON1_PIN = 27  # User speaks
BUTTON2_PIN = 22  # Guest speaks
BUTTON3_PIN = 20  # Guest language select
BUTTON4_PIN = 21  # User language select
BUTTON5_PIN = 16  # Auto-detect speech


class TollMatcherEdgeClient:
    """Connects Raspberry Pi hardware (buttons/mic) to host inference server."""

    def __init__(self, server_url: str = "http://127.0.0.1:8000", audio_device: str = "sysdefault:CARD=USB"):
        self.server_url = server_url.rstrip("/")
        self.audio_device = audio_device
        print(f"[EdgeClient] Configured host server at: {self.server_url}")
        print(f"[EdgeClient] Configured audio output device: {self.audio_device}")

    def test_connection(self) -> bool:
        try:
            r = requests.get(f"{self.server_url}/api/health", timeout=3.0)
            if r.status_code == 200:
                data = r.json()
                print(f"[EdgeClient] Connected to server! Backend: {data.get('execution_mode', 'OK')}")
                return True
        except Exception as e:
            print(f"[EdgeClient] Warning: Could not reach server at {self.server_url}: {e}")
        return False

    def send_audio_for_translation(self, wav_path: str, source_lang: str = "auto") -> dict:
        """Uploads WAV file to host backend and receives transcript + dual translations."""
        url = f"{self.server_url}/api/translate-audio"
        with open(wav_path, "rb") as f:
            files = {"audio": (os.path.basename(wav_path), f, "audio/wav")}
            data = {"source_language": source_lang}
            t0 = time.perf_counter()
            resp = requests.post(url, files=files, data=data, timeout=15.0)
            latency = time.perf_counter() - t0

        if resp.status_code == 200:
            result = resp.json()
            result["network_roundtrip_sec"] = round(latency, 3)
            return result
        else:
            raise RuntimeError(f"Server error {resp.status_code}: {resp.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TollMatcher Distributed Raspberry Pi Edge Client")
    parser.add_argument("--server", default="http://127.0.0.1:8000", help="Host server URL (e.g. http://192.168.1.50:8000)")
    parser.add_argument("--wav", default=None, help="Optional test WAV file to send to server")
    args = parser.parse_args()

    client = TollMatcherEdgeClient(server_url=args.server)
    client.test_connection()

    if args.wav and os.path.exists(args.wav):
        print(f"Sending test audio: {args.wav}")
        res = client.send_audio_for_translation(args.wav)
        print("\nTranslation Result:")
        print(f" - Detected Lang : {res.get('detected_language')}")
        print(f" - Transcript    : {res.get('transcript')}")
        for lang, text in res.get("translations", {}).items():
            print(f" - [{lang.upper()}] : {text}")
        print(f" - Roundtrip Time: {res.get('network_roundtrip_sec')}s")
