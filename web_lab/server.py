"""
web_lab/server.py
==================
HTTP Server for TollMatcher Speech Translation Lab.

Provides REST API endpoints and serves the modern laboratory web interface.
Built using Python's standard library ThreadingHTTPServer for maximum portability
and zero external server dependencies, with full multipart audio upload support.
"""

import os
import sys
import json
import time
import email
import base64
import mimetypes
from urllib.parse import urlparse, parse_qs
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from web_lab.backend_adapter import get_adapter

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class SpeechTranslationLabHandler(SimpleHTTPRequestHandler):
    """Request handler for TollMatcher Speech Translation Lab."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PROJECT_ROOT, **kwargs)

    def _send_json(self, data: dict, status: int = 200, is_head: bool = False):
        """Sends a JSON response with CORS headers."""
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        if not is_head:
            self.wfile.write(body)

    def _send_error_json(self, message: str, status: int = 400):
        """Sends an error response in JSON format."""
        self._send_json({"status": "error", "message": message}, status=status)

    def do_OPTIONS(self):
        """Handles CORS preflight requests."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_HEAD(self):
        """Handles HEAD requests identically to GET without writing the response body."""
        self._handle_get(is_head=True)

    def do_GET(self):
        """Routes GET requests to API endpoints or static files."""
        self._handle_get(is_head=False)

    def _handle_get(self, is_head: bool = False):
        """Processes GET and HEAD requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        if not path:
            path = "/"

        # API Endpoints
        if path == "/api/health":
            adapter = get_adapter()
            self._send_json({
                "status": "ok",
                "app": "TollMatcher — Speech Translation Lab",
                "version": "1.0.0",
                "execution_mode": adapter.execution_mode,
                "models": {
                    "asr_model": adapter.asr_model_name,
                    "translation_engine": adapter.mt_model_name,
                    "faster_whisper_available": adapter.has_faster_whisper,
                    "argos_available": adapter.has_argos,
                    "marian_available": adapter.has_marian
                },
                "dataset_counts": {
                    "flores200_aligned_groups": len(adapter.pilot_groups),
                    "speech_examples": len(adapter.speech_examples)
                }
            }, is_head=is_head)
            return

        elif path == "/api/languages":
            adapter = get_adapter()
            self._send_json(adapter.get_supported_languages(), is_head=is_head)
            return

        elif path == "/api/benchmark-examples":
            adapter = get_adapter()
            self._send_json({
                "pilot_groups": adapter.pilot_groups,
                "speech_examples": adapter.speech_examples
            }, is_head=is_head)
            return

        elif path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return

        # Serve static frontend files
        if path == "/" or path == "/index.html":
            index_path = os.path.join(STATIC_DIR, "index.html")
            if os.path.exists(index_path):
                self._serve_file(index_path, "text/html; charset=utf-8", is_head=is_head)
                return

        elif path.startswith("/static/"):
            rel_file = path[len("/static/"):]
            file_path = os.path.join(STATIC_DIR, rel_file)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                mime, _ = mimetypes.guess_type(file_path)
                self._serve_file(file_path, mime or "application/octet-stream", is_head=is_head)
                return

        elif path.startswith("/benchmark/audio/"):
            rel_file = path.lstrip("/")
            file_path = os.path.join(PROJECT_ROOT, rel_file)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                self._serve_file(file_path, "audio/wav", is_head=is_head)
                return

        self._send_error_json("Resource not found", 404)

    def do_POST(self):
        """Routes POST requests to translation and evaluation endpoints."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        content_length = int(self.headers.get("Content-Length", 0))
        content_type = self.headers.get("Content-Type", "")

        adapter = get_adapter()

        # 1. Translate Audio Endpoint
        if path == "/api/translate-audio":
            if content_length == 0:
                self._send_error_json("Empty audio payload received.", 400)
                return

            try:
                audio_bytes = b""
                filename = "recording.wav"
                manual_language = "auto"

                if "multipart/form-data" in content_type:
                    boundary = content_type.split("boundary=")[-1].strip().encode()
                    raw_body = self.rfile.read(content_length)
                    parts = raw_body.split(b"--" + boundary)

                    for part in parts:
                        if b'name="audio"' in part:
                            header_end = part.find(b"\r\n\r\n")
                            if header_end != -1:
                                audio_part = part[header_end + 4:]
                                if audio_part.endswith(b"\r\n"):
                                    audio_part = audio_part[:-2]
                                audio_bytes = audio_part

                            if b'filename="' in part:
                                fn_start = part.find(b'filename="') + 10
                                fn_end = part.find(b'"', fn_start)
                                if fn_end != -1:
                                    filename = part[fn_start:fn_end].decode("utf-8", errors="ignore")

                        elif b'name="manual_language"' in part:
                            header_end = part.find(b"\r\n\r\n")
                            if header_end != -1:
                                lang_part = part[header_end + 4:].decode("utf-8", errors="ignore").strip()
                                if lang_part.endswith("\r\n"):
                                    lang_part = lang_part[:-2]
                                manual_language = lang_part

                elif "application/json" in content_type:
                    raw_body = self.rfile.read(content_length)
                    body_json = json.loads(raw_body.decode("utf-8"))
                    manual_language = body_json.get("manual_language", "auto")
                    filename = body_json.get("filename", "recording.wav")
                    if "audio_base64" in body_json:
                        import base64
                        b64_str = body_json["audio_base64"]
                        if "," in b64_str:
                            b64_str = b64_str.split(",", 1)[1]
                        audio_bytes = base64.b64decode(b64_str)
                    elif "benchmark_audio_id" in body_json:
                        audio_id = body_json["benchmark_audio_id"]
                        matched = next((ex for ex in adapter.speech_examples if ex.get("audio_example_id") == audio_id), None)
                        if matched:
                            wav_path = os.path.join(PROJECT_ROOT, matched.get("audio_file_path", ""))
                            if os.path.exists(wav_path):
                                with open(wav_path, "rb") as wf:
                                    audio_bytes = wf.read()
                                filename = os.path.basename(wav_path)
                else:
                    audio_bytes = self.rfile.read(content_length)

                if not audio_bytes or len(audio_bytes) < 100:
                    self._send_error_json("Audio data is empty or too small.", 400)
                    return

                result = adapter.translate_audio(
                    audio_bytes,
                    filename=filename,
                    manual_language=manual_language if manual_language != "auto" else None
                )
                self._send_json({"status": "success", "result": result})

            except Exception as e:
                self._send_error_json(f"Audio processing failed: {str(e)}", 500)
            return

        # 2. Translate Text Endpoint
        elif path == "/api/translate-text":
            try:
                raw_body = self.rfile.read(content_length)
                body_json = json.loads(raw_body.decode("utf-8"))
                text = body_json.get("text", "").strip()
                source_lang = body_json.get("source_language", "en").strip().lower()

                if not text:
                    self._send_error_json("Text cannot be empty.", 400)
                    return

                if source_lang not in ["en", "de", "fr"]:
                    self._send_error_json(f"Invalid source language '{source_lang}'. Must be 'en', 'de', or 'fr'.", 400)
                    return

                result = adapter.translate_text(text, source_lang)
                self._send_json({"status": "success", "result": result})

            except Exception as e:
                self._send_error_json(f"Text translation failed: {str(e)}", 500)
            return

        # 3. Evaluate Sample Endpoint
        elif path == "/api/evaluate-sample":
            try:
                raw_body = self.rfile.read(content_length)
                body_json = json.loads(raw_body.decode("utf-8"))
                group_id = body_json.get("group_id") or body_json.get("sample_id")
                source_lang = body_json.get("source_language", "en")
                hyp_transcript = body_json.get("hypothesis_transcript") or body_json.get("predicted_transcript")
                hyp_translations = body_json.get("hypothesis_translations") or body_json.get("predicted_translations")

                if not group_id:
                    self._send_error_json("group_id or sample_id is required.", 400)
                    return

                result = adapter.evaluate_benchmark_sample(
                    group_id=group_id,
                    source_lang=source_lang,
                    hypothesis_transcript=hyp_transcript,
                    hypothesis_translations=hyp_translations
                )
                self._send_json({"status": "success", "result": result})

            except Exception as e:
                self._send_error_json(f"Evaluation failed: {str(e)}", 500)
            return

        self._send_error_json("Endpoint not found", 404)

    def _serve_file(self, file_path: str, content_type: str, is_head: bool = False):
        """Helper to stream static files."""
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if not is_head:
                self.wfile.write(content)
        except Exception as e:
            self._send_error_json(f"Failed to read file: {e}", 500)

    def log_message(self, format, *args):
        """Keep console logging clean and informative."""
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {args[0]} - {args[1]} {args[2]}")


def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Starts the TollMatcher Speech Translation Lab web server."""
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, SpeechTranslationLabHandler)
    print("=" * 70)
    print("  TollMatcher — Speech Translation Lab (Web Tester)")
    print(f"  Running at: http://{host}:{port}/")
    print("  Endpoints:")
    print("    • Web Interface:        http://{host}:{port}/")
    print("    • Health Check:         http://{host}:{port}/api/health")
    print("    • Supported Languages:  http://{host}:{port}/api/languages")
    print("    • Audio Translation:    POST http://{host}:{port}/api/translate-audio")
    print("    • Text Translation:     POST http://{host}:{port}/api/translate-text")
    print("    • Benchmark Examples:   GET  http://{host}:{port}/api/benchmark-examples")
    print("    • Benchmark Evaluate:   POST http://{host}:{port}/api/evaluate-sample")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
        httpd.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TollMatcher Speech Translation Lab Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port)
