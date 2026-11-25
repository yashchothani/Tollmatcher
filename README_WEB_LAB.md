# TollMatcher — Speech Translation Lab (Web Tester)

A local, temporary web application for testing and evaluating the TollMatcher English–German–French speech translation dataset and pipeline.

---

## 1. Objective & Capabilities

**TollMatcher — Speech Translation Lab** provides an interactive research-grade dashboard for testing:
- **Audio Capture & Recording:** Live microphone recording via browser `MediaRecorder` API or audio file upload (`.wav`, `.mp3`, `.webm`).
- **Tri-lingual Multilingual Support:** English (EN), German (DE), and French (FR).
- **Automatic Language Detection:** Whisper-based acoustic language detection with manual language override options (`Auto`, `en`, `de`, `fr`).
- **Complementary Dual Translations:** Never translates into itself:
  - **English (EN) Input** $\rightarrow$ **German (DE)** & **French (FR)**
  - **German (DE) Input** $\rightarrow$ **English (EN)** & **French (FR)**
  - **French (FR) Input** $\rightarrow$ **English (EN)** & **German (DE)**
- **System & Latency Evaluation:** Real-time calculation of ASR inference time, Translation time, Total pipeline latency, Audio duration, and Real-Time Factor (RTF).
- **FLORES-200 Dataset Benchmark Evaluation:** Side-by-side comparison of model transcripts and translations against official human annotations with automated Word Error Rate (WER), Character Error Rate (CER), SacreBLEU, and chrF metrics.
- **Exporting Results:** Single-click export of evaluation sessions to structured JSON and CSV reports.

---

## 2. Architecture & Reuse of Existing Models

The web lab is built as a non-invasive adapter that directly bridges existing project components without replacing the original pipeline:

```
                          ┌──────────────────────────────────────────────┐
                          │     Browser Web UI (HTML5 / Vanilla CSS)     │
                          │   Live Mic Capture (MediaRecorder) / Upload  │
                          └──────────────────────┬───────────────────────┘
                                                 │ HTTP REST API (port 8000)
                                                 ▼
                          ┌──────────────────────────────────────────────┐
                          │        web_lab/server.py (HTTP Server)       │
                          │  /api/translate-audio    /api/translate-text │
                          │  /api/health             /api/languages      │
                          │  /api/benchmark-examples /api/evaluate-sample│
                          └──────────────────────┬───────────────────────┘
                                                 │
                                                 ▼
                          ┌──────────────────────────────────────────────┐
                          │    web_lab/backend_adapter.py (Adapter)      │
                          └──────┬───────────────────────┬───────────────┘
                                 │                       │
      ┌──────────────────────────┴────────┐   ┌──────────┴──────────────────────────┐
      │   Production Neural Models        │   │   Offline Benchmark Reference       │
      │   (Used when packages installed)  │   │   (Zero-dependency fallback)        │
      ├───────────────────────────────────┤   ├─────────────────────────────────────┤
      │ • faster-whisper (tiny int8 CPU)  │   │ • 36 Aligned FLORES-200 Groups      │
      │   (from main/Main_Argos.py:74)    │   │   (benchmark/data/pilot_dataset)    │
      │ • ArgosTranslate                  │   │ • 18 Praat-annotated WAV clips      │
      │   (from main/translate_text_argos)│   │   (benchmark/data/speech_dataset)   │
      │ • MarianMT (Helsinki-NLP opus-mt) │   │ • Evaluation Metrics Engine         │
      │   (from main/translate_text_Marian│   │   (src/metrics/asr & translation)   │
      └───────────────────────────────────┘   └─────────────────────────────────────┘
```

---

## 3. Quick Start Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.12, 3.14).
- No external web frameworks required (runs out of the box with standard library).

### Launching the Web Lab
From the repository root:

```bash
# 1. Start the TollMatcher Speech Translation Lab server with neural Whisper & Argos:
.venv/bin/python3 web_lab/server.py --port 8000

# (Or with standard python3 for zero-dependency benchmark reference mode):
python3 web_lab/server.py --port 8000

# 2. Open the web interface in your browser:
# http://127.0.0.1:8000/
```

To run on a custom port or host:
```bash
.venv/bin/python3 web_lab/server.py --host 0.0.0.0 --port 8080
```

---

## 4. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the interactive laboratory web interface. |
| `GET` | `/api/health` | Returns backend availability, active execution mode, model names, and dataset statistics. |
| `GET` | `/api/languages` | Returns the 3 supported languages (`en`, `de`, `fr`) and the 6-way directional routing table. |
| `POST` | `/api/translate-audio` | Accepts multipart audio (`audio` file) + optional `manual_language`. Returns transcript, detected language, 2 translations, and latency metadata. |
| `POST` | `/api/translate-text` | Accepts JSON `{"text": "...", "source_language": "..."}`. Returns original text, source language, 2 translations, and latency. |
| `GET` | `/api/benchmark-examples` | Returns the curated 36 FLORES-200 aligned groups and 18 audio speech clips. |
| `POST` | `/api/evaluate-sample` | Evaluates a sample against ground-truth references, calculating WER, CER, BLEU, and chrF. |

---

## 5. Running Automated Tests

To run the dedicated web lab test suite:
```bash
python3 -m unittest tests/test_web_lab.py
```

To run the entire project test suite:
```bash
python3 -m unittest discover tests
```

---

## 6. Execution Modes & Environment Disclosure

- **Hardware Isolation:** The existing `main/Main_Argos.py` contains Raspberry Pi GPIO hooks (`from gpiozero import Button`) and ALSA sound card locks (`aplay -L` looking for `UACDemoV10`). The `web_lab/backend_adapter.py` safely isolates these hooks so the web tester can run on development laptops (Mac/Linux/Windows) without hardware peripherals.
- **Production Neural Mode:** If `faster_whisper` and `argostranslate` are installed in the active virtual environment, the adapter executes real live neural inference on the CPU.
- **Offline Benchmark Mode:** If running in an environment without pre-downloaded weights, the adapter uses the verified FLORES-200 aligned sentences (`pilot_dataset.jsonl`) and acoustic recordings (`TM_A01.wav` to `TM_A18.wav`), clearly indicating `Benchmark Reference Pipeline (Offline Evaluation)` in the UI header.
- **Security & Privacy:** Audio recordings are processed locally in temporary files and immediately deleted after transcription. No audio or transcripts are sent to third-party cloud services.
