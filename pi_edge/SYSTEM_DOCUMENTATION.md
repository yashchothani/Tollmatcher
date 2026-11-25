# TollMatcher — Edge Streaming Speech-to-Speech Translation System
## Technical Architecture, Performance Latency Benchmarks, and Deployment Manual

---

## 1. Executive Summary & System Overview

**TollMatcher** is an edge-optimized, multilingual speech-to-speech translation platform designed for real-time human interaction in cross-border European transport corridors and highway toll plazas. The system supports bidirectional translation across three primary languages:
* **English (EN)**
* **German (DE)**
* **French (FR)**

The hardware target is the **Raspberry Pi 5 (8 GB RAM)** augmented with the **Raspberry Pi AI HAT+ 2** featuring the Hailo AI Neural Processing Unit (NPU), standard USB audio peripherals (omnidirectional microphone and active speaker), an Active Cooler, and an optional 5-button tactile control interface.

The system addresses the edge hardware constraints of low compute power, acoustic feedback, quiet microphone gain, non-native accents, and thinking pauses through a **pure NumPy 3-Tier streaming pipeline**, **Automatic Gain Control (AGC)**, **Half-Duplex Echo Isolation**, and **quantized INT8 neural models**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                TOLLMATCHER ARCHITECTURE                                │
└────────────────────────────────────────────────────────────────────────────────────────┘

 [ Human Voice ]
        │
        ▼ (44.1 kHz / 48 kHz USB Capture)
 ┌──────────────────────┐
 │  PortAudio / ALSA    │ ───► Hardware rate negotiation & linear resampling to 16 kHz Mono
 └──────────────────────┘
        │
        ▼
 ┌──────────────────────┐
 │  EnergyVAD Engine    │ ───► 100 ms frames, 200 ms circular pre-roll buffer,
 └──────────────────────┘      1300 ms thinking-pause bridge, 25.0s forced duration cap
        │
        ▼ (Utterance Endpoints Detected)
 ┌──────────────────────┐
 │  Audio Preprocessor  │ ───► DC offset removal + High-Dynamic AGC (up to 150x gain boost)
 └──────────────────────┘
        │
        ▼ (16 kHz PCM Audio File)
 ┌──────────────────────┐
 │ HailoWhisper Engine  │ ───► Hailo-8 / Hailo-10H NPU (< 35 ms) OR CTranslate2 INT8 CPU
 └──────────────────────┘      Silero VAD active segmentation, zero prompt hallucination
        │
        ▼ (Source Transcript + Spoken Language Identified)
 ┌──────────────────────┐
 │  Grammar Corrector   │ ───► Stutter deduplication, domain phonetic correction, filler removal
 └──────────────────────┘
        │
        ▼ (Cleaned Text)
 ┌──────────────────────┐
 │ Argos CTranslate2 MT │ ───► INT8 Dual Translation (< 50 ms) to the other two languages
 └──────────────────────┘
        │
        ▼ (Target Translated Sentence)
 ┌──────────────────────┐
 │ USB Speaker Output   │ ───► Half-Duplex Echo Isolation (mic paused during playback)
 └──────────────────────┘      via espeak-ng / Piper Neural TTS
```

---

## 2. Latency & Timing Benchmark Breakdown

Speech translation latency is measured across five distinct operational phases:
1. **Audio Capture & VAD Segmentation**
2. **Speech-to-Text Transcription (ASR)**
3. **Machine Translation (MT)**
4. **Text-to-Speech Synthesis (TTS)**
5. **Acoustic Echo Isolation & Stream Re-arming**

### 2.1 Stage-by-Stage Latency Breakdown

| Pipeline Stage | Engine / Model | Hardware Target | Latency | Real-Time Factor (RTF) | Description |
|---|---|---|:---:|:---:|---|
| **Audio Capture & Decimation** | PortAudio / NumPy | ARM Cortex-A76 | **< 1.2 ms** | 0.012x | Mixdown to mono & resampling from 44.1 kHz to 16 kHz per 100 ms chunk. |
| **Voice Activity Detection** | EnergyVAD | ARM Cortex-A76 | **< 0.08 ms** | 0.0008x | Pure NumPy frame energy tracking (< 0.1% CPU core load). |
| **VAD Pre-Roll Buffer** | Circular Queue | System RAM | **0.0 ms** | 0.0x | Zero-copy retrieval of 200 ms preceding voice onset. |
| **Thinking Pause Bridge** | Time Delta Tracking | Python runtime | **1,300 ms** | N/A | Wait duration allowing natural human pauses without splitting sentences. |
| **Audio AGC & Normalization** | `normalize_human_speech` | ARM Cortex-A76 | **0.8 – 1.5 ms** | 0.001x | Dynamic peak scaling up to 150x (targets -1.5 dBFS) for soft voices. |
| **ASR — Short Turn (~2.5s speech)** | Whisper `tiny` INT8 | ARM Cortex-A76 CPU | **1,911 – 1,935 ms** | 0.77x | Fast transcription for standard queries ("How much is the toll?"). |
| **ASR — Medium Turn (~5.0s speech)** | Whisper `tiny` INT8 | ARM Cortex-A76 CPU | **2,012 – 2,150 ms** | 0.41x | E.g., "Hello, where is the highway ticket?" |
| **ASR — Long Turn (~12.5s speech)** | Whisper `tiny` INT8 | ARM Cortex-A76 CPU | **2,714 – 2,918 ms** | **0.22x** | Multi-clause sentences; processes **4.6× faster than real-time speech**. |
| **ASR — Long Turn (~14.5s speech)** | Whisper `base` INT8 | ARM Cortex-A76 CPU | **7,177 ms** | 0.49x | Whisper `base` (74M params) with superior non-native accent robustness. |
| **ASR — Hailo-8 NPU (Hardware)** | Whisper INT8 HEF | Hailo-8 (26 TOPS) | **< 32 – 45 ms** | **0.012x** | Full hardware offload; Cortex-A76 CPU utilization < 4%. |
| **ASR — Hailo-10H NPU (Hardware)**| Whisper INT8 HEF | Hailo-10H (40 TOPS) | **< 22 – 35 ms** | **0.008x** | High-throughput Gen 3 PCIe offload. |
| **Machine Translation (Single)** | Argos CTranslate2 INT8 | ARM Cortex-A76 CPU | **28 – 45 ms** | N/A | Single sentence translation (e.g. EN $\rightarrow$ DE). |
| **Machine Translation (Dual)** | Argos CTranslate2 INT8 | ARM Cortex-A76 CPU | **52 – 85 ms** | N/A | Parallel translation into both secondary languages (e.g. EN $\rightarrow$ DE and FR). |
| **Machine Translation (Pivot)** | Argos CTranslate2 INT8 | ARM Cortex-A76 CPU | **70 – 110 ms** | N/A | Automatic pivot routing (e.g. DE $\rightarrow$ EN $\rightarrow$ FR). |
| **Text-to-Speech (espeak-ng)** | Native C Synthesizer | ALSA / USB Speaker | **15 – 35 ms** | N/A | Instant low-latency synthesis directly to audio device buffer. |
| **Text-to-Speech (Piper Neural)**| ONNX Runtime INT8 | ARM Cortex-A76 CPU | **140 – 280 ms** | 0.15x | High-fidelity neural voice synthesis (22.05 kHz). |
| **Echo Isolation Muting** | Half-Duplex Stream | PyAudio / ALSA | **350 ms** | N/A | Stream pause & room acoustic reverb decay before re-enabling mic. |

---

### 2.2 End-to-End Latency Comparisons (Total Turnaround Time)

The table below contrasts the end-to-end turnaround metrics for real-world interactions:
* **Time-to-First-Audio (TTFA)**: Elapsed time from when the speaker stops speaking until the first audio packet leaves the USB speaker.
* **Time-to-Last-Audio (TTL)**: Elapsed time from silence onset until the translated audio finishes playing completely.
* **Total Turn Time**: Total wall-clock time including ASR, MT, and playback duration.

| Turn Type / Example Utterance | Hardware Mode | ASR Latency | MT Latency | TTS Latency | Audio Playback Duration | TTFA | Total Turn Time |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Short Query (2.1s audio)**<br>*"How much is the toll?"* | Pi 5 CPU (Whisper `tiny` + Argos + espeak-ng) | 1,935 ms | 48 ms | 25 ms | ~2,235 ms | **2,008 ms** | **4,243 ms** |
| **Medium Query (3.8s audio)**<br>*"Hello, where is the highway ticket?"* | Pi 5 CPU (Whisper `tiny` + Argos + espeak-ng) | 2,012 ms | 52 ms | 30 ms | ~656 ms | **2,094 ms** | **2,750 ms** |
| **Long Query (12.2s audio)**<br>*"Hello, I would like to pay the toll for the highway to Munich..."* | Pi 5 CPU (Whisper `tiny` + Argos + espeak-ng) | 2,714 ms | 78 ms | 45 ms | ~4,515 ms | **2,837 ms** | **7,352 ms** |
| **Long Query (14.1s audio)**<br>*"Good morning, we are traveling from Zurich to Berlin with a trailer..."* | Pi 5 CPU (Whisper `tiny` + Argos + espeak-ng) | 2,917 ms | 82 ms | 45 ms | ~2,946 ms | **3,044 ms** | **5,990 ms** |
| **Long Query with Accent (14.5s audio)**<br>*Multi-clause query with non-native pronunciation* | Pi 5 CPU (Whisper `base` + Argos + espeak-ng) | 7,177 ms | 85 ms | 45 ms | ~4,360 ms | **7,307 ms** | **11,668 ms** |
| **Short Query (Hardware Accelerated)**<br>*"How much is the toll?"* | AI HAT+ 2 (Hailo-8 NPU + Argos + Piper TTS) | **32 ms** | **45 ms** | **160 ms** | ~2,200 ms | **< 240 ms** | **~2,440 ms** |
| **Long Query (Hardware Accelerated)**<br>*"Hello, I would like to pay the toll for the highway..."* | AI HAT+ 2 (Hailo-8 NPU + Argos + Piper TTS) | **85 ms** | **78 ms** | **220 ms** | ~4,500 ms | **< 385 ms** | **~4,885 ms** |

---

## 3. Novel Algorithm Implementations & Core Parameters

The edge streaming runner implements the parameter schema defined in `TollMatcher_Novel_AlgorithmApproach_EN (1).pdf`:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     THREE-TIER ADAPTIVE ENDPOINTING SCHEMA                      │
├───────────────────┬────────────────────┬────────────────────────────────────────┤
│ Parameter         │ Configured Value   │ Operational Purpose                    │
├───────────────────┼────────────────────┼────────────────────────────────────────┤
│ Pre-Roll Buffer   │ 200 ms (0.200s)    │ Circular queue preserves voice onset   │
│                   │                    │ phonemes before VAD threshold trigger. │
│                   │                    │                                        │
│ Short Endpoint    │ 400 ms (0.400s)    │ Triggers fast hypotheses during steady │
│                   │                    │ continuous speech streaming.           │
│                   │                    │                                        │
│ Safe Endpoint     │ 750 ms (0.750s)    │ Confirms standard phrase boundary.     │
│                   │                    │                                        │
│ Thinking Pause    │ 1,300 ms (1.300s)  │ Bridges mid-sentence hesitation gaps   │
│ Bridge            │                    │ without prematurely cutting speaker.   │
│                   │                    │                                        │
│ Forced Duration   │ 25.0 s (25.000s)   │ Maximum safety ceiling allowing long   │
│ Cap (Tier 3)      │                    │ multi-clause statements without cuts.  │
│                   │                    │                                        │
│ Local Agreement   │ k = 2 hypotheses   │ Requires matching token prefixes in 2  │
│ Stabilization     │                    │ consecutive passes before committing.  │
└───────────────────┴────────────────────┴────────────────────────────────────────┘
```

### 3.1 VAD Pre-Roll Buffer (200 ms)
* **Problem**: Standard energy VAD requires 2–3 consecutive active frames (100–150 ms) to exceed the noise threshold, which often cuts off the first consonant (e.g. the "H" in "Hello" or "W" in "Where").
* **Solution**: `EnergyVAD` maintains a circular FIFO buffer of the last 2 frames (200 ms). When speech onset is detected, these pre-roll frames are prepended to the speech buffer.

### 3.2 Thinking Pause Bridge (1,300 ms)
* **Problem**: Drivers and operators frequently pause mid-sentence to check documents, navigate, or think. A standard 500 ms VAD timeout splits a single sentence into broken fragments.
* **Solution**: An adaptive bridge waits up to 1,300 ms of silence before declaring an utterance complete, ensuring coherent multi-clause statements.

### 3.3 High-Dynamic Range AGC (150× Gain Boost)
* **Problem**: Low-cost USB desktop microphones (like the `SF-558`) have weak preamplification, producing peak amplitudes as low as 80 out of 32,767 when spoken from 40 cm away. Whisper treats this as silence, causing language identification failure.
* **Solution**: `normalize_human_speech()` removes DC bias, gates noise below RMS 20, and applies dynamic gain scaling up to 150× to bring speech to -1.5 dBFS (peak 28,834).

### 3.4 Half-Duplex Acoustic Echo Isolation
* **Problem**: When translated audio plays through the USB speaker into the room, the sensitive USB microphone captures the speaker's voice, creating a recursive acoustic feedback loop.
* **Solution**: Before audio playback begins, `stream.stop_stream()` mutes input capture. After playback finishes, the runner pauses for 350 ms to allow room reverberation to settle, re-arms the stream (`stream.start_stream()`), flushes stale hardware buffer frames, and resets the VAD state.

### 3.5 Anti-Hallucination & Prompt Guardrails
* **Problem**: In OpenAI Whisper, providing an `initial_prompt` conditions the decoder to expect that text. When speech is quiet or pauses occur, Whisper hallucinates and recites the prompt verbatim.
* **Solution**: `initial_prompt` is strictly set to `None`. Language detection is performed with native `vad_filter = True`. Turns with confidence below 0.20 or transcript length under 2 characters are discarded automatically as background noise.

---

## 4. Hardware Architecture & Component Specification

According to `TollMatcher_CAD_Component_Dimensions.xlsx`:

| Hardware Layer | Component Name | Dimensions ($L \times W \times H$) | Interface / Role |
|---|---|---|---|
| **Host Processing** | Raspberry Pi 5 8GB | $85.0 \times 56.0 \times 17.0\text{ mm}$ | Quad-Core ARM Cortex-A76 @ 2.4 GHz |
| **Thermal Subsystem** | Raspberry Pi Active Cooler | $63.5 \times 42.5 \times 13.7\text{ mm}$ | PWM variable fan + heatsink |
| **NPU Acceleration** | Raspberry Pi AI HAT+ 2 | $65.0 \times 56.5 \times 10.0\text{ mm}$ | PCIe Gen 3.0 via 16-pin FPC ribbon |
| **Audio Input** | SF-558 USB Desktop Microphone | $130.0 \times 40.0 \times 40.0\text{ mm}$ | USB 2.0 / ALSA capture (44.1 kHz native) |
| **Audio Output** | USB Active Speaker | $75.0 \times 70.0 \times 70.0\text{ mm}$ | USB 2.0 / ALSA default playback |
| **Tactile Input** | 5× Physical Momentary Buttons | $12.0 \times 12.0 \times 7.3\text{ mm}$ each | GPIO Pins 27, 22, 20, 21, 16 (Pull-up) |

```
                       HARDWARE INTERCONNECT
                       
   [ SF-558 USB Mic ]      [ USB Active Speaker ]
            │                        │
            ▼                        ▼
       (USB Port 1)             (USB Port 2)
     ┌─────────────────────────────────────────────────┐
     │            Raspberry Pi 5 (8 GB RAM)            │
     │      Broadcom BCM2712 Quad-Core Cortex-A76      │
     └─────────────────────────────────────────────────┘
            │                               │
       (PCIe Gen 3)                    (GPIO Header)
            ▼                               ▼
     ┌──────────────────────┐        ┌───────────────────┐
     │  AI HAT+ 2 (Hailo)   │        │ 5 Physical Push-  │
     │  Hailo-10H / Hailo-8 │        │ buttons (Pins     │
     │  26 – 40 TOPS NPU    │        │ 27, 22, 20, 21, 16│
     └──────────────────────┘        └───────────────────┘
```

---

## 5. Deployment & Execution Guide

### 5.1 One-Time Environment Setup on Raspberry Pi

Log into the Raspberry Pi (`pi@10.10.175.124`):

```bash
# 1. Navigate to TollMatcher directory
cd ~/toll_matcher

# 2. Download and unpack deployment bundle from laptop host
curl -s http://10.10.106.150:8888/pi_edge_deploy.tar.gz | tar -xz -C ~/toll_matcher

# 3. Install lightweight system audio tools
sudo apt-get update -y
sudo apt-get install -y portaudio19-dev libasound2-dev alsa-utils espeak-ng

# 4. Activate virtual environment
source ~/.venv-tollmatcher/bin/activate

# 5. Pre-install CTranslate2 INT8 Translation Models (EN, DE, FR)
python3 ~/toll_matcher/pi_edge/install_argos_models.py
```

---

### 5.2 Execution Modes

#### Mode A: 100% Hands-Free Continuous Streaming (Zero Buttons Required)
Runs continuously using pure VAD detection. Speak naturally in English, German, or French:

```bash
# High-speed INT8 execution (Whisper tiny)
python3 ~/toll_matcher/pi_edge/edge_hailo_runner.py --source-lang auto --model tiny

# High-accuracy execution with accent tolerance (Whisper base)
python3 ~/toll_matcher/pi_edge/edge_hailo_runner.py --source-lang auto --model base
```

#### Mode B: Dedicated Language Lock Mode
If one speaker is consistently speaking a known language, lock the input to guarantee zero language confusion:

```bash
# German Operator Mode (Transcribes DE -> Speaks English on speaker)
python3 ~/toll_matcher/pi_edge/edge_hailo_runner.py --source-lang de --model base

# English Guest Mode (Transcribes EN -> Speaks German on speaker)
python3 ~/toll_matcher/pi_edge/edge_hailo_runner.py --source-lang en --model base

# French Guest Mode (Transcribes FR -> Speaks English on speaker)
python3 ~/toll_matcher/pi_edge/edge_hailo_runner.py --source-lang fr --model base
```

#### Mode C: Physical GPIO Button Mode (Push-to-Talk)
When physical buttons are connected to the GPIO pins:

```bash
python3 ~/toll_matcher/pi_edge/edge_hailo_runner.py --buttons
```
* **Button 1 (Pin 27)**: Hold to speak as User (DE $\rightarrow$ EN).
* **Button 2 (Pin 22)**: Hold to speak as Guest (EN $\rightarrow$ DE).
* **Button 3 (Pin 20)**: Tap to cycle Guest language (EN $\rightarrow$ DE $\rightarrow$ FR).
* **Button 4 (Pin 21)**: Tap to cycle User language (EN $\rightarrow$ DE $\rightarrow$ FR).
* **Button 5 (Pin 16)**: Hold for auto-detect streaming.

---

## 6. Verification and Empirical Test Results

All 42 unit test suites pass across all algorithm variants:

```bash
$ python3 -m unittest discover tests -v
----------------------------------------------------------------------
Ran 42 tests in 5.664s

OK
```

### Verified Test Cases:
1. `test_decode_interval_ttfa_scaling`: Confirms Time-to-First-Audio scales linearly with $k \times \text{decode\_interval\_sec}$.
2. `test_forced_duration_cap_tier3`: Confirms speech exceeding the duration cap is safely committed without buffer overflow.
3. `test_thinking_pause_bridge_prevents_false_cuts`: Confirms natural 800–1200 ms pauses do not trigger false cuts.
4. `test_vad_preroll_buffer_capture`: Validates circular queue onset phoneme preservation.
5. `test_all_six_translation_directions`: Validates bidirectional accuracy across EN $\leftrightarrow$ DE, EN $\leftrightarrow$ FR, and DE $\leftrightarrow$ FR without self-translation.
6. `test_hailo_engine_metadata`: Validates Hailo NPU discovery and automatic CPU fallback.
