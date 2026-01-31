# Tollmatcher — Offline Speech Translation System

Real-time bidirectional speech translation device running fully offline on Raspberry Pi. Designed for face-to-face conversations between two people speaking different languages.

---

## Features

- **Fully offline** — no internet required after setup
- **Bidirectional translation** — both users can speak and hear translations
- **Hardware buttons** — simple physical interface for language selection and recording
- **Dual microphone/speaker setup** — separate audio paths for guest and user
- **10 languages supported** — Arabic, Catalan, Chinese, Dutch, English, French, German, Italian, Spanish, Ukrainian
- **Auto-start on boot** — runs as systemd service
- **Performance optimized** — audio preprocessing, memory cleanup, translation caching

---

## Hardware Requirements

- Raspberry Pi (tested on Pi 5)
- 4 GPIO buttons:
  - **Button 1** (GPIO 27) — Guest microphone trigger
  - **Button 2** (GPIO 22) — User microphone trigger
  - **Button 3** (GPIO 20) — Source language selector
  - **Button 4** (GPIO 21) — Target language selector
- 2 USB microphones (guest: SF-558, user: EPOS PC 7 USB)
- 2 audio output devices (configurable via environment variables)

---

## Quick Start

### 1. Clone and setup

```bash
git clone <your-repo-url> ~/Tollmatcher
cd ~/Tollmatcher
chmod +x setup.sh
./setup.sh
```

This will:

- Create virtual environment at `~/tollmatcher-env`
- Install all dependencies
- Copy and enable systemd service
- Start Tollmatcher automatically

### 2. Install translation models

```bash
source ~/tollmachter-env/bin/activate

# Install Argos Translate models (example for German-English)
argospm install translate-en_de
argospm install translate-de_en
argospm install translate-en_fr
argospm install translate-fr_en
# ... install other language pairs as needed

# Verify installed models
argospm list
```

### 3. Verify service status

```bash
sudo systemctl status tollmatcher.service
```

### 4. About the start script (auto-start on boot)

`main/start_tollmatcher.sh` is meant for auto-start when the Raspberry Pi boots (e.g. called by the systemd service). It:

- Waits 5 seconds so USB/audio can initialize before the app runs
- Activates the virtual environment
- Sets audio device env vars (CONFIRM_AUDIO_DEVICE, GUEST_TTS_DEVICE, USER_TTS_DEVICE)
- Runs `Main_Argos.py` from the correct directory

Edit the paths inside the script for your system (venv path, project path). To use it for auto-start, point the systemd service at this script (see _Configuration → Systemd service_). You can also run it manually for testing:

```bash
chmod +x main/start_tollmatcher.sh
./main/start_tollmatcher.sh
```

---

## Translation Engines

The project includes two main translation scripts. The default systemd service uses `Main_Argos.py`.

### Main_Argos.py (Default)

- **Engine:** [Argos Translate](https://www.argosopentech.com/argospm/index/)
- **Languages:** 10 (ar, ca, de, en, es, fr, it, nl, uk, zh)
- **Translation strategy:** Direct when available, otherwise via English bridge
- **Features:**
  - **Single-instance lock** — only one process can run; second start exits with "Another instance is already running"
  - **USB audio guard** — waits for `sysdefault:CARD=USB` before playing any audio (avoids first sound on wrong device)
  - Translation path caching (direct/via-English)
  - Audio optimization (librosa trim + resample to 16kHz)
  - Memory cleanup and performance logging
  - Configurable audio devices via environment variables

### Main_MarianMT_7languages.py (Alternative)

- **Engine:** [Helsinki-NLP MarianMT](https://huggingface.co/models?sort=trending&search=opus-mt-de-)
- **Languages:** 7 (ca, de, en, es, fr, it, uk)
- **Translation strategy:** Direct translation for all pairs
- **Models location:** `~/.cache/huggingface/hub`
- **Note:** Requires all MarianMT models pre-downloaded

---

## Usage

### Button Controls

| Button                  | Function                                          |
| ----------------------- | ------------------------------------------------- |
| **Button 1** (hold >1s) | Record guest speech → translate → play to user    |
| **Button 2** (hold >1s) | Record user speech → translate → play to guest    |
| **Button 3** (press)    | Cycle through source languages (guest's language) |
| **Button 4** (press)    | Cycle through target languages (user's language)  |

### Audio Feedback

- Language selection plays confirmation audio from `languages/` folder
- Error conditions play `languages/error.wav`
- Default startup plays `languages/default.wav`

### Workflow

1. Power on → service auto-starts
2. Press **Button 4** to select user's language (plays confirmation)
3. Press **Button 3** to select guest's language (plays confirmation)
4. Guest holds **Button 1** and speaks → release after 1s → translation plays on user's speaker
5. User holds **Button 2** and speaks → release after 1s → translation plays on guest's speaker

---

## Configuration

### Audio Devices

Set environment variables before running (or in `main/start_tollmatcher.sh`, or in the systemd service):

```bash
export CONFIRM_AUDIO_DEVICE="sysdefault:CARD=USB"
export GUEST_TTS_DEVICE="sysdefault:CARD=USB"
export USER_TTS_DEVICE="sysdefault:CARD=Device"
```

The start script uses these defaults; Main_Argos.py waits for `sysdefault:CARD=USB` at startup before playing any audio.

### Systemd Service

The service runs when the Pi boots. Edit `/etc/systemd/system/tollmatcher.service` to:

- Use the start script for auto-start (recommended): `ExecStart=/path/to/Tollmatcher/main/start_tollmatcher.sh` so USB/audio get time to initialize and env vars are set
- Or run Main_Argos.py directly
- Change which script runs (Main_Argos.py or Main_MarianMT_7languages.py)
- Set environment variables
- Adjust restart behavior

After changes:

```bash
sudo systemctl daemon-reload
sudo systemctl restart tollmatcher.service
```

---

## Project Structure

```
Tollmatcher/
├── main/
│   ├── Main_Argos.py              # Main script (Argos Translate + faster-whisper)
│   ├── start_tollmatcher.sh       # Start script for auto-start on boot (venv + env + Main_Argos.py)
│   ├── Main_MarianMT_7languages.py # Alternative (MarianMT)
│   └── [other test/debug scripts]
├── languages/                      # Audio confirmation files (*.wav)
├── piper_rebuild/piper/            # Piper TTS engine + voice models
├── guest_mic_recordings/           # Guest audio recordings
├── user_mic_recordings/            # User audio recordings
├── output/                         # Generated TTS audio files
├── system/
│   └── tollmatcher.service          # Systemd service file
├── requirements.txt                # Python dependencies
├── setup.sh                        # Automated setup script
└── README.md
```

---

## Technology Stack

- **STT (Speech-to-Text):** [faster-whisper](https://github.com/guillaumekln/faster-whisper) (tiny/tiny.en models)
- **Translation:** [Argos Translate](https://github.com/argosopentech/argos-translate) or [MarianMT](https://huggingface.co/Helsinki-NLP)
- **TTS (Text-to-Speech):** [Piper](https://github.com/rhasspy/piper)
- **Audio:** PyAudio, librosa, soundfile
- **Hardware:** gpiozero for Raspberry Pi GPIO

---

## Performance Optimizations

- **Model warmup** on startup to reduce first-run latency
- **Audio preprocessing** (silence trimming, 16kHz resampling)
- **Translation caching** (avoid repeated path lookups)
- **Memory cleanup** (gc + psutil monitoring)
- **Optimized transcription params** (beam_size=1, no word timestamps)

---

## Troubleshooting

### Service not starting

```bash
sudo journalctl -u tollmatcher.service -f
```

### Audio device issues

List available devices:

```bash
aplay -L
arecord -L
```

### Missing translation models

```bash
source ~/tollmachter-env/bin/activate
argospm list  # Check installed models
argospm install translate-<source>_<target>
```

### Button not responding

Check GPIO connections and pull-up resistor configuration in code.

### "Another instance is already running"

Main_Argos.py uses a single-instance lock. Stop the other process (e.g. `sudo systemctl stop tollmatcher.service`) or remove the lock only if you are sure no instance is running: `rm -f /tmp/tollmatcher.lock`.

---

## Development

### Run manually (for testing)

Either use the start script (after editing paths inside it):

```bash
./main/start_tollmatcher.sh
```

Or run the main script directly:

```bash
source ~/tollmachter-env/bin/activate
cd ~/Tollmatcher/main
python Main_Argos.py
```

### Single instance

Only one instance of Main_Argos.py can run at a time (lock file: `/tmp/tollmatcher.lock`). If the service is running and you start manually (or the other way around), the second start exits with "Another instance is already running. Exiting." Stop the service before a manual run, or exit the manual run before starting the service.

### Stop the service first (when running manually)

```bash
sudo systemctl stop tollmatcher.service
```

---

## License

[Add your license here]

## Credits

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) by Guillaume Klein
- [Argos Translate](https://github.com/argosopentech/argos-translate) by Argos Open Tech
- [Piper TTS](https://github.com/rhasspy/piper) by Rhasspy
- [MarianMT](https://huggingface.co/Helsinki-NLP) by Helsinki-NLP
