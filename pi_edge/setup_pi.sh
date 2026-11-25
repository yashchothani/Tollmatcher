#!/usr/bin/env bash
# ==============================================================================
# setup_pi.sh — Automated Setup Script for TollMatcher on Raspberry Pi 5
# ==============================================================================

set -e

echo "========================================================================"
echo ">>> TOLLMATCHER RASPBERRY PI 5 EDGE ENVIRONMENT SETUP <<<"
echo "========================================================================"

# 1. System package dependencies
echo "[1/4] Installing required system libraries (ALSA, PortAudio, GCC)..."
sudo apt-get update -y
sudo apt-get install -y \
    python3-dev \
    python3-venv \
    portaudio19-dev \
    libasound2-dev \
    libatlas-base-dev \
    alsa-utils \
    espeak-ng \
    sox \
    libsox-fmt-all

# 2. Virtual environment setup
VENV_DIR="$HOME/.venv-tollmatcher"
echo "[2/4] Setting up isolated Python virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# 3. Install lightweight edge dependencies
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[3/4] Installing lightweight edge dependencies from $SCRIPT_DIR/requirements_pi.txt..."
pip install -r "$SCRIPT_DIR/requirements_pi.txt"

# 4. Pre-download translation models
echo "[4/4] Pre-downloading INT8 Argos translation models (EN, DE, FR)..."
python3 -c "
import argostranslate.package
argostranslate.package.update_package_index()
available = argostranslate.package.get_available_packages()
pairs = [('en', 'de'), ('de', 'en'), ('en', 'fr'), ('fr', 'en')]
for src, tgt in pairs:
    pkg = next((p for p in available if p.from_code == src and p.to_code == tgt), None)
    if pkg:
        print(f'Installing {src}->{tgt}...')
        download_path = pkg.download()
        argostranslate.package.install_from_path(download_path)
print('Argos INT8 packages installed successfully!')
"

echo "========================================================================"
echo ">>> SETUP COMPLETED SUCCESSFULLY! <<<"
echo "To run the standalone edge engine on Pi:"
echo "  source $VENV_DIR/bin/activate"
echo "  python3 $SCRIPT_DIR/edge_streaming_runner.py"
echo "========================================================================"
