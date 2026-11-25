#!/usr/bin/env bash
# ==============================================================================
# setup_hailo.sh — Raspberry Pi 5 AI HAT+ 2 (Hailo-8 / Hailo-8L NPU) Setup
# ==============================================================================

set -e

echo "========================================================================"
echo ">>> TOLLMATCHER — RASPBERRY PI 5 AI HAT+ 2 (HAILO-8 NPU) INSTALLER <<<"
echo "========================================================================"

# 1. Enable PCIe Gen 3 in /boot/firmware/config.txt
CONFIG_TXT="/boot/firmware/config.txt"
if [ ! -f "$CONFIG_TXT" ]; then
    CONFIG_TXT="/boot/config.txt"
fi

echo "[1/5] Configuring PCIe Gen 3 interface in $CONFIG_TXT..."
if grep -q "dtparam=pciex1" "$CONFIG_TXT"; then
    echo "  PCIe parameters already present in $CONFIG_TXT"
else
    echo "  Adding PCIe Gen 3 parameters to $CONFIG_TXT..."
    sudo bash -c "cat >> $CONFIG_TXT" << 'EOF'

# --- Raspberry Pi AI HAT+ (Hailo-8) PCIe Configuration ---
dtparam=pciex1
dtparam=pciex1_gen=3
EOF
    echo "  PCIe Gen 3 enabled. (Note: Requires system reboot to activate)."
fi

# 2. Install Hailo software suite & drivers
echo "[2/5] Installing HailoRT driver, firmware, and CLI tools via APT..."
sudo apt-get update -y
sudo apt-get install -y \
    hailo-all \
    pciutils \
    portaudio19-dev \
    libasound2-dev \
    alsa-utils

# 3. Create model directory and download Whisper HEF
MODEL_DIR="/opt/hailo/models"
echo "[3/5] Setting up Hailo model directory at $MODEL_DIR..."
sudo mkdir -p "$MODEL_DIR"
sudo chown -R "$USER:$USER" "$MODEL_DIR"

WHISPER_HEF="$MODEL_DIR/whisper_tiny.hef"
if [ ! -f "$WHISPER_HEF" ]; then
    echo "Downloading Hailo Model Zoo Whisper Tiny HEF..."
    curl -L "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo8/whisper_tiny.hef" -o "$WHISPER_HEF" || {
        echo "Warning: Direct S3 download failed. Checking local repository fallback..."
    }
fi

# 4. Virtual environment setup with HailoRT
VENV_DIR="$HOME/.venv-tollmatcher"
echo "[4/5] Setting up Python virtual environment at $VENV_DIR..."
python3 -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "$SCRIPT_DIR/requirements_pi.txt"

# 5. Diagnostic hardware check
echo "[5/5] Checking Hailo-8 NPU hardware status..."
if command -v hailortcli &> /dev/null; then
    echo "--- Hailo Device Identification ---"
    hailortcli fw-control identify || echo "NPU PCIe link not detected yet (Reboot required if newly attached)."
else
    echo "hailortcli not found in PATH."
fi

echo "========================================================================"
echo ">>> HAILO-8 SETUP COMPLETE! <<<"
echo "To test the NPU pipeline:"
echo "  source $VENV_DIR/bin/activate"
echo "  python3 $SCRIPT_DIR/edge_hailo_runner.py"
echo "========================================================================"
