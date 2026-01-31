#!/bin/bash

set -e

echo ">>> Creating virtual environment at ~/tollmatcher-env..."
python3 -m venv ~/tollmatcher-env
source ~/tollmatcher-env/bin/activate

echo ">>> Installing dependencies from requirements.txt..."
pip install -r "$(dirname "$0")/requirements.txt"

echo ">>> Copying systemd service file..."
sudo cp "$(dirname "$0")/system/tollmatcher.service" /etc/systemd/system/tollmatcher.service

echo ">>> Reloading systemd..."
sudo systemctl daemon-reexec
sudo systemctl daemon-reload

echo ">>> Enabling and starting the service..."
sudo systemctl enable tollmatcher.service
sudo systemctl start tollmatcher.service

echo "Setup complete. The service is now running."
