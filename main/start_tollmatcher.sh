#!/bin/bash

# Give USB/audio time to initialize
sleep 5

# Activate virtual environment (CONFIRMED PATH)
source /home/tollmatcher2/tollmachter-env/bin/activate

# Audio devices
export CONFIRM_AUDIO_DEVICE="sysdefault:CARD=USB"
export GUEST_TTS_DEVICE="sysdefault:CARD=USB"
export USER_TTS_DEVICE="sysdefault:CARD=Device"
# Go to correct directory
cd /home/tollmatcher2/offline_translator/main

# Start application
exec python3 -u Main_Argos.py
