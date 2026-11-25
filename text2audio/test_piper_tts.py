import subprocess
import os
import simpleaudio as sa

base_dir = "/home/tollmatcher2/offline_translator/text2audio/piper"
if not os.path.isdir(base_dir):
    base_dir = os.path.join(os.path.dirname(__file__), "piper")

model_path = os.path.join(base_dir, "en_US-kristin-medium.onnx")
config_path = os.path.join(base_dir, "en_US-kristin-medium.onnx.json")
piper_bin = os.path.join(base_dir, "piper", "piper")
output_path = "speech.wav"

while True:
    text = input("Enter text: ")
    if text.lower() == "bye":
        break

    result = subprocess.run([
        piper_bin,        
        "--model", model_path,
        "--config", config_path,
        "--output_file", output_path,
        "--text", text
    ])

    if os.path.exists(output_path):
        wave_obj = sa.WaveObject.from_wave_file(output_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    else:
        print("Failed to generate audio. Check if model and config are valid.")

