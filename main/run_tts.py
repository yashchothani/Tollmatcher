import subprocess
import time
import os

PIPER_PATH = "/home/tollmatcher2/offline_translator/piper_rebuild/piper/piper"
OUTPUT_DIR = "/home/tollmatcher2/offline_translator/output"

VOICE_MODELS = {
    "en": {
        "model": "en_US-john-medium.onnx",
        "config": "en_US-john-medium.onnx.json"
    },
    "de": {
        "model": "de_DE-thorsten-medium.onnx",
        "config": "de_DE-thorsten-medium.onnx.json"
    },
    "es": {
        "model": "es_ES-carlfm-x_low.onnx",
        "config": "es_ES-carlfm-x_low.onnx.json"
    },
    "fr": {
        "model": "fr_FR-tom-medium.onnx",
        "config": "fr_FR-tom-medium.onnx.json"
    },
    "uk": {
        "model": "uk_UA-lada-x_low.onnx",
        "config": "uk_UA-lada-x_low.onnx.json"
    },    
    "it": {
        "model": "it_IT-paola-medium.onnx",
        "config": "it_IT-paola-medium.onnx.json"
    },    
    "ca": {
        "model": "ca_ES-upc_ona-x_low.onnx",
        "config": "ca_ES-upc_ona-x_low.onnx.json"
    },    
    "ar": {
        "model": "ar_JO-kareem-low.onnx",
        "config": "ar_JO-kareem-low.onnx.json"
    },    
    "zh": {
        "model": "zh_CN-huayan-medium.onnx",
        "config": "zh_CN-huayan-medium.onnx.json"
    },
    "nl": {
        "model": "nl_NL-pim-medium.onnx",
        "config": "nl_NL-pim-medium.onnx.json"
    }
}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lang_choice = input("Enter language (en, de, es, fr, uk, it, ca, ar, nl, zh): ").strip().lower()
    if lang_choice not in VOICE_MODELS:
        print("Invalid language")
        return

    text = input("Enter text to synthesize: ").strip()
    if not text:
        print("Empty input")
        return

    voice = VOICE_MODELS[lang_choice]
    timestamp = int(time.time())
    output_file = os.path.join(OUTPUT_DIR, f"{lang_choice}_{timestamp}.wav")

    command = [
        PIPER_PATH,
        "--model", f"/home/tollmatcher2/offline_translator/piper_rebuild/piper/{voice['model']}",
        "--config", f"/home/tollmatcher2/offline_translator/piper_rebuild/piper/{voice['config']}",
        "--output_file", output_file,
    ]

    print("Synthesizing...")
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    process.communicate(input=text.encode("utf-8"))
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
