import subprocess
import os

PIPER_PATH = "/home/tollmatcher2/offline_translator/piper_rebuild/piper/piper"
OUTPUT_DIR = "/home/tollmatcher2/offline_translator/output"

NL_MODEL = {
    "model": "nl_NL-pim-medium.onnx",
    "config": "nl_NL-pim-medium.onnx.json"
}

sentences = {
    "zh": "De taal van uw gast is Chinees.",
    "en": "De taal van uw gast is Engels.",
    "de": "De taal van uw gast is Duits.",
    "es": "De taal van uw gast is Spaans.",
    "fr": "De taal van uw gast is Frans.",
    "uk": "De taal van uw gast is Oekraïens.",
    "it": "De taal van uw gast is Italiaans.",
    "ca": "De taal van uw gast is Catalaans.",
    "ar": "De taal van uw gast is Arabisch."
}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for lang_code, text in sentences.items():
        output_file = os.path.join(OUTPUT_DIR, f"nl_{lang_code}.wav")
        command = [
            PIPER_PATH,
            "--model", f"/home/tollmatcher2/offline_translator/piper_rebuild/piper/{NL_MODEL['model']}",
            "--config", f"/home/tollmatcher2/offline_translator/piper_rebuild/piper/{NL_MODEL['config']}",
            "--output_file", output_file,
        ]

        print(f"Synthesizing for {lang_code} → {output_file}")
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        process.communicate(input=text.encode("utf-8"))
        print(f"Saved: {output_file}")

if __name__ == "__main__":
    main()
