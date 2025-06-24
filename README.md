# Offline Translator

This is an **offline speech translation system** that runs automatically at system startup using `systemd`.

##  Auto Start on Boot

Enable and start the service with:

```bash
sudo systemctl enable offline_translator.service
sudo systemctl start offline_translator.service
```

##  Main Scripts

This project provides two main Python scripts. Each script supports offline speech translation and can run independently.

### 1. `Main_MarianMT_7languages.py`

- Uses **Helsinki-NLP MarianMT** models from Hugging Face.( https://huggingface.co/models?sort=trending&search=opus-mt-de- )
- Supports **7 languages**: ( models saved in /home/tollmatcher2/.cache/huggingface/hub )
  - English (`en`)
  - German (`de`)
  - French (`fr`)
  - Spanish (`es`)
  - Italian (`it`)
  - Ukrainian (`uk`)
  - Catalan (`ca`)
- Provides **direct translation** between any two supported languages (no intermediate step).
- All models are pre-downloaded and fully offline.

### 2. `Main_Argos.py`

- Uses **Argos Translate** engine. ( https://www.argosopentech.com/argospm/index/ )
- Supports **10 languages**, including some not available in MarianMT.
- Most translation pairs are handled **via English** as an intermediate language.
  - Example: `de → en → fr`
- Works fully offline once models are installed with:

```bash
argospm install translate-en_de
argospm install translate-de_en
argospm install translate-en_fr
argospm install translate-fr_en
```
  
- Installed models could be check with:

```bash
argospm list
```

##  Usage Summary

- Press and hold button 0 more than 3 seconds to shut down, more than 5 seconds to restart.
- Use hardware buttons to trigger speech recording (Whisper) , translation (MarianMT or Argos), and TTS playback (Piper: https://github.com/rhasspy/piper/blob/master/VOICES.md ).
- Select guest and user's languages via button 3 and 4.
- Record using button 1 (guest) or button 2 (user) via their corresponding micphones.
- Translated speech is spoken aloud via their corresponding speakers.
