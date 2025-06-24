# Offline Translator

This is an **offline speech translation system** designed to run automatically at system startup using `systemd`.

## Auto Start on Boot

Enable and start the service with:

```bash
sudo systemctl enable offline_translator.service
sudo systemctl start offline_translator.service

## Main Scripts

There are two main Python files in this project:
Both scripts can run independently to perform offline speech translation.

- **`Main_MarianMT_7languages.py`**  
  Supports 7 languages with direct translation between any two supported languages.
- **`Main_Argos.py`**  
  Supports 10 languages, but most translations are done via English as an intermediate language.


