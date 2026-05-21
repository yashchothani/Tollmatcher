# -*- coding: utf-8 -*-
import os
import platform
import shutil

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    WHITE   = "\033[97m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    MAGENTA = "\033[95m"
    BLUE    = "\033[94m"
    RED     = "\033[91m"
    GRAY    = "\033[90m"

if platform.system() == "Windows":
    os.system("")

LANG_NAMES = {
    "ar": "Arabic",    "ca": "Catalan",  "de": "German",
    "en": "English",   "es": "Spanish",  "fr": "French",
    "it": "Italian",   "nl": "Dutch",    "uk": "Ukrainian",
    "zh": "Chinese",
}

LANG_FLAGS = {
    "ar": "[AR]", "ca": "[CA]", "de": "[DE]",
    "en": "[EN]", "es": "[ES]", "fr": "[FR]",
    "it": "[IT]", "nl": "[NL]", "uk": "[UA]",
    "zh": "[ZH]",
}

def _term_width():
    return shutil.get_terminal_size((60, 20)).columns

def _center(text, width):
    return str(text).center(width)

def _top(color, w):
    print(color + "+" + "-" * (w - 2) + "+" + C.RESET)

def _mid(color, w):
    print(color + "+" + "-" * (w - 2) + "+" + C.RESET)

def _bot(color, w):
    print(color + "+" + "-" * (w - 2) + "+" + C.RESET)

def _row(color, text, w):
    inner = w - 4
    print(color + "| " + str(text)[:inner].ljust(inner) + " |" + C.RESET)

def show_banner(guest_lang, user_lang):
    gname = LANG_NAMES.get(guest_lang, guest_lang)
    uname = LANG_NAMES.get(user_lang, user_lang)
    gflag = LANG_FLAGS.get(guest_lang, "")
    uflag = LANG_FLAGS.get(user_lang, "")
    w = min(_term_width(), 70)
    print()
    _top(C.CYAN, w)
    _row(C.CYAN, C.BOLD + C.WHITE + _center("TOLLMATCHER", w - 4) + C.RESET, w)
    _mid(C.CYAN, w)
    _row(C.CYAN, f"  Guest : {gflag} {gname} ({guest_lang})", w)
    _row(C.CYAN, f"  User  : {uflag} {uname} ({user_lang})", w)
    _mid(C.CYAN, w)
    keys = [
        ("BTN1", "Guest speaks --> translated to User"),
        ("BTN2", "User speaks  --> translated to Guest"),
        ("BTN3", "Cycle Guest language"),
        ("BTN4", "Cycle User language"),
    ]
    for k, desc in keys:
        _row(C.CYAN, f"  [{k}] {desc}", w)
    _bot(C.CYAN, w)
    print()

def show_language_selection(guest_lang, user_lang, changed="guest"):
    gname = LANG_NAMES.get(guest_lang, guest_lang)
    uname = LANG_NAMES.get(user_lang, user_lang)
    gflag = LANG_FLAGS.get(guest_lang, "")
    uflag = LANG_FLAGS.get(user_lang, "")
    w = min(_term_width(), 70)
    print()
    _top(C.MAGENTA, w)
    _row(C.MAGENTA, C.BOLD + C.WHITE + _center(" LANGUAGE CHANGED ", w - 4) + C.RESET, w)
    _mid(C.MAGENTA, w)
    gm = ">>> " if changed == "guest" else "    "
    um = ">>> " if changed == "user"  else "    "
    _row(C.MAGENTA, C.GREEN + gm + C.BOLD + f"GUEST : {gflag} {gname} ({guest_lang})" + C.RESET, w)
    _row(C.MAGENTA, C.BLUE  + um + C.BOLD + f"USER  : {uflag} {uname} ({user_lang})"  + C.RESET, w)
    _row(C.MAGENTA, C.YELLOW + f"  Direction:  {guest_lang} --> {user_lang}" + C.RESET, w)
    _bot(C.MAGENTA, w)
    print()

def show_speaking(role, lang):
    name  = LANG_NAMES.get(lang, lang)
    flag  = LANG_FLAGS.get(lang, "")
    color = C.GREEN if role == "GUEST" else C.BLUE
    w = min(_term_width(), 70)
    print()
    _top(color, w)
    _row(color, C.BOLD + C.WHITE + _center(f"[MIC] {role} IS SPEAKING - {flag} {name} ({lang})", w - 4) + C.RESET, w)
    _row(color, _center("Recording... (release button to stop)", w - 4), w)
    _bot(color, w)
    print()

def show_processing(role, src, tgt):
    sname = LANG_NAMES.get(src, src)
    tname = LANG_NAMES.get(tgt, tgt)
    color = C.GREEN if role == "GUEST" else C.BLUE
    w = min(_term_width(), 70)
    print()
    _top(color, w)
    _row(color, C.BOLD + C.YELLOW + _center(f"[>>] Processing {role}  ({sname} --> {tname})", w - 4) + C.RESET, w)
    _row(color, C.DIM + _center("STT --> Translation --> TTS ...", w - 4) + C.RESET, w)
    _bot(color, w)
    print()

def show_translation_result(role, src, tgt, original, translated, stt_s, tr_s, tts_s):
    sname  = LANG_NAMES.get(src, src)
    tname  = LANG_NAMES.get(tgt, tgt)
    sflag  = LANG_FLAGS.get(src, "")
    tflag  = LANG_FLAGS.get(tgt, "")
    color  = C.GREEN if role == "GUEST" else C.BLUE
    w = min(_term_width(), 70)
    inner  = w - 4
    print()
    _top(color, w)
    _row(color, C.BOLD + C.WHITE + _center(f"[OK] {role} TRANSLATION COMPLETE", w - 4) + C.RESET, w)
    _mid(color, w)
    _row(color, C.YELLOW + C.BOLD + f"  {sflag} {sname} ({src})  -->  {tflag} {tname} ({tgt})" + C.RESET, w)
    _mid(color, w)
    _row(color, C.BOLD + f"  Original  ({sname}):" + C.RESET, w)
    for chunk in _wrap(original, inner):
        _row(color, C.WHITE + "  " + chunk + C.RESET, w)
    _mid(color, w)
    _row(color, C.BOLD + f"  Translated  ({tname}):" + C.RESET, w)
    for chunk in _wrap(translated, inner):
        _row(color, C.CYAN + "  " + chunk + C.RESET, w)
    _mid(color, w)
    timing = f"  STT {stt_s:.1f}s  |  Translate {tr_s:.1f}s  |  TTS {tts_s:.1f}s  |  Total {stt_s+tr_s+tts_s:.1f}s"
    _row(color, C.GRAY + timing + C.RESET, w)
    _bot(color, w)
    print()

def show_idle(guest_lang, user_lang):
    gname = LANG_NAMES.get(guest_lang, guest_lang)
    uname = LANG_NAMES.get(user_lang, user_lang)
    gflag = LANG_FLAGS.get(guest_lang, "")
    uflag = LANG_FLAGS.get(user_lang, "")
    w = min(_term_width(), 70)
    print()
    print(C.GRAY + "-" * w + C.RESET)
    print(C.GRAY + f"  Ready  |  GUEST: {gflag} {gname} ({guest_lang})  -->  USER: {uflag} {uname} ({user_lang})" + C.RESET)
    print(C.GRAY + "  [BTN1] Guest  [BTN2] User  [BTN3] Guest-Lang  [BTN4] User-Lang" + C.RESET)
    print(C.GRAY + "-" * w + C.RESET)
    print()

def show_error(message):
    w = min(_term_width(), 70)
    print()
    _top(C.RED, w)
    _row(C.RED, C.BOLD + C.WHITE + f"  [!] {message}" + C.RESET, w)
    _bot(C.RED, w)
    print()

def show_info(message):
    print(C.GRAY + f"  [i] {message}" + C.RESET)

def _wrap(text, width):
    words   = text.split()
    lines   = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= width:
            current = (current + " " + word).strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]
