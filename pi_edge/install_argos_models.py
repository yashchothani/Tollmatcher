"""
pi_edge/install_argos_models.py
================================
Installs and verifies the CTranslate2 INT8 translation packages for:
- English <-> German (en <-> de)
- English <-> French (en <-> fr)
"""

import sys
import argostranslate.package
import argostranslate.translate


def main():
    print("=" * 70)
    print("TOLLMATCHER — INSTALLING INT8 TRANSLATION MODELS (EN, DE, FR)")
    print("=" * 70)

    print("[1/3] Updating Argos package index...")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()

    target_pairs = [
        ("en", "de"),
        ("de", "en"),
        ("en", "fr"),
        ("fr", "en")
    ]

    installed = argostranslate.translate.get_installed_languages()
    installed_codes = {l.code for l in installed}
    print(f"Current installed language codes: {list(installed_codes)}")

    for src, tgt in target_pairs:
        # Check if already installed
        src_obj = next((l for l in installed if l.code == src), None)
        tgt_obj = next((l for l in installed if l.code == tgt), None)
        if src_obj and tgt_obj and src_obj.get_translation(tgt_obj):
            print(f"[OK] Package {src} -> {tgt} is already installed.")
            continue

        pkg = next((p for p in available if p.from_code == src and p.to_code == tgt), None)
        if pkg:
            print(f"[Downloading] {src.upper()} -> {tgt.upper()} package ({pkg.package_version})...")
            download_path = pkg.download()
            print(f"[Installing] {src.upper()} -> {tgt.upper()} from {download_path}...")
            argostranslate.package.install_from_path(download_path)
            print(f"[Success] Installed {src.upper()} -> {tgt.upper()}")
        else:
            print(f"[Warning] No online package found for {src} -> {tgt}")

    # Verification
    print("\n[3/3] Verifying active translation routes:")
    installed = argostranslate.translate.get_installed_languages()
    lang_map = {l.code: l for l in installed}
    for src, tgt in target_pairs:
        s = lang_map.get(src)
        t = lang_map.get(tgt)
        if s and t and s.get_translation(t):
            sample = "Hello" if src == "en" else "Hallo" if src == "de" else "Bonjour"
            tr = s.get_translation(t).translate(sample)
            print(f"  [Active] {src.upper()} -> {tgt.upper()}: '{sample}' -> '{tr}'")
        else:
            print(f"  [MISSING] {src.upper()} -> {tgt.upper()}")

    print("=" * 70)


if __name__ == "__main__":
    main()
