#!/usr/bin/env python3
"""
src/generate_inspection_table.py
================================
Generates a human-readable markdown table of all 36 curated sentence groups
in the pilot dataset for inspection and documentation.
"""

import json
import os

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSONL_PATH = os.path.join(WORKSPACE_ROOT, "benchmark", "data", "pilot_dataset.jsonl")
DOC_PATH = os.path.join(WORKSPACE_ROOT, "benchmark", "docs", "pilot_dataset_inspection.md")


def generate_table():
    os.makedirs(os.path.dirname(DOC_PATH), exist_ok=True)
    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    lines = []
    lines.append("# TollMatcher Benchmark — Stage 2 Pilot Dataset Inspection Table\n")
    lines.append("This document provides a human-readable audit table for the 36 curated, multi-way aligned sentence groups from **FLORES-200**.\n")
    lines.append("Each group expresses one underlying meaning across English, German, and French, supporting evaluation in all six translation directions:\n")
    lines.append("- $\\text{EN} \\rightarrow \\text{DE}$, $\\text{DE} \\rightarrow \\text{EN}$")
    lines.append("- $\\text{EN} \\rightarrow \\text{FR}$, $\\text{FR} \\rightarrow \\text{EN}$")
    lines.append("- $\\text{DE} \\rightarrow \\text{FR}$, $\\text{FR} \\rightarrow \\text{DE}$\n")
    lines.append("| ID | Split | Source ID | Length | Difficulty | Domain / Topic | English (`eng_Latn`) | German (`deu_Latn`) | French (`fra_Latn`) |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :--- | :--- | :--- | :--- |")

    for r in records:
        gid = r["sentence_group_id"]
        split = r["original_split"]
        src = r["original_source_id"]
        lcat = r["sentence_length_category"].capitalize()
        dcat = r["linguistic_difficulty_category"].replace("_", " ").capitalize()
        dom_top = f"{r.get('domain', '')} / {r.get('topic', '')}"
        en = r["text_en"].replace("|", "\\|")
        de = r["text_de"].replace("|", "\\|")
        fr = r["text_fr"].replace("|", "\\|")
        lines.append(f"| **{gid}** | `{split}` | `{src}` | {lcat} | {dcat} | {dom_top} | {en} | {de} | {fr} |")

    with open(DOC_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Generated inspection markdown at {DOC_PATH} ({len(records)} entries)")


if __name__ == "__main__":
    generate_table()
