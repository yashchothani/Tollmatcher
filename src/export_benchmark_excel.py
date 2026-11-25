#!/usr/bin/env python3
"""
src/export_benchmark_excel.py
=============================
Generates a comprehensive, professionally styled Excel workbook (.xlsx) and
complementary CSV files for the TollMatcher multilingual benchmark datasets:
1. "Aligned Sentences (3-Lang)": Multi-way parallel ground truth (EN, DE, FR) for 36 sentence groups.
2. "Directional Pairs (216)": All 6 translation directions (EN<->DE, EN<->FR, DE<->FR) for direct evaluation.
3. "Speech Audio Benchmark": 16kHz speech recordings metadata with timestamps & pause annotations.
4. "Summary & Manifest": High-level distribution, schema specifications, and licensing.
"""

import json
import os
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOT_JSONL = os.path.join(WORKSPACE_ROOT, "benchmark", "data", "pilot_dataset.jsonl")
MANIFEST_JSON = os.path.join(WORKSPACE_ROOT, "benchmark", "data", "pilot_manifest.json")
SPEECH_JSONL = os.path.join(WORKSPACE_ROOT, "benchmark", "data", "speech_dataset.jsonl")

EXCEL_OUTPUT = os.path.join(WORKSPACE_ROOT, "benchmark", "data", "TollMatcher_Benchmark_Datasets.xlsx")
CSV_ALIGNED_OUTPUT = os.path.join(WORKSPACE_ROOT, "benchmark", "data", "aligned_sentences_benchmark.csv")
CSV_DIRECTIONAL_OUTPUT = os.path.join(WORKSPACE_ROOT, "benchmark", "data", "directional_pairs_benchmark.csv")


def create_benchmark_excel():
    print(f"Loading benchmark datasets from {PILOT_JSONL}...")
    with open(PILOT_JSONL, "r", encoding="utf-8") as f:
        pilot_records = [json.loads(line) for line in f if line.strip()]

    speech_records = []
    if os.path.isfile(SPEECH_JSONL):
        with open(SPEECH_JSONL, "r", encoding="utf-8") as f:
            speech_records = [json.loads(line) for line in f if line.strip()]

    manifest_data = {}
    if os.path.isfile(MANIFEST_JSON):
        with open(MANIFEST_JSON, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    # Styles
    navy_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    dark_teal_fill = PatternFill(start_color="165B33", end_color="165B33", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    bold_font = Font(name="Calibri", size=11, bold=True)
    regular_font = Font(name="Calibri", size=10)
    thin_border_side = Side(border_style="thin", color="D9D9D9")
    cell_border = Border(left=thin_border_side, right=thin_border_side, top=thin_border_side, bottom=thin_border_side)
    alt_row_fill = PatternFill(start_color="F9FAFB", end_color="F9FAFB", fill_type="solid")

    # -------------------------------------------------------------
    # Sheet 1: Aligned Sentences (3-Lang Parallel Ground Truth)
    # -------------------------------------------------------------
    ws1 = wb.create_sheet(title="Aligned Sentences (3-Lang)")
    ws1.views.sheetView[0].showGridLines = True

    headers_ws1 = [
        "Group ID", "Split", "FLORES ID", "Length", "Difficulty", "Domain", "Topic",
        "English Ground Truth (eng_Latn)", "German Ground Truth (deu_Latn)", "French Ground Truth (fra_Latn)",
        "Supported Directions", "Linguistic & Challenge Notes", "Reference Source URL"
    ]
    ws1.append(headers_ws1)

    for cell in ws1[1]:
        cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws1.row_dimensions[1].height = 28

    for row_idx, r in enumerate(pilot_records, start=2):
        row_vals = [
            r["sentence_group_id"],
            r["original_split"],
            r["original_source_id"],
            r["sentence_length_category"].capitalize(),
            r["linguistic_difficulty_category"].replace("_", " ").capitalize(),
            r.get("domain", ""),
            r.get("topic", ""),
            r["text_en"],
            r["text_de"],
            r["text_fr"],
            ", ".join(r["supported_directions"]),
            r.get("notes", ""),
            r.get("url", "")
        ]
        ws1.append(row_vals)
        row = ws1[row_idx]
        is_alt = (row_idx % 2 == 1)
        for c_idx, cell in enumerate(row):
            cell.font = regular_font
            cell.border = cell_border
            if is_alt:
                cell.fill = alt_row_fill
            # Alignments
            if c_idx in [0, 1, 2, 3, 4]:
                cell.alignment = Alignment(horizontal="center", vertical="top")
            elif c_idx in [7, 8, 9, 11]:
                cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
            else:
                cell.alignment = Alignment(horizontal="left", vertical="top")
        ws1.row_dimensions[row_idx].height = 42

    ws1.freeze_panes = "A2"

    col_widths_ws1 = {
        "A": 12, "B": 10, "C": 12, "D": 12, "E": 14, "F": 14, "G": 16,
        "H": 40, "I": 40, "J": 40, "K": 26, "L": 35, "M": 28
    }
    for col, width in col_widths_ws1.items():
        ws1.column_dimensions[col].width = width

    # -------------------------------------------------------------
    # Sheet 2: Directional Pairs (216 Directional Ground-Truth Pairs)
    # -------------------------------------------------------------
    ws2 = wb.create_sheet(title="Directional Pairs (216)")
    ws2.views.sheetView[0].showGridLines = True

    headers_ws2 = [
        "Pair ID", "Group ID", "Direction", "Source Lang", "Target Lang",
        "Source Sentence (Input)", "Target Reference (Ground Truth)",
        "Length", "Difficulty", "Topic", "Challenge Notes"
    ]
    ws2.append(headers_ws2)

    for cell in ws2[1]:
        cell.fill = navy_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws2.row_dimensions[1].height = 28

    lang_map = {
        "en": ("text_en", "English"),
        "de": ("text_de", "German"),
        "fr": ("text_fr", "French")
    }
    directions = [
        ("EN->DE", "en", "de"),
        ("DE->EN", "de", "en"),
        ("EN->FR", "en", "fr"),
        ("FR->EN", "fr", "en"),
        ("DE->FR", "de", "fr"),
        ("FR->DE", "fr", "de"),
    ]

    pair_counter = 1
    directional_rows_for_csv = []

    for r in pilot_records:
        gid = r["sentence_group_id"]
        lcat = r["sentence_length_category"].capitalize()
        dcat = r["linguistic_difficulty_category"].replace("_", " ").capitalize()
        topic = f"{r.get('domain', '')} / {r.get('topic', '')}"
        notes = r.get("notes", "")

        for dir_code, src_l, tgt_l in directions:
            pair_id = f"TM_P{pair_counter:03d}"
            pair_counter += 1
            src_text = r[lang_map[src_l][0]]
            tgt_text = r[lang_map[tgt_l][0]]

            row_vals = [
                pair_id, gid, dir_code, lang_map[src_l][1], lang_map[tgt_l][1],
                src_text, tgt_text, lcat, dcat, topic, notes
            ]
            ws2.append(row_vals)
            directional_rows_for_csv.append(row_vals)

            cur_row = ws2[pair_counter]  # pair_counter is current row index because starts at 1 and incremented
            is_alt = (pair_counter % 2 == 1)
            for c_idx, cell in enumerate(cur_row):
                cell.font = regular_font
                cell.border = cell_border
                if is_alt:
                    cell.fill = alt_row_fill
                if c_idx in [0, 1, 2, 3, 4, 7, 8]:
                    cell.alignment = Alignment(horizontal="center", vertical="top")
                elif c_idx in [5, 6, 10]:
                    cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="top")
            ws2.row_dimensions[pair_counter].height = 36

    ws2.freeze_panes = "A2"
    col_widths_ws2 = {
        "A": 12, "B": 12, "C": 14, "D": 14, "E": 14,
        "F": 45, "G": 45, "H": 12, "I": 14, "J": 22, "K": 35
    }
    for col, width in col_widths_ws2.items():
        ws2.column_dimensions[col].width = width

    # -------------------------------------------------------------
    # Sheet 3: Speech Audio Benchmark (30 recordings)
    # -------------------------------------------------------------
    ws3 = wb.create_sheet(title="Speech Audio Benchmark (30)")
    ws3.views.sheetView[0].showGridLines = True

    headers_ws3 = [
        "Audio Example ID", "Linked Group ID", "Language", "Speaker ID", "Audio File Path",
        "Duration (s)", "Speech Start (s)", "Speech End (s)", "Thinking Pause (s)",
        "Exact Speech Transcript", "Status / Provenance"
    ]
    ws3.append(headers_ws3)

    for cell in ws3[1]:
        cell.fill = dark_teal_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws3.row_dimensions[1].height = 28

    for row_idx, r in enumerate(speech_records, start=2):
        pauses = r.get("internal_pause_annotations", [])
        pause_str = ""
        if pauses:
            pause_str = ", ".join([f"{p['pause_start_sec']}s-{p['pause_end_sec']}s ({p.get('duration_sec', 0):.2f}s)" for p in pauses])
        else:
            pause_str = "None"

        row_vals = [
            r.get("audio_example_id", ""),
            ", ".join(r.get("reference_translation_ids", [])),
            r.get("transcript_language", "").upper(),
            r.get("speaker_id", ""),
            r.get("audio_file_path", ""),
            round(r.get("duration_seconds", 0.0), 3),
            round(r.get("speech_start_time", 0.0), 3),
            round(r.get("speech_end_time", 0.0), 3),
            pause_str,
            r.get("transcript", ""),
            r.get("annotation_status", "")
        ]
        ws3.append(row_vals)
        row = ws3[row_idx]
        is_alt = (row_idx % 2 == 1)
        for c_idx, cell in enumerate(row):
            cell.font = regular_font
            cell.border = cell_border
            if is_alt:
                cell.fill = alt_row_fill
            if c_idx in [0, 1, 2, 3, 5, 6, 7]:
                cell.alignment = Alignment(horizontal="center", vertical="top")
            elif c_idx in [9]:
                cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
            else:
                cell.alignment = Alignment(horizontal="left", vertical="top")
        ws3.row_dimensions[row_idx].height = 32

    ws3.freeze_panes = "A2"
    col_widths_ws3 = {
        "A": 18, "B": 16, "C": 12, "D": 20, "E": 32,
        "F": 14, "G": 16, "H": 16, "I": 24, "J": 45, "K": 26
    }
    for col, width in col_widths_ws3.items():
        ws3.column_dimensions[col].width = width

    # -------------------------------------------------------------
    # Sheet 4: Summary & Specifications
    # -------------------------------------------------------------
    ws4 = wb.create_sheet(title="Summary & Specifications")
    ws4.views.sheetView[0].showGridLines = True

    ws4.column_dimensions["A"].width = 28
    ws4.column_dimensions["B"].width = 50

    title_cell = ws4["A1"]
    title_cell.value = "TollMatcher Multilingual Benchmark Specification"
    title_cell.font = Font(name="Calibri", size=14, bold=True, color="1F4E79")
    ws4.row_dimensions[1].height = 30

    specs = [
        ("Benchmark Dataset Name", manifest_data.get("dataset_name", "TollMatcher FLORES-200 Pilot Benchmark")),
        ("Dataset Version", manifest_data.get("version", "1.0.0")),
        ("Target Core Languages", "3 Languages: English (eng_Latn), German (deu_Latn), French (fra_Latn)"),
        ("Ground-Truth Type", "Multi-way Bi-directional Aligned Parallel Text (FLORES-200)"),
        ("Total Sentence Groups", f"{len(pilot_records)} unique multi-way parallel groups"),
        ("Total Directional Pairs", f"{len(directional_rows_for_csv)} bidirectional translation pairs (6 directions)"),
        ("Supported Directions", "EN->DE, DE->EN, EN->FR, FR->EN, DE->FR, FR->DE"),
        ("Total Audio Benchmark Files", f"{len(speech_records)} WAV recordings (16kHz mono 16-bit PCM)"),
        ("Split Distribution", "18 'dev' split, 18 'devtest' split"),
        ("Sentence Length Distribution", "12 Short (<12 words), 12 Medium (12-22 words), 12 Long (>22 words)"),
        ("Linguistic Difficulty Distribution", "9 Easy, 10 Medium, 8 Hard, 9 Stress-test"),
        ("Domains / Topics", "Wikinews (Accident, Health, Tech, Sports, Politics) & Wikivoyage (Travel, Tourism)"),
        ("Licensing & Attribution", "CC-BY-SA-4.0 (FLORES-200 / Meta NLLB Team, 2022)"),
        ("Hardware Target", "Raspberry Pi 5 (ARM Cortex-A76), Dual USB Audio Devices"),
        ("Primary Evaluation Metrics", "Translation: BLEU, chrF++ | ASR: WER, CER | Latency: TTFA, TTL, RTF")
    ]

    for idx, (param, val) in enumerate(specs, start=3):
        c_param = ws4.cell(row=idx, column=1, value=param)
        c_val = ws4.cell(row=idx, column=2, value=val)
        c_param.font = bold_font
        c_param.border = cell_border
        c_param.fill = PatternFill(start_color="F2F4F7", end_color="F2F4F7", fill_type="solid")
        c_val.font = regular_font
        c_val.border = cell_border
        c_val.alignment = Alignment(wrap_text=True)
        ws4.row_dimensions[idx].height = 24

    # Save Excel
    os.makedirs(os.path.dirname(EXCEL_OUTPUT), exist_ok=True)
    wb.save(EXCEL_OUTPUT)
    print(f"[SUCCESS] Saved Excel benchmark workbook to {EXCEL_OUTPUT}")

    # Also save clean CSVs for easy terminal/pandas/data pipelines
    import csv
    with open(CSV_ALIGNED_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers_ws1)
        for r in pilot_records:
            writer.writerow([
                r["sentence_group_id"], r["original_split"], r["original_source_id"],
                r["sentence_length_category"].capitalize(),
                r["linguistic_difficulty_category"].replace("_", " ").capitalize(),
                r.get("domain", ""), r.get("topic", ""),
                r["text_en"], r["text_de"], r["text_fr"],
                ", ".join(r["supported_directions"]),
                r.get("notes", ""), r.get("url", "")
            ])
    print(f"[SUCCESS] Saved Aligned CSV to {CSV_ALIGNED_OUTPUT}")

    with open(CSV_DIRECTIONAL_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers_ws2)
        for row in directional_rows_for_csv:
            writer.writerow(row)
    print(f"[SUCCESS] Saved Directional Pairs CSV to {CSV_DIRECTIONAL_OUTPUT}")


if __name__ == "__main__":
    create_benchmark_excel()
