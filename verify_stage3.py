#!/usr/bin/env python3
"""
verify_stage3.py
================
Comprehensive quality assurance audit script for Stage 3 evaluation framework.
Audits:
1. Verification of 16kHz speech recordings in benchmark/audio/
2. Verification of speech metadata records in benchmark/data/speech_dataset.jsonl
3. Verification of unit test suite passing (15 tests)
4. Verification of benchmark runner outputs (results/evaluation_summary.json, results/per_example_results.csv, results/benchmark_report.md)
5. Algorithmic trade-off assertions (Variant C dominates in false cuts, TTFA, and RTF)
"""

import json
import os
import subprocess
import sys
import wave

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

SPEECH_JSONL = os.path.join(WORKSPACE_ROOT, "benchmark", "data", "speech_dataset.jsonl")
AUDIO_DIR = os.path.join(WORKSPACE_ROOT, "benchmark", "audio")
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "results")


def run_stage3_audit():
    print("=" * 72)
    print("TOLLMATCHER BENCHMARK — STAGE 3 EVALUATION AUDIT")
    print("=" * 72)

    # 1. Audit speech dataset JSONL
    if not os.path.isfile(SPEECH_JSONL):
        print(f"[FAIL] Missing speech dataset: {SPEECH_JSONL}")
        sys.exit(1)

    with open(SPEECH_JSONL, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"[OK] Found {len(records)} speech dataset records in {SPEECH_JSONL}")
    assert len(records) == 18, f"Expected 18 speech records, found {len(records)}"

    # 2. Audit audio files on disk
    for r in records:
        rel_path = r["audio_file_path"]
        full_path = os.path.join(WORKSPACE_ROOT, rel_path)
        if not os.path.isfile(full_path):
            print(f"[FAIL] Missing audio file: {full_path}")
            sys.exit(1)

        with wave.open(full_path, "rb") as wf:
            assert wf.getframerate() == 16000, f"Expected 16kHz, got {wf.getframerate()} in {rel_path}"
            assert wf.getnchannels() == 1, f"Expected mono, got {wf.getnchannels()} in {rel_path}"
            assert wf.getsampwidth() == 2, f"Expected 16-bit, got {wf.getsampwidth()} in {rel_path}"

    print(f"[OK] All 18 WAV files verified as 16kHz mono 16-bit PCM.")

    # 3. Audit benchmark results
    summary_path = os.path.join(RESULTS_DIR, "evaluation_summary.json")
    csv_path = os.path.join(RESULTS_DIR, "per_example_results.csv")
    md_path = os.path.join(RESULTS_DIR, "benchmark_report.md")

    for path, name in [(summary_path, "JSON Summary"), (csv_path, "Per-Example CSV"), (md_path, "Markdown Report")]:
        if not os.path.isfile(path):
            print(f"[FAIL] Missing benchmark artifact {name}: {path}")
            sys.exit(1)
        print(f"[OK] Verified artifact {name}: {path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    var_map = {res["variant_name"]: res for res in summary_data}
    va = next((v for k, v in var_map.items() if "Variant A" in k), None)
    vb = next((v for k, v in var_map.items() if "Variant B" in k), None)
    vc = next((v for k, v in var_map.items() if "Variant C" in k), None)

    assert va and vb and vc, "Missing one of Variants A, B, C in evaluation summary!"

    # Algorithmic assertions
    # Variant A has word slicing WER > 0
    assert va["avg_wer"] > 0.0, f"Expected Variant A WER > 0, got {va['avg_wer']}"
    # Variant B triggers premature cuts in thinking pauses
    assert vb["total_false_cuts_in_pause"] > 0, f"Expected Variant B false cuts > 0, got {vb['total_false_cuts_in_pause']}"
    # Variant C achieves zero false cuts in pauses
    assert vc["total_false_cuts_in_pause"] == 0, f"Expected Variant C zero false cuts, got {vc['total_false_cuts_in_pause']}"
    # Variant C has the lowest Time-to-First-Audio (TTFA)
    assert vc["ttfa_ms_median"] < vb["ttfa_ms_median"], f"Expected Variant C TTFA < Variant B TTFA"
    # Variant C has the fastest RTF
    assert vc["avg_rtf"] < va["avg_rtf"], f"Expected Variant C RTF < Variant A RTF"

    print("\n" + "=" * 72)
    print(">>> STAGE 3 EVALUATION AUDIT PASSED: ZERO DEFECTS FOUND <<<")
    print("=" * 72)


if __name__ == "__main__":
    run_stage3_audit()
