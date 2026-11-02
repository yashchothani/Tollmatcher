#!/usr/bin/env python3
"""
evaluate_variants.py
====================
Main evaluation driver script for TollMatcher benchmark.
Executes the full evaluation matrix comparing:
- Variant A: Fixed 1.0-second chunking
- Variant B: VAD + fixed 600ms silence endpointing
- Variant C: VAD + Local Agreement (k=2) + adaptive endpointing

Evaluates:
- ASR quality (WER, CER)
- Translation fidelity (SacreBLEU, chrF++)
- Segmentation accuracy (F1 at tau=200ms and 500ms, False Cut Rate in thinking pauses)
- Latency (Time-to-First-Audio, Total Turnaround Latency, P95 tail latency)
- Resource efficiency (Real-Time Factor, Memory RSS)
"""

import json
import os
import sys
import time

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

PILOT_JSONL = os.path.join(WORKSPACE_ROOT, "benchmark", "data", "pilot_dataset.jsonl")
SPEECH_JSONL = os.path.join(WORKSPACE_ROOT, "benchmark", "data", "speech_dataset.jsonl")
RESULTS_DIR = os.path.join(WORKSPACE_ROOT, "results")

from src.runners.variant_a import VariantARunner
from src.runners.variant_b import VariantBRunner
from src.runners.variant_c import VariantCRunner
from src.reporting.aggregate import aggregate_variant_results
from src.reporting.report_generator import generate_benchmark_reports
from src.metrics.runtime import get_hardware_info, get_peak_memory_mb


import argparse

def run_benchmark(selected_variants=None):
    print("=" * 80)
    print("TOLLMATCHER MULTILINGUAL STREAMING SPEECH TRANSLATION BENCHMARK")
    print("=" * 80)

    # 1. Load datasets
    if not os.path.isfile(PILOT_JSONL) or not os.path.isfile(SPEECH_JSONL):
        print(f"[ERROR] Required datasets missing: {PILOT_JSONL} or {SPEECH_JSONL}")
        sys.exit(1)

    with open(PILOT_JSONL, "r", encoding="utf-8") as f:
        pilot_groups = {json.loads(line)["sentence_group_id"]: json.loads(line) for line in f if line.strip()}

    with open(SPEECH_JSONL, "r", encoding="utf-8") as f:
        speech_records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(pilot_groups)} FLORES-200 aligned text groups.")
    print(f"Loaded {len(speech_records)} timestamped speech recordings.")

    available_runners = {
        "A": VariantARunner(chunk_sec=1.0),
        "B": VariantBRunner(endpoint_silence_sec=0.600),
        "C": VariantCRunner(k_agreement=2, base_silence_sec=0.400, max_pause_bridge_sec=1.000)
    }

    if not selected_variants:
        selected_variants = ["A", "B", "C"]

    runners = [available_runners[k.upper()] for k in selected_variants if k.upper() in available_runners]
    if not runners:
        print("[ERROR] No valid runners selected. Choose from: A, B, C")
        sys.exit(1)

    all_results = []
    variant_results = {r.name: [] for r in runners}

    # Target language pairs for each source speech language
    target_lang_map = {
        "en": ["de", "fr"],
        "de": ["en", "fr"],
        "fr": ["en", "de"]
    }

    start_bench_time = time.perf_counter()

    for runner in runners:
        print(f"\nExecuting {runner.name}...")
        for spk_rec in speech_records:
            ref_id = spk_rec["reference_translation_ids"][0]
            text_group = pilot_groups.get(ref_id)
            if not text_group:
                continue

            src_lang = spk_rec["transcript_language"]
            for tgt_lang in target_lang_map.get(src_lang, []):
                res = runner.run_example(spk_rec, text_group, tgt_lang)
                variant_results[runner.name].append(res)
                all_results.append(res)

    total_bench_wall_sec = time.perf_counter() - start_bench_time
    peak_rss = get_peak_memory_mb()

    # Aggregate summaries
    aggregated = []
    for runner in runners:
        agg = aggregate_variant_results(variant_results[runner.name])
        aggregated.append(agg)

    # Generate Reports
    report_files = generate_benchmark_reports(aggregated, all_results, RESULTS_DIR)

    # Print executive summary
    print("\n" + "=" * 88)
    print("BENCHMARK EXECUTION SUMMARY")
    print("=" * 88)
    print(f"Total Evaluations Completed : {len(all_results)} runs")
    print(f"Total Benchmark Wall Time   : {total_bench_wall_sec:.2f}s")
    print(f"Peak Host Process RSS       : {peak_rss:.2f} MB")
    print("\n" + "-" * 88)

    col_w = 24
    header_str = f"{'Metric':<34}" + "".join([f" | {r.name[:col_w]:<{col_w}}" for r in runners])
    print(header_str)
    print("-" * 88)

    metric_defs = [
        ("Translation chrF++ (0-100)", lambda m: f"{m['avg_chrf']:.2f}"),
        ("Translation BLEU (0-100)", lambda m: f"{m['avg_bleu']:.2f}"),
        ("ASR WER (lower is better)", lambda m: f"{m['avg_wer']:.3f}"),
        ("ASR CER (lower is better)", lambda m: f"{m['avg_cer']:.3f}"),
        ("False Cuts in Thinking Pauses", lambda m: f"{m['total_false_cuts_in_pause']}" + (" (Zero Cuts!)" if m['total_false_cuts_in_pause'] == 0 else " (Premature!)" if m['total_false_cuts_in_pause'] > 15 else "")),
        ("Segmentation F1 (tau=200ms)", lambda m: f"{m['avg_f1_200ms']:.3f}"),
        ("Segmentation F1 (tau=500ms)", lambda m: f"{m['avg_f1_500ms']:.3f}"),
        ("Time-to-First-Audio (TTFA Median)", lambda m: f"{m['ttfa_ms_median']:.1f} ms"),
        ("Turnaround Latency (TTL Median)", lambda m: f"{m['ttl_ms_median']:.1f} ms"),
        ("TTL 95th Percentile (P95)", lambda m: f"{m['ttl_ms_p95']:.1f} ms"),
        ("Real-Time Factor (RTF on RPi)", lambda m: f"{m['avg_rtf']:.3f}")
    ]

    for label, extractor in metric_defs:
        vals = [f"{extractor(m):<{col_w}}" for m in aggregated]
        print(f"{label:<34}" + "".join([f" | {v}" for v in vals]))

    print("-" * 88)
    print(f"Generated artifacts in: {RESULTS_DIR}")
    print(f" - JSON: {report_files['json_path']}")
    print(f" - CSV : {report_files['csv_path']}")
    print(f" - MD  : {report_files['md_path']}")
    print("=" * 88)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TollMatcher Streaming Speech Translation Benchmark Runner")
    parser.add_argument("--variants", nargs="+", choices=["A", "B", "C", "a", "b", "c"], default=None,
                        help="Select algorithm variants to evaluate (A=Fixed 1s, B=VAD Pause EP [Summersemester 2026], C=Adaptive Local Agreement [Proposed])")
    parser.add_argument("--compare", action="store_true",
                        help="Directly compare Previous Algorithm (B) vs Proposed Algorithm (C)")
    parser.add_argument("--proposed", action="store_true",
                        help="Run only Proposed Algorithm (Variant C)")
    parser.add_argument("--previous", action="store_true",
                        help="Run only Previous Algorithm from Summersemester 2026 (Variant B)")

    args = parser.parse_args()

    selected = None
    if args.proposed:
        selected = ["C"]
    elif args.previous:
        selected = ["B"]
    elif args.compare:
        selected = ["B", "C"]
    elif args.variants:
        selected = [v.upper() for v in args.variants]

    run_benchmark(selected)
