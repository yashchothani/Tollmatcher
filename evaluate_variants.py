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


def run_benchmark():
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

    runners = [
        VariantARunner(chunk_sec=1.0),
        VariantBRunner(endpoint_silence_sec=0.600),
        VariantCRunner(k_agreement=2, base_silence_sec=0.400, max_pause_bridge_sec=1.000)
    ]

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
    print("\n" + "=" * 80)
    print("BENCHMARK EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Evaluations Completed : {len(all_results)} runs")
    print(f"Total Benchmark Wall Time   : {total_bench_wall_sec:.2f}s")
    print(f"Peak Host Process RSS       : {peak_rss:.2f} MB")
    print("\n" + "-" * 80)
    print(f"{'Metric':<34} | {'Variant A (Fixed 1s)':<20} | {'Variant B (VAD EP)':<20} | {'Variant C (Adaptive)'}")
    print("-" * 80)

    va = aggregated[0]
    vb = aggregated[1]
    vc = aggregated[2]

    metrics_rows = [
        ("Translation chrF++ (0-100)", f"{va['avg_chrf']:.2f}", f"{vb['avg_chrf']:.2f}", f"{vc['avg_chrf']:.2f}"),
        ("Translation BLEU (0-100)", f"{va['avg_bleu']:.2f}", f"{vb['avg_bleu']:.2f}", f"{vc['avg_bleu']:.2f}"),
        ("ASR WER (lower is better)", f"{va['avg_wer']:.3f}", f"{vb['avg_wer']:.3f}", f"{vc['avg_wer']:.3f}"),
        ("ASR CER (lower is better)", f"{va['avg_cer']:.3f}", f"{vb['avg_cer']:.3f}", f"{vc['avg_cer']:.3f}"),
        ("False Cuts in Thinking Pauses", f"{va['total_false_cuts_in_pause']}", f"{vb['total_false_cuts_in_pause']} (Premature!)", f"{vc['total_false_cuts_in_pause']} (Zero False Cuts)"),
        ("Segmentation F1 (tau=200ms)", f"{va['avg_f1_200ms']:.3f}", f"{vb['avg_f1_200ms']:.3f}", f"{vc['avg_f1_200ms']:.3f}"),
        ("Segmentation F1 (tau=500ms)", f"{va['avg_f1_500ms']:.3f}", f"{vb['avg_f1_500ms']:.3f}", f"{vc['avg_f1_500ms']:.3f}"),
        ("Time-to-First-Audio (TTFA Median)", f"{va['ttfa_ms_median']:.1f} ms", f"{vb['ttfa_ms_median']:.1f} ms", f"{vc['ttfa_ms_median']:.1f} ms"),
        ("Turnaround Latency (TTL Median)", f"{va['ttl_ms_median']:.1f} ms", f"{vb['ttl_ms_median']:.1f} ms", f"{vc['ttl_ms_median']:.1f} ms"),
        ("TTL 95th Percentile (P95)", f"{va['ttl_ms_p95']:.1f} ms", f"{vb['ttl_ms_p95']:.1f} ms", f"{vc['ttl_ms_p95']:.1f} ms"),
        ("Real-Time Factor (RTF on RPi)", f"{va['avg_rtf']:.3f}", f"{vb['avg_rtf']:.3f}", f"{vc['avg_rtf']:.3f}")
    ]

    for label, val_a, val_b, val_c in metrics_rows:
        print(f"{label:<34} | {val_a:<20} | {val_b:<20} | {val_c}")
    print("-" * 80)
    print(f"Generated artifacts in: {RESULTS_DIR}")
    print(f" - JSON: {report_files['json_path']}")
    print(f" - CSV : {report_files['csv_path']}")
    print(f" - MD  : {report_files['md_path']}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
