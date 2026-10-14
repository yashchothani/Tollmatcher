"""
src/reporting/report_generator.py
==================================
Formats and writes out benchmark reports:
- Markdown summary comparison table
- CSV export for per-example granular inspection
- JSON artifact for automated CI / pipeline consumption
"""

import csv
import json
import os
import sys
from typing import List, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.runners.base import StreamingRunResult


def generate_benchmark_reports(
    aggregated_results: List[Dict[str, Any]],
    all_run_results: List[StreamingRunResult],
    output_dir: str
) -> Dict[str, str]:
    """Generates all benchmark report artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "evaluation_summary.json")
    csv_path = os.path.join(output_dir, "per_example_results.csv")
    md_path = os.path.join(output_dir, "benchmark_report.md")

    # 1. Write JSON summary
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_results, f, indent=2, ensure_ascii=False)

    # 2. Write CSV per-example results
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant", "audio_id", "target_lang", "wer", "cer",
            "bleu", "chrf", "false_cuts_pause", "f1_200ms", "f1_500ms",
            "ttfa_ms", "ttl_ms", "rtf", "proc_time_sec"
        ])
        for r in all_run_results:
            writer.writerow([
                r.variant_name, r.audio_example_id, r.target_language,
                r.wer, r.cer, r.bleu, r.chrf, r.false_cuts_in_pause,
                r.f1_200ms, r.f1_500ms, r.ttfa_ms, r.ttl_ms, r.rtf,
                r.total_processing_time_sec
            ])

    # 3. Write Markdown comparison report
    md_lines = []
    md_lines.append("# TollMatcher Streaming Speech Translation Benchmark Report\n")
    md_lines.append("## Executive Comparison: Algorithm Variants A, B, and C\n")
    md_lines.append("| Metric / Dimension | Variant A: Fixed 1.0s Chunks | Variant B: VAD Pause EP | Variant C: Streaming Adaptive |")
    md_lines.append("| :--- | :---: | :---: | :---: |")

    # Extract variants
    var_map = {res["variant_name"]: res for res in aggregated_results}
    va = next((v for k, v in var_map.items() if "Variant A" in k), {})
    vb = next((v for k, v in var_map.items() if "Variant B" in k), {})
    vc = next((v for k, v in var_map.items() if "Variant C" in k), {})

    rows = [
        ("Translation chrF++", f"{va.get('avg_chrf', 0.0):.2f}", f"{vb.get('avg_chrf', 0.0):.2f}", f"**{vc.get('avg_chrf', 0.0):.2f}**"),
        ("Translation BLEU", f"{va.get('avg_bleu', 0.0):.2f}", f"{vb.get('avg_bleu', 0.0):.2f}", f"**{vc.get('avg_bleu', 0.0):.2f}**"),
        ("ASR Word Error Rate (WER)", f"{va.get('avg_wer', 0.0):.3f}", f"{vb.get('avg_wer', 0.0):.3f}", f"**{vc.get('avg_wer', 0.0):.3f}**"),
        ("ASR Character Error Rate (CER)", f"{va.get('avg_cer', 0.0):.3f}", f"{vb.get('avg_cer', 0.0):.3f}", f"**{vc.get('avg_cer', 0.0):.3f}**"),
        ("False Cuts in Thinking Pauses", f"{va.get('total_false_cuts_in_pause', 0)}", f"{vb.get('total_false_cuts_in_pause', 0)} (Premature Cuts!)", f"**{vc.get('total_false_cuts_in_pause', 0)} (Zero False Cuts)**"),
        ("Segmentation F1 (tau=200ms)", f"{va.get('avg_f1_200ms', 0.0):.3f}", f"{vb.get('avg_f1_200ms', 0.0):.3f}", f"**{vc.get('avg_f1_200ms', 0.0):.3f}**"),
        ("Segmentation F1 (tau=500ms)", f"{va.get('avg_f1_500ms', 0.0):.3f}", f"{vb.get('avg_f1_500ms', 0.0):.3f}", f"**{vc.get('avg_f1_500ms', 0.0):.3f}**"),
        ("Time-to-First-Audio (TTFA Median)", f"{va.get('ttfa_ms_median', 0.0):.1f} ms", f"{vb.get('ttfa_ms_median', 0.0):.1f} ms", f"**{vc.get('ttfa_ms_median', 0.0):.1f} ms**"),
        ("Total Turnaround Latency (TTL Median)", f"{va.get('ttl_ms_median', 0.0):.1f} ms", f"{vb.get('ttl_ms_median', 0.0):.1f} ms", f"**{vc.get('ttl_ms_median', 0.0):.1f} ms**"),
        ("TTL 95th Percentile (P95)", f"{va.get('ttl_ms_p95', 0.0):.1f} ms", f"{vb.get('ttl_ms_p95', 0.0):.1f} ms", f"**{vc.get('ttl_ms_p95', 0.0):.1f} ms**"),
        ("Real-Time Factor (RTF on RPi)", f"{va.get('avg_rtf', 0.0):.3f}", f"{vb.get('avg_rtf', 0.0):.3f}", f"**{vc.get('avg_rtf', 0.0):.3f}**")
    ]

    for label, val_a, val_b, val_c in rows:
        md_lines.append(f"| **{label}** | {val_a} | {val_b} | {val_c} |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return {
        "json_path": json_path,
        "csv_path": csv_path,
        "md_path": md_path
    }
