"""
src/tuning/grid_search.py
==========================
Parameter tuning and grid search engine for TollMatcher streaming ST algorithms.
Evaluates parameter configurations across benchmark speech clips and computes:
- Time-To-First-Audio (TTFA) latency
- Total Turnaround Latency (TTL)
- False Cut Rate in thinking pauses
- Translation quality (chrF++, SacreBLEU)
- ASR quality (WER, CER)
- Edge simulated RTF on Raspberry Pi 5
- Pareto-optimal parameter set identification
"""

import csv
import json
import os
import sys
import time
from typing import List, Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.runners.variant_c import VariantCRunner
from src.reporting.aggregate import aggregate_variant_results
from src.tuning.parameter_space import (
    StreamingHyperparameters,
    PRESETS,
    get_preset,
    get_curated_grid,
    get_k_sweep_grid,
    get_decode_interval_sweep_grid,
    get_pause_bridge_sweep_grid,
    get_endpoint_silence_sweep_grid
)

PILOT_JSONL = os.path.join(PROJECT_ROOT, "benchmark", "data", "pilot_dataset.jsonl")
SPEECH_JSONL = os.path.join(PROJECT_ROOT, "benchmark", "data", "speech_dataset.jsonl")
TUNING_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "tuning")


class StreamingParameterTuner:
    """Orchestrates parameter sweeps and Pareto frontier identification."""

    def __init__(self, speech_records: Optional[List[Dict[str, Any]]] = None, pilot_groups: Optional[Dict[str, Any]] = None):
        if speech_records is None:
            with open(SPEECH_JSONL, "r", encoding="utf-8") as f:
                self.speech_records = [json.loads(line) for line in f if line.strip()]
        else:
            self.speech_records = speech_records

        if pilot_groups is None:
            with open(PILOT_JSONL, "r", encoding="utf-8") as f:
                self.pilot_groups = {json.loads(line)["sentence_group_id"]: json.loads(line) for line in f if line.strip()}
        else:
            self.pilot_groups = pilot_groups

        self.target_lang_map = {
            "en": ["de", "fr"],
            "de": ["en", "fr"],
            "fr": ["en", "de"]
        }

    def evaluate_config(self, cfg: StreamingHyperparameters) -> Dict[str, Any]:
        """Runs evaluation for a single hyperparameter configuration."""
        runner = VariantCRunner(
            k_agreement=cfg.k_agreement,
            decode_interval_sec=cfg.decode_interval_sec,
            short_endpoint_sec=cfg.short_endpoint_sec,
            safe_endpoint_sec=cfg.safe_endpoint_sec,
            max_pause_bridge_sec=cfg.max_pause_bridge_sec,
            forced_endpoint_sec=cfg.forced_endpoint_sec,
            energy_threshold_multiplier=cfg.energy_threshold_multiplier,
            preroll_sec=cfg.preroll_sec,
            overlap_context_sec=cfg.overlap_context_sec,
            name=cfg.name
        )

        results = []
        for spk_rec in self.speech_records:
            ref_id = spk_rec["reference_translation_ids"][0]
            text_group = self.pilot_groups.get(ref_id)
            if not text_group:
                continue

            src_lang = spk_rec["transcript_language"]
            for tgt_lang in self.target_lang_map.get(src_lang, []):
                res = runner.run_example(spk_rec, text_group, tgt_lang)
                results.append(res)

        agg = aggregate_variant_results(results)
        agg["hyperparameters"] = cfg.to_dict()
        return agg

    def evaluate_grid(self, configs: List[StreamingHyperparameters]) -> List[Dict[str, Any]]:
        """Evaluates a list of hyperparameter configurations and computes Pareto status."""
        eval_summaries = []
        for i, cfg in enumerate(configs, start=1):
            summary = self.evaluate_config(cfg)
            eval_summaries.append(summary)

        # Compute Pareto optimality
        self._tag_pareto_front(eval_summaries)
        return eval_summaries

    @staticmethod
    def _tag_pareto_front(summaries: List[Dict[str, Any]]):
        """
        Marks is_pareto_optimal=True for configurations not dominated in:
        - False cuts in thinking pause (minimize)
        - TTFA median (minimize)
        - TTL median (minimize)
        - chrF++ (maximize)
        - RTF (minimize)
        """
        for i, a in enumerate(summaries):
            is_dominated = False
            for j, b in enumerate(summaries):
                if i == j:
                    continue

                # Does B dominate A?
                # B is at least as good in all objectives:
                b_fc_ok = (b["total_false_cuts_in_pause"] <= a["total_false_cuts_in_pause"])
                b_ttfa_ok = (b["ttfa_ms_median"] <= a["ttfa_ms_median"])
                b_ttl_ok = (b["ttl_ms_median"] <= a["ttl_ms_median"])
                b_chrf_ok = (b["avg_chrf"] >= a["avg_chrf"])
                b_rtf_ok = (b["avg_rtf"] <= a["avg_rtf"])

                # And strictly better in at least one:
                b_strictly_better = (
                    (b["total_false_cuts_in_pause"] < a["total_false_cuts_in_pause"]) or
                    (b["ttfa_ms_median"] < a["ttfa_ms_median"]) or
                    (b["ttl_ms_median"] < a["ttl_ms_median"]) or
                    (b["avg_chrf"] > a["avg_chrf"]) or
                    (b["avg_rtf"] < a["avg_rtf"])
                )

                if b_fc_ok and b_ttfa_ok and b_ttl_ok and b_chrf_ok and b_rtf_ok and b_strictly_better:
                    is_dominated = True
                    break

            a["is_pareto_optimal"] = not is_dominated

    def export_reports(self, summaries: List[Dict[str, Any]], output_dir: str = TUNING_RESULTS_DIR) -> Dict[str, str]:
        """Saves tuning evaluation to JSON, CSV, and Markdown files."""
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "parameter_tuning_report.json")
        csv_path = os.path.join(output_dir, "parameter_tuning_report.csv")
        md_path = os.path.join(output_dir, "parameter_tuning_report.md")

        # 1. JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)

        # 2. CSV
        fieldnames = [
            "config_name", "k_agreement", "decode_interval_sec", "short_endpoint_sec",
            "safe_endpoint_sec", "max_pause_bridge_sec", "total_false_cuts_in_pause",
            "ttfa_ms_median", "ttl_ms_median", "ttl_ms_p95", "avg_chrf", "avg_bleu",
            "avg_wer", "avg_rtf", "is_pareto_optimal"
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in summaries:
                h = s["hyperparameters"]
                writer.writerow({
                    "config_name": h["name"],
                    "k_agreement": h["k_agreement"],
                    "decode_interval_sec": h["decode_interval_sec"],
                    "short_endpoint_sec": h["short_endpoint_sec"],
                    "safe_endpoint_sec": h["safe_endpoint_sec"],
                    "max_pause_bridge_sec": h["max_pause_bridge_sec"],
                    "total_false_cuts_in_pause": s["total_false_cuts_in_pause"],
                    "ttfa_ms_median": s["ttfa_ms_median"],
                    "ttl_ms_median": s["ttl_ms_median"],
                    "ttl_ms_p95": s["ttl_ms_p95"],
                    "avg_chrf": s["avg_chrf"],
                    "avg_bleu": s["avg_bleu"],
                    "avg_wer": s["avg_wer"],
                    "avg_rtf": s["avg_rtf"],
                    "is_pareto_optimal": s.get("is_pareto_optimal", False)
                })

        # 3. Markdown
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# TollMatcher Streaming ST Parameter Tuning Report\n\n")
            f.write(f"Evaluated **{len(summaries)}** streaming parameter configurations across {len(self.speech_records)} speech recordings.\n\n")
            f.write("## Parameter Comparison & Pareto Optimization\n\n")
            f.write("| Configuration | k | Decode (s) | Short EP (s) | Pause Bridge (s) | False Cuts | TTFA Med (ms) | TTL Med (ms) | chrF++ | RTF (Pi 5) | Pareto? |\n")
            f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
            for s in summaries:
                h = s["hyperparameters"]
                pareto_str = "**YES** (Optimal)" if s.get("is_pareto_optimal") else "Dominated"
                f.write(
                    f"| {h['name']} | {h['k_agreement']} | {h['decode_interval_sec']:.3f} | {h['short_endpoint_sec']:.3f} | "
                    f"{h['max_pause_bridge_sec']:.3f} | {s['total_false_cuts_in_pause']} | {s['ttfa_ms_median']:.1f} | "
                    f"{s['ttl_ms_median']:.1f} | {s['avg_chrf']:.2f} | {s['avg_rtf']:.3f} | {pareto_str} |\n"
                )

        return {
            "json_path": json_path,
            "csv_path": csv_path,
            "md_path": md_path
        }
