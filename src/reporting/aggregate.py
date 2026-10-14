"""
src/reporting/aggregate.py
===========================
Aggregates run results across examples, directions, and variants.
Calculates statistical summaries and builds Pareto quality vs latency metrics.
"""

import os
import sys
from typing import List, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.metrics.latency import compute_latency_distribution
from src.runners.base import StreamingRunResult


def aggregate_variant_results(results: List[StreamingRunResult]) -> Dict[str, Any]:
    """Computes aggregate benchmark metrics for a single algorithm variant."""
    if not results:
        return {}

    n = len(results)
    avg_wer = sum(r.wer for r in results) / n
    avg_cer = sum(r.cer for r in results) / n
    avg_bleu = sum(r.bleu for r in results) / n
    avg_chrf = sum(r.chrf for r in results) / n

    total_false_cuts = sum(r.false_cuts_in_pause for r in results)
    avg_f1_200 = sum(r.f1_200ms for r in results) / n
    avg_f1_500 = sum(r.f1_500ms for r in results) / n
    avg_rtf = sum(r.rtf for r in results) / n

    ttfa_list = [r.ttfa_ms for r in results]
    ttl_list = [r.ttl_ms for r in results]

    ttfa_dist = compute_latency_distribution(ttfa_list)
    ttl_dist = compute_latency_distribution(ttl_list)

    return {
        "variant_name": results[0].variant_name,
        "total_evaluations": n,
        "avg_wer": round(avg_wer, 4),
        "avg_cer": round(avg_cer, 4),
        "avg_bleu": round(avg_bleu, 2),
        "avg_chrf": round(avg_chrf, 2),
        "total_false_cuts_in_pause": total_false_cuts,
        "avg_f1_200ms": round(avg_f1_200, 4),
        "avg_f1_500ms": round(avg_f1_500, 4),
        "avg_rtf": round(avg_rtf, 4),
        "ttfa_ms_median": ttfa_dist["median"],
        "ttfa_ms_mean": ttfa_dist["mean"],
        "ttfa_ms_p95": ttfa_dist["p95"],
        "ttl_ms_median": ttl_dist["median"],
        "ttl_ms_mean": ttl_dist["mean"],
        "ttl_ms_p95": ttl_dist["p95"]
    }
