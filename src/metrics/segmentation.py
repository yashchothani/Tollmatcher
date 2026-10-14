"""
src/metrics/segmentation.py
============================
Evaluates acoustic/streaming segmentation boundaries:
- Precision, Recall, and F1 score against gold boundaries with tolerance windows tau (200ms, 500ms)
- False Cut Rate (FCR) inside annotated thinking pauses
- Mean Absolute Boundary Timing Error (MAE)
"""

from typing import List, Dict, Any, Tuple


def evaluate_segmentation_boundaries(
    predicted_boundaries_sec: List[float],
    gold_boundaries_sec: List[float],
    tau_sec: float = 0.200
) -> Dict[str, float]:
    """
    Evaluates predicted utterance boundary timestamps against gold boundaries using
    nearest-neighbor matching within a tolerance window tau_sec.
    """
    if len(gold_boundaries_sec) == 0:
        return {
            "precision": 1.0 if len(predicted_boundaries_sec) == 0 else 0.0,
            "recall": 1.0,
            "f1": 1.0 if len(predicted_boundaries_sec) == 0 else 0.0,
            "mae_ms": 0.0
        }

    if len(predicted_boundaries_sec) == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mae_ms": 0.0
        }

    matched_gold = set()
    matched_pred = set()
    errors_ms = []

    # Greedy matching from predicted to nearest gold within tolerance
    for p_idx, p_time in enumerate(predicted_boundaries_sec):
        best_g_idx = None
        best_dist = float("inf")
        for g_idx, g_time in enumerate(gold_boundaries_sec):
            if g_idx in matched_gold:
                continue
            dist = abs(p_time - g_time)
            if dist <= tau_sec and dist < best_dist:
                best_dist = dist
                best_g_idx = g_idx

        if best_g_idx is not None:
            matched_gold.add(best_g_idx)
            matched_pred.add(p_idx)
            errors_ms.append(best_dist * 1000.0)

    tp = len(matched_pred)
    fp = len(predicted_boundaries_sec) - tp
    fn = len(gold_boundaries_sec) - len(matched_gold)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mae_ms = sum(errors_ms) / len(errors_ms) if errors_ms else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mae_ms": round(mae_ms, 2)
    }


def compute_false_cuts_in_pauses(
    predicted_boundaries_sec: List[float],
    internal_pauses: List[Dict[str, Any]]
) -> int:
    """
    Counts the number of predicted boundaries that fall prematurely inside annotated
    intra-utterance thinking pauses (or hesitation intervals).
    """
    false_cuts = 0
    for cut_time in predicted_boundaries_sec:
        for pause in internal_pauses:
            p_start = pause["pause_start_sec"]
            p_end = pause["pause_end_sec"]
            # A boundary cut falling inside an internal pause is an error
            if p_start <= cut_time < p_end:
                false_cuts += 1
                break
    return false_cuts
