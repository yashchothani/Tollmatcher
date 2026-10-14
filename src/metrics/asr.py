"""
src/metrics/asr.py
===================
Automatic Speech Recognition evaluation metrics:
- Word Error Rate (WER)
- Character Error Rate (CER)
- Normalized text edit distance
- Standard Whisper-compatible text normalizer
"""

import re
import unicodedata
from typing import List, Tuple


def normalize_text(text: str) -> str:
    """Standardizes casing, punctuation, and unicode accents for ASR evaluation."""
    if not text:
        return ""
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    # Remove standard punctuation except apostrophes in words
    text = re.sub(r"[^\w\s']", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein_distance(seq1: List[str], seq2: List[str]) -> Tuple[int, int, int, int]:
    """
    Computes Levenshtein edit distance between two sequences.
    Returns (distance, substitutions, deletions, insertions).
    """
    n, m = len(seq1), len(seq2)
    # dp[i][j] stores (cost, S, D, I)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                cost_sub, s, d, ins = dp[i - 1][j - 1]
                sub_opt = (cost_sub + 1, s + 1, d, ins)

                cost_del, s, d, ins = dp[i - 1][j]
                del_opt = (cost_del + 1, s, d + 1, ins)

                cost_ins, s, d, ins = dp[i][j - 1]
                ins_opt = (cost_ins + 1, s, d, ins + 1)

                best = min(sub_opt, del_opt, ins_opt, key=lambda x: x[0])
                dp[i][j] = best

    return dp[n][m]


def compute_wer(reference: str, hypothesis: str) -> float:
    """Computes Word Error Rate (WER) = (S + D + I) / N_ref."""
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    dist, _, _, _ = levenshtein_distance(ref_words, hyp_words)
    return round(dist / len(ref_words), 4)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Computes Character Error Rate (CER) = (S + D + I) / N_chars_ref."""
    ref_norm = normalize_text(reference).replace(" ", "")
    hyp_norm = normalize_text(hypothesis).replace(" ", "")

    ref_chars = list(ref_norm)
    hyp_chars = list(hyp_norm)

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    dist, _, _, _ = levenshtein_distance(ref_chars, hyp_chars)
    return round(dist / len(ref_chars), 4)
