"""
src/metrics/translation.py
===========================
Machine translation evaluation metrics in zero-dependency native Python:
- SacreBLEU (sentence-level and corpus-level with dynamic n-gram capping for short turns)
- chrF and chrF++ (character n-gram F-score + word n-grams with beta=2.0)
- Brevity penalty calculation
"""

import collections
import math
import re
from typing import List, Tuple, Dict, Any


def tokenize_words(text: str) -> List[str]:
    """Tokenizes text into words and punctuation tokens."""
    if not text:
        return []
    # Separate punctuation
    text = re.sub(r"([.,!?:;\"'()\[\]«»„“”])", r" \1 ", text)
    tokens = text.strip().split()
    return tokens


def get_ngrams(tokens: List[str], n: int) -> collections.Counter:
    """Extracts n-grams from a list of tokens."""
    if len(tokens) < n:
        return collections.Counter()
    return collections.Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(hypothesis: str, reference: str, max_order: int = 4, smooth: bool = True) -> float:
    """
    Computes sentence-level BLEU score (0.0 to 100.0).
    Features dynamic order smoothing: For short turns (< 4 words), caps max_order
    to avoid spurious 0.0 scores due to missing 4-grams.
    """
    hyp_tokens = tokenize_words(hypothesis.lower())
    ref_tokens = tokenize_words(reference.lower())

    hyp_len = len(hyp_tokens)
    ref_len = len(ref_tokens)

    if hyp_len == 0:
        return 0.0

    # Dynamic order capping for short sentences
    effective_max_order = min(max_order, hyp_len, ref_len)
    if effective_max_order == 0:
        return 0.0

    precisions = []
    for n in range(1, effective_max_order + 1):
        hyp_ngrams = get_ngrams(hyp_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)

        total_hyp = sum(hyp_ngrams.values())
        if total_hyp == 0:
            p_n = 0.0
        else:
            clipped_matches = 0
            for ng, count in hyp_ngrams.items():
                clipped_matches += min(count, ref_ngrams.get(ng, 0))

            if smooth and clipped_matches == 0:
                # Add-1 epsilon smoothing
                p_n = 1.0 / (2.0 * total_hyp)
            else:
                p_n = clipped_matches / total_hyp

        precisions.append(p_n)

    # Brevity penalty
    if hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (ref_len / hyp_len))

    # Geometric mean of precisions
    log_sum = 0.0
    for p in precisions:
        if p <= 0.0:
            return 0.0
        log_sum += math.log(p)

    geo_mean = math.exp(log_sum / effective_max_order)
    bleu_score = bp * geo_mean * 100.0
    return round(bleu_score, 2)


def get_char_ngrams(text: str, n: int) -> collections.Counter:
    """Extracts character n-grams ignoring whitespaces."""
    chars = text.replace(" ", "")
    if len(chars) < n:
        return collections.Counter()
    return collections.Counter(chars[i:i + n] for i in range(len(chars) - n + 1))


def compute_chrf(hypothesis: str, reference: str, char_order: int = 6, word_order: int = 2, beta: float = 2.0) -> float:
    """
    Computes chrF++ score (0.0 to 100.0) combining character and word n-grams.
    Beta = 2.0 weights recall higher than precision.
    """
    if not hypothesis or not reference:
        return 0.0 if not (not hypothesis and not reference) else 100.0

    hyp_clean = hypothesis.lower()
    ref_clean = reference.lower()

    precisions = []
    recalls = []

    # 1. Character n-grams (1 to 6)
    for n in range(1, char_order + 1):
        hyp_ng = get_char_ngrams(hyp_clean, n)
        ref_ng = get_char_ngrams(ref_clean, n)

        tot_hyp = sum(hyp_ng.values())
        tot_ref = sum(ref_ng.values())

        if tot_hyp == 0 or tot_ref == 0:
            continue

        matches = sum(min(count, ref_ng.get(ng, 0)) for ng, count in hyp_ng.items())
        prec = matches / tot_hyp if tot_hyp > 0 else 0.0
        rec = matches / tot_ref if tot_ref > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)

    # 2. Word n-grams (1 to 2) for chrF++
    hyp_words = tokenize_words(hyp_clean)
    ref_words = tokenize_words(ref_clean)
    for n in range(1, word_order + 1):
        hyp_w_ng = get_ngrams(hyp_words, n)
        ref_w_ng = get_ngrams(ref_words, n)

        tot_hyp = sum(hyp_w_ng.values())
        tot_ref = sum(ref_w_ng.values())

        if tot_hyp == 0 or tot_ref == 0:
            continue

        matches = sum(min(count, ref_w_ng.get(ng, 0)) for ng, count in hyp_w_ng.items())
        prec = matches / tot_hyp if tot_hyp > 0 else 0.0
        rec = matches / tot_ref if tot_ref > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)

    if not precisions or not recalls:
        return 0.0

    avg_prec = sum(precisions) / len(precisions)
    avg_rec = sum(recalls) / len(recalls)

    if avg_prec + avg_rec == 0.0:
        return 0.0

    beta_sq = beta ** 2
    f_score = (1.0 + beta_sq) * (avg_prec * avg_rec) / ((beta_sq * avg_prec) + avg_rec)
    return round(f_score * 100.0, 2)
