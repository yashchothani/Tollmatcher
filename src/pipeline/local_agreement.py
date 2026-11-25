"""
src/pipeline/local_agreement.py
================================
Local Agreement streaming ASR stabilization algorithm:
- Tracks sliding window of k partial hypotheses (default k=2)
- Computes Longest Common Prefix (LCP) across consecutive streaming iterations
- Commits prefix words once they stabilize across k steps
- Retains uncommitted suffix words in active hypothesis buffer
"""

from typing import List, Tuple, Dict, Any, Optional
import re


class LocalAgreementStabilizer:
    """Stabilizes streaming incremental ASR hypotheses via k-agreement."""

    def __init__(self, k_agreement: int = 2, normalize_punctuation: bool = True):
        self.k_agreement = max(1, k_agreement)
        self.normalize_punctuation = normalize_punctuation
        self.history: List[List[str]] = []
        self.committed_words: List[str] = []

    @property
    def committed_text(self) -> str:
        return " ".join(self.committed_words)

    @property
    def uncommitted_count(self) -> int:
        latest = self.history[-1] if self.history else []
        return max(0, len(latest) - len(self.committed_words))

    def is_stable(self, threshold_k: Optional[int] = None) -> bool:
        """
        Returns True if the stabilizer has accumulated at least k hypotheses
        and the latest hypothesis uncommitted buffer is empty or stabilized.
        """
        req_k = threshold_k if threshold_k is not None else self.k_agreement
        if len(self.history) < req_k:
            return False
        # If all words in latest hypothesis are committed, it's 100% stable
        latest = self.history[-1] if self.history else []
        if len(latest) == len(self.committed_words):
            return True
        # If the last two hypotheses are identical in uncommitted words
        if len(self.history) >= 2 and self.history[-1] == self.history[-2]:
            return True
        return False

    def _strip_punct(self, word: str) -> str:
        return re.sub(r"^[^\w\s]+|[^\w\s]+$", "", word).lower()

    def _words_match(self, w1: str, w2: str) -> bool:
        if w1 == w2:
            return True
        if self.normalize_punctuation:
            return self._strip_punct(w1) == self._strip_punct(w2)
        return False

    def _longest_common_prefix(self, seqs: List[List[str]]) -> List[str]:
        if not seqs:
            return []
        prefix = []
        min_len = min(len(s) for s in seqs)
        for i in range(min_len):
            first = seqs[0][i]
            if all(self._words_match(s[i], first) for s in seqs):
                # Prefer the most recent capitalization/punctuation
                prefix.append(seqs[-1][i])
            else:
                break
        return prefix

    def update(self, partial_hypothesis_text: str) -> Dict[str, Any]:
        """
        Ingests a new partial ASR hypothesis string.
        Returns dict with newly_committed_words, all_committed_text, and uncommitted_text.
        """
        words = partial_hypothesis_text.strip().split()
        self.history.append(words)
        if len(self.history) > self.k_agreement:
            self.history.pop(0)

        newly_committed = []
        if len(self.history) >= self.k_agreement:
            common = self._longest_common_prefix(self.history)
            # Words beyond what was already committed
            current_comm_len = len(self.committed_words)
            if len(common) > current_comm_len:
                new_words = common[current_comm_len:]
                self.committed_words.extend(new_words)
                newly_committed = new_words

        # Uncommitted words are from the latest hypothesis
        latest_words = self.history[-1] if self.history else []
        uncommitted_words = latest_words[len(self.committed_words):]

        return {
            "newly_committed": newly_committed,
            "all_committed_text": " ".join(self.committed_words),
            "uncommitted_text": " ".join(uncommitted_words),
            "latest_hypothesis": " ".join(latest_words),
            "is_stable": self.is_stable()
        }

    def flush(self) -> str:
        """Forces commitment of any remaining uncommitted words at utterance completion."""
        if self.history:
            latest = self.history[-1]
            current_len = len(self.committed_words)
            if len(latest) > current_len:
                remaining = latest[current_len:]
                self.committed_words.extend(remaining)
        result = " ".join(self.committed_words)
        self.history.clear()
        self.committed_words.clear()
        return result


# Alias for backward compatibility
LocalAgreementMatcher = LocalAgreementStabilizer

