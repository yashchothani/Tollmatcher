"""
src/pipeline/local_agreement.py
================================
Local Agreement streaming ASR stabilization algorithm:
- Tracks sliding window of k partial hypotheses (default k=2)
- Computes Longest Common Prefix (LCP) across consecutive streaming iterations
- Commits prefix words once they stabilize across k steps
- Retains uncommitted suffix words in active hypothesis buffer
"""

from typing import List, Tuple, Dict, Any


class LocalAgreementStabilizer:
    """Stabilizes streaming incremental ASR hypotheses via k-agreement."""

    def __init__(self, k_agreement: int = 2):
        self.k_agreement = k_agreement
        self.history: List[List[str]] = []
        self.committed_words: List[str] = []

    def _longest_common_prefix(self, seqs: List[List[str]]) -> List[str]:
        if not seqs:
            return []
        prefix = []
        min_len = min(len(s) for s in seqs)
        for i in range(min_len):
            first = seqs[0][i]
            if all(s[i] == first for s in seqs):
                prefix.append(first)
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
            "latest_hypothesis": " ".join(latest_words)
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
