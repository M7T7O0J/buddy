from __future__ import annotations

import math
from typing import Sequence


def recall_at_k(ranks: Sequence[int], k: int) -> float:
    """Recall@k for a single relevant set per query using first-hit rank.

    ranks: 1-based rank of first relevant result; 0 means miss.
    """
    hits = sum(1 for r in ranks if 1 <= r <= k)
    return hits / max(1, len(ranks))


def mrr(ranks: Sequence[int]) -> float:
    """Mean reciprocal rank. ranks: 1-based, 0 for miss."""
    return sum((1.0 / r) for r in ranks if r > 0) / max(1, len(ranks))


def ndcg_at_k(relevances: Sequence[Sequence[int]], k: int) -> float:
    """nDCG@k for binary/multi-grade relevances per query.

    relevances[i] is relevance list in predicted rank order (len>=k).
    """
    def dcg(rel: Sequence[int]) -> float:
        s = 0.0
        for i, r in enumerate(rel[:k], start=1):
            s += (2.0**r - 1.0) / math.log2(i + 1.0)
        return s

    scores = []
    for rel in relevances:
        ideal = sorted(rel, reverse=True)
        denom = dcg(ideal) or 1.0
        scores.append(dcg(rel) / denom)
    return sum(scores) / max(1, len(scores))
