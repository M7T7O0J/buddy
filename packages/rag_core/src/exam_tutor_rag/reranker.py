from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from exam_tutor_common.logging import get_logger
from exam_tutor_common.schemas import RetrievedChunk

log = get_logger(__name__)


@dataclass(frozen=True)
class RerankConfig:
    enabled: bool = False
    model: str = "BAAI/bge-reranker-v2-m3"
    top_m: int = 30
    batch_size: int = 16


class CrossEncoderReranker:
    """Optional cross-encoder reranker.

    Runs *after* vector search to reorder top-M candidates by relevance to the query.
    """

    def __init__(self, cfg: RerankConfig):
        self.cfg = cfg
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("sentence-transformers is required for reranking") from e
        self._model = CrossEncoder(self.cfg.model)
        return self._model

    def rerank(self, *, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        if not self.cfg.enabled or not chunks:
            return chunks

        top_m = max(1, int(self.cfg.top_m))
        head = chunks[:top_m]
        tail = chunks[top_m:]

        try:
            model = self._load()
        except Exception as e:
            log.warning("reranker_load_failed", extra={"error": str(e), "model": self.cfg.model})
            return chunks

        pairs = [(query, c.content) for c in head]
        try:
            scores = model.predict(pairs, batch_size=max(1, int(self.cfg.batch_size)))
        except Exception as e:
            log.warning("reranker_predict_failed", extra={"error": str(e), "model": self.cfg.model})
            return chunks

        # Attach rerank_score to metadata and sort by it.
        updated: List[RetrievedChunk] = []
        for c, s in zip(head, scores):
            md = dict(c.metadata or {})
            md["rerank_model"] = self.cfg.model
            md["rerank_score"] = float(s)
            updated.append(c.model_copy(update={"metadata": md}))

        updated.sort(key=lambda c: float((c.metadata or {}).get("rerank_score", 0.0)), reverse=True)
        return updated + tail

