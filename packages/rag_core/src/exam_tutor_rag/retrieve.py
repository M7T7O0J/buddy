from __future__ import annotations


from exam_tutor_common.schemas import RetrieveRequest, RetrieveResponse

from .embed_client import EmbeddingsClient
from .reranker import CrossEncoderReranker
from .vector_store import PgVectorStore


class Retriever:
    def __init__(
        self,
        *,
        embedder: EmbeddingsClient,
        store: PgVectorStore,
        exclude_tags: list[str] | None = None,
        reranker: CrossEncoderReranker | None = None,
        hnsw_ef_search: int | None = None,
        min_quality_score: float | None = None,
    ):
        self.embedder = embedder
        self.store = store
        self.exclude_tags = {t.strip().lower() for t in (exclude_tags or []) if t.strip()}
        self.reranker = reranker
        self.hnsw_ef_search = hnsw_ef_search
        self.min_quality_score = min_quality_score

    def retrieve(self, req: RetrieveRequest) -> RetrieveResponse:
        q_emb = self.embedder.embed(req.query)
        chunks = self.store.search(
            query_embedding=q_emb,
            exam=req.exam,
            subject=req.subject,
            topic=req.topic,
            doc_type=req.doc_type,
            year=req.year,
            top_k=req.top_k,
            exclude_tags=sorted(self.exclude_tags) if self.exclude_tags else None,
            min_quality_score=self.min_quality_score,
            hnsw_ef_search=self.hnsw_ef_search,
        )

        if self.reranker is not None:
            chunks = self.reranker.rerank(query=req.query, chunks=chunks)

        # top_n trimming and simple score ordering already done
        return RetrieveResponse(chunks=chunks[: req.top_n])
