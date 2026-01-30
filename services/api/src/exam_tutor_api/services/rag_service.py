from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List
from uuid import UUID

from exam_tutor_common.logging import get_logger
from exam_tutor_common.schemas import ChatRequest, RetrievedChunk, RetrieveRequest

from exam_tutor_rag.prompt_builder import build_prompt, to_openai_messages
from exam_tutor_rag.retrieve import Retriever
from exam_tutor_rag.reranker import CrossEncoderReranker, RerankConfig
from exam_tutor_rag.vector_store import PgVectorStore
from exam_tutor_rag.embed_client import EmbeddingsClient

from ..settings import settings
from ..clients.llm_client import LLMClient, LLMConfig
from .memory_service import MemoryService
from .tutor_modes import style_hint

log = get_logger(__name__)


@dataclass(frozen=True)
class RagResult:
    used_chunks: List[RetrievedChunk]
    prompt_messages: List[Dict[str, str]]


class RagService:
    def __init__(self):
        # Lazy-initialise heavy clients to keep import-time fast and make /health robust
        self.embedder = EmbeddingsClient(
            base_url=settings.EMBEDDINGS_BASE_URL,
            model=settings.EMBEDDINGS_MODEL,
            timeout_s=settings.EMBEDDINGS_TIMEOUT_S,
        )
        self.store = PgVectorStore(settings.DATABASE_URL)
        exclude = [t.strip() for t in (settings.RETRIEVE_EXCLUDE_TAGS or "").split(",") if t.strip()]
        ef_search = int(settings.RETRIEVE_HNSW_EF_SEARCH or 0)
        min_quality = float(settings.RETRIEVE_MIN_QUALITY_SCORE)
        min_quality_score = None if min_quality < 0 else min_quality
        reranker = CrossEncoderReranker(
            RerankConfig(
                enabled=bool(settings.RERANK_ENABLED),
                model=settings.RERANK_MODEL,
                top_m=settings.RERANK_TOP_M,
                batch_size=settings.RERANK_BATCH_SIZE,
            )
        )
        self.retriever = Retriever(
            embedder=self.embedder,
            store=self.store,
            exclude_tags=exclude,
            reranker=reranker,
            hnsw_ef_search=(ef_search if ef_search > 0 else None),
            min_quality_score=min_quality_score,
        )

        self.memory = MemoryService(max_messages=12)
        self.llm = LLMClient(
            LLMConfig(
                mode=settings.LLM_MODE,
                base_url=settings.LLM_BASE_URL,
                model=settings.LLM_MODEL,
                api_key=settings.LLM_API_KEY,
                timeout_s=settings.LLM_TIMEOUT_S,
            )
        )

    def build(self, *, req: ChatRequest, conversation_id: UUID) -> RagResult:
        exam = req.exam or settings.DEFAULT_EXAM
        mode = req.mode
        mem = self.memory.get_memory_block(conversation_id)

        # Retrieval
        retrieve_req = RetrieveRequest(
            query=req.message,
            exam=exam,
            subject=req.subject,
            topic=req.topic,
            doc_type=req.doc_type,
            year=req.year,
            top_k=settings.RETRIEVE_TOP_K,
            top_n=settings.RETRIEVE_TOP_N,
        )
        retrieved = self.retriever.retrieve(retrieve_req).chunks

        # Filter weak retrieval
        used = [c for c in retrieved if c.score >= settings.RETRIEVE_MIN_SCORE]
        used = self._cap_sources_for_prompt(used)
        sources = [c.model_dump() for c in used]

        # Build prompt
        parts = build_prompt(
            mode=mode,
            exam=exam,
            language=req.language or settings.DEFAULT_LANGUAGE,
            question=req.message + "\n\nSTYLE NOTE: " + style_hint(mode, exam),
            sources=sources,
            memory=mem if mem else None,
        )
        messages = to_openai_messages(parts)
        return RagResult(used_chunks=used, prompt_messages=messages)

    def stream_answer(self, *, prompt_messages: List[Dict[str, str]]) -> Iterator[str]:
        return self.llm.stream_chat(prompt_messages)

    def _cap_sources_for_prompt(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        if not chunks:
            return []

        max_chunks = max(1, int(settings.PROMPT_SOURCES_MAX_CHUNKS))
        max_tokens = max(1, int(settings.PROMPT_SOURCES_MAX_TOKENS))

        capped: List[RetrievedChunk] = []
        total = 0

        for c in chunks:
            if len(capped) >= max_chunks:
                break

            remaining = max_tokens - total
            if remaining <= 0:
                break

            estimate = c.token_count or max(1, len(c.content) // 4)
            if estimate <= remaining:
                capped.append(c)
                total += estimate
                continue

            # If the top chunk is too large (common with OCR PDFs), truncate it to fit.
            if not capped:
                suffix = "\n\n[TRUNCATED]\n"
                max_chars = max(1, remaining * 4 - len(suffix))
                truncated = c.content[:max_chars].rstrip() + suffix
                capped.append(c.model_copy(update={"content": truncated, "token_count": remaining}))
                total += remaining
            break

        if len(capped) < len(chunks):
            log.info(
                "prompt_sources_capped",
                extra={
                    "kept": len(capped),
                    "dropped": len(chunks) - len(capped),
                    "token_budget": max_tokens,
                    "token_total_est": total,
                },
            )

        return capped


_rag_singleton: RagService | None = None


def get_rag_service() -> RagService:
    global _rag_singleton
    if _rag_singleton is None:
        _rag_singleton = RagService()
    return _rag_singleton
