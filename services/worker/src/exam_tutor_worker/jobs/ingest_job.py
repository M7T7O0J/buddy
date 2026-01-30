from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx

from exam_tutor_common.logging import get_logger
from exam_tutor_rag.embed_client import EmbeddingsClient
from exam_tutor_rag.vector_store import ChunkRow, PgVectorStore

from ..db import pool
from ..settings import settings
from ..pipelines.extractors.docling_extractor import DoclingExtractor
from ..pipelines.cleanup import cleanup_markdown
from ..pipelines.normalize import normalize_markdown
from ..pipelines.chunker import TokenAwareChunker
from ..pipelines.chunk_filter import FilterConfig, filter_chunks

log = get_logger(__name__)

_SOURCE_URL_METADATA_KEY = "source_url"


def _set_job_status(document_id: UUID, status: str, error: Optional[str] = None) -> None:
    with pool.connection() as conn:
        conn.execute(
            """                UPDATE ingestion_jobs
            SET status=%s, error=%s, updated_at=now(),
                started_at = CASE WHEN %s='running' THEN now() ELSE started_at END,
                finished_at = CASE WHEN %s IN ('done','failed') THEN now() ELSE finished_at END
            WHERE document_id=%s
            """,
            (status, error, status, status, document_id),
        )
        conn.commit()


def _get_document_meta(document_id: UUID) -> Dict[str, Any]:
    with pool.connection() as conn:
        row = conn.execute(
            """                SELECT source, title, exam, subject, topic, doc_type, year, metadata
            FROM documents WHERE id=%s
            """,
            (document_id,),
        ).fetchone()
    if not row:
        raise ValueError("document not found")
    source, title, exam, subject, topic, doc_type, year, metadata = row
    return {
        "source": source,
        "title": title,
        "exam": exam,
        "subject": subject,
        "topic": topic,
        "doc_type": doc_type,
        "year": year,
        "metadata": metadata or {},
    }

def _download_to_path(*, url: str, path: str, timeout_s: float = 120.0) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.partial"
    headers = {
        "User-Agent": "veda-worker/1.0 (+https://example.invalid)",
        "Accept": "*/*",
    }

    last_exc: Exception | None = None
    for attempt in range(1, 6):
        try:
            with httpx.Client(timeout=timeout_s, follow_redirects=True, headers=headers) as client:
                with client.stream("GET", url) as r:
                    r.raise_for_status()
                    content_type = (r.headers.get("content-type") or "").lower()
                    if path.lower().endswith(".pdf") and ("html" in content_type or "text/" in content_type):
                        raise ValueError(
                            f"source_url did not look like a PDF (content-type={content_type!r}); "
                            "provide a direct PDF download URL"
                        )

                    with open(tmp_path, "wb") as f:
                        for chunk in r.iter_bytes():
                            if chunk:
                                f.write(chunk)
            os.replace(tmp_path, path)
            return
        except Exception as e:
            last_exc = e
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            if attempt >= 5:
                break
            time.sleep(0.5 * (2 ** (attempt - 1)))

    assert last_exc is not None
    raise last_exc


def _ensure_source_present(meta: Dict[str, Any]) -> None:
    source = meta["source"]
    if source.startswith("http://") or source.startswith("https://"):
        return
    if os.path.exists(source):
        return
    url = (meta.get("metadata") or {}).get(_SOURCE_URL_METADATA_KEY)
    if not url:
        raise FileNotFoundError(f"source file not found: {source}")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError("metadata.source_url must start with http:// or https://")
    log.info("downloading_missing_source", extra={"source": source})
    _download_to_path(url=url, path=source)


def run_ingest_job(document_id: str) -> None:
    """RQ job entrypoint."""
    doc_uuid = UUID(document_id)
    _set_job_status(doc_uuid, "running")

    embedder: Optional[EmbeddingsClient] = None
    try:
        meta = _get_document_meta(doc_uuid)
        _ensure_source_present(meta)

        extractor = DoclingExtractor()
        extracted = extractor.extract(meta["source"])
        md = normalize_markdown(extracted.markdown)
        md = cleanup_markdown(md)

        chunker = TokenAwareChunker(
            tokenizer_name=settings.TOKENIZER_NAME,
            min_tokens=settings.CHUNK_MIN_TOKENS,
            max_tokens=settings.CHUNK_MAX_TOKENS,
            overlap_tokens=settings.CHUNK_OVERLAP_TOKENS,
            parent_section_level=settings.CHUNK_PARENT_SECTION_LEVEL,
        )
        chunks = chunker.chunk(md)
        chunks, stats = filter_chunks(
            chunks,
            cfg=FilterConfig(
                enabled=settings.CHUNK_FILTER_ENABLED,
                min_tokens=settings.CHUNK_FILTER_MIN_TOKENS,
                max_chunks_per_doc=settings.CHUNK_FILTER_MAX_CHUNKS_PER_DOC,
                max_chunks_per_parent=settings.CHUNK_FILTER_MAX_CHUNKS_PER_PARENT,
            ),
        )
        log.info(
            "chunk_filter_stats",
            extra={
                "document_id": document_id,
                "total_in": stats.total_in,
                "total_out": stats.total_out,
                "dropped": stats.dropped,
                "dropped_by_tag": stats.dropped_by_tag,
            },
        )

        embedder = EmbeddingsClient(
            base_url=settings.EMBEDDINGS_BASE_URL,
            model=settings.EMBEDDINGS_MODEL,
            timeout_s=settings.EMBEDDINGS_TIMEOUT_S,
        )
        store = PgVectorStore(settings.DATABASE_URL)

        # Batch embeddings to dramatically reduce overhead (HTTP + model encode).
        # Keep batch size conservative for CPU (embeddings server).
        BATCH_SIZE = 16
        rows: List[ChunkRow] = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            embs = embedder.embed_many([c.text for c in batch])
            for ch, emb in zip(batch, embs):
                chunk_metadata = dict(meta.get("metadata") or {})
                chunk_metadata.update(ch.metadata or {})
                rows.append(
                    ChunkRow(
                        document_id=doc_uuid,
                        chunk_index=ch.index,
                        content=ch.text,
                        token_count=ch.token_count,
                        embedding=emb,
                        exam=meta["exam"],
                        subject=meta.get("subject"),
                        topic=meta.get("topic"),
                        doc_type=meta.get("doc_type"),
                        year=meta.get("year"),
                        source_title=meta["title"],
                        metadata=chunk_metadata,
                    )
                )

        store.upsert_chunks(doc_uuid, rows)
        store.close()
        _set_job_status(doc_uuid, "done")

    except Exception as e:
        log.exception("ingest_failed", extra={"document_id": document_id})
        _set_job_status(doc_uuid, "failed", error=str(e))
        raise
    finally:
        if embedder is not None:
            try:
                embedder.close()
            except Exception:
                pass
