from __future__ import annotations

import os
from uuid import UUID

from fastapi import APIRouter, HTTPException

from exam_tutor_common.schemas import IngestRequest, IngestResponse, IngestStatusResponse
from exam_tutor_common.logging import get_logger

from psycopg.types.json import Jsonb
from uuid6 import uuid7

from ..db import pool
from ..redis_queue import ingest_queue
from ..settings import settings

log = get_logger(__name__)
router = APIRouter()

_SOURCE_URL_METADATA_KEY = "source_url"


def _safe_detail(*, prefix: str, exc: Exception) -> str:
    if settings.APP_ENV.lower() in {"prod", "production"}:
        return prefix
    msg = str(exc).strip().replace("\n", " ")
    if len(msg) > 400:
        msg = msg[:400] + "…"
    return f"{prefix}: {type(exc).__name__}: {msg}" if msg else f"{prefix}: {type(exc).__name__}"


def _mark_job_failed(*, document_id: UUID, error: str) -> None:
    with pool.connection() as conn:
        conn.execute(
            """                UPDATE ingestion_jobs
            SET status='failed', error=%s, updated_at=now(), finished_at=now()
            WHERE document_id=%s
            """,
            (error, document_id),
        )
        conn.commit()


def _resolve_source(source: str) -> str:
    # MVP: allow relative paths under the configured local doc root.
    if source.startswith("http://") or source.startswith("https://"):
        return source
    # Local files under LOCAL_DOC_ROOT.
    if os.path.isabs(source):
        abs_path = os.path.abspath(source)
    else:
        abs_path = os.path.abspath(os.path.join(settings.LOCAL_DOC_ROOT, source))
    _assert_local_path_under_doc_root(abs_path)
    return abs_path


def _assert_local_path_under_doc_root(abs_path: str) -> None:
    doc_root = os.path.realpath(os.path.abspath(settings.LOCAL_DOC_ROOT))
    target = os.path.realpath(os.path.abspath(abs_path))
    if os.path.commonpath([doc_root, target]) != doc_root:
        raise HTTPException(status_code=400, detail="source must be under LOCAL_DOC_ROOT when source_url is provided")


@router.post("/v1/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    doc_id = uuid7()
    source = _resolve_source(req.source)

    metadata = dict(req.metadata)
    if req.source_url:
        if not (req.source_url.startswith("http://") or req.source_url.startswith("https://")):
            raise HTTPException(status_code=400, detail="source_url must start with http:// or https://")
        if not (source.startswith("http://") or source.startswith("https://")):
            metadata.setdefault(_SOURCE_URL_METADATA_KEY, req.source_url)

    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """                        INSERT INTO documents (id, source, title, exam, subject, topic, doc_type, year, metadata)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        doc_id,
                        source,
                        req.title,
                        req.exam,
                        req.subject,
                        req.topic,
                        req.doc_type,
                        req.year,
                        Jsonb(metadata),
                    ),
                )
                cur.execute(
                    """                        INSERT INTO ingestion_jobs (document_id, status)
                    VALUES (%s, 'queued')
                    """,
                    (doc_id,),
                )
            conn.commit()
    except Exception as e:
        log.exception("ingest_db_insert_failed", extra={"document_id": str(doc_id)})
        raise HTTPException(status_code=500, detail=_safe_detail(prefix="db_insert_failed", exc=e)) from e

    # enqueue background job
    try:
        ingest_queue.enqueue(
            "exam_tutor_worker.jobs.ingest_job.run_ingest_job",
            str(doc_id),
        )
    except Exception as e:
        log.exception("ingest_enqueue_failed", extra={"document_id": str(doc_id)})
        _mark_job_failed(document_id=doc_id, error=_safe_detail(prefix="enqueue_failed", exc=e))
        raise HTTPException(status_code=500, detail=_safe_detail(prefix="enqueue_failed", exc=e)) from e

    return IngestResponse(document_id=doc_id, status="queued")


@router.get("/v1/ingest/{document_id}/status", response_model=IngestStatusResponse)
def ingest_status(document_id: UUID) -> IngestStatusResponse:
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT status, error FROM ingestion_jobs WHERE document_id=%s",
            (document_id,),
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="document not found")

    status, error = row
    return IngestStatusResponse(document_id=document_id, status=status, error=error)
