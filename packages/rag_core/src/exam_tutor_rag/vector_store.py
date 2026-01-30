from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
from uuid import UUID

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from psycopg.types.json import Jsonb

from exam_tutor_common.schemas import RetrievedChunk


@dataclass(frozen=True)
class ChunkRow:
    document_id: UUID
    chunk_index: int
    content: str
    token_count: int
    embedding: Sequence[float]
    exam: str
    subject: Optional[str]
    topic: Optional[str]
    doc_type: Optional[str]
    year: Optional[int]
    source_title: str
    metadata: Dict[str, Any]

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


class PgVectorStore:
    """Minimal pgvector-backed store using psycopg3 + pgvector-python.

    Uses cosine distance (<=>) with HNSW index.
    """

    def __init__(self, database_url: str):
        def _configure(conn: psycopg.Connection) -> None:
            register_vector(conn)

        self.pool = ConnectionPool(conninfo=database_url, max_size=10, configure=_configure)

    def close(self) -> None:
        self.pool.close()

    def upsert_chunks(self, document_id: UUID, chunks: List[ChunkRow]) -> None:
        sql = """            INSERT INTO chunks (
          document_id, chunk_index, content, token_count, embedding,
          exam, subject, topic, doc_type, year, metadata, content_hash
        )
        VALUES (
          %(document_id)s, %(chunk_index)s, %(content)s, %(token_count)s, %(embedding)s,
          %(exam)s, %(subject)s, %(topic)s, %(doc_type)s, %(year)s, %(metadata)s, %(content_hash)s
        )
        ON CONFLICT (document_id, chunk_index)
        DO UPDATE SET
          content = EXCLUDED.content,
          token_count = EXCLUDED.token_count,
          embedding = EXCLUDED.embedding,
          exam = EXCLUDED.exam,
          subject = EXCLUDED.subject,
          topic = EXCLUDED.topic,
          doc_type = EXCLUDED.doc_type,
          year = EXCLUDED.year,
          metadata = EXCLUDED.metadata,
          content_hash = EXCLUDED.content_hash;
        """
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                for c in chunks:
                    cur.execute(sql, {
                        "document_id": document_id,
                        "chunk_index": c.chunk_index,
                        "content": c.content,
                        "token_count": c.token_count,
                        "embedding": Vector(c.embedding),
                        "exam": c.exam,
                        "subject": c.subject,
                        "topic": c.topic,
                        "doc_type": c.doc_type,
                        "year": c.year,
                        "metadata": Jsonb(c.metadata),
                        "content_hash": c.content_hash,
                    })
            conn.commit()

    def search(
        self,
        *,
        query_embedding: Sequence[float],
        exam: Optional[str],
        subject: Optional[str],
        topic: Optional[str],
        doc_type: Optional[str],
        year: Optional[int],
        top_k: int,
        exclude_tags: Optional[List[str]] = None,
        min_quality_score: Optional[float] = None,
        hnsw_ef_search: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        where = []
        # Use pgvector's Vector type so psycopg binds the parameter as `vector`
        # (avoids array typing issues like `vector <=> double precision[]`).
        params: Dict[str, Any] = {"q": Vector(query_embedding), "k": top_k}

        if exam:
            where.append("c.exam = %(exam)s")
            params["exam"] = exam
        if subject:
            where.append("c.subject = %(subject)s")
            params["subject"] = subject
        if topic:
            where.append("c.topic = %(topic)s")
            params["topic"] = topic
        if doc_type:
            where.append("c.doc_type = %(doc_type)s")
            params["doc_type"] = doc_type
        if year:
            where.append("c.year = %(year)s")
            params["year"] = year

        if exclude_tags:
            where.append("NOT (COALESCE(c.metadata->'tags', '[]'::jsonb) ?| %(exclude_tags)s)")
            params["exclude_tags"] = [t.strip().lower() for t in exclude_tags if t.strip()]

        if min_quality_score is not None:
            where.append("COALESCE((c.metadata->>'quality_score')::double precision, 1.0) >= %(min_quality_score)s")
            params["min_quality_score"] = float(min_quality_score)

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        # cosine distance: smaller is closer; we convert to similarity score in Python.
        sql = f"""            SELECT
          c.id,
          c.document_id,
          d.title AS source_title,
          c.content,
          c.token_count,
          (c.embedding <=> %(q)s) AS distance,
          c.exam, c.subject, c.topic, c.doc_type, c.year,
          c.metadata
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        {where_sql}
        ORDER BY c.embedding <=> %(q)s
        LIMIT %(k)s;
        """

        results: List[RetrievedChunk] = []
        with self.pool.connection() as conn:
            if hnsw_ef_search is not None and int(hnsw_ef_search) > 0:
                # Older pgvector versions or non-HNSW indexes may not support this GUC.
                # Don't fail retrieval if tuning isn't available, but clear the transaction if it errors.
                try:
                    with conn.cursor() as cur:
                        cur.execute("SET LOCAL hnsw.ef_search = %s;", (int(hnsw_ef_search),))
                except Exception:
                    conn.rollback()
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        for (chunk_id, document_id, source_title, content, token_count, distance, ex, sub, top, dt, yr, md) in rows:
            # Convert cosine distance (smaller is better) to a stable, positive similarity score in (0, 1].
            # This avoids negative scores from `1 - distance` and behaves better with downstream thresholds.
            score = float(1.0 / (1.0 + float(distance)))
            results.append(RetrievedChunk(
                chunk_id=int(chunk_id),
                document_id=document_id,
                source_title=source_title,
                content=content,
                token_count=int(token_count) if token_count is not None else None,
                score=score,
                exam=ex,
                subject=sub,
                topic=top,
                doc_type=dt,
                year=yr,
                metadata=md or {},
            ))
        return results
