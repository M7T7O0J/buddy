from __future__ import annotations

from fastapi import APIRouter

import httpx

from ..db import pool
from ..redis_queue import redis_conn
from ..settings import settings

router = APIRouter()


@router.get("/health")
def health(deep: bool = False):
    if not deep:
        return {"status": "ok"}

    out = {"status": "ok", "checks": {"db": "unknown", "redis": "unknown", "embeddings": "unknown"}}

    try:
        with pool.connection() as conn:
            conn.execute("SELECT 1;").fetchone()
        out["checks"]["db"] = "ok"
    except Exception as e:
        out["status"] = "degraded"
        out["checks"]["db"] = f"error: {type(e).__name__}"

    try:
        redis_conn.ping()
        out["checks"]["redis"] = "ok"
    except Exception as e:
        out["status"] = "degraded"
        out["checks"]["redis"] = f"error: {type(e).__name__}"

    try:
        base = (settings.EMBEDDINGS_BASE_URL or "").rstrip("/")
        url = f"{base}/docs"
        r = httpx.get(url, timeout=3.0)
        out["checks"]["embeddings"] = "ok" if r.status_code < 500 else f"error: HTTP {r.status_code}"
        if r.status_code >= 500:
            out["status"] = "degraded"
    except Exception as e:
        out["status"] = "degraded"
        out["checks"]["embeddings"] = f"error: {type(e).__name__}"

    return out
