from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from .settings import settings


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)


pool = ConnectionPool(conninfo=settings.DATABASE_URL, max_size=10, configure=_configure)
