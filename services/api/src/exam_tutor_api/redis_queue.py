from __future__ import annotations

import redis
from rq import Queue

from .settings import settings

redis_conn = redis.from_url(settings.REDIS_URL)
ingest_queue = Queue("ingest", connection=redis_conn, default_timeout=60 * 60)
