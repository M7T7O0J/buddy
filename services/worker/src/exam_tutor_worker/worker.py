from __future__ import annotations

import os

import redis
from rq import Queue
from rq.worker import SimpleWorker, Worker

from exam_tutor_common.logging import configure_logging
from .settings import settings

configure_logging()

listen = ["ingest"]
redis_conn = redis.from_url(settings.REDIS_URL)

if __name__ == "__main__":
    queues = [Queue(name, connection=redis_conn) for name in listen]
    # RQ's default Worker uses `os.fork()`, which is unavailable on Windows.
    # SimpleWorker executes jobs in-process and works cross-platform.
    worker_cls = SimpleWorker if os.name == "nt" else Worker
    worker = worker_cls(queues, connection=redis_conn)
    worker.work()
