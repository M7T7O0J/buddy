FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends         git       && rm -rf /var/lib/apt/lists/*

COPY packages/common /app/packages/common
COPY packages/rag_core /app/packages/rag_core
COPY services/worker /app/services/worker

RUN pip install --no-cache-dir -e /app/packages/common       && pip install --no-cache-dir -e /app/packages/rag_core       && pip install --no-cache-dir -e /app/services/worker

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "exam_tutor_worker.worker"]
