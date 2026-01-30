FROM python:3.11-slim

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends         curl       && rm -rf /var/lib/apt/lists/*

COPY packages/common /app/packages/common
COPY packages/rag_core /app/packages/rag_core
COPY services/api /app/services/api

RUN pip install --no-cache-dir -e /app/packages/common       && pip install --no-cache-dir -e /app/packages/rag_core       && pip install --no-cache-dir -e /app/services/api

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "exam_tutor_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
