FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends         git       && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi uvicorn[standard] sentence-transformers numpy

COPY services/embeddings/src /app/src

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "embeddings_server.main:app", "--host", "0.0.0.0", "--port", "8001"]
