# Exam Tutor MVP (GATE + UPSC) — RAG + vLLM + Docling

A production-lean MVP codebase for an **exam tutor** supporting **GATE** and **UPSC** using:
- **Docling** for document extraction (PDF/DOCX/HTML/etc.)
- **RAG** (chunk → embed → pgvector → retrieve)
- **vLLM** for fast inference (OpenAI-compatible `/v1/chat/completions`)
- **FastAPI** for the API (chat/retrieve/ingest/feedback)
- **RQ + Redis** for background ingestion jobs
- **Optional local embedding server** (FastAPI + SentenceTransformers)

## What you get
- Streaming chat (`/v1/chat`) with citations
- Retrieval debug endpoint (`/v1/retrieve`)
- Ingestion endpoints (`/v1/ingest`, `/v1/ingest/{document_id}/status`) that enqueue Docling-based extraction
- A clean multi-exam metadata schema (exam/subject/topic/doc_type/year/etc.)
- A CI-friendly evaluation CLI (retrieval metrics; optional RAGAS stub)

## Quickstart (local dev)
Requirements: Docker + Docker Compose

1) Copy env
```bash
cp .env.example .env
```
Then edit `.env` and set the Docker Compose endpoints:
- `DATABASE_URL=postgresql://postgres:postgres@db:5432/exam_tutor`
- `REDIS_URL=redis://redis:6379/0`

2) Start services (db + redis + embeddings + api + worker)
```bash
docker compose up --build
```

3) Ingest a document (example)
```bash
curl -sS http://localhost:8000/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "source": "samples/gate_sample.md",
    "title": "GATE Sample Notes",
    "exam": "GATE_DA",
    "subject": "Probability",
    "topic": "Bayes Theorem",
    "doc_type": "notes",
    "year": 2025
  }'
```

4) Ask a question (streaming SSE)
```bash
curl -N http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
 -d '{
    "message": "Explain Bayes theorem with an example.",
    "exam": "GATE_DA",
    "mode": "doubt"
  }'
```

## Local dev (Python venv)
If you prefer to run the Python services directly (while still using Docker for Postgres/Redis):

1) Create and activate a venv, then install deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```
Then edit `.env` and set the Docker Compose endpoints:
- `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/exam_tutor`
- `REDIS_URL=redis://localhost:6379/0`

2) Start infra containers (db + redis; add `embeddings` if you prefer the Dockerized embedder)
```bash
docker compose up db redis
# or: docker compose up db redis embeddings
```

3) Run the Python services from your venv
```bash
# embeddings server (if you don't use the Docker one)
python -m uvicorn --app-dir services/embeddings/src embeddings_server.main:app --port 8001

# worker (Docling extraction + chunk/embedding pipeline)
python -m exam_tutor_worker.worker

# API
python -m uvicorn exam_tutor_api.main:app --reload --port 8000
```

## Hosted DB/Redis (Neon + Upstash recommended for WSL)
If you want to run the API/worker locally but use hosted Postgres and Redis:

Important:
- Embedding vector dimension must match your database schema (`chunks.embedding`).
- On WSL, some hosted Postgres providers may resolve to IPv6-only endpoints; prefer a provider with an IPv4 endpoint (Neon works well on free tier).

1) Create a Postgres DB with pgvector and run the schema SQL:
   - For **bge-m3 (1024-dim)**: run `infra/sql/001_init.sql`
   - For **all-MiniLM-L6-v2 (384-dim)**: run `infra/sql/001_init_384.sql`

2) Update `.env`:
   - `DATABASE_URL` to your Postgres connection string (typically with `sslmode=require`). Wrap the value in quotes.
   - `REDIS_URL` to your Upstash Redis URL (use `rediss://`).
   - `EMBEDDINGS_BASE_URL` to `http://localhost:8001` (or your hosted embeddings URL).
   - `EMBEDDINGS_MODEL` to match your chosen schema (1024-dim vs 384-dim).
   - `LLM_MODE=vllm`, `LLM_BASE_URL`, and `LLM_MODEL` if you want real inference (see terminal workflow below).

3) Install deps and start services locally:
```bash
python -m pip install -e packages/common -e packages/rag_core -e services/api -e services/worker
python -m exam_tutor_worker.worker
python -m uvicorn --app-dir services/embeddings/src embeddings_server.main:app --port 8001
python -m uvicorn exam_tutor_api.main:app --reload --port 8000
```

Notes:
- If you see `python-dotenv could not parse statement...`, ensure every non-empty line in `.env` is either `KEY=VALUE` or starts with `#`.
- Avoid `source .env` in your shell: values like URLs can contain `&` which breaks shell parsing. The services load `.env` themselves.
- Prefer `python -m uvicorn ...` to ensure you use the active virtualenv on Windows.
- Quick `.env` format check: `python scripts/check_env_format.py .env`
- PyTorch wheels: do not mix CUDA variants (e.g. `+cu121`) with CPU wheels. CUDA 12.1 wheels only go up to torch 2.5.1, but the embeddings server may require torch >=2.6 due to upstream security hardening; use CPU wheels or a newer CUDA index (e.g. `cu124`/`cu126`).

## vLLM inference
This repo includes an inference container scaffold in `services/inference/`.
For local CPU-only dev, the API can run without vLLM by setting `LLM_MODE=mock`.

For real inference:
- deploy vLLM on a GPU VM
- set `LLM_BASE_URL` and `LLM_MODEL` in `.env`

### Terminal workflow: full pipeline (hosted Postgres/Redis + local embeddings + local vLLM)
This is the setup we validated end-to-end:
- Hosted Postgres (Supabase/Neon/etc.) + hosted Redis (Upstash/etc.)
- Local embeddings server on `http://127.0.0.1:8003`
- Local vLLM server on `http://127.0.0.1:8002`
- Local worker + API

Prereqs:
- Create a venv and install deps:
  ```bash
  cd /path/to/veda
  python3.13 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- `.env` points to your hosted `DATABASE_URL` and `REDIS_URL` (and your DB has `CREATE EXTENSION vector;` applied).
- `.env` also includes:
  - `EMBEDDINGS_BASE_URL=http://127.0.0.1:8003`
  - `LLM_MODE=vllm`
  - `LLM_BASE_URL=http://127.0.0.1:8002/v1`
  - `LLM_MODEL=Qwen/Qwen3-4B`
  - `LLM_API_KEY=EMPTY`
  - Chunking + filtering (recommended to reduce “front matter” noise and improve retrieval):
    - `CHUNK_PARENT_SECTION_LEVEL=2` (section-bounded hierarchical chunking by headings)
    - `CHUNK_FILTER_ENABLED=true`
    - `CHUNK_FILTER_MIN_TOKENS=40`
    - `CHUNK_FILTER_MAX_CHUNKS_PER_DOC=2000`
    - `CHUNK_FILTER_MAX_CHUNKS_PER_PARENT=400`
    - `RETRIEVE_EXCLUDE_TAGS=front_matter,boilerplate,image_only,duplicate,low_signal`
  - Optional reranking (improves relevance after pgvector search; costs extra CPU time):
    - `RERANK_ENABLED=false` (set `true` only if the reranker model is available locally; it may try to download on first run)
    - `RERANK_MODEL=BAAI/bge-reranker-v2-m3`
    - `RERANK_TOP_M=30`
    - `RERANK_BATCH_SIZE=16`
- If your DB password contains special characters (notably `@`), URL-encode it in `DATABASE_URL`.

Notes on the retrieval stack:
- Embeddings: `EMBEDDINGS_MODEL` is used for both ingestion and query embedding. If you change it, you must re-ingest/re-embed your documents.
- Vector search: Supabase Postgres uses pgvector HNSW cosine distance (`chunks_embedding_hnsw`).
- Chunk hygiene: the worker tags/drops common front-matter/boilerplate chunks during ingestion and the API excludes tagged chunks during retrieval.
- Reranking (optional): runs after pgvector search to reorder the top results for better relevance.

Terminal 1 — embeddings (CPU; keep it running)
```bash
cd /path/to/veda
source .venv/bin/activate
CUDA_VISIBLE_DEVICES="" EMBEDDINGS_MODEL=BAAI/bge-m3 \
python -m uvicorn --app-dir services/embeddings/src embeddings_server.main:app --host 127.0.0.1 --port 8003
```

Terminal 2 — vLLM (GPU; tuned to avoid OOM on 12GB cards)
```bash
cd /path/to/veda
source .venv/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --port 8002 \
  --dtype float16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.88 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 1024 \
  --swap-space 4 \
  --enforce-eager \
  --attention-config.backend TRITON_ATTN
```

Terminal 3 — worker (CPU; keep it running)
```bash
cd /path/to/veda
source .venv/bin/activate
CUDA_VISIBLE_DEVICES="" python -m exam_tutor_worker.worker
```

Terminal 4 — API
```bash
cd /path/to/veda
source .venv/bin/activate
python -m uvicorn exam_tutor_api.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 5 — ingest → poll → retrieve → chat
```bash
# Ingest a sample doc under ./data/samples
DOC_ID="$(curl -fsS http://127.0.0.1:8000/v1/ingest \
  -H 'Content-Type: application/json' \
  -d '{"source":"samples/gate_sample.md","title":"GATE Sample Notes","exam":"GATE_DA","subject":"Probability","topic":"Bayes Theorem","doc_type":"notes","year":2025}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["document_id"])')"
echo "DOC_ID=$DOC_ID"

# Poll status until it becomes "done"
watch -n 2 "curl -fsS http://127.0.0.1:8000/v1/ingest/$DOC_ID/status; echo"
```

Tip: If you want to ingest a local PDF by filename but also provide a download fallback, pass `source_url`.
```bash
DOC_ID="$(curl -fsS http://127.0.0.1:8000/v1/ingest \
  -H 'Content-Type: application/json' \
  -d '{"source":"my_doc.pdf","source_url":"https://example.com/my_doc.pdf","title":"My Doc","exam":"UPSC","doc_type":"notes","year":2025}' \
  | python -c 'import sys,json; print(json.load(sys.stdin)["document_id"])')"
echo "DOC_ID=$DOC_ID"
```

Once status is `done`:
```bash
# retrieval (debug)
curl -fsS http://127.0.0.1:8000/v1/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"Summarize Bayes theorem and give a simple example.","exam":"GATE_DA","top_k":20,"top_n":8}' \
  | python -m json.tool

# chat (TutorMode must be one of: doubt | practice | pyq)
curl -fsS http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Explain Bayes theorem with an example.","exam":"GATE_DA","mode":"doubt","stream":false}' \
  | python -m json.tool
```

Tip: for faster perceived responses, use streaming:
```bash
curl -N http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Explain Bayes theorem with an example.","exam":"GATE_DA","mode":"doubt","stream":true}'
```

Optional: chat directly with vLLM from your terminal (no RAG)
```bash
cd /path/to/veda
source .venv/bin/activate
python scripts/vllm_chat.py --base-url http://127.0.0.1:8002/v1 --model Qwen/Qwen3-4B
```

## Repository map
- `services/api`       FastAPI API service
- `services/worker`    RQ worker for ingestion (Docling)
- `services/embeddings` Local embedding server (SentenceTransformers; configurable via `EMBEDDINGS_MODEL`)
- `packages/common`    Shared Pydantic schemas + logging helpers
- `packages/rag_core`  Retrieval + prompt building + vector store
- `infra/sql`          DB schema + pgvector extension + index

## Notes
- **GATE** answers emphasize step-by-step solving + formulas.
- **UPSC** answers emphasize structured writing (intro/body/conclusion) + examples.
- Citations use chunk IDs and source titles (no page-level mapping by default).

## Linting & tests
From an active venv:
```bash
ruff check .
pytest -q
```
You can also use `make format`, `make lint`, and `make test`.

---
MIT license for code in this repo. You are responsible for licensing of your study materials.
