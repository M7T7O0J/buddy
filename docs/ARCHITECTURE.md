# Architecture

## Runtime request flow (Chat)
1. Client calls `POST /v1/chat` with (message, exam, mode, optional filters).
2. API fetches recent conversation memory from Postgres.
3. API retrieves relevant chunks:
   - embeds query via Embeddings service
   - vector search in Postgres (pgvector) filtered by exam/subject/topic
4. API builds a tutor prompt with grounding policy + retrieved context.
5. API calls vLLM (OpenAI-compatible) and streams tokens back to the client (SSE).
6. API writes messages + telemetry events.

## Ingestion flow
1. Client calls `POST /v1/ingest` with source path/URL + metadata.
2. API inserts `documents` + `ingestion_jobs` and enqueues an RQ job.
3. Worker:
   - Docling converts document → Markdown
   - normalization
   - structure-aware chunking (token-aware)
   - embeddings
   - upsert to `chunks` table
   - update `ingestion_jobs` status
