# API Contracts

## POST /v1/chat
Streams an SSE response with tokens and a final JSON payload.

Request:
- message: string (required)
- conversation_id: uuid (optional)
- exam: string (optional)
- mode: doubt|practice|pyq (optional)
- subject/topic/year/doc_type (optional filters)

Response (SSE events):
- event: token, data: {"delta":"..."}
- event: final, data: {"answer":"...","citations":[...],"used_chunks":[...],...}

## POST /v1/ingest
Request:
- source: string (local path under ./data or URL)
- metadata fields: title, exam, subject, topic, doc_type, year
Response:
- document_id, status

## POST /v1/retrieve
Debug endpoint returning retrieved chunks.
