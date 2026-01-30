CREATE EXTENSION IF NOT EXISTS vector;

-- Users
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY,
  email TEXT UNIQUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Conversations and messages
CREATE TABLE IF NOT EXISTS conversations (
  id UUID PRIMARY KEY,
  user_id UUID NULL REFERENCES users(id) ON DELETE SET NULL,
  title TEXT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
  id BIGSERIAL PRIMARY KEY,
  conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('system','user','assistant')),
  content TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Documents and ingestion jobs
CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY,
  source TEXT NOT NULL,
  title TEXT NOT NULL,
  exam TEXT NOT NULL,
  subject TEXT NULL,
  topic TEXT NULL,
  doc_type TEXT NULL,
  year INT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
  document_id UUID PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
  status TEXT NOT NULL CHECK (status IN ('queued','running','done','failed')),
  error TEXT NULL,
  started_at TIMESTAMPTZ NULL,
  finished_at TIMESTAMPTZ NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Chunks (pgvector)
CREATE TABLE IF NOT EXISTS chunks (
  id BIGSERIAL PRIMARY KEY,
  document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index INT NOT NULL,
  content TEXT NOT NULL,
  token_count INT NOT NULL,
  embedding VECTOR(384) NOT NULL,
  exam TEXT NOT NULL,
  subject TEXT NULL,
  topic TEXT NULL,
  doc_type TEXT NULL,
  year INT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  content_hash TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(document_id, chunk_index)
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS chunks_exam_idx ON chunks (exam);
CREATE INDEX IF NOT EXISTS chunks_subject_idx ON chunks (subject);
CREATE INDEX IF NOT EXISTS chunks_topic_idx ON chunks (topic);
CREATE INDEX IF NOT EXISTS chunks_doc_type_idx ON chunks (doc_type);
CREATE INDEX IF NOT EXISTS chunks_year_idx ON chunks (year);

-- Retrieval helpers (optional but recommended)
-- Speeds up excluding chunks via metadata.tags and applying quality_score thresholds.
CREATE INDEX IF NOT EXISTS chunks_metadata_tags_gin ON chunks USING gin ((metadata->'tags'));
CREATE INDEX IF NOT EXISTS chunks_metadata_quality_score_idx ON chunks (((metadata->>'quality_score')::double precision));

-- Optional: lexical retrieval support (Postgres FTS). Enable in code if/when you add hybrid search.
CREATE INDEX IF NOT EXISTS chunks_content_fts_idx ON chunks USING gin (to_tsvector('simple', content));

-- Vector index (cosine distance)
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
  ON chunks USING hnsw (embedding vector_cosine_ops);

-- Events (feedback, telemetry)
CREATE TABLE IF NOT EXISTS events (
  id BIGSERIAL PRIMARY KEY,
  user_id UUID NULL REFERENCES users(id) ON DELETE SET NULL,
  conversation_id UUID NULL REFERENCES conversations(id) ON DELETE SET NULL,
  event_type TEXT NOT NULL,
  payload JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
