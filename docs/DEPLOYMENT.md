# Deployment (MVP)

Recommended split:
- GPU VM: vLLM inference server (services/inference)
- CPU service: API + Worker
- Managed DB: Supabase Postgres + pgvector (or self-host Postgres)
- Managed Redis: Upstash/Redis Cloud (or self-host)

Minimal:
- Use docker-compose on a single VM for early MVP users.

Production:
- Put API behind a reverse proxy (Caddy/Nginx)
- Enable HTTPS
- Add rate limiting and auth
- Add observability (Sentry + OpenTelemetry)
