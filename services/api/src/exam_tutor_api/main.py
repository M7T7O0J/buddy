from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from exam_tutor_common.logging import configure_logging

from .middleware import RequestIDMiddleware
from .settings import settings
from .routes.health import router as health_router
from .routes.chat import router as chat_router
from .routes.retrieve import router as retrieve_router
from .routes.ingest import router as ingest_router
from .routes.feedback import router as feedback_router

configure_logging()

app = FastAPI(title="Exam Tutor MVP", version="0.1.0")

@app.get("/")
def root():
    return {"status": "ok", "health": "/health", "docs": "/docs"}

app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"] ,
)

app.include_router(health_router)
app.include_router(chat_router)
app.include_router(retrieve_router)
app.include_router(ingest_router)
app.include_router(feedback_router)
