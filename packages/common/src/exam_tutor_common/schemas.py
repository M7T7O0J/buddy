from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from .constants import TutorMode


class Citation(BaseModel):
    chunk_id: int
    source_title: str
    note: Optional[str] = None


class RetrievedChunk(BaseModel):
    chunk_id: int
    document_id: UUID
    source_title: str
    content: str
    token_count: Optional[int] = None
    score: float
    exam: str
    subject: Optional[str] = None
    topic: Optional[str] = None
    doc_type: Optional[str] = None
    year: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrieveRequest(BaseModel):
    query: str
    exam: Optional[str] = None
    subject: Optional[str] = None
    topic: Optional[str] = None
    doc_type: Optional[str] = None
    year: Optional[int] = None
    top_k: int = 20
    top_n: int = 8


class RetrieveResponse(BaseModel):
    chunks: List[RetrievedChunk]


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    exam: Optional[str] = None
    mode: TutorMode = TutorMode.doubt
    language: str = "en"

    # optional retrieval filters
    subject: Optional[str] = None
    topic: Optional[str] = None
    doc_type: Optional[str] = None
    year: Optional[int] = None

    stream: bool = True


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    used_chunks: List[RetrievedChunk] = Field(default_factory=list)
    conversation_id: UUID


class IngestRequest(BaseModel):
    source: str  # local path under ./data or URL
    source_url: Optional[str] = None  # optional: download URL if local file is missing
    title: str
    exam: str
    subject: Optional[str] = None
    topic: Optional[str] = None
    doc_type: Optional[str] = None
    year: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: UUID
    status: str


class IngestStatusResponse(BaseModel):
    document_id: UUID
    status: str
    error: Optional[str] = None


class FeedbackRequest(BaseModel):
    conversation_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    rating: int = Field(ge=0, le=1, description="0=bad, 1=good")
    notes: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
