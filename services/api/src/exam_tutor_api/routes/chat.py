from __future__ import annotations

import json
from typing import Iterator

from fastapi import APIRouter, Request
from fastapi import HTTPException
import httpx
from sse_starlette.sse import EventSourceResponse

from exam_tutor_common.schemas import ChatRequest, ChatResponse
from exam_tutor_common.logging import get_logger

from ..services.rag_service import get_rag_service

log = get_logger(__name__)
router = APIRouter()


def _sse(event: str, data: dict) -> dict:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


@router.post("/v1/chat")
def chat(req: ChatRequest, request: Request):
    rag = get_rag_service()
    # Ensure conversation
    conversation_id = rag.memory.ensure_conversation(
        conversation_id=req.conversation_id,
        user_id=req.user_id,
    )

    # Build prompt from existing memory (exclude the current user turn).
    built = rag.build(req=req, conversation_id=conversation_id)
    used_chunks = built.used_chunks

    # Persist user message after prompt assembly to avoid duplication in memory block.
    rag.memory.add_message(conversation_id, "user", req.message)

    def build_final(answer: str) -> ChatResponse:
        # naive citations: include top used chunks (you can improve this later by parsing [chunk:..] tags)
        citations = [
            {"chunk_id": c.chunk_id, "source_title": c.source_title}
            for c in used_chunks[:5]
        ]
        return ChatResponse(
            conversation_id=conversation_id,
            answer=answer,
            citations=citations,
            used_chunks=used_chunks,
        )

    if not req.stream:
        try:
            answer = "".join(rag.stream_answer(prompt_messages=built.prompt_messages)).strip()
        except httpx.HTTPStatusError as e:
            detail = e.response.text if e.response is not None else str(e)
            raise HTTPException(status_code=502, detail=f"LLM request failed: {detail}") from e
        rag.memory.add_message(conversation_id, "assistant", answer)
        return build_final(answer)

    def event_generator() -> Iterator[dict]:
        # stream tokens
        answer_parts = []
        try:
            for delta in rag.stream_answer(prompt_messages=built.prompt_messages):
                answer_parts.append(delta)
                yield _sse("token", {"delta": delta})
        except httpx.HTTPStatusError as e:
            detail = e.response.text if e.response is not None else str(e)
            yield _sse("error", {"detail": f"LLM request failed: {detail}"})
            return

        answer = "".join(answer_parts).strip()

        # persist assistant message
        rag.memory.add_message(conversation_id, "assistant", answer)

        final = build_final(answer)
        yield _sse("final", final.model_dump(mode="json"))

    return EventSourceResponse(event_generator())
