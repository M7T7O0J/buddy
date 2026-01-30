from __future__ import annotations

from fastapi import APIRouter

from exam_tutor_common.schemas import FeedbackRequest

from ..db import pool

router = APIRouter()


@router.post("/v1/feedback")
def feedback(req: FeedbackRequest):
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO events (user_id, conversation_id, event_type, payload) VALUES (%s, %s, %s, %s::jsonb)",
            (req.user_id, req.conversation_id, "feedback", req.model_dump_json()),
        )
        conn.commit()
    return {"status": "ok"}
