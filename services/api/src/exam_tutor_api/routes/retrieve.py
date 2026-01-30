from __future__ import annotations

from fastapi import APIRouter
from fastapi import HTTPException

from exam_tutor_common.schemas import RetrieveRequest, RetrieveResponse

from ..services.rag_service import get_rag_service
from ..settings import settings

router = APIRouter()


@router.post("/v1/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    rag = get_rag_service()
    try:
        return rag.retriever.retrieve(req)
    except Exception as e:
        if settings.APP_ENV.lower() in {"prod", "production"}:
            raise HTTPException(status_code=500, detail="retrieve_failed") from e
        msg = str(e).strip().replace("\n", " ")
        if len(msg) > 400:
            msg = msg[:400] + "…"
        raise HTTPException(status_code=500, detail=f"retrieve_failed: {type(e).__name__}: {msg}") from e
