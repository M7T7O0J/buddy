from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from exam_tutor_common.logging import get_logger

from ..db import pool

log = get_logger(__name__)


class MemoryService:
    """Conversation memory.

    MVP approach:
    - store all messages
    - for prompting, return last N messages as a compact text block
    - (optional) upgrade later to summary + recent turns
    """

    def __init__(self, max_messages: int = 12):
        self.max_messages = max_messages

    def ensure_conversation(self, *, conversation_id: Optional[UUID], user_id: Optional[UUID]) -> UUID:
        from uuid6 import uuid7
        cid = conversation_id or uuid7()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM conversations WHERE id=%s", (cid,))
                exists = cur.fetchone() is not None
                if not exists:
                    cur.execute(
                        "INSERT INTO conversations (id, user_id) VALUES (%s, %s)",
                        (cid, user_id),
                    )
            conn.commit()
        return cid

    def add_message(self, conversation_id: UUID, role: str, content: str) -> None:
        with pool.connection() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)",
                (conversation_id, role, content),
            )
            conn.commit()

    def get_memory_block(self, conversation_id: UUID) -> str:
        with pool.connection() as conn:
            rows = conn.execute(
                """                    SELECT role, content
                FROM messages
                WHERE conversation_id = %s
                ORDER BY id DESC
                LIMIT %s
                """,
                (conversation_id, self.max_messages),
            ).fetchall()

        rows = list(reversed(rows))
        lines: List[str] = []
        for role, content in rows:
            lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines).strip()
