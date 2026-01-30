from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterator, List

import httpx

from exam_tutor_common.logging import get_logger


log = get_logger(__name__)


@dataclass(frozen=True)
class LLMConfig:
    mode: str  # mock | vllm
    base_url: str
    model: str
    api_key: str
    timeout_s: float


class LLMClient:
    """LLM client.

    - `mode=mock`: returns a deterministic stub response (useful for local dev).
    - `mode=vllm`: calls an OpenAI-compatible vLLM server.
    """

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def stream_chat(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        if self.cfg.mode.lower() == "mock":
            yield "(mock) I can answer once vLLM is connected. Retrieved sources are available; please configure LLM_MODE=vllm."
            return

        url = f"{self.cfg.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"} if self.cfg.api_key else {}

        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 600,
            "stream": True,
        }

        with httpx.Client(timeout=self.cfg.timeout_s) as client:
            with client.stream("POST", url, headers=headers, json=payload) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    # OpenAI streaming uses SSE lines: 'data: {...}'
                    if line.startswith("data:"):
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                            delta = obj["choices"][0]["delta"].get("content")
                            if delta:
                                yield delta
                        except Exception:
                            # best-effort parsing; log and continue
                            log.warning("stream_parse_failed", extra={"raw": line[:200]})
                            continue
