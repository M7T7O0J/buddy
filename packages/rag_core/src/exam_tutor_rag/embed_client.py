from __future__ import annotations

import httpx


class EmbeddingsClient:
    def __init__(self, base_url: str, model: str, timeout_s: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout_s
        self._client = httpx.Client(timeout=self.timeout)

    def close(self) -> None:
        self._client.close()

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": texts}
        r = self._client.post(f"{self.base_url}/v1/embeddings", json=payload)
        r.raise_for_status()
        data = r.json()
        # server returns [{"index": i, "embedding": ...}, ...]
        embs = [None] * len(texts)
        for item in data.get("data", []):
            idx = int(item.get("index"))
            if 0 <= idx < len(embs):
                embs[idx] = item.get("embedding")

        missing = [i for i, e in enumerate(embs) if e is None]
        if missing:
            raise RuntimeError(
                f"embeddings response missing {len(missing)} item(s): indices={missing}; status={r.status_code}"
            )

        return embs  # type: ignore[return-value]
