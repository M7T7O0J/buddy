from __future__ import annotations

import os
from typing import Any, Dict, List, Union

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


class EmbeddingsRequest(BaseModel):
    model: str
    input: Union[str, List[str]]


app = FastAPI(title="Embeddings Server", version="0.1.0")


@app.on_event("startup")
def _load_model() -> None:
    model_name = os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-m3")
    # Load once on startup (may take time on first run due to HF download)
    app.state.model_name = model_name
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "Failed to import `sentence_transformers` / `transformers`.\n"
            "Common cause on Windows: mismatched PyTorch and Torchvision wheels.\n"
            "Fix by reinstalling them together (pick ONE):\n"
            "  CPU: python -m pip uninstall -y torch torchvision torchaudio; "
            "python -m pip install \"torch>=2.6\" torchvision torchaudio\n"
            "  CUDA: python -m pip uninstall -y torch torchvision torchaudio; "
            "python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126\n"
        ) from e

    # Transformers may refuse to load `.bin` checkpoints on older torch versions due to CVE hardening.
    # We try safetensors-only first (when the model provides it), then fall back to the default loader.
    try:
        app.state.model = SentenceTransformer(model_name, model_kwargs={"use_safetensors": True})
        return
    except OSError:
        # Model doesn't provide safetensors in the expected location.
        pass
    except ValueError as e:
        msg = str(e).lower()
        if "vulnerability" in msg or "upgrade torch" in msg or "torch.load" in msg:
            raise RuntimeError(
                "Embeddings model load blocked by torch/transformers safety checks. "
                "Upgrade torch to >=2.6 (recommended) or switch to a model that supports safetensors."
            ) from e
        raise

    try:
        app.state.model = SentenceTransformer(model_name)
    except ValueError as e:
        msg = str(e).lower()
        if "vulnerability" in msg or "upgrade torch" in msg or "torch.load" in msg:
            raise RuntimeError(
                "Embeddings model load blocked by torch/transformers safety checks. "
                "Upgrade torch to >=2.6 (recommended) or switch to a model that supports safetensors."
            ) from e
        raise


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.clip(n, 1e-12, None)
    return v / n


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingsRequest) -> Dict[str, Any]:
    model = app.state.model
    inp = req.input
    texts = [inp] if isinstance(inp, str) else inp
    embs = model.encode(texts, normalize_embeddings=True)  # normalize for cosine
    if isinstance(embs, list):
        embs = np.array(embs)
    embs = np.asarray(embs, dtype=np.float32)

    data = []
    for i, e in enumerate(embs):
        data.append({"object": "embedding", "index": i, "embedding": e.tolist()})

    return {"object": "list", "model": req.model, "data": data}
