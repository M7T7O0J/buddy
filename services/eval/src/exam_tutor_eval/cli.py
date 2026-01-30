from __future__ import annotations

import json
from pathlib import Path
from typing import List

import typer

from exam_tutor_common.schemas import RetrieveRequest
from exam_tutor_rag.embed_client import EmbeddingsClient
from exam_tutor_rag.vector_store import PgVectorStore
from exam_tutor_rag.retrieve import Retriever

from .retrieval_metrics import recall_at_k, mrr


app = typer.Typer(add_completion=False)


@app.command()
def retrieval(
    gold_path: str = typer.Option(..., help="Path to gold.jsonl"),
    database_url: str = typer.Option(..., envvar="DATABASE_URL"),
    embeddings_base_url: str = typer.Option(..., envvar="EMBEDDINGS_BASE_URL"),
    embeddings_model: str = typer.Option("BAAI/bge-m3", envvar="EMBEDDINGS_MODEL"),
    top_k: int = typer.Option(20),
    top_n: int = typer.Option(8),
    min_recall_at_5: float = typer.Option(0.5),
):
    """Compute retrieval metrics against a gold set.

    gold.jsonl format (one per line):
    {"query":"...","exam":"GATE_DA","expected_chunk_ids":[1,2,3]}
    """
    embedder = EmbeddingsClient(embeddings_base_url, embeddings_model, timeout_s=60.0)
    store = PgVectorStore(database_url)
    retriever = Retriever(embedder=embedder, store=store)

    ranks: List[int] = []
    total = 0

    with Path(gold_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj["query"]
            exam = obj.get("exam")
            expected = set(obj.get("expected_chunk_ids", []))

            res = retriever.retrieve(RetrieveRequest(query=q, exam=exam, top_k=top_k, top_n=top_n))
            got = [c.chunk_id for c in res.chunks]

            rank = 0
            for i, cid in enumerate(got, start=1):
                if cid in expected:
                    rank = i
                    break
            ranks.append(rank)
            total += 1

    r5 = recall_at_k(ranks, 5)
    score_mrr = mrr(ranks)

    typer.echo(json.dumps({"count": total, "recall@5": r5, "mrr": score_mrr}, indent=2))

    # CI-friendly exit
    if r5 < min_recall_at_5:
        raise typer.Exit(code=2)
