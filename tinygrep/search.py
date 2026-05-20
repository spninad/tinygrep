from __future__ import annotations

import numpy as np

from .db import Database
from .providers.base import EmbeddingProvider


def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """query: (D,), matrix: (N, D) -> scores: (N,)"""
    q_norm = np.linalg.norm(query)
    m_norms = np.linalg.norm(matrix, axis=1)
    denom = m_norms * q_norm
    denom = np.where(denom == 0, 1e-9, denom)
    return (matrix @ query) / denom


def search(
    query: str,
    db: Database,
    provider: EmbeddingProvider,
    top_k: int = 10,
) -> list[dict]:
    """
    Returns top-k results as list of dicts with keys:
        file, chunk_index, score, snippet
    """
    chunks = db.get_all_chunks(provider.provider_name, provider.model_id)
    if not chunks:
        return []

    embeddings = np.stack([c["embedding"] for c in chunks])  # (N, D)
    query_emb = np.array(provider.embed([query], input_type="query")[0], dtype=np.float32)

    scores = cosine_similarity_matrix(query_emb, embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        c = chunks[int(idx)]
        results.append(
            {
                "file": c["path"],
                "chunk_index": c["chunk_index"],
                "score": float(scores[idx]),
                "snippet": _snippet(c["text"]),
                "text": c["text"],
            }
        )
    return results


def _snippet(text: str, max_chars: int = 280) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " …"
