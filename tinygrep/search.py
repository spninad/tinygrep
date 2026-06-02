from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from .db import Database
from .providers.base import EmbeddingProvider


def cosine_similarity_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """query: (D,), matrix: (N, D) -> scores: (N,)"""
    q_norm = np.linalg.norm(query)
    m_norms = np.linalg.norm(matrix, axis=1)
    denom = m_norms * q_norm
    denom = np.where(denom == 0, 1e-9, denom)
    return (matrix @ query) / denom


def _parse_query(query: str) -> tuple[list[str], str]:
    """Split query into (required_phrases, free_text).

    Quoted substrings become required phrases; everything else is free text
    for BM25/semantic ranking.
    """
    phrases = re.findall(r'"([^"]+)"', query)
    free = re.sub(r'"[^"]+"', "", query).strip()
    return phrases, free


def _apply_required_phrases(chunks: list[dict], phrases: list[str]) -> list[dict]:
    if not phrases:
        return chunks
    lower_phrases = [p.lower() for p in phrases]
    return [c for c in chunks if all(p in c["text"].lower() for p in lower_phrases)]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]; return zeros if all scores are equal."""
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def search(
    query: str,
    db: Database,
    provider: EmbeddingProvider,
    top_k: int = 10,
    mode: str = "semantic",
    folder_prefix: str | None = None,
) -> list[dict]:
    """
    Returns top-k results as list of dicts with keys:
        file, chunk_index, score, snippet, text

    mode: "semantic" | "keyword" | "hybrid"
    folder_prefix: if set, only chunks whose stored path starts with this
                   prefix (relative to the index root) are considered.
    """
    chunks = db.get_all_chunks(provider.provider_name, provider.model_id)
    if folder_prefix:
        prefix = folder_prefix.rstrip("/") + "/"
        chunks = [c for c in chunks if c["path"].startswith(prefix)]
    if not chunks:
        return []

    required_phrases, free_query = _parse_query(query)
    chunks = _apply_required_phrases(chunks, required_phrases)
    if not chunks:
        return []

    # Use free_query for ranking when phrases were present; fall back to full query.
    rank_query = free_query if free_query else query

    if mode == "keyword":
        scores = _bm25_scores(rank_query, chunks)
    elif mode == "hybrid":
        sem = _semantic_scores(query, chunks, provider)
        kw = _bm25_scores(rank_query, chunks)
        scores = (_normalize(sem) + _normalize(kw)) / 2.0
    else:
        scores = _semantic_scores(query, chunks, provider)

    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            "file": chunks[int(i)]["path"],
            "chunk_index": chunks[int(i)]["chunk_index"],
            "score": float(scores[i]),
            "snippet": _snippet(chunks[int(i)]["text"]),
            "text": chunks[int(i)]["text"],
        }
        for i in top_indices
    ]


def _semantic_scores(query: str, chunks: list[dict], provider: EmbeddingProvider) -> np.ndarray:
    embeddings = np.stack([c["embedding"] for c in chunks])
    query_emb = np.array(provider.embed([query], input_type="query")[0], dtype=np.float32)
    return cosine_similarity_matrix(query_emb, embeddings)


def _bm25_scores(query: str, chunks: list[dict]) -> np.ndarray:
    tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return np.array(bm25.get_scores(_tokenize(query)), dtype=np.float32)


def _snippet(text: str, max_chars: int = 280) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " …"
