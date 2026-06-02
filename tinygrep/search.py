from __future__ import annotations

import re
from collections import defaultdict

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


def _query_terms(query: str) -> list[str]:
    tokens = [t for t in _tokenize(query) if len(t) > 2]
    return tokens or _tokenize(query)


def _normalize_space(text: str) -> str:
    return " ".join(text.lower().split())


def _token_coverage_scores(query: str, chunks: list[dict]) -> np.ndarray:
    terms = _query_terms(query)
    if not terms:
        return np.zeros(len(chunks), dtype=np.float32)

    query_set = set(terms)
    scores = np.zeros(len(chunks), dtype=np.float32)
    for i, chunk in enumerate(chunks):
        chunk_tokens = set(_tokenize(chunk["text"]))
        scores[i] = len(query_set & chunk_tokens) / len(query_set)
    return scores


def _path_coverage_scores(query: str, chunks: list[dict]) -> np.ndarray:
    terms = _query_terms(query)
    if not terms:
        return np.zeros(len(chunks), dtype=np.float32)

    query_set = set(terms)
    scores = np.zeros(len(chunks), dtype=np.float32)
    for i, chunk in enumerate(chunks):
        path_tokens = set(_tokenize(chunk["path"]))
        scores[i] = len(query_set & path_tokens) / len(query_set)
    return scores


def _all_terms_present_scores(query: str, chunks: list[dict]) -> np.ndarray:
    terms = _query_terms(query)
    if not terms:
        return np.zeros(len(chunks), dtype=np.float32)

    query_set = set(terms)
    scores = np.zeros(len(chunks), dtype=np.float32)
    for i, chunk in enumerate(chunks):
        chunk_tokens = set(_tokenize(chunk["text"]))
        scores[i] = float(query_set.issubset(chunk_tokens))
    return scores


def _phrase_overlap_scores(query: str, chunks: list[dict]) -> np.ndarray:
    terms = _query_terms(query)
    if not terms:
        return np.zeros(len(chunks), dtype=np.float32)
    if len(terms) == 1:
        return _exact_match_scores(query, chunks)

    phrases = [" ".join(terms[i : i + 2]) for i in range(len(terms) - 1)]
    scores = np.zeros(len(chunks), dtype=np.float32)
    for i, chunk in enumerate(chunks):
        text = _normalize_space(chunk["text"])
        matches = sum(1 for phrase in phrases if phrase in text)
        scores[i] = matches / len(phrases)
    return scores


def _exact_match_scores(query: str, chunks: list[dict]) -> np.ndarray:
    normalized_query = _normalize_space(query)
    if not normalized_query:
        return np.zeros(len(chunks), dtype=np.float32)

    scores = np.zeros(len(chunks), dtype=np.float32)
    for i, chunk in enumerate(chunks):
        text = _normalize_space(chunk["text"])
        path = _normalize_space(chunk["path"])
        scores[i] = float(normalized_query in text or normalized_query in path)
    return scores


def _hybrid_scores(query: str, chunks: list[dict], provider: EmbeddingProvider) -> np.ndarray:
    semantic = _normalize(_semantic_scores(query, chunks, provider))
    keyword = _normalize(_bm25_scores(query, chunks))
    token_coverage = _token_coverage_scores(query, chunks)
    phrase_overlap = _phrase_overlap_scores(query, chunks)
    path_coverage = _path_coverage_scores(query, chunks)
    all_terms = _all_terms_present_scores(query, chunks)
    exact_matches = _exact_match_scores(query, chunks)

    scores = (
        0.35 * semantic
        + 0.15 * keyword
        + 0.20 * token_coverage
        + 0.10 * phrase_overlap
        + 0.05 * path_coverage
        + 0.10 * all_terms
        + 0.20 * exact_matches
    )
    return _normalize(scores)


def _select_top_indices(scores: np.ndarray, chunks: list[dict], top_k: int) -> tuple[list[int], np.ndarray]:
    adjusted = scores.copy()
    remaining = set(range(len(chunks)))
    seen_per_file: dict[str, int] = defaultdict(int)
    selected: list[int] = []

    while remaining and len(selected) < top_k:
        best_idx = max(
            remaining,
            key=lambda i: adjusted[i] - 0.03 * seen_per_file[chunks[i]["path"]],
        )
        path = chunks[best_idx]["path"]
        duplicate_penalty = 0.03 * seen_per_file[path]
        adjusted[best_idx] = max(0.0, adjusted[best_idx] - duplicate_penalty)
        selected.append(best_idx)
        remaining.remove(best_idx)
        seen_per_file[path] += 1

    return selected, adjusted


def search(
    query: str,
    db: Database,
    provider: EmbeddingProvider,
    top_k: int = 10,
    mode: str = "hybrid",
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
    rank_query = free_query if free_query else " ".join(required_phrases) or query

    if mode == "keyword":
        scores = _bm25_scores(rank_query, chunks)
        top_indices = np.argsort(scores)[::-1][:top_k].tolist()
        display_scores = scores
    elif mode == "hybrid":
        scores = _hybrid_scores(rank_query, chunks, provider)
        top_indices, display_scores = _select_top_indices(scores, chunks, top_k)
    else:
        scores = _semantic_scores(rank_query, chunks, provider)
        top_indices = np.argsort(scores)[::-1][:top_k].tolist()
        display_scores = scores

    return [
        {
            "file": chunks[int(i)]["path"],
            "chunk_index": chunks[int(i)]["chunk_index"],
            "score": float(display_scores[i]),
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
