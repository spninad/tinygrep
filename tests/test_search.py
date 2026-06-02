from __future__ import annotations

import numpy as np

from tinygrep.db import Database
from tinygrep.providers.base import EmbeddingProvider
from tinygrep.search import _select_top_indices, search


class FakeProvider(EmbeddingProvider):
    def __init__(self, query_embedding: list[float], model: str = "test-model"):
        self.query_embedding = query_embedding
        self._model = model

    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]:
        assert input_type == "query"
        return [self.query_embedding for _ in texts]

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "fake"


def _upsert(
    db: Database,
    path: str,
    chunks: list[tuple[str, list[float]]],
    provider: EmbeddingProvider,
) -> None:
    db.upsert_document(
        path=path,
        file_hash=f"hash:{path}",
        provider=provider.provider_name,
        model=provider.model_id,
        chunks=[
            (idx, text, np.array(embedding, dtype=np.float32))
            for idx, (text, embedding) in enumerate(chunks)
        ],
    )


def test_hybrid_promotes_exact_match(tmp_path):
    provider = FakeProvider([1.0, 0.0])
    db_path = tmp_path / "index.db"

    with Database(db_path) as db:
        _upsert(
            db,
            "semantic.md",
            [("This passage discusses nearby concepts in broad terms only.", [1.0, 0.0])],
            provider,
        )
        _upsert(
            db,
            "exact.md",
            [("Alpha beta gamma is the phrase you are looking for.", [0.72, 0.28])],
            provider,
        )

        semantic_results = search("alpha beta gamma", db, provider, mode="semantic", top_k=2)
        hybrid_results = search("alpha beta gamma", db, provider, mode="hybrid", top_k=2)

    assert semantic_results[0]["file"] == "semantic.md"
    assert hybrid_results[0]["file"] == "exact.md"


def test_required_phrase_filter_is_respected(tmp_path):
    provider = FakeProvider([1.0, 0.0])
    db_path = tmp_path / "index.db"

    with Database(db_path) as db:
        _upsert(
            db,
            "hit.md",
            [("Flow matching helps diffusion models learn transport paths.", [0.9, 0.1])],
            provider,
        )
        _upsert(
            db,
            "miss.md",
            [("Diffusion models can be trained without the required words.", [1.0, 0.0])],
            provider,
        )

        results = search('"flow matching" diffusion', db, provider, mode="hybrid", top_k=5)

    assert [result["file"] for result in results] == ["hit.md"]


def test_soft_file_diversification_reorders_close_scores():
    chunks = [
        {"path": "repeat.md"},
        {"path": "repeat.md"},
        {"path": "other.md"},
    ]
    scores = np.array([0.95, 0.94, 0.93], dtype=np.float32)

    top_indices, adjusted = _select_top_indices(scores, chunks, top_k=3)

    assert top_indices == [0, 2, 1]
    assert adjusted[1] < adjusted[2]
