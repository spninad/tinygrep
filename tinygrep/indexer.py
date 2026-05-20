from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

from .config import Config
from .db import Database
from .providers.base import EmbeddingProvider

_CHARS_PER_TOKEN = 4  # rough heuristic for token estimation


def _token_batches(
    texts: list[str], max_items: int, max_tokens: int
) -> Iterator[list[str]]:
    """
    Yield batches that respect both a maximum item count and a maximum
    estimated token count. Uses len(text) / _CHARS_PER_TOKEN as the estimate.
    """
    batch: list[str] = []
    batch_tokens = 0
    for text in texts:
        estimated = max(1, len(text) // _CHARS_PER_TOKEN)
        if batch and (len(batch) >= max_items or batch_tokens + estimated > max_tokens):
            yield batch
            batch = []
            batch_tokens = 0
        batch.append(text)
        batch_tokens += estimated
    if batch:
        yield batch


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks, respecting paragraph boundaries where possible.
    Returns at least one chunk even for short texts.
    """
    paragraphs = re.split(r"\n\n+", text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if not paragraphs:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > chunk_size and current:
            chunk_text_str = "\n\n".join(current)
            chunks.append(chunk_text_str)
            # Carry over a tail of text for overlap
            tail = chunk_text_str[-overlap:] if overlap else ""
            current = [tail] if tail else []
            current_len = len(tail)
        current.append(para)
        current_len += para_len + 2  # +2 for the \n\n separator

    if current:
        chunks.append("\n\n".join(current))

    return chunks or [text]


def walk_folder(folder: Path, extensions: list[str]) -> list[Path]:
    ext_set = {e.lower() for e in extensions}
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in ext_set
        and ".tinygrep" not in p.parts
    )


def index_folder(
    folder: Path,
    root: Path,
    db: Database,
    provider: EmbeddingProvider,
    cfg: Config,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Walk *folder*, embed its files, and store them in the DB.
    Paths stored in the DB are relative to *root* (the index store root),
    so they remain stable no matter which subdirectory index was called from.
    *folder* must be equal to or a subdirectory of *root*.
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

    files = walk_folder(folder, cfg.index.file_extensions)
    stats = {"total": len(files), "skipped": 0, "indexed": 0, "failed": 0}

    # Identify which files actually need embedding
    pending: list[tuple[Path, str, str]] = []  # (path, rel_path, content)
    for file_path in files:
        rel = str(file_path.relative_to(root))
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            stats["failed"] += 1
            continue

        file_hash = sha256(content)
        existing = db.get_document(rel)

        if (
            not force
            and existing
            and existing["hash"] == file_hash
            and existing["provider"] == provider.provider_name
            and existing["model"] == provider.model_id
        ):
            stats["skipped"] += 1
            continue

        pending.append((file_path, rel, content))

    if not pending:
        return stats

    # Collect all chunks across all pending files
    all_chunks: list[tuple[str, str, int, str]] = []  # (rel_path, hash, chunk_idx, text)
    file_hashes: dict[str, str] = {}

    for file_path, rel, content in pending:
        file_hash = sha256(content)
        file_hashes[rel] = file_hash
        chunks = chunk_text(content, cfg.index.chunk_size, cfg.index.chunk_overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append((rel, file_hash, i, chunk))

    # Embed in batches
    import numpy as np

    texts = [c[3] for c in all_chunks]
    embeddings: list[list[float]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=True,
        disable=not verbose,
    ) as progress:
        task = progress.add_task(
            f"Embedding {len(texts)} chunks from {len(pending)} files...",
            total=len(texts),
        )
        for batch in _token_batches(texts, provider.batch_size, provider.max_tokens_per_batch):
            embs = provider.embed(batch, input_type="document")
            embeddings.extend(embs)
            progress.advance(task, len(batch))

    # Write to DB, grouped by file
    emb_arrays = [np.array(e, dtype=np.float32) for e in embeddings]
    by_file: dict[str, list[tuple[int, str, object]]] = {}
    for (rel, fhash, idx, text), emb in zip(all_chunks, emb_arrays):
        by_file.setdefault(rel, []).append((idx, text, emb))

    for rel, chunks_with_emb in by_file.items():
        db.upsert_document(
            path=rel,
            file_hash=file_hashes[rel],
            provider=provider.provider_name,
            model=provider.model_id,
            chunks=chunks_with_emb,
        )
        stats["indexed"] += 1

    return stats
