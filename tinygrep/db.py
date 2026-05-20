from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np


class Database:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY,
                path        TEXT    NOT NULL UNIQUE,
                hash        TEXT    NOT NULL,
                provider    TEXT    NOT NULL,
                model       TEXT    NOT NULL,
                indexed_at  REAL    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY,
                doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                text        TEXT    NOT NULL,
                embedding   BLOB    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
        """)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_document(self, path: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT id, path, hash, provider, model, indexed_at FROM documents WHERE path = ?",
            (path,),
        ).fetchone()
        if row:
            return {"id": row[0], "path": row[1], "hash": row[2],
                    "provider": row[3], "model": row[4], "indexed_at": row[5]}
        return None

    def get_all_documents(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, path, hash, provider, model, indexed_at FROM documents ORDER BY path"
        ).fetchall()
        return [
            {"id": r[0], "path": r[1], "hash": r[2],
             "provider": r[3], "model": r[4], "indexed_at": r[5]}
            for r in rows
        ]

    def get_all_chunks(self, provider: str, model: str) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT c.id, d.path, c.chunk_index, c.text, c.embedding
            FROM chunks c
            JOIN documents d ON c.doc_id = d.id
            WHERE d.provider = ? AND d.model = ?
            ORDER BY d.path, c.chunk_index
            """,
            (provider, model),
        ).fetchall()
        return [
            {
                "id": r[0],
                "path": r[1],
                "chunk_index": r[2],
                "text": r[3],
                "embedding": np.frombuffer(r[4], dtype=np.float32).copy(),
            }
            for r in rows
        ]

    def get_document_embeddings(self, provider: str, model: str) -> dict[str, np.ndarray]:
        """Returns mean-pooled embedding per document path."""
        chunks = self.get_all_chunks(provider, model)
        by_doc: dict[str, list[np.ndarray]] = {}
        for c in chunks:
            by_doc.setdefault(c["path"], []).append(c["embedding"])
        return {path: np.mean(embs, axis=0) for path, embs in by_doc.items()}

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert_document(
        self,
        path: str,
        file_hash: str,
        provider: str,
        model: str,
        chunks: list[tuple[int, str, np.ndarray]],
    ) -> None:
        """chunks: list of (chunk_index, text, embedding_array)"""
        self.conn.execute("DELETE FROM documents WHERE path = ?", (path,))
        cursor = self.conn.execute(
            "INSERT INTO documents (path, hash, provider, model, indexed_at) VALUES (?, ?, ?, ?, ?)",
            (path, file_hash, provider, model, time.time()),
        )
        doc_id = cursor.lastrowid
        self.conn.executemany(
            "INSERT INTO chunks (doc_id, chunk_index, text, embedding) VALUES (?, ?, ?, ?)",
            [
                (doc_id, idx, text, arr.astype(np.float32).tobytes())
                for idx, text, arr in chunks
            ],
        )
        self.conn.commit()

    def delete_document(self, path: str) -> bool:
        cursor = self.conn.execute("DELETE FROM documents WHERE path = ?", (path,))
        self.conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *args) -> None:
        self.close()
