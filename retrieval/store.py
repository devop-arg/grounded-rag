"""Storage for the hybrid index: SQLite (metadata + FTS5) and numpy (vectors).

pgvector is not available here (it would need sudo / a PostgreSQL extension),
so the two retrieval signals live behind one small interface:

* ``data/index.sqlite``: ``chunks`` table plus an FTS5 table over page title,
  section path and chunk text, ranked with ``bm25()``.
* ``data/vectors.npy`` (float32, L2 normalized) + ``data/vector_ids.json``:
  dense vectors searched with a dot product.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_PATH = DATA_DIR / "index.sqlite"
VECTORS_PATH = DATA_DIR / "vectors.npy"
VECTOR_IDS_PATH = DATA_DIR / "vector_ids.json"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
FTS_TOKENIZER = "porter unicode61 remove_diacritics 2"

SCHEMA = """
CREATE TABLE chunks (
    rowid INTEGER PRIMARY KEY,
    chunk_id TEXT UNIQUE NOT NULL,
    page_file TEXT NOT NULL,
    page_title TEXT NOT NULL,
    section_path TEXT NOT NULL,
    anchor TEXT NOT NULL,
    url TEXT NOT NULL,
    text TEXT NOT NULL,
    n_words INTEGER NOT NULL
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_id UNINDEXED,
    page_title,
    section_path,
    text,
    tokenize='{tokenizer}'
);
"""


class Store(Protocol):
    """Interface shared by index backends."""

    def lexical(self, query: str, k: int) -> list[tuple[str, float]]: ...  # (chunk_id, score)

    def vector(self, qvec: np.ndarray, k: int) -> list[tuple[str, float]]: ...

    def get(self, ids: list[str]) -> list[dict]: ...


def fts_query(query: str) -> str:
    """Turn a user query into a safe FTS5 MATCH expression.

    Every token is wrapped in double quotes (quotes escaped by doubling) and the
    terms are OR-ed.  The raw user string is never passed to ``MATCH``, so
    operators like ``*``, ``-``, ``NEAR`` or stray quotes are treated as text.
    """
    tokens = [token for token in TOKEN_RE.findall(query) if token]
    if not tokens:
        return ""
    quoted = ['"' + token.replace('"', '""') + '"' for token in tokens]
    return " OR ".join(quoted)


class SqliteNumpyStore:
    """SQLite + numpy implementation of :class:`Store`."""

    def __init__(
        self,
        index_path: Path | str = INDEX_PATH,
        vectors_path: Path | str = VECTORS_PATH,
        vector_ids_path: Path | str = VECTOR_IDS_PATH,
    ) -> None:
        self.index_path = Path(index_path)
        self.vectors_path = Path(vectors_path)
        self.vector_ids_path = Path(vector_ids_path)
        if not self.index_path.exists():
            raise FileNotFoundError(f"index not found: {self.index_path} (run `python -m ingest`)")
        self._vectors: np.ndarray | None = None
        self._vector_ids: list[str] | None = None

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        chunks: Sequence[dict],
        vectors: np.ndarray | None = None,
        index_path: Path | str = INDEX_PATH,
        vectors_path: Path | str = VECTORS_PATH,
        vector_ids_path: Path | str = VECTOR_IDS_PATH,
    ) -> "SqliteNumpyStore":
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()

        conn = sqlite3.connect(tmp_path)
        try:
            conn.executescript(SCHEMA.format(tokenizer=FTS_TOKENIZER))
            with conn:
                for rowid, chunk in enumerate(chunks, start=1):
                    conn.execute(
                        "INSERT INTO chunks (rowid, chunk_id, page_file, page_title, section_path, anchor, url, text, n_words)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            rowid,
                            chunk["id"],
                            chunk["page_file"],
                            chunk["page_title"],
                            json.dumps(chunk["section_path"]),
                            chunk["anchor"],
                            chunk["url"],
                            chunk["text"],
                            chunk["n_words"],
                        ),
                    )
                    conn.execute(
                        "INSERT INTO chunks_fts (rowid, chunk_id, page_title, section_path, text)"
                        " VALUES (?, ?, ?, ?, ?)",
                        (
                            rowid,
                            chunk["id"],
                            chunk["page_title"],
                            " > ".join(chunk["section_path"]),
                            chunk["text"],
                        ),
                    )
            conn.execute("INSERT INTO chunks_fts (chunks_fts) VALUES ('optimize')")
            conn.commit()
        finally:
            conn.close()
        tmp_path.replace(index_path)

        if vectors is not None:
            vectors_path = Path(vectors_path)
            vector_ids_path = Path(vector_ids_path)
            vectors_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(vectors_path, np.asarray(vectors, dtype=np.float32))
            vector_ids_path.write_text(json.dumps([chunk["id"] for chunk in chunks]))

        return cls(index_path=index_path, vectors_path=vectors_path, vector_ids_path=vector_ids_path)

    # ------------------------------------------------------------- vector data
    def _load_vectors(self) -> None:
        if self._vectors is not None:
            return
        if not self.vectors_path.exists() or not self.vector_ids_path.exists():
            self._vectors = np.zeros((0, 0), dtype=np.float32)
            self._vector_ids = []
            return
        self._vectors = np.load(self.vectors_path).astype(np.float32, copy=False)
        self._vector_ids = json.loads(self.vector_ids_path.read_text())

    @property
    def size(self) -> int:
        with closing(sqlite3.connect(self.index_path)) as conn:
            (count,) = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(count)

    # ------------------------------------------------------------------ query
    def lexical(self, query: str, k: int) -> list[tuple[str, float]]:
        match = fts_query(query)
        if not match or k <= 0:
            return []
        with closing(sqlite3.connect(self.index_path)) as conn:
            rows = conn.execute(
                "SELECT chunk_id, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ?"
                " ORDER BY bm25(chunks_fts) LIMIT ?",
                (match, k),
            ).fetchall()
        # bm25() returns negative scores, smaller is better: flip the sign.
        return [(chunk_id, -float(score)) for chunk_id, score in rows]

    def vector(self, qvec: np.ndarray, k: int) -> list[tuple[str, float]]:
        self._load_vectors()
        assert self._vectors is not None and self._vector_ids is not None
        if not len(self._vector_ids) or k <= 0:
            return []
        query = np.asarray(qvec, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return []
        query = query / norm
        scores = self._vectors @ query
        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self._vector_ids[i], float(scores[i])) for i in top]

    def get(self, ids: list[str]) -> list[dict]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with closing(sqlite3.connect(self.index_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT chunk_id, page_file, page_title, section_path, anchor, url, text, n_words"
                f" FROM chunks WHERE chunk_id IN ({placeholders})",
                list(ids),
            ).fetchall()
        found = {row["chunk_id"]: _row_to_dict(row) for row in rows}
        return [found[chunk_id] for chunk_id in ids if chunk_id in found]


def _row_to_dict(row: sqlite3.Row) -> dict:
    data = dict(row)
    data["section_path"] = json.loads(data["section_path"])
    return data


def iter_chunks(path: Path | str = CHUNKS_PATH) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
