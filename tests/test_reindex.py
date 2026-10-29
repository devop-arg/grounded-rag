"""``python -m retrieval.reindex``: new files only, batch by batch, resumable."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from retrieval.reindex import ReindexError, build_index
from retrieval.remote import RemoteEmbeddings
from retrieval.store import vector_ids_path_for, vectors_path_for

KEY = "sk-reindex-test"
BASE_URL = "https://llm.example/v1"
MODEL = "vendor/embed-model"


def make_chunks(count: int) -> list[dict]:
    return [
        {
            "id": f"c{index}",
            "page_file": f"page{index}.html",
            "page_title": f"Page {index}",
            "section_path": ["Section"],
            "anchor": f"a{index}",
            "url": f"https://example.test/page{index}.html#a{index}",
            "text": f"text {index}",
            "n_words": 2,
        }
        for index in range(count)
    ]


class FakeEmbeddings(RemoteEmbeddings):
    """Answers every batch with a deterministic vector and counts the passages."""

    def __init__(self, dims: int = 3, fail_at: int | None = None) -> None:
        super().__init__(BASE_URL, KEY, MODEL)
        self.dims = dims
        self.fail_at = fail_at
        self.batches: list[int] = []
        self.passages = 0

    def _post(self, path: str, payload: dict) -> dict:
        size = len(payload["input"])
        if self.fail_at is not None and self.passages >= self.fail_at:
            from retrieval.remote import RemoteError

            raise RemoteError("embeddings request failed with HTTP 500")
        self.batches.append(size)
        self.passages += size
        return {
            "data": [
                {"index": index, "embedding": [float(self.passages - size + index + 1), 0.5, 0.25][: self.dims]}
                for index in range(size)
            ],
            "usage": {"prompt_tokens": size * 10, "cost": 0.0001 * size},
        }


def test_reindex_writes_new_files_and_leaves_the_local_index_alone(tmp_path) -> None:
    local_vectors = tmp_path / "vectors.npy"
    local_ids = tmp_path / "vector_ids.json"
    np.save(local_vectors, np.ones((2, 2), dtype=np.float32))
    local_ids.write_text(json.dumps(["keep1", "keep2"]))
    before = (local_vectors.read_bytes(), local_ids.read_bytes())

    summary = build_index(make_chunks(5), FakeEmbeddings(), name="alt", data_dir=tmp_path, batch_size=2, log=lambda message: None)

    assert summary["chunks"] == 5 and summary["dim"] == 3
    assert summary["batches"] == 3
    assert summary["tokens"] == 50 and summary["cost_usd"] == pytest.approx(0.0005)
    vectors_path = vectors_path_for("alt", tmp_path)
    ids_path = vector_ids_path_for("alt", tmp_path)
    assert np.load(vectors_path).shape == (5, 3)
    assert json.loads(ids_path.read_text()) == [f"c{index}" for index in range(5)]
    # The local pair is byte-identical: nothing was touched or deleted.
    assert (local_vectors.read_bytes(), local_ids.read_bytes()) == before
    # The partial file and its progress file are gone after a clean finish.
    assert not (tmp_path / "vectors-alt.part.npy").exists()
    assert not (tmp_path / "vectors-alt.progress.json").exists()


def test_an_interrupted_run_resumes_without_paying_for_the_same_passages(tmp_path) -> None:
    chunks = make_chunks(6)
    broken = FakeEmbeddings(fail_at=4)
    with pytest.raises(ReindexError, match="resume"):
        build_index(chunks, broken, name="alt", data_dir=tmp_path, batch_size=2, sleep=lambda seconds: None, log=lambda message: None)
    assert (tmp_path / "vectors-alt.part.npy").exists()
    progress = json.loads((tmp_path / "vectors-alt.progress.json").read_text())
    assert progress["done"] == 4
    assert not vectors_path_for("alt", tmp_path).exists()

    resumed = FakeEmbeddings()
    summary = build_index(chunks, resumed, name="alt", data_dir=tmp_path, batch_size=2, log=lambda message: None)
    assert summary["chunks"] == 6
    # Only the two remaining chunks were embedded (plus the failed attempts of the first run).
    assert resumed.passages == 2
    assert np.load(vectors_path_for("alt", tmp_path)).shape == (6, 3)


def test_a_different_model_restarts_instead_of_resuming(tmp_path) -> None:
    chunks = make_chunks(4)
    with pytest.raises(ReindexError):
        build_index(chunks, FakeEmbeddings(fail_at=2), name="alt", data_dir=tmp_path, batch_size=2, sleep=lambda seconds: None, log=lambda message: None)

    other = FakeEmbeddings()
    other.model = "vendor/other-model"
    summary = build_index(chunks, other, name="alt", data_dir=tmp_path, batch_size=2, log=lambda message: None)
    assert summary["chunks"] == 4
    assert other.passages == 4  # started from scratch


def test_a_batch_that_fails_every_attempt_keeps_the_partial_index(tmp_path) -> None:
    with pytest.raises(ReindexError):
        build_index(make_chunks(2), FakeEmbeddings(fail_at=0), name="alt", data_dir=tmp_path, batch_size=1, sleep=lambda seconds: None, log=lambda message: None)
    assert not vectors_path_for("alt", tmp_path).exists()
