"""Local embeddings with fastembed (multilingual-e5-small).

The box has ~1 GB of free RAM and no swap, so the model runs with two threads
and small batches, and its files are cached under ``data/models/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = DATA_DIR / "models"

MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
#: fastembed's registry only ships e5 *large*. The small checkpoint is registered
#: from its ONNX export on the Hub; the int8 file is used because the box has
#: ~1 GB of free RAM, no swap, and the fp32 file peaks at ~870 MB of RSS.
MODEL_SOURCE = "Xenova/multilingual-e5-small"
MODEL_FILE = "onnx/model_quantized.onnx"

#: Batches are small on purpose: 32 chunks peak at ~1.5 GB of RSS and this box
#: has ~1 GB free with no swap (measured: 4 -> 670 MB, 8 -> 846 MB, 32 -> 1577 MB).
BATCH_SIZE = 4
THREADS = 2
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


def passage_text(page_title: str, section_path: Sequence[str], text: str) -> str:
    """Text embedded for a chunk (title + breadcrumb + body, prefixed by caller)."""
    path = " > ".join(section_path)
    return f"{page_title} — {path}\n{text}"


class Embedder:
    """Thin wrapper around ``fastembed.TextEmbedding``."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        cache_dir: Path | str = MODELS_DIR,
        threads: int = THREADS,
        batch_size: int = BATCH_SIZE,
        show_download_progress: bool = False,
    ) -> None:
        from fastembed import TextEmbedding
        from fastembed.common.model_description import ModelSource, PoolingType

        # fastembed's registry ships multilingual-e5 only in "large"; the small
        # checkpoint is registered here from its official ONNX export.
        registered = {model.model.lower() for model in TextEmbedding._list_supported_models()}
        if model_name.lower() not in registered:
            TextEmbedding.add_custom_model(
                model=model_name,
                pooling=PoolingType.MEAN,
                normalization=True,
                sources=ModelSource(hf=MODEL_SOURCE),
                dim=EMBEDDING_DIM,
                model_file=MODEL_FILE,
                description="Multilingual E5 small, 384 dims, 512 token input (int8 ONNX)",
                license="mit",
            )

        self.model_name = model_name
        self.batch_size = batch_size
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir),
            threads=threads,
            show_download_progress=show_download_progress,
            enable_cpu_mem_arena=False,
        )

    def _encode(self, texts: Iterable[str]) -> np.ndarray:
        vectors = [np.asarray(vector, dtype=np.float32) for vector in self.model.embed(list(texts), batch_size=self.batch_size)]
        if not vectors:
            return np.zeros((0, 0), dtype=np.float32)
        matrix = np.vstack(vectors)
        return l2_normalize(matrix)

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(PASSAGE_PREFIX + text for text in texts)

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode([QUERY_PREFIX + query])[0]


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def chunk_documents(chunks: Iterable[dict]) -> list[str]:
    return [passage_text(chunk["page_title"], chunk["section_path"], chunk["text"]) for chunk in chunks]
