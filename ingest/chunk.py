"""Split parsed pages into retrieval chunks.

Pure, testable functions: they take :class:`ingest.parse.Page` /
:class:`ingest.parse.Unit` objects and return plain dicts ready for
``chunks.jsonl``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from .parse import Block, Page, Unit

DOCS_BASE = "https://www.postgresql.org/docs/17/"
MIN_WORDS = 300
MAX_WORDS = 600


def block_words(block: Block) -> int:
    return len(block.text.split())


def blocks_words(blocks: Sequence[Block]) -> int:
    return sum(block_words(block) for block in blocks)


def blocks_text(blocks: Sequence[Block]) -> str:
    return "\n\n".join(block.text for block in blocks)


def _new_group(unit: Unit) -> dict:
    return {
        "anchor": unit.anchor,
        "path": list(unit.path),
        "blocks": list(unit.blocks),
        "words": unit.n_words,
    }


def group_units(units: Iterable[Unit], min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> list[dict]:
    """Merge consecutive small units of one page (anchor of the first is kept)."""
    groups: list[dict] = []
    current: dict | None = None
    for unit in units:
        if current is None:
            current = _new_group(unit)
            continue
        if (current["words"] < min_words or unit.n_words < min_words) and current["words"] + unit.n_words <= max_words:
            current["blocks"].extend(unit.blocks)
            current["words"] += unit.n_words
        else:
            groups.append(current)
            current = _new_group(unit)
    if current is not None:
        groups.append(current)
    return groups


def split_blocks(
    blocks: Sequence[Block], min_words: int = MIN_WORDS, max_words: int = MAX_WORDS
) -> list[tuple[list[Block], int]]:
    """Split a block list on block boundaries, never inside a ``pre`` block."""
    pieces: list[list] = []
    current: list[Block] = []
    current_words = 0
    for block in blocks:
        words = block_words(block)
        if not current:
            current, current_words = [block], words
            continue
        if current_words + words <= max_words:
            current.append(block)
            current_words += words
        else:
            pieces.append([current, current_words])
            current, current_words = [block], words
    if current:
        pieces.append([current, current_words])

    # A short tail merged back into the previous piece keeps chunks near target.
    if len(pieces) > 1 and pieces[-1][1] < min_words and pieces[-2][1] + pieces[-1][1] <= max_words:
        tail = pieces.pop()
        pieces[-1][0].extend(tail[0])
        pieces[-1][1] += tail[1]

    return [(blocks_, words) for blocks_, words in pieces]


def chunk_id(page_file: str, anchor: str, n: int) -> str:
    return f"{page_file}#{anchor}#{n}"


def chunk_url(page_file: str, anchor: str) -> str:
    base = f"{DOCS_BASE}{page_file}"
    return f"{base}#{anchor}" if anchor else base


def page_chunks(page: Page, min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> list[dict]:
    """Turn one parsed page into a list of chunk dicts."""
    chunks: list[dict] = []
    counters: dict[str, int] = defaultdict(int)
    for group in group_units(page.units, min_words=min_words, max_words=max_words):
        for blocks, words in split_blocks(group["blocks"], min_words=min_words, max_words=max_words):
            text = blocks_text(blocks)
            if not text:
                continue
            n = counters[group["anchor"]]
            counters[group["anchor"]] = n + 1
            chunks.append(
                {
                    "id": chunk_id(page.page_file, group["anchor"], n),
                    "page_file": page.page_file,
                    "page_title": page.page_title,
                    "section_path": group["path"],
                    "anchor": group["anchor"],
                    "url": chunk_url(page.page_file, group["anchor"]),
                    "text": text,
                    "n_words": words,
                }
            )
    return chunks
