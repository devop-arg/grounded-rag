"""Parsing and chunking tests (no network, no model)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingest.chunk import chunk_id, page_chunks, split_blocks
from ingest.parse import Block, Page, Unit, parse_html

FIXTURE = Path(__file__).parent / "fixtures" / "sample.html"


@pytest.fixture(scope="module")
def page() -> Page:
    parsed = parse_html(FIXTURE.read_text(encoding="utf-8"), "sample.html")
    assert parsed is not None
    return parsed


def test_page_title(page: Page) -> None:
    assert page.page_title == "Test Page"


def test_units_are_h2_and_h3_sections(page: Page) -> None:
    assert [unit.anchor for unit in page.units] == ["TEST-SECT1", "TEST-SECT2"]


def test_section_path(page: Page) -> None:
    first, second = page.units
    assert first.path == ["1. Test Section"]
    assert second.path == ["1. Test Section", "1.1. Small Sub"]


def test_h4_stays_inside_h3(page: Page) -> None:
    text = "\n".join(block.text for block in page.units[1].blocks)
    assert "1.1.1. Deep Sub" in text
    assert "Deeply nested text stays inside the h3 chunk." in text
    assert "This paragraph explains the first behaviour" not in text


def test_navigation_and_boilerplate_removed(page: Page) -> None:
    text = "\n".join(block.text for unit in page.units for block in unit.blocks)
    for noise in ("Table of Contents", "Submit correction", "About PostgreSQL", "Previous", "Copyright"):
        assert noise not in text


def test_code_block_kept(page: Page) -> None:
    code = [block for block in page.units[0].blocks if block.kind == "code"]
    assert len(code) == 1
    assert code[0].text.splitlines()[0] == "CREATE INDEX test_idx ON test_table (id)"
    assert "    WHERE id > 100;" in code[0].text


def test_table_cells_joined_with_pipe(page: Page) -> None:
    text = "\n".join(block.text for block in page.units[0].blocks)
    assert "Column | Meaning" in text
    assert "id | row identifier" in text


def test_small_page_is_one_chunk(page: Page) -> None:
    chunks = page_chunks(page)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["id"] == "sample.html#TEST-SECT1#0"
    assert chunk["anchor"] == "TEST-SECT1"
    assert chunk["url"] == "https://www.postgresql.org/docs/17/sample.html#TEST-SECT1"
    assert chunk["section_path"] == ["1. Test Section"]
    assert chunk["n_words"] == len(chunk["text"].split())


def test_ids_are_stable(page: Page) -> None:
    assert [chunk["id"] for chunk in page_chunks(page)] == [chunk["id"] for chunk in page_chunks(page)]


def test_chunk_id_format() -> None:
    assert chunk_id("dml.html", "DML-INSERT", 2) == "dml.html#DML-INSERT#2"


def make_unit(anchor: str, paragraphs: int, words: int, code_words: int = 0) -> Unit:
    blocks = [Block("text", " ".join(f"w{i}" for i in range(words - 1)) + " end") for _ in range(paragraphs)]
    if code_words:
        blocks.append(Block("code", "SELECT " + " ".join(f"c{i}" for i in range(code_words))))
    return Unit(anchor=anchor, title=f"Section {anchor}", path=[f"Section {anchor}"], blocks=blocks)


def test_large_section_is_split_on_block_boundaries() -> None:
    unit = make_unit("BIG", paragraphs=40, words=50)  # 2000 words
    page = Page(page_file="big.html", page_title="Big", units=[unit])
    chunks = page_chunks(page)

    assert len(chunks) >= 4
    assert all(chunk["n_words"] <= 600 for chunk in chunks)
    assert all(chunk["anchor"] == "BIG" for chunk in chunks)
    assert [chunk["id"] for chunk in chunks] == [f"big.html#BIG#{n}" for n in range(len(chunks))]
    # every word is kept, nothing duplicated
    assert sum(chunk["n_words"] for chunk in chunks) == 2000


def test_oversized_single_block_is_kept_whole() -> None:
    unit = make_unit("CODE", paragraphs=0, words=0, code_words=700)
    page = Page(page_file="code.html", page_title="Code", units=[unit])
    chunks = page_chunks(page)

    assert len(chunks) == 1
    assert chunks[0]["n_words"] == 701
    assert chunks[0]["text"].startswith("SELECT c0")
    assert len(chunks[0]["text"].split()) == 701


def test_consecutive_small_units_merge_and_keep_first_anchor() -> None:
    units = [make_unit("A", paragraphs=5, words=40), make_unit("B", paragraphs=5, words=40)]
    page = Page(page_file="merge.html", page_title="Merge", units=units)
    chunks = page_chunks(page)

    assert len(chunks) == 1
    assert chunks[0]["anchor"] == "A"
    assert chunks[0]["id"] == "merge.html#A#0"
    assert chunks[0]["n_words"] == 400


def test_no_anchor_means_url_without_fragment() -> None:
    unit = make_unit("", paragraphs=2, words=200)
    page = Page(page_file="plain.html", page_title="Plain", units=[unit])
    chunk = page_chunks(page)[0]

    assert chunk["url"] == "https://www.postgresql.org/docs/17/plain.html"
    assert "#" not in chunk["url"]


def test_split_blocks_does_not_split_code_blocks() -> None:
    blocks = [Block("code", " ".join(f"x{i}" for i in range(700)))]
    blocks += [Block("text", " ".join(f"y{i}" for i in range(50))) for _ in range(3)]
    pieces = split_blocks(blocks)

    assert pieces[0][0][0].kind == "code"
    assert pieces[0][1] >= 700


def test_toc_only_page_is_discarded() -> None:
    html = (
        "<html><head><title>PostgreSQL: Documentation: 17: Index</title></head><body>"
        '<div id="docContent"><div class="chapter" id="DDL">'
        '<div class="titlepage"><div><div><h1>Chapter 5. Data Definition</h1></div></div></div>'
        '<div class="toc"><p>Table of Contents</p><ul><li>5.1. Tables</li></ul></div>'
        "</div></div></body></html>"
    )
    assert parse_html(html, "ddl.html") is None
