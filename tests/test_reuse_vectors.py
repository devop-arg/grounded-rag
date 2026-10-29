"""The alphabetical index is parsed (crawl seed) but never chunked."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ingest.chunk import page_chunks
from ingest.parse import is_book_index, parse_html
from ingest.reuse import VectorReuseError, reconcile_vectors

FIXTURE = Path(__file__).parent / "fixtures" / "sample.html"

INDEX_HTML = """<html><head><title>PostgreSQL: Documentation: 17: Index</title></head>
<body>
<div id="docContent">
  <div class="index" id="BOOKINDEX">
    <div class="titlepage"><div><div><h1 class="title">Index</h1></div></div></div>
    <div class="index">
      <p class="indexdiv-quicklinks"><a href="#indexdiv-A">A</a> | <a href="#indexdiv-B">B</a></p>
      <div class="indexdiv" id="indexdiv-A">
        <h3>A</h3>
        <dl>
          <dt id="ientry-id-1">abort, <a class="indexterm" href="sql-abort.html">ABORT</a></dt>
          <dt id="ientry-id-2">access method, <a class="indexterm" href="indexam.html">Index Access Method</a></dt>
          <dt id="ientry-id-3">access privileges, <a class="indexterm" href="ddl-priv.html">Privileges</a></dt>
          <dt id="ientry-id-4">aggregate function, <a class="indexterm" href="functions-aggregate.html">Aggregate Functions</a></dt>
          <dt id="ientry-id-5">alias, <a class="indexterm" href="sql-select.html">SELECT</a></dt>
          <dt id="ientry-id-6">all, <a class="indexterm" href="sql-select.html">SELECT</a></dt>
          <dt id="ientry-id-7">analyze, <a class="indexterm" href="sql-analyze.html">ANALYZE</a></dt>
          <dt id="ientry-id-8">archive mode, <a class="indexterm" href="continuous-archiving.html">Continuous Archiving</a></dt>
          <dt id="ientry-id-9">array, <a class="indexterm" href="arrays.html">Arrays</a></dt>
          <dt id="ientry-id-10">authentication, <a class="indexterm" href="client-authentication.html">Client Authentication</a></dt>
          <dt id="ientry-id-11">autovacuum, <a class="indexterm" href="routine-vacuuming.html">Routine Vacuuming</a></dt>
          <dt id="ientry-id-12">backup, <a class="indexterm" href="backup.html">Backup and Restore</a></dt>
          <dt id="ientry-id-13">bit string, <a class="indexterm" href="datatype-bit.html">Bit String Types</a></dt>
          <dt id="ientry-id-14">boolean, <a class="indexterm" href="datatype-boolean.html">Boolean Type</a></dt>
        </dl>
      </div>
    </div>
  </div>
</div>
</body></html>"""


def test_bookindex_fixture_has_documentation_units() -> None:
    page = parse_html(FIXTURE.read_text(encoding="utf-8"), "sample.html")
    assert page is not None
    assert page.is_index is False
    assert page_chunks(page)


def test_index_page_is_flagged_and_yields_no_chunks() -> None:
    page = parse_html(INDEX_HTML, "bookindex.html")
    assert page is not None
    assert page.is_index is True
    assert page.units  # still parsed: it is a crawl seed
    assert page_chunks(page) == []


def test_any_page_whose_body_is_an_alphabetical_index_is_excluded() -> None:
    # Same body under another file name: the rule is structural, not the name.
    page = parse_html(INDEX_HTML, "some-index.html")
    assert page is not None
    assert page.is_index is True
    assert page_chunks(page) == []


def test_is_book_index_needs_the_indexdiv_structure() -> None:
    soup = parse_html(FIXTURE.read_text(encoding="utf-8"), "sample.html")
    assert soup is not None and soup.is_index is False
    html = "<html><body><div class='index'><p>not an index</p></div></body></html>"
    from bs4 import BeautifulSoup

    assert is_book_index(BeautifulSoup(html, "lxml"), "other.html") is False


def test_reconcile_keeps_identical_vectors_untouched() -> None:
    ids = ["a#x#0", "a#x#1"]
    vectors = np.arange(8, dtype=np.float32).reshape(2, 4)
    out, out_ids, mode = reconcile_vectors(ids, ids, vectors)
    assert mode == "identical"
    assert out_ids == ids
    assert out is vectors


def test_reconcile_filters_rows_by_id_in_chunk_order() -> None:
    stored_ids = ["a#x#0", "b#y#0", "c#z#0"]
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    new_ids = ["c#z#0", "a#x#0"]
    out, out_ids, mode = reconcile_vectors(new_ids, stored_ids, vectors)
    assert mode == "filtered"
    assert out_ids == ["c#z#0", "a#x#0"]
    assert out.tolist() == [[2.0, 2.0], [1.0, 0.0]]
    assert out.dtype == np.float32


def test_reconcile_aborts_on_a_new_id_without_vector() -> None:
    stored_ids = ["a#x#0", "b#y#0"]
    vectors = np.asarray([[1.0], [2.0]], dtype=np.float32)
    with pytest.raises(VectorReuseError, match="no vector"):
        reconcile_vectors(["a#x#0", "d#new#0"], stored_ids, vectors)


def test_reconcile_aborts_when_rows_and_ids_disagree() -> None:
    with pytest.raises(VectorReuseError, match="rows"):
        reconcile_vectors(["a#x#0"], ["a#x#0", "b#y#0"], np.ones((1, 3), dtype=np.float32))
