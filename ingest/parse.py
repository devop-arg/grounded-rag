"""Parse PostgreSQL documentation pages into section units.

A page is split into *units*: one per section whose heading is an ``h2`` or
``h3`` (``div.sect1``/``div.sect2``/``div.refsect1``/``div.refsect2``...).
Deeper sections (``h4`` and below) stay inside their ``h3`` unit, with their
own heading kept as a line of text.

Only the documentation content is kept: the postgresql.org template (navbar,
footer, version banner) and the ``Submit correction`` block are dropped, as
well as the documentation navigation (``div.navheader``, ``div.navfooter``,
``div.toc``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup

SECTION_CLASS_RE = re.compile(r"^(?:sect[1-5]|refsect[1-3]|simplesect)$")
TITLE_PREFIX_RE = re.compile(r"^PostgreSQL:\s*Documentation:\s*[\d.]+:\s*")

#: Sections at this nesting depth or shallower (h2/h3) become their own unit.
UNIT_DEPTH = 2
#: Word budget per table block: keeps big tables below the chunk size target.
TABLE_WORDS = 300

SKIP_CLASSES = {"navheader", "navfooter", "toc", "titlepage"}
SKIP_TAGS = {"script", "style", "nav", "header", "footer", "noscript"}
SKIP_IDS = {"docComments", "docContent-nav"}

#: Tags whose own inline text becomes one block.
TEXT_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "dt", "dd", "li", "caption", "figcaption"}
#: Tags that contain nested blocks (so only their inline text is taken first).
CONTAINER_TEXT_TAGS = {"li", "dd"}
#: Block level elements that end a parent's inline text.
NESTED_BLOCK_TAGS = {"p", "ul", "ol", "dl", "pre", "table", "div", "blockquote", "li"}


@dataclass
class Block:
    """A paragraph, code block or table row list inside a unit."""

    kind: str  # "text" or "code"
    text: str


@dataclass
class Unit:
    """One h2/h3 section, or the whole page when it has none."""

    anchor: str
    title: str
    path: list[str] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)

    @property
    def n_words(self) -> int:
        return sum(len(block.text.split()) for block in self.blocks)


@dataclass
class Page:
    page_file: str
    page_title: str
    units: list[Unit] = field(default_factory=list)
    #: The alphabetical index of the manual (``bookindex.html``): parsed for the
    #: crawl, never chunked (see :func:`ingest.chunk.page_chunks`).
    is_index: bool = False

    @property
    def n_words(self) -> int:
        return sum(unit.n_words for unit in self.units)


def normalize_text(text: str) -> str:
    """Collapse whitespace (including NBSP) into single spaces."""
    return " ".join(text.replace("\xa0", " ").split())


def normalize_code(text: str) -> str:
    """Normalize a code block while keeping its line structure."""
    lines = [line.rstrip() for line in text.replace("\xa0", " ").splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _classes(tag) -> set[str]:
    value = tag.get("class") or []
    return set(value)


def _is_section(tag) -> bool:
    return tag.name == "div" and any(SECTION_CLASS_RE.match(c) for c in _classes(tag))


def is_discarded(tag) -> bool:
    if tag.name in SKIP_TAGS:
        return True
    if tag.get("id") in SKIP_IDS:
        return True
    return bool(_classes(tag) & SKIP_CLASSES)


def doc_root(soup: BeautifulSoup):
    """Return the documentation container, skipping the site template."""
    root = soup.find(id="docContent")
    if root is None:
        root = soup.body or soup
    return root


def _section_title(section) -> str:
    titlepage = section.find("div", class_="titlepage", recursive=False)
    if titlepage is not None:
        heading = titlepage.find(["h1", "h2", "h3", "h4", "h5", "h6"])
        if heading is not None:
            return normalize_text(heading.get_text(" "))
    heading = section.find(["h1", "h2", "h3", "h4", "h5", "h6"], recursive=False)
    if heading is not None:
        return normalize_text(heading.get_text(" "))
    return ""


def _inline_text(tag) -> str:
    """Text of ``tag`` ignoring nested block level elements."""
    parts: list[str] = []
    for child in tag.children:
        if getattr(child, "name", None) in NESTED_BLOCK_TAGS:
            continue
        parts.append(child.get_text(" ") if getattr(child, "name", None) else str(child))
    return normalize_text(" ".join(parts))


def _table_rows(table) -> tuple[str, list[str]]:
    """Return ``(header_row, body_rows)`` as ``cell | cell`` lines."""
    header = ""
    rows: list[str] = []
    for row in table.find_all("tr"):
        cells = [normalize_text(cell.get_text(" ")) for cell in row.find_all(["th", "td"], recursive=False)]
        cells = [cell for cell in cells if cell]
        if not cells:
            continue
        line = " | ".join(cells)
        if not header and row.find("th") is not None and not rows:
            header = line
            continue
        rows.append(line)
    return header, rows


def _table_text(table, max_words: int = TABLE_WORDS) -> list[str]:
    """Table as ``cell | cell`` lines, grouped in blocks of about ``max_words``.

    The header row is repeated in every block so each piece stays readable, and
    no block grows far past the chunk size target.
    """
    header, rows = _table_rows(table)
    blocks: list[str] = []
    current: list[str] = []
    current_words = 0
    for line in rows:
        words = len(line.split())
        if current and current_words + words > max_words:
            blocks.append("\n".join(([header] if header else []) + current))
            current, current_words = [], 0
        current.append(line)
        current_words += words
    if current:
        blocks.append("\n".join(([header] if header else []) + current))
    return blocks


def _collect_blocks(node, blocks: list[Block], depth_of, skip_own_heading: bool = False) -> None:
    for child in node.find_all(recursive=False):
        if not getattr(child, "name", None) or is_discarded(child):
            continue
        if _is_section(child):
            if depth_of(child) <= UNIT_DEPTH:
                continue  # it is its own unit
            title = _section_title(child)
            if title:
                blocks.append(Block("text", title))
            _collect_blocks(child, blocks, depth_of, skip_own_heading=True)
            continue
        name = child.name
        if skip_own_heading and name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            # The section heading is already part of section_path.
            skip_own_heading = False
            continue
        skip_own_heading = False
        if name == "pre":
            text = normalize_code(child.get_text("\n"))
            if text:
                blocks.append(Block("code", text))
        elif name == "table":
            for text in _table_text(child):
                blocks.append(Block("text", text))
        elif name in TEXT_TAGS:
            text = _inline_text(child)
            if text:
                blocks.append(Block("text", text))
            if name in CONTAINER_TEXT_TAGS:
                _collect_blocks(child, blocks, depth_of)
        else:
            _collect_blocks(child, blocks, depth_of)


def _section_depths(root) -> dict[int, int]:
    """Map ``id(element)`` to its section nesting depth (top level == 1)."""
    depths: dict[int, int] = {}

    def walk(node, depth: int) -> None:
        for child in node.find_all("div", recursive=False):
            if _is_section(child):
                depths[id(child)] = depth + 1
                walk(child, depth + 1)
            else:
                walk(child, depth)

    walk(root, 0)
    return depths


def extract_units(root) -> list[Unit]:
    depths = _section_depths(root)
    units: list[Unit] = []

    def visit(node, path: list[str]) -> None:
        for child in node.find_all(recursive=False):
            if not getattr(child, "name", None) or is_discarded(child):
                continue
            if _is_section(child):
                if depths.get(id(child), 1) <= UNIT_DEPTH:
                    title = _section_title(child)
                    blocks: list[Block] = []
                    _collect_blocks(child, blocks, lambda el: depths.get(id(el), 99), skip_own_heading=True)
                    units.append(
                        Unit(
                            anchor=child.get("id") or "",
                            title=title,
                            path=path + [title] if title else list(path),
                            blocks=blocks,
                        )
                    )
                    visit(child, path + [title] if title else path)
                continue
            visit(child, path)

    visit(root, [])
    return [unit for unit in units if unit.blocks]


def page_title_of(soup: BeautifulSoup, page_file: str) -> str:
    if soup.title and soup.title.string:
        title = TITLE_PREFIX_RE.sub("", normalize_text(soup.title.string))
        if title:
            return title
    return page_file.removesuffix(".html")


def is_book_index(soup: BeautifulSoup, page_file: str) -> bool:
    """True for the manual's alphabetical index (``bookindex.html``).

    The index is fetched as a crawl seed but must not become retrieval chunks:
    its 27 fragments are ``term, page`` links and match almost any query.  The
    rule is the file name plus the DocBook-XSL shape of the body: an
    alphabetical index is a ``div.index`` made of ``div.indexdiv`` letter
    sections (with a ``p.indexdiv-quicklinks`` A-Z bar).
    """
    if page_file == "bookindex.html":
        return True
    if soup.find("p", class_="indexdiv-quicklinks") is not None:
        return True
    index = soup.find("div", class_="index")
    return index is not None and index.find("div", class_="indexdiv") is not None


def parse_html(html: str, page_file: str) -> Page | None:
    """Parse one documentation page. Returns ``None`` for TOC-only pages."""
    soup = BeautifulSoup(html, "lxml")
    # The docs site appends a "#" permalink inside every heading.
    for link in soup.select("a.id_link"):
        link.decompose()
    root = doc_root(soup)
    title = page_title_of(soup, page_file)
    is_index = is_book_index(soup, page_file)
    units = extract_units(root)

    if not units:
        blocks: list[Block] = []
        _collect_blocks(root, blocks, lambda el: 99)
        anchor = ""
        for candidate in ("chapter", "appendix", "part", "preface", "refentry", "book", "article"):
            found = root.find("div", class_=candidate)
            if found is not None and found.get("id"):
                anchor = found["id"]
                break
        if blocks:
            units = [Unit(anchor=anchor, title=title, path=[title], blocks=blocks)]

    page = Page(page_file=page_file, page_title=title, units=units, is_index=is_index)
    if page.n_words < 40:
        return None
    return page


def parse_file(path: Path) -> Page | None:
    return parse_html(path.read_text("utf-8", "replace"), path.name)
