"""Fetch the PostgreSQL 17 documentation sources for ingestion.

The ``ftp.postgresql.org`` source tarballs ship DocBook/SGML sources only (no
generated HTML) and this machine has no DocBook toolchain, so the HTML is
fetched from the official documentation site, which serves the exact
DocBook-XSL output structure the parser expects (``div.sect1``/``div.refsect1``
with ids, ``div.navheader``, ``div.toc``, ``pre``, ``table``...).

The source tarball is still downloaded and its published ``.sha256`` verified,
because it pins the exact release the docs site's ``/docs/17/`` branch maps to.
If a future tarball does ship ``doc/src/sgml/html/*.html``, those members are
extracted instead (only that subtree, rejecting unsafe paths).
"""

from __future__ import annotations

import hashlib
import http.client
import json
import re
import sys
import tarfile
import time
import urllib.request
from collections import deque
from pathlib import Path, PurePosixPath

BASE_URL = "https://ftp.postgresql.org/pub/source/"
VERSION_DIR_RE = re.compile(r"v(17\.\d+)/")
HTML_MEMBER_RE = re.compile(r"^postgresql-17\.\d+/doc/src/sgml/html/[^/]+\.html$")

DOCS_HOST = "www.postgresql.org"
DOCS_PATH = "/docs/17/"
SEED_PAGES = ("index.html", "bookindex.html")
USER_AGENT = "grounded-rag-ingest/0.1 (+https://github.com/devop-arg)"
HREF_RE = re.compile(r'href="([^"]+)"')
PAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]*\.html$")
SOURCE_MARKER = ".source.json"


def _fetch(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def _head_ok(url: str, timeout: int = 30) -> bool:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def list_versions() -> list[str]:
    """Return the 17.x versions published in the source directory, newest first."""
    html = _fetch(BASE_URL).decode("utf-8", "replace")
    versions = sorted(set(VERSION_DIR_RE.findall(html)), key=lambda v: tuple(int(p) for p in v.split(".")))
    versions.reverse()
    return versions


def latest_version() -> str:
    """Newest 17.x version that has both a tarball and a published .sha256."""
    for version in list_versions():
        name = f"postgresql-{version}.tar.bz2"
        url = f"{BASE_URL}v{version}/{name}"
        if not _head_ok(url):
            continue
        try:
            _fetch(url + ".sha256")
        except Exception:
            continue
        return version
    raise RuntimeError("no 17.x release with tarball and .sha256 found")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_sha256_file(text: str, filename: str) -> str:
    """Extract the expected digest for ``filename`` from a .sha256 file body."""
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) == 1:
            return parts[0].lower()
        if parts[-1].lstrip("*") == filename:
            return parts[0].lower()
    raise ValueError(f"no digest for {filename} in sha256 file")


def download(version: str, raw_dir: Path, log=print) -> tuple[Path, str]:
    """Download the tarball for ``version`` into ``raw_dir`` (idempotent).

    Returns ``(path, sha256)`` where the digest has been checked against the
    published ``.sha256`` file.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    name = f"postgresql-{version}.tar.bz2"
    url = f"{BASE_URL}v{version}/{name}"
    dest = raw_dir / name

    expected = parse_sha256_file(_fetch(url + ".sha256").decode("utf-8", "replace"), name)

    if dest.exists():
        actual = sha256_file(dest)
        if actual == expected:
            log(f"download: {name} already present, sha256 verified ({actual})")
            return dest, actual
        log(f"download: {name} present but digest mismatch, re-downloading")
        dest.unlink()

    log(f"download: fetching {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    started = time.time()
    with urllib.request.urlopen(url, timeout=300) as resp, tmp.open("wb") as fh:
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            fh.write(block)
    actual = sha256_file(tmp)
    if actual != expected:
        tmp.unlink()
        raise RuntimeError(f"sha256 mismatch for {name}: expected {expected}, got {actual}")
    tmp.rename(dest)
    log(f"download: {name} verified in {time.time() - started:.1f}s ({dest.stat().st_size} bytes)")
    return dest, actual


def _safe_member(name: str) -> bool:
    pure = PurePosixPath(name)
    return not pure.is_absolute() and ".." not in pure.parts


def extract_html(tarball: Path, html_dir: Path, log=print) -> int:
    """Extract ``doc/src/sgml/html/*.html`` members of ``tarball`` into ``html_dir``.

    Only that subtree is extracted, one file at a time (nothing is written
    outside ``html_dir``) and unsafe member names are rejected.  Returns the
    number of pages written, which is 0 for the current 17.x tarballs.
    """
    html_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with tarfile.open(tarball, "r:bz2") as tar:
        for member in tar:
            if not member.isfile() or not HTML_MEMBER_RE.match(member.name):
                continue
            if not _safe_member(member.name):
                continue
            source = tar.extractfile(member)
            if source is None:
                continue
            (html_dir / PurePosixPath(member.name).name).write_bytes(source.read())
            written += 1
    if written:
        log(f"extract: {written} pages from {tarball.name}")
    else:
        log(f"extract: {tarball.name} contains no doc/src/sgml/html/*.html members")
    return written


def _open() -> http.client.HTTPSConnection:
    return http.client.HTTPSConnection(DOCS_HOST, timeout=30)


def _get_page(conn: http.client.HTTPSConnection, name: str, log=print, retries: int = 4):
    """Fetch one docs page with politeness. Returns ``(conn, body)`` or ``(conn, None)``."""
    path = DOCS_PATH + name
    for attempt in range(retries):
        try:
            conn.request("GET", path, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
            resp = conn.getresponse()
            body = resp.read()
            status = resp.status
            if status == 200:
                return conn, body
            if status == 404:
                log(f"fetch: 404 {path}")
                return conn, None
            if status == 429 or 500 <= status < 600:
                wait = 2 ** attempt
                log(f"fetch: HTTP {status} for {path}, retrying in {wait}s")
                time.sleep(wait)
                conn.close()
                conn = _open()
                continue
            log(f"fetch: unexpected HTTP {status} for {path}")
            return conn, None
        except (http.client.HTTPException, OSError) as exc:
            wait = 2 ** attempt
            log(f"fetch: {type(exc).__name__} on {path}, retrying in {wait}s")
            time.sleep(wait)
            try:
                conn.close()
            except Exception:
                pass
            conn = _open()
    log(f"fetch: giving up on {path}")
    return conn, None


def normalize_link(href: str) -> str | None:
    """Return the docs page name for an in-directory link, else ``None``."""
    href = href.strip()
    if not href or href.startswith(("#", "mailto:", "javascript:", "data:")):
        return None
    href = href.split("#", 1)[0].split("?", 1)[0]
    if not href:
        return None
    if href.startswith("/"):
        if not href.startswith(DOCS_PATH):
            return None
        name = href[len(DOCS_PATH) :]
    else:
        if "://" in href or href.startswith("//"):
            return None
        name = href
    name = name.removeprefix("./")
    if not PAGE_NAME_RE.match(name):
        return None
    return name


def fetch_docs_site(html_dir: Path, refresh: bool = False, pause: float = 0.3, log=print) -> int:
    """Crawl ``/docs/17/`` and cache every page in ``html_dir``.

    Only same-directory ``*.html`` links are followed, one connection is reused,
    there is a pause between requests and already cached pages are not re-fetched
    unless ``refresh`` is set.  Returns the number of pages available on disk.
    """
    html_dir.mkdir(parents=True, exist_ok=True)
    queue: deque[str] = deque(SEED_PAGES)
    seen: set[str] = set()
    fetched = cached = failed = 0
    conn: http.client.HTTPSConnection | None = None
    started = time.time()

    try:
        while queue:
            name = queue.popleft()
            if name in seen:
                continue
            seen.add(name)
            target = html_dir / name
            body = None
            if not refresh and target.exists() and target.stat().st_size:
                cached += 1
                html = target.read_text("utf-8", "replace")
            else:
                if conn is None:
                    conn = _open()
                conn, body = _get_page(conn, name, log=log)
                if body is None:
                    failed += 1
                    continue
                target.write_bytes(body)
                html = body.decode("utf-8", "replace")
                fetched += 1
                time.sleep(pause)
                if fetched % 200 == 0:
                    log(f"fetch: {fetched} pages fetched, {len(seen)} discovered, {len(queue)} queued")
            for href in HREF_RE.findall(html):
                link = normalize_link(href)
                if link and link not in seen:
                    queue.append(link)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    pages = len(list(html_dir.glob("*.html")))
    log(
        f"fetch: {pages} pages on disk ({fetched} downloaded, {cached} cached, {failed} failed) "
        f"in {time.time() - started:.1f}s"
    )
    return pages


def ensure_docs(
    raw_dir: Path,
    html_dir: Path,
    refresh: bool = False,
    log=print,
    download_tarball: bool = True,
) -> dict:
    """Make the documentation HTML available and return provenance metadata."""
    version = latest_version()
    info: dict = {"version": version, "docs_path": DOCS_PATH, "tarball": None}
    log(f"version: latest 17.x release is {version}")

    if download_tarball:
        tarball, digest = download(version, raw_dir, log=log)
        info["tarball"] = {"name": tarball.name, "sha256": digest, "bytes": tarball.stat().st_size}
        pages = extract_html(tarball, html_dir, log=log)
        source = "tarball"
    else:
        log("download: skipped (--skip-download), version taken from the ftp listing")
        pages = 0
        source = None

    if pages == 0:
        pages = fetch_docs_site(html_dir, refresh=refresh, log=log)
        source = f"https://{DOCS_HOST}{DOCS_PATH}"

    (html_dir / SOURCE_MARKER).write_text(
        json.dumps({"version": version, "source": source, "pages": pages}, indent=2)
    )
    info["source"] = source
    info["pages"] = pages
    return info


if __name__ == "__main__":
    print(json.dumps(ensure_docs(Path("data/raw"), Path("data/html")), indent=2))
    sys.exit(0)
