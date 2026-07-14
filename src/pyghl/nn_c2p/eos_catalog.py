from __future__ import annotations

import bz2
from collections import deque
import curses
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
import tarfile
from typing import Callable, Iterable
from urllib.parse import unquote, urldefrag, urljoin, urlparse
from urllib.request import build_opener, HTTPRedirectHandler


MICROPHYSICS_URL = "https://stellarcollapse.org/microphysics.html"
LEGACY_CATALOG_URL = "https://stellarcollapse.org/equationofstate.html"
ALLOWED_CATALOG_HOST = "stellarcollapse.org"
ALLOWED_DOWNLOAD_HOSTS = {
    ALLOWED_CATALOG_HOST,
    "stockholmuniversity.app.box.com",
    "stockholmuniversity.box.com",
}
NETWORK_TIMEOUT_SECONDS = 30.0
MAX_CATALOG_BYTES = 5 * 1024 * 1024
MAX_CATALOG_PAGES = 64
MAX_COMPRESSED_BYTES = 6 * 1024 * 1024 * 1024
MAX_DECOMPRESSED_BYTES = 16 * 1024 * 1024 * 1024
TRANSFER_CHUNK_BYTES = 1024 * 1024
_EOS_FILENAME = re.compile(r"[A-Za-z0-9._+-]+\.h5(?:\.tar)?\.bz2\Z")


def _is_allowed_download_host(hostname: str | None) -> bool:
    return hostname in ALLOWED_DOWNLOAD_HOSTS or bool(
        hostname and hostname.endswith(".boxcloud.com")
    )


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urlparse(newurl)
        if (
            parsed.scheme != "https"
            or not _is_allowed_download_host(parsed.hostname)
            or parsed.port is not None
        ):
            raise ValueError(f"Refusing EOS redirect to untrusted URL: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_URL_OPENER = build_opener(_SafeRedirectHandler())


def _open_url(url: str, *, timeout: float):
    return _URL_OPENER.open(url, timeout=timeout)


@dataclass(frozen=True)
class EOSTable:
    family: str
    description: str
    filename: str
    url: str


@dataclass(frozen=True)
class EOSCatalogPage:
    family: str
    url: str


def _normalize_text(parts: Iterable[str]) -> str:
    return " ".join("".join(parts).split())


def _archive_filename(href: str, link_text: str) -> str | None:
    candidates = (
        _normalize_text([link_text]),
        unquote(urlparse(href).path.rsplit("/", 1)[-1]),
    )
    return next((value for value in candidates if _EOS_FILENAME.fullmatch(value)), None)


def _validated_table_url(
    raw_url: str,
    *,
    expected_filename: str | None = None,
) -> tuple[str, str]:
    parsed = urlparse(raw_url)
    if (
        parsed.scheme != "https"
        or not _is_allowed_download_host(parsed.hostname)
        or parsed.port is not None
    ):
        hosts = ", ".join(sorted(ALLOWED_DOWNLOAD_HOSTS))
        raise ValueError(
            "EOS download URL must use HTTPS from an allowed host "
            f"({hosts}, or Box download hosts): {raw_url}"
        )
    filename = expected_filename or unquote(parsed.path.rsplit("/", 1)[-1])
    if not _EOS_FILENAME.fullmatch(filename):
        raise ValueError(f"Invalid StellarCollapse EOS filename: {filename!r}")
    path_filename = unquote(parsed.path.rsplit("/", 1)[-1])
    if _EOS_FILENAME.fullmatch(path_filename) and path_filename != filename:
        raise ValueError("EOS catalog filename does not match its download URL")
    return raw_url, filename


class _EOSPageLinkParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links: list[tuple[str, str, str]] = []
        self._href: str | None = None
        self._title = ""
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attributes = dict(attrs)
        self._href = attributes.get("href")
        self._title = attributes.get("title") or ""
        self._parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._href is not None:
            self.links.append((self._href, _normalize_text(self._parts), self._title))
            self._href = None


def parse_eos_family_pages(
    html: str,
    *,
    base_url: str = MICROPHYSICS_URL,
) -> list[EOSCatalogPage]:
    parser = _EOSPageLinkParser()
    parser.feed(html)
    parser.close()
    pages: list[EOSCatalogPage] = []
    seen: set[str] = set()
    for href, text, title in parser.links:
        if "eos" not in f"{text} {title}".lower() and "equation of state" not in text.lower():
            continue
        url = urldefrag(urljoin(base_url, href)).url
        parsed = urlparse(url)
        path_suffix = Path(parsed.path).suffix
        if (
            parsed.scheme != "https"
            or parsed.hostname != ALLOWED_CATALOG_HOST
            or parsed.port is not None
            or path_suffix not in {"", ".html"}
            or url in seen
        ):
            continue
        seen.add(url)
        pages.append(EOSCatalogPage(family=text or title, url=url))
    return pages


def _normalize_lines(parts: Iterable[str]) -> list[str]:
    return [" ".join(line.split()) for line in "".join(parts).splitlines() if line.strip()]


class _CatalogParser(HTMLParser):
    def __init__(self, base_url: str, default_family: str | None):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.tables: list[EOSTable] = []
        self.family = default_family or "Uncategorized EOS"
        self._cells: list[list[str]] | None = None
        self._cell_parts: list[str] | None = None
        self._links: list[tuple[str, str]] = []
        self._anchor_href: str | None = None
        self._anchor_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._cells = []
            self._links = []
        elif tag in {"td", "th"} and self._cells is not None:
            self._cell_parts = []
        elif tag == "a" and self._cells is not None:
            self._anchor_href = dict(attrs).get("href")
            self._anchor_parts = []
        elif tag == "br" and self._cell_parts is not None:
            self._cell_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)
        if self._anchor_href is not None:
            self._anchor_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._anchor_href is not None:
            self._links.append((self._anchor_href, _normalize_text(self._anchor_parts)))
            self._anchor_href = None
            self._anchor_parts = []
        elif tag in {"td", "th"} and self._cell_parts is not None:
            assert self._cells is not None
            self._cells.append(_normalize_lines(self._cell_parts))
            self._cell_parts = None
        elif tag == "tr" and self._cells is not None:
            self._finish_row()
            self._cells = None

    def _finish_row(self) -> None:
        eos_links = [
            (href, link_text, filename)
            for href, link_text in self._links
            if (filename := _archive_filename(href, link_text)) is not None
        ]
        if not eos_links:
            family_links = [
                text.rstrip("*").strip()
                for _href, text in self._links
                if "EOS" in text
            ]
            if len(self._cells or ()) == 1 and family_links:
                self.family = " / ".join(dict.fromkeys(family_links))
            return

        descriptions = (self._cells or [[]])[0]
        for index, (href, link_text, filename) in enumerate(eos_links):
            url, filename = _validated_table_url(
                urljoin(self.base_url, href),
                expected_filename=filename,
            )
            description = (
                descriptions[min(index, len(descriptions) - 1)]
                if descriptions
                else link_text or filename
            )
            self.tables.append(
                EOSTable(
                    family=self.family,
                    description=description or filename,
                    filename=filename,
                    url=url,
                )
            )


def parse_eos_catalog(
    html: str,
    *,
    base_url: str = LEGACY_CATALOG_URL,
    default_family: str | None = None,
) -> list[EOSTable]:
    parser = _CatalogParser(base_url, default_family)
    parser.feed(html)
    parser.close()
    return parser.tables


def _fetch_catalog_html(
    url: str,
    *,
    open_url: Callable[..., object],
) -> str:
    requested = urlparse(url)
    if (
        requested.scheme != "https"
        or requested.hostname != ALLOWED_CATALOG_HOST
        or requested.port is not None
    ):
        raise ValueError(f"EOS catalog page must use https://{ALLOWED_CATALOG_HOST}: {url}")
    with open_url(
        url,
        timeout=NETWORK_TIMEOUT_SECONDS,
    ) as response:  # type: ignore[attr-defined]
        final_url = response.geturl()  # type: ignore[attr-defined]
        parsed = urlparse(final_url)
        if parsed.scheme != "https" or parsed.hostname != ALLOWED_CATALOG_HOST:
            raise ValueError(f"StellarCollapse catalog redirected to untrusted URL: {final_url}")
        payload = response.read(MAX_CATALOG_BYTES + 1)  # type: ignore[attr-defined]
        if len(payload) > MAX_CATALOG_BYTES:
            raise OSError(f"EOS catalog exceeds safety limit of {MAX_CATALOG_BYTES} bytes")
        return payload.decode("utf-8")


def fetch_eos_catalog(
    *,
    open_url: Callable[..., object] = _open_url,
) -> list[EOSTable]:
    index_html = _fetch_catalog_html(MICROPHYSICS_URL, open_url=open_url)
    pages = parse_eos_family_pages(index_html)
    if not pages:
        raise RuntimeError(f"No EOS family pages found at {MICROPHYSICS_URL}")

    tables: list[EOSTable] = []
    seen: set[tuple[str, str]] = set()
    page_queue = deque(pages)
    queued_urls = {page.url for page in pages}
    visited_pages = 0
    while page_queue:
        if visited_pages >= MAX_CATALOG_PAGES:
            raise RuntimeError(
                f"EOS catalog crawl exceeds safety limit of {MAX_CATALOG_PAGES} pages"
            )
        page = page_queue.popleft()
        visited_pages += 1
        html = _fetch_catalog_html(page.url, open_url=open_url)
        for table in parse_eos_catalog(
            html,
            base_url=page.url,
            default_family=page.family,
        ):
            key = (table.filename, table.url)
            if key not in seen:
                seen.add(key)
                tables.append(table)
        for discovered in parse_eos_family_pages(html, base_url=page.url):
            if discovered.url not in queued_urls:
                queued_urls.add(discovered.url)
                page_queue.append(discovered)
    if not tables:
        raise RuntimeError(f"No EOS tables found from {MICROPHYSICS_URL}")
    return tables


def _copy_with_limit(source, destination, *, limit: int) -> None:
    total = 0
    while chunk := source.read(TRANSFER_CHUNK_BYTES):
        total += len(chunk)
        if total > limit:
            raise OSError(f"EOS table exceeds safety limit of {limit} bytes")
        destination.write(chunk)


def download_eos_table(
    table: EOSTable,
    *,
    destination_dir: Path,
    open_url: Callable[..., object] = _open_url,
) -> Path:
    url, _filename = _validated_table_url(
        table.url,
        expected_filename=table.filename,
    )

    destination_dir.mkdir(parents=True, exist_ok=True)
    if table.filename.endswith(".h5.tar.bz2"):
        destination_name = table.filename.removesuffix(".tar.bz2")
    else:
        destination_name = table.filename.removesuffix(".bz2")
    destination = destination_dir / destination_name
    if destination.is_file():
        print(f"Using existing EOS table: {destination}")
        return destination
    if destination.exists():
        raise FileExistsError(f"EOS destination exists but is not a file: {destination}")

    archive_part = destination_dir / f"{table.filename}.part"
    destination_part = destination_dir / f"{destination.name}.part"
    archive_part.unlink(missing_ok=True)
    destination_part.unlink(missing_ok=True)

    try:
        print(f"Downloading {table.filename} from {urlparse(url).hostname} ...")
        with open_url(
            url,
            timeout=NETWORK_TIMEOUT_SECONDS,
        ) as response:  # type: ignore[attr-defined]
            _validated_table_url(
                response.geturl(),  # type: ignore[attr-defined]
                expected_filename=table.filename,
            )
            with archive_part.open("wb") as archive:
                _copy_with_limit(response, archive, limit=MAX_COMPRESSED_BYTES)

        print(f"Decompressing {table.filename} ...")
        with destination_part.open("wb") as output:
            if table.filename.endswith(".h5.tar.bz2"):
                hdf5_members = 0
                with tarfile.open(archive_part, mode="r|bz2") as archive:
                    for member in archive:
                        if not member.isfile() or not Path(member.name).name.endswith(".h5"):
                            continue
                        hdf5_members += 1
                        if hdf5_members > 1:
                            raise OSError(
                                "APR EOS archive must contain exactly one regular HDF5 file"
                            )
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            raise OSError("Could not read HDF5 file from APR EOS archive")
                        with extracted:
                            _copy_with_limit(
                                extracted,
                                output,
                                limit=MAX_DECOMPRESSED_BYTES,
                            )
                if hdf5_members != 1:
                    raise OSError(
                        "APR EOS archive must contain exactly one regular HDF5 file"
                    )
            else:
                with bz2.open(archive_part, "rb") as archive:
                    _copy_with_limit(archive, output, limit=MAX_DECOMPRESSED_BYTES)
        destination_part.replace(destination)
    finally:
        archive_part.unlink(missing_ok=True)
        destination_part.unlink(missing_ok=True)

    print(f"EOS table ready: {destination}")
    return destination


def choice_matches(table: EOSTable, query: str) -> bool:
    terms = query.strip().lower().split()
    haystack = f"{table.family} {table.description} {table.filename}".lower()
    return all(term in haystack for term in terms)


def _add_clipped(screen, row: int, column: int, text: str, width: int, attrs: int = 0) -> None:
    if width <= 0:
        return
    clipped = text if len(text) <= width else text[: max(0, width - 3)] + "..."
    try:
        screen.addstr(row, column, clipped.ljust(width), attrs)
    except curses.error:
        pass


def _run_eos_picker(screen, tables: list[EOSTable]) -> EOSTable:
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    screen.keypad(True)
    selected = 0
    top = 0
    query = ""

    while True:
        visible = [table for table in tables if choice_matches(table, query)]
        selected = min(selected, max(0, len(visible) - 1))
        rows, columns = screen.getmaxyx()
        screen.erase()

        if rows < 12 or columns < 72:
            _add_clipped(screen, 0, 0, "Terminal too small; resize to at least 72x12.", columns)
            screen.refresh()
            if screen.getch() in (3, 27):
                raise KeyboardInterrupt
            continue

        screen.border()
        _add_clipped(screen, 0, 2, " StellarCollapse EOS tables ", columns - 4, curses.A_BOLD)
        _add_clipped(
            screen,
            1,
            2,
            "Arrows/PgUp/PgDn scroll  Home/End jump  Type to filter  Enter select  Esc cancel",
            columns - 4,
            curses.A_DIM,
        )
        _add_clipped(screen, 2, 2, f"Filter: {query or '[all]'}", columns - 4, curses.A_BOLD)

        list_top = 4
        list_height = max(1, rows - 7)
        if selected < top:
            top = selected
        if selected >= top + list_height:
            top = selected - list_height + 1
        top = max(0, min(top, max(0, len(visible) - list_height)))
        end = min(len(visible), top + list_height)
        _add_clipped(
            screen,
            3,
            2,
            f"Showing {top + 1 if visible else 0}-{end} of {len(visible)}",
            columns - 4,
            curses.A_DIM,
        )

        if not visible:
            _add_clipped(screen, list_top, 4, "No matches. Backspace edits filter.", columns - 8)
        else:
            row_width = columns - 4
            label_width = max(20, row_width // 2)
            for row, table in enumerate(visible[top:end], start=list_top):
                index = top + row - list_top
                marker = ">" if index == selected else " "
                detail = f"{table.family} | {table.filename}"
                text = f"{marker} {table.description[:label_width].ljust(label_width)}  {detail}"
                attrs = curses.A_REVERSE | curses.A_BOLD if index == selected else 0
                _add_clipped(screen, row, 2, text, row_width, attrs)

            current = visible[selected]
            _add_clipped(
                screen,
                rows - 3,
                2,
                f"{current.family} | {current.filename}",
                columns - 4,
                curses.A_DIM,
            )
        _add_clipped(
            screen,
            rows - 2,
            2,
            "Enter selects highlighted table.",
            columns - 4,
            curses.A_DIM,
        )
        screen.refresh()

        key = screen.getch()
        if key in (3, 27):
            raise KeyboardInterrupt
        if key == curses.KEY_UP:
            selected = max(0, selected - 1)
        elif key == curses.KEY_DOWN:
            selected = min(max(0, len(visible) - 1), selected + 1)
        elif key == curses.KEY_PPAGE:
            selected = max(0, selected - list_height)
        elif key == curses.KEY_NPAGE:
            selected = min(max(0, len(visible) - 1), selected + list_height)
        elif key == curses.KEY_HOME:
            selected = 0
        elif key == curses.KEY_END:
            selected = max(0, len(visible) - 1)
        elif key in (curses.KEY_BACKSPACE, 8, 127):
            query = query[:-1]
            selected = top = 0
        elif key in (10, 13, curses.KEY_ENTER) and visible:
            return visible[selected]
        elif 32 <= key <= 126:
            query += chr(key)
            selected = top = 0


def select_eos_table(tables: list[EOSTable]) -> EOSTable:
    if not tables:
        raise RuntimeError("No StellarCollapse EOS tables available for selection.")
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError("StellarCollapse EOS selector requires an interactive terminal.")
    try:
        return curses.wrapper(_run_eos_picker, tables)
    except curses.error as exc:
        raise RuntimeError("Could not start StellarCollapse terminal selector.") from exc


def choose_and_download_eos(
    *,
    destination_dir: Path | None = None,
    fetch_catalog: Callable[[], list[EOSTable]] = fetch_eos_catalog,
    select_table: Callable[[list[EOSTable]], EOSTable] = select_eos_table,
    download_table: Callable[..., Path] = download_eos_table,
) -> Path:
    tables = fetch_catalog()
    selected = select_table(tables)
    return download_table(selected, destination_dir=destination_dir or Path.cwd())
