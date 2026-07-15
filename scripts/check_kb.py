#!/usr/bin/env python3
"""Conservative structural checks for the pyghl Markdown knowledge base."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import unquote, urlsplit


INLINE_LINK_RE = re.compile(
    r"(?<!!)\[[^\]\n]+\]\(\s*(?:<([^>\n]*)>|([^\s)]*))(?:\s+[^)\n]+)?\s*\)"
)
REFERENCE_LINK_RE = re.compile(r"(?<!!)\[([^\]\n]+)\]\[([^\]\n]*)\]")
REFERENCE_DEF_RE = re.compile(
    r"^\s{0,3}\[([^\]]+)\]:\s*(?:<([^>]+)>|(\S+))(?:\s+.*)?$"
)
ATX_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})(.*)$")
INLINE_CODE_RE = re.compile(r"(`+)(.*?)\1")
FRESHNESS_ASSIGNMENT_RE = re.compile(
    r"(?i)(?:`\s*)?\b(last_verified|source_digest|source_checksum|"
    r"source_mtime|kb_fingerprint)\b(?:\s*`)?\s*(?::|=)\s*\S+"
)
NEGATION_RE = re.compile(
    r"(?i)(?:\bdo\s+not\b|\bmust\s+not\b|\bnever\b|\bnot\s+store(?:d)?\b|"
    r"\bno\s+affirmative\b|\breject(?:ed|s|ing)?\b|\bforbid(?:den|s|ding)?\b|"
    r"\bprohibit(?:ed|s|ing)?\b)"
)
FILE_URI_RE = re.compile(r"(?i)(?<![\w])file:(?://)?[^\s)>`]+")
EXTERNAL_SCHEMES = {"http", "https", "mailto"}


@dataclass(frozen=True, order=True)
class Diagnostic:
    path: str
    line: int
    code: str
    message: str
    severity: str = "ERROR"

    def render(self) -> str:
        return f"{self.severity} {self.path}:{self.line} {self.code} {self.message}"


@dataclass
class Page:
    relpath: str
    path: Path
    lines: list[str]
    visible_lines: list[str]
    headings: dict[str, int]


@dataclass(frozen=True)
class Link:
    source: str
    line: int
    destination: str


def strip_fenced_code(lines: list[str]) -> list[str]:
    """Blank fenced blocks while preserving line numbers."""
    visible: list[str] = []
    fence_char = ""
    fence_length = 0
    for line in lines:
        match = FENCE_RE.match(line)
        if not fence_char:
            if match:
                marker = match.group(1)
                fence_char = marker[0]
                fence_length = len(marker)
                visible.append("")
            else:
                visible.append(line)
            continue
        closing = re.match(rf"^\s{{0,3}}{re.escape(fence_char)}{{{fence_length},}}\s*$", line)
        visible.append("")
        if closing:
            fence_char = ""
            fence_length = 0
    return visible


def strip_inline_code(line: str) -> str:
    return INLINE_CODE_RE.sub(lambda match: " " * len(match.group(0)), line)


def heading_slug(text: str) -> str:
    text = INLINE_CODE_RE.sub(lambda match: match.group(2), text)
    text = re.sub(r"<[^>]*>", "", text).strip().lower()
    result: list[str] = []
    for char in text:
        if char.isalnum() or char in "_-":
            result.append(char)
        elif char.isspace():
            result.append("-")
    return "".join(result)


def collect_headings(visible_lines: list[str]) -> dict[str, int]:
    headings: dict[str, int] = {}
    counts: dict[str, int] = {}
    for line_number, line in enumerate(visible_lines, 1):
        match = ATX_HEADING_RE.match(line)
        if not match:
            continue
        base = heading_slug(match.group(2))
        count = counts.get(base, 0)
        counts[base] = count + 1
        slug = base if count == 0 else f"{base}-{count}"
        headings[slug] = line_number
    return headings


def normalize_reference_id(value: str) -> str:
    return " ".join(value.split()).casefold()


def extract_links(page: Page, diagnostics: list[Diagnostic]) -> list[Link]:
    definitions: dict[str, tuple[str, int]] = {}
    definition_lines: set[int] = set()
    for line_number, line in enumerate(page.visible_lines, 1):
        match = REFERENCE_DEF_RE.match(line)
        if not match:
            continue
        identifier = normalize_reference_id(match.group(1))
        destination = match.group(2) or match.group(3)
        if identifier in definitions:
            diagnostics.append(Diagnostic(
                page.relpath, line_number, "REFERENCE_DUPLICATE",
                f"duplicate reference definition [{match.group(1)}]",
            ))
        else:
            definitions[identifier] = (destination, line_number)
        definition_lines.add(line_number)

    links: list[Link] = []
    for line_number, raw_line in enumerate(page.visible_lines, 1):
        if line_number in definition_lines:
            continue
        line = strip_inline_code(raw_line)
        for match in INLINE_LINK_RE.finditer(line):
            links.append(Link(page.relpath, line_number, match.group(1) or match.group(2)))
        for match in REFERENCE_LINK_RE.finditer(line):
            identifier_text = match.group(2) or match.group(1)
            identifier = normalize_reference_id(identifier_text)
            if identifier not in definitions:
                diagnostics.append(Diagnostic(
                    page.relpath, line_number, "REFERENCE_UNDEFINED",
                    f"reference [{identifier_text}] has no definition",
                ))
                continue
            destination, _ = definitions[identifier]
            links.append(Link(page.relpath, line_number, destination))
    return links


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def case_mismatch(root: Path, relative: PurePosixPath) -> str | None:
    current = root
    actual: list[str] = []
    for part in relative.parts:
        try:
            names = [entry.name for entry in current.iterdir()]
        except OSError:
            return None
        if part in names:
            chosen = part
        else:
            matches = sorted(name for name in names if name.casefold() == part.casefold())
            if not matches:
                return None
            chosen = matches[0]
        actual.append(chosen)
        current /= chosen
    actual_path = "/".join(actual)
    requested = relative.as_posix()
    return actual_path if actual_path != requested else None


def run_git(root: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    env["GIT_NO_LAZY_FETCH"] = "1"
    env["GIT_TERMINAL_PROMPT"] = "0"
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )


class PinValidator:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.nested = root / "extern" / "GRHayL"
        self._state: tuple[str, str] | None = None
        self._targets: dict[str, tuple[bool | None, str]] = {}

    def _pin_state(self) -> tuple[str, str]:
        if self._state is not None:
            return self._state
        result = run_git(self.root, ["ls-tree", "HEAD", "--", "extern/GRHayL"])
        match = re.fullmatch(
            r"160000 commit ([0-9a-fA-F]{40,64})\textern/GRHayL\n?", result.stdout
        )
        if result.returncode != 0 or not match:
            self._state = ("skip", "parent gitlink pin unavailable")
            return self._state
        pin = match.group(1)
        nested_probe = run_git(self.nested, ["rev-parse", "--git-dir"])
        if nested_probe.returncode != 0:
            self._state = ("skip", "nested repository unavailable")
            return self._state
        object_probe = run_git(self.nested, ["cat-file", "-e", f"{pin}^{{commit}}"])
        if object_probe.returncode != 0:
            self._state = ("skip", f"parent pin object {pin} unavailable in nested repository")
            return self._state
        self._state = ("ready", pin)
        return self._state

    def validate(self, target: str) -> tuple[bool | None, str]:
        if target in self._targets:
            return self._targets[target]
        state, detail = self._pin_state()
        if state == "skip":
            answer = (None, detail)
        else:
            probe = run_git(self.nested, ["cat-file", "-e", f"{detail}:{target}"])
            if probe.returncode == 0:
                answer = (True, f"target exists at parent pin {detail}")
            else:
                answer = (False, f"target absent at parent pin {detail}: {target}")
        self._targets[target] = answer
        return answer


def page_structure(page: Page, diagnostics: list[Diagnostic]) -> None:
    h1_lines: list[int] = []
    first_nonblank = 0
    for line_number, line in enumerate(page.visible_lines, 1):
        if not first_nonblank and line.strip():
            first_nonblank = line_number
        match = ATX_HEADING_RE.match(line)
        if match and len(match.group(1)) == 1:
            h1_lines.append(line_number)
    if len(h1_lines) != 1:
        diagnostics.append(Diagnostic(
            page.relpath, h1_lines[0] if h1_lines else 1, "H1_COUNT",
            f"expected exactly one H1; found {len(h1_lines)}",
        ))
        return
    h1_line = h1_lines[0]
    if first_nonblank != h1_line:
        diagnostics.append(Diagnostic(
            page.relpath, h1_line, "H1_POSITION", "H1 must be first nonblank line",
        ))
    purpose_found = False
    for line_number in range(h1_line + 1, min(len(page.visible_lines), h1_line + 6) + 1):
        line = page.visible_lines[line_number - 1].strip()
        if not line:
            continue
        if ATX_HEADING_RE.match(line) or line.startswith(("|", "- ", "* ", ">", "1. ")):
            break
        purpose_found = bool(re.search(r"[A-Za-z0-9]", strip_inline_code(line)))
        if purpose_found:
            break
    if not purpose_found:
        diagnostics.append(Diagnostic(
            page.relpath, h1_line, "EARLY_PURPOSE",
            "expected purpose/scope prose within five lines after H1",
        ))


def check_freshness(page: Page, diagnostics: list[Diagnostic]) -> None:
    for line_number, line in enumerate(page.visible_lines, 1):
        for match in FRESHNESS_ASSIGNMENT_RE.finditer(line):
            prefix = line[:match.start()]
            prefix = re.split(r"[.!?;]", prefix)[-1]
            if NEGATION_RE.search(prefix):
                continue
            diagnostics.append(Diagnostic(
                page.relpath, line_number, "FRESHNESS_FIELD",
                f"affirmative stored field assignment forbidden: {match.group(1).lower()}",
            ))


def check_forbidden_paths(root: Path, page: Page, diagnostics: list[Diagnostic]) -> None:
    root_text = root.resolve().as_posix()
    workspace_re = re.compile(
        rf"(?<![A-Za-z0-9_.-]){re.escape(root_text)}(?:/[^\s)>`]*)?(?![A-Za-z0-9_.-])"
    )
    for line_number, raw_line in enumerate(page.visible_lines, 1):
        match = FILE_URI_RE.search(raw_line)
        if match:
            diagnostics.append(Diagnostic(
                page.relpath, line_number, "FILE_URI",
                f"file URI forbidden: {match.group(0)}",
            ))
        workspace_match = workspace_re.search(raw_line)
        if workspace_match:
            diagnostics.append(Diagnostic(
                page.relpath, line_number, "WORKSPACE_ABSOLUTE",
                f"workspace-absolute path forbidden: {workspace_match.group(0)}",
            ))


def load_pages(root: Path, diagnostics: list[Diagnostic]) -> dict[str, Page]:
    paths = [root / "AGENTS.md"]
    wiki = root / "wiki"
    if wiki.is_dir():
        paths.extend(sorted(
            path for path in wiki.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".md"
        ))
    pages: dict[str, Page] = {}
    for path in paths:
        relpath = path.relative_to(root).as_posix()
        if not path.is_file():
            diagnostics.append(Diagnostic(relpath, 1, "PAGE_MISSING", "required page missing"))
            continue
        try:
            resolved_page = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            diagnostics.append(Diagnostic(
                relpath, 1, "PAGE_RESOLVE", f"cannot resolve checked page: {exc}",
            ))
            continue
        if not is_within(resolved_page, root.resolve()):
            diagnostics.append(Diagnostic(
                relpath, 1, "PAGE_ESCAPE", "checked page resolves outside repository",
            ))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            diagnostics.append(Diagnostic(relpath, 1, "PAGE_READ", f"cannot read UTF-8 page: {exc}"))
            continue
        lines = text.splitlines()
        visible_lines = strip_fenced_code(lines)
        pages[relpath] = Page(
            relpath, path, lines, visible_lines, collect_headings(visible_lines)
        )
    return pages


def check_link(
    root: Path,
    pages: dict[str, Page],
    link: Link,
    pin_validator: PinValidator,
    diagnostics: list[Diagnostic],
) -> str | None:
    raw = link.destination.strip()
    if not raw:
        diagnostics.append(Diagnostic(link.source, link.line, "LINK_EMPTY", "empty link destination"))
        return None
    decoded = unquote(raw)
    if "\x00" in decoded:
        diagnostics.append(Diagnostic(link.source, link.line, "LINK_INVALID", "NUL in link destination"))
        return None
    if PureWindowsPath(decoded).is_absolute() or decoded.startswith(("/", "\\\\")):
        diagnostics.append(Diagnostic(
            link.source, link.line, "LINK_ABSOLUTE", f"absolute local path forbidden: {raw}",
        ))
        return None
    try:
        split = urlsplit(decoded)
    except ValueError as exc:
        diagnostics.append(Diagnostic(
            link.source, link.line, "LINK_INVALID", f"cannot parse link destination {raw}: {exc}",
        ))
        return None
    scheme = split.scheme.lower()
    if scheme in EXTERNAL_SCHEMES:
        return None
    if scheme == "file":
        return None  # Whole-page URI scan already emitted the stable diagnostic.
    if scheme:
        diagnostics.append(Diagnostic(
            link.source, link.line, "LINK_SCHEME", f"unsupported URI scheme: {scheme}",
        ))
        return None
    if split.netloc:
        diagnostics.append(Diagnostic(
            link.source, link.line, "LINK_ABSOLUTE", f"network/absolute local path forbidden: {raw}",
        ))
        return None
    if split.query:
        diagnostics.append(Diagnostic(
            link.source, link.line, "LINK_QUERY", f"query component unsupported: {raw}",
        ))
        return None

    source_dir = PurePosixPath(link.source).parent
    path_part = split.path
    requested = source_dir / PurePosixPath(path_part) if path_part else PurePosixPath(link.source)
    normalized_parts: list[str] = []
    escaped = False
    for part in requested.parts:
        if part in ("", "."):
            continue
        if part == "..":
            if normalized_parts:
                normalized_parts.pop()
            else:
                escaped = True
        else:
            normalized_parts.append(part)
    if escaped:
        diagnostics.append(Diagnostic(
            link.source, link.line, "LINK_ESCAPE", f"link escapes repository: {raw}",
        ))
        return None
    relative = PurePosixPath(*normalized_parts)
    target_path = root.joinpath(*relative.parts)
    resolved_root = root.resolve()
    try:
        resolved_target = target_path.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        diagnostics.append(Diagnostic(
            link.source, link.line, "LINK_RESOLVE", f"cannot resolve {raw}: {exc}",
        ))
        return None
    if not is_within(resolved_target, resolved_root):
        diagnostics.append(Diagnostic(
            link.source, link.line, "LINK_ESCAPE", f"resolved link escapes repository: {raw}",
        ))
        return None

    relative_text = relative.as_posix()
    if relative_text == "extern/GRHayL" or relative_text.startswith("extern/GRHayL/"):
        target = relative_text.removeprefix("extern/GRHayL/")
        if not target or target == "extern/GRHayL":
            diagnostics.append(Diagnostic(
                link.source, link.line, "PIN_TARGET", "delegated GRHayL link must name a pinned target",
            ))
            return None
        valid, detail = pin_validator.validate(target)
        if valid is None:
            diagnostics.append(Diagnostic(
                link.source, link.line, "PIN_PROOF_SKIPPED", detail, severity="SKIP",
            ))
        elif not valid:
            diagnostics.append(Diagnostic(link.source, link.line, "PIN_TARGET_MISSING", detail))
        return None

    mismatch = case_mismatch(root, relative)
    if mismatch:
        diagnostics.append(Diagnostic(
            link.source, link.line, "LINK_CASE",
            f"link casing differs from filesystem: {relative_text} != {mismatch}",
        ))
        return None
    if not target_path.exists():
        diagnostics.append(Diagnostic(
            link.source, link.line, "LINK_MISSING", f"local target missing: {relative_text}",
        ))
        return None
    if split.fragment:
        if relative_text not in pages:
            diagnostics.append(Diagnostic(
                link.source, link.line, "FRAGMENT_TARGET",
                f"fragment target is not a checked Markdown page: {relative_text}",
            ))
        else:
            fragment = unquote(split.fragment)
            if fragment not in pages[relative_text].headings:
                diagnostics.append(Diagnostic(
                    link.source, link.line, "FRAGMENT_MISSING",
                    f"heading fragment missing in {relative_text}: #{fragment}",
                ))
    return relative_text if relative_text in pages else None


def check_repository(root: Path) -> list[Diagnostic]:
    root = root.resolve()
    diagnostics: list[Diagnostic] = []
    pages = load_pages(root, diagnostics)
    folded: dict[str, list[str]] = {}
    for relpath in pages:
        folded.setdefault(relpath.casefold(), []).append(relpath)
    for paths in folded.values():
        if len(paths) > 1:
            message = "case-colliding pages: " + ", ".join(sorted(paths))
            for relpath in sorted(paths):
                diagnostics.append(Diagnostic(relpath, 1, "PAGE_CASE_COLLISION", message))

    edges: dict[str, set[str]] = {relpath: set() for relpath in pages}
    pin_validator = PinValidator(root)
    for relpath in sorted(pages):
        page = pages[relpath]
        page_structure(page, diagnostics)
        check_freshness(page, diagnostics)
        check_forbidden_paths(root, page, diagnostics)
        for link in extract_links(page, diagnostics):
            target = check_link(root, pages, link, pin_validator, diagnostics)
            if target:
                edges[relpath].add(target)

    if "AGENTS.md" in pages:
        reached = {"AGENTS.md"}
        pending = ["AGENTS.md"]
        while pending:
            source = pending.pop()
            for target in sorted(edges.get(source, ())):
                if target not in reached:
                    reached.add(target)
                    pending.append(target)
        for relpath in sorted(set(pages) - reached):
            diagnostics.append(Diagnostic(
                relpath, 1, "ORPHAN", "page is not reachable from AGENTS.md",
            ))
    return sorted(diagnostics)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: checker script parent repository)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    diagnostics = check_repository(args.root)
    for diagnostic in diagnostics:
        print(diagnostic.render())
    error_count = sum(item.severity == "ERROR" for item in diagnostics)
    skip_count = sum(item.severity == "SKIP" for item in diagnostics)
    print(f"KB check: {error_count} error(s), {skip_count} proof skip(s)")
    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
