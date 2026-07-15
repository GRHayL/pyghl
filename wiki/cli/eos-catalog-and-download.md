# EOS Catalog and Download Trust Boundary

This page owns remote EOS catalog crawl/parser, terminal selection, download/decompression, cache, limits, cleanup, and mocked evidence. Authority is `src/pyghl/nn_c2p/eos_catalog.py` and `tests/test_eos_catalog.py`; no live availability or remote content claim is made.

## Remote Flow

When `pyghl train` omits EOS path, `choose_and_download_eos()`:

1. Fetches fixed StellarCollapse microphysics HTTPS index.
2. Parses anchor text/title containing EOS terms into same-host family pages.
3. Breadth-first crawls discovered family pages, assigns top-level category, parses table rows, validates table filenames/URLs, and de-duplicates by `(filename,url)`.
4. Uses a curses category screen, then searchable table screen.
5. Downloads selected archive into current or explicit directory, decompresses to a partial destination, publishes final file, and removes partials.

Catalog contents, URLs, and hard-coded Stockholm warning can change independently of repository. Source-visible constants and mocked HTML prove code decisions only; they do not prove current service availability, HTTP status, advertised size, archive content, or warning accuracy.

## Catalog Trust Decisions

- Requested catalog pages must use HTTPS, exact hostname `stellarcollapse.org`, and no explicit port.
- Discovered catalog links must keep that scheme/host/no-port boundary and path suffix empty or `.html`; URL fragments are removed and URLs visited once.
- Custom `_SafeRedirectHandler` allows redirects only to HTTPS approved download hosts, with no explicit port. Catalog response is additionally checked for HTTPS and exact catalog hostname. The response check itself does not retest port, so injected/custom `open_url` proof differs from default-opener proof.
- Each catalog page is read at most `MAX_CATALOG_BYTES + 1` and rejected above 5 MiB. Crawl stops at 64 visited pages. Limit is per page, not cumulative bytes.
- Table filename must match `[A-Za-z0-9._+-]+\.h5(?:\.tar)?\.bz2`; separators and unrelated suffixes are excluded. If URL basename itself looks like an archive filename, it must equal catalog filename.
- Download URLs require HTTPS, no explicit port, and exact allowed host (`stellarcollapse.org`, two Stockholm Box hosts) or any hostname ending `.boxcloud.com`. Opaque Box share paths are accepted when catalog supplies valid expected filename.

These are allowlist/shape checks, not content authenticity. No signature or trusted digest is supplied by catalog flow.

## Download and Response Revalidation

`download_eos_table()` validates selected URL and filename before filesystem
changes. Destination directory is created. Derived archive/destination basenames
come from restricted filename. Any existing final path for which
`Path.is_file()` is true is reused without network or content validation; this
includes a symlink resolving to a regular file, potentially outside the
destination directory. If `Path.is_file()` is false but `Path.exists()` is true,
the destination raises. A dangling symlink makes both predicates false, so the
download proceeds and final publication replaces the symlink itself with the
downloaded file.

Before transfer, both `<archive>.part` and `<destination>.part` are unlinked if present. Final response URL is fully revalidated against allowed download scheme/host/port and expected filename. Redirect handler separately validates each default-opener redirect. Response `Content-Length` is used only for progress when parseable and at most 6 GiB; missing/invalid/oversized header switches to unknown-total progress rather than authorizing size.

Compressed bytes stream in 64 KiB chunks to archive partial and are rejected after crossing 6 GiB. A progress object writes percentage/size/speed/ETA when total is known, or moving marker/size/speed/elapsed when unknown; terminal updates default to 0.2 seconds and nonterminal logs to 5 seconds. Exceptional transfer closes terminal progress line.

## Decompression and Publication

Plain `.h5.bz2` uses `bz2.open` and bounded copy. `.h5.tar.bz2` opens `tarfile` in streaming `r|bz2` mode, ignores non-regular/non-`.h5` members, requires exactly one regular member whose basename ends `.h5`, reads its bytes through `extractfile`, and never extracts member path/metadata to filesystem. Both paths reject output after crossing 16 GiB.

Code does not check HDF5 signature/schema, archive member count beyond qualifying
HDF5 members, member-declared size before reading, checksum, or EOS validity.
Selected output is written only to controlled same-directory destination
partial. After successful decompression, `Path.replace()` publishes it at final
destination. No lock or second existence check protects publication, so a
destination created after the initial cache check can be replaced. There is no
fsync/durability protocol.

A `finally` block removes archive and destination partials on success, parsing/decompression failure, cancellation, and other `BaseException` unwinding. It does not remove a pre-existing cached final file, published final file, or directory it created. Startup removal means unrelated files with exact controlled `.part` names are deleted.

| Artifact | Producer | Consumer | Location | Tracked/user-owned | Mutation/overwrite | Guard/cleanup | Strongest proof |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Catalog HTML in memory | HTTPS response | HTML parsers/crawler | Memory | Ephemeral/network | Per request | Host/redirect/page/page-count gates; context close | Source path `implemented`; parser/crawl assertions present |
| Archive partial | downloader | bz2/tar reader | Destination directory, `*.part` | Temporary | Exact old part unlinked; new `wb` | URL response + 6 GiB limit; `finally` unlink | Source path `implemented`; synthetic-byte and cleanup assertions present |
| Destination partial | decompressor | `Path.replace()` | Destination directory, `*.part` | Temporary | Exact old part unlinked; new `wb` | One selected stream + 16 GiB limit; `finally` unlink | Source path `implemented`; synthetic-byte and cleanup assertions present |
| Final EOS cache | publication or pre-existing file/symlink target | train/user | Destination path; a reused symlink may resolve elsewhere | User-owned/cache | `is_file()` true reuses path; other `exists()` true raises; a dangling symlink passes both gates and is replaced; publication can also replace a concurrently created destination | No cache content, symlink-target, lock, or second-existence validation; partials cleaned | Source path `implemented`; synthetic cache/publication assertions present |
| Progress output | downloader | Terminal/log user | stderr | Ephemeral | Repeated updates | Throttled; close/finish newline handling | Source path `implemented`; rendering assertions present |

## Terminal Selection and Cancellation

Selection requires both stdin and stdout TTY; otherwise `RuntimeError`. Category picker supports arrows/home/end/enter; table picker supports arrows, paging, home/end, backspace, printable type-to-filter across family/description/filename, and enter. Ctrl-C character or Escape raises `KeyboardInterrupt`; `pyghl train` converts this selection cancellation to status 130. Small terminals request resize and allow cancellation. Curses startup errors become `RuntimeError`.

Source hard-codes a warning for Stockholm hosts and renders it at category/table levels. That warning is volatile UI text, not probed availability.

## Mocked Proof and Gaps

`tests/test_eos_catalog.py` contains 22 test methods using in-memory response
objects and temporary directories. Their assertions cover:

- known/unknown-total progress rendering;
- index/family/table parsing, nested crawl, category retention, duplicate aggregation, external-host rejection, and oversized HTML;
- synthetic bzip2 and tar+bzip2 decode, progress from response size, existing-file cache, directory collision, and partial cleanup after invalid bzip2;
- filter matching, category/table picker behavior, hard-coded warning, and fetch-select-download orchestration.

Direct tests do not exercise default network opener, `_SafeRedirectHandler`, redirect chains, final-response revalidation, 64-page limit, compressed/decompressed byte-limit failure, multiple qualifying tar members, cancellation during transfer/decompression, final replace failure, or real HDF5. Mocked synthetic success proves control flow and bytes, not live network/security completeness/EOS validity.

No live fetch or download ran for KB work. Current upstream catalog, Stockholm/Box availability, TLS endpoint, archive sizes, EOS validity, and disk-space requirements remain gaps.

## Change Impact

Host/scheme/redirect changes require adversarial URL tests for requested, discovered, redirected, and response URLs. Parser changes require hostile/malformed HTML fixtures. Limit/archive/publication changes require boundary, cleanup, cancellation, multiple-member, and disk-failure tests. Cache changes must preserve explicit user ownership and avoid silent replacement.

## External Ground Truth

- [Python `urllib.request` redirect handler API](https://docs.python.org/3/library/urllib.request.html#httpredirecthandler-objects) defines `redirect_request`; project adds stricter host/scheme/port policy.
- [Python `tarfile` stream modes](https://docs.python.org/3/library/tarfile.html#tarfile.open) defines `r|bz2`; project reads one member stream and does not call filesystem extraction APIs.
- [Python tar extraction security guidance](https://docs.python.org/3/library/tarfile.html#tarfile-extraction-filter) explains why untrusted archive handling needs limits and inspection; project-specific constraints are listed above.
- [Python `bz2.open`](https://docs.python.org/3/library/bz2.html#bz2.open) defines binary decompression used for plain archives.
- [Python `Path.is_file`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.is_file)
  documents normal symlink following. [Python `Path.replace`](https://docs.python.org/3/library/pathlib.html#pathlib.Path.replace)
  delegates replacement semantics; [Python `os.replace`](https://docs.python.org/3/library/os.html#os.replace)
  documents destination replacement and atomic success on POSIX.
