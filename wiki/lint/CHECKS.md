# Knowledge-Base Structural Checks

This page defines the deterministic structural-checker contract for `AGENTS.md`
and `wiki/**/*.md`. It owns machine-checkable Markdown boundaries, diagnostics,
local commands, parent-pin proof behavior, and explicit manual-review limits;
repository source and exact artifacts remain authority for product semantics.

## Supported Markdown Subset

The checker intentionally does not claim full CommonMark or GitHub Flavored
Markdown parsing. Machine checks support:

- UTF-8 files, ATX headings (`#` through `######`), and backtick or tilde fenced
  code blocks whose closing marker uses the same character and at least the
  opening length;
- one-line inline links in the form `[label](destination)` or
  `[label](<destination>)`, with an optional one-line title;
- one-line full or collapsed reference links (`[label][id]`, `[label][]`) and
  one-line definitions (`[id]: destination`); reference identifiers compare
  case-insensitively after whitespace normalization; and
- heading fragments derived from checked ATX headings: inline-code content is
  preserved while its delimiters are removed, `<...>` spans are removed, and
  text is stripped and lowercased; Unicode alphanumeric characters, `_`, and
  `-` are retained; whitespace becomes `-`; other characters are removed; and
  repeated slugs gain `-1`, `-2`, and so on.

Nested link labels/destinations, multiline links, shortcut references,
autolinks, raw HTML anchors, Setext headings, and arbitrary CommonMark/GFM
extensions remain outside the machine contract. Link-like text inside fenced or
inline code is ignored. Use supported relative links when graph validation is
required.

## Hard Checks

`scripts/check_kb.py` performs read-only, Python-standard-library checks and
sorts all diagnostics by path, line, code, and message. Any `ERROR` produces a
nonzero exit status.

- `AGENTS.md` is the graph root. Every `wiki/**/*.md` page must be reachable
  through checked Markdown links; circular-only and orphan pages fail.
- Every checked page has exactly one ATX H1 as its first nonblank line. Purpose
  or scope prose starts within five lines after that H1.
- Local links must stay within the repository after lexical normalization and
  filesystem resolution. Checked pages cannot resolve outside the repository;
  targets must exist with exact case. Case-colliding KB page paths fail.
- Fragments must name a supported ATX-heading slug in another checked Markdown
  page or the source page. Queries on local links are unsupported and fail.
- Absolute local paths, workspace-absolute links, `file:` URIs, unsupported URI
  schemes, broken references, and empty destinations fail. `http`, `https`, and
  `mailto` destinations are external and are not fetched or validated.
- Affirmative stored assignments or fields for `last_verified`,
  `source_digest`, `source_checksum`, `source_mtime`, and `kb_fingerprint` fail.
  Negated governance prose such as “do not store `last_verified`” remains
  permitted. Product canonical/file MD5 and SHA-1 values, `perm_sha1_16`, C
  header-guard SHA-1, artifact timestamps, and provenance are product contracts,
  not forbidden KB-freshness metadata.

Representative stable diagnostic codes are `LINK_MISSING`, `LINK_ESCAPE`,
`LINK_ABSOLUTE`, `LINK_CASE`, `FRAGMENT_MISSING`, `ORPHAN`, `H1_COUNT`,
`EARLY_PURPOSE`, `FRESHNESS_FIELD`, `PIN_TARGET_MISSING`, and
`PIN_PROOF_SKIPPED`; `WORKSPACE_ABSOLUTE` identifies stored paths rooted at the
current checkout. Regression tests assert that every listed representative code
appears in at least one fixture. Selected diagnostic substrings are asserted for
parent-pin cases, and sorted output is asserted for one two-link `LINK_MISSING`
scenario; wording and ordering are not frozen for every code. New hard rules
require matching fixtures and this contract to change together.

## Parent-Pin Link Proof

Links under `extern/GRHayL/<target>` are delegated evidence, not ordinary
working-tree links. Checker obtains parent-recorded gitlink with
`git ls-tree HEAD -- extern/GRHayL`, then asks nested object database for
`<pin>^{commit}` and exactly `<pin>:<target>` using `git cat-file -e`. It never
accepts target presence only in current, advanced, or dirty nested checkout.

If nested repository or parent-pin object is unavailable, checker emits explicit
nonfatal `SKIP ... PIN_PROOF_SKIPPED`; local structure may pass, but pinned-target
proof did not occur. Once pin object is available, target absence is hard
`PIN_TARGET_MISSING`. Parent-pin discovery failure also skips proof rather than
substituting current checkout. Initialize/fetch only through an separately
authorized workflow; checker itself performs no checkout, reset, write, or
network operation.

## Manual Checks

Machine success does not establish:

- truth, completeness, ownership, hop-budget usefulness, or source semantics;
- semantics of backticked source paths/symbols or whether cited lines still
  implement prose claims;
- external-link availability/content, full Markdown rendering, raw HTML anchor
  behavior, YAML validity, workflow platform acceptance, or hosted execution;
- product build/import/test behavior, artifact contents, numerical validity,
  package/release publication, remote-service behavior, or branch protection;
  or
- whether a skipped parent-pin proof would succeed after required objects are
  acquired.

Review changed owner leaves against exact source/configuration/tests and the
parent-pinned GRHayL objects. Then route downstream review through [change
impact](../change-impact.md) and operational risk through [workflows](../workflows.md).

## Local Commands

Run exact checker suite, then full graph:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_kb_checks -v
PYTHONDONTWRITEBYTECODE=1 python scripts/check_kb.py
git diff --check -- scripts/check_kb.py tests/test_kb_checks.py wiki/lint/CHECKS.md .github/workflows/kb.yml
```

Checker and tests must leave root, nested repository, index, untracked files,
ignored files, and generated artifacts unchanged. A successful local run proves
only implemented fixture behavior and current checked graph. Workflow text
proves configuration only; YAML/platform acceptance and hosted success require
separate hosted evidence.

## External Ground Truth

- [Python `pathlib`](https://docs.python.org/3/library/pathlib.html) distinguishes
  pure lexical paths from filesystem paths and recommends `Path.resolve()` when
  eliminating `..` while walking arbitrary paths; checker combines lexical and
  resolved containment checks.
- [Python `subprocess`](https://docs.python.org/3/library/subprocess.html) defines
  `subprocess.run()` argument-list execution and return-code/output handling used
  for read-only Git probes.
- [Python `unittest`](https://docs.python.org/3/library/unittest.html) defines
  direct module selection used by local and workflow commands.
- [Python `tempfile`](https://docs.python.org/3/library/tempfile.html) defines
  `TemporaryDirectory` cleanup used to isolate regression repositories.
- [`git ls-tree`](https://git-scm.com/docs/git-ls-tree) defines tree-entry output,
  including mode, object type, object name, and path used to obtain parent pin.
- [`git cat-file`](https://git-scm.com/docs/git-cat-file) defines `-e` as a
  zero-status existence/validity probe used for pin commits and pinned targets.
- [Git environment documentation](https://git-scm.com/docs/git) defines
  `GIT_NO_LAZY_FETCH`, used so missing partial-clone objects produce local proof
  skips instead of on-demand network access.
- [Official `actions/checkout`](https://github.com/actions/checkout) documents
  `submodules: recursive` and its recursive submodule acquisition behavior.
- [GitHub workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)
  defines path-filtered events, `workflow_dispatch`, and token `permissions` used
  by KB workflow configuration.
