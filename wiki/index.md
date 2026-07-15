# pyghl Knowledge-Base Index

This page defines knowledge-base authority, evidence vocabulary, content rules,
safe use, and broad routes. It is a router and governance owner; implementation
contracts belong in maps, domain hubs, and leaves linked below.

## Route by Question

| Question | Read first | Then reach |
| --- | --- | --- |
| What does an unfamiliar term mean and who owns it? | [Catalog](catalog.md) | One canonical map, hub, or leaf |
| Which file owns this behavior? | [Source map](source-map.md) | Exact repo-root source path |
| Is a Python/C/CLI surface registered or exported? | [Public API map](public-api-map.md) | Binding, NN, or CLI owner |
| What proves a behavior? | [Test map](test-map.md) | Exact assertion, artifact inspection, run, or explicit gap |
| What changes with this path? | [Change-impact map](change-impact.md) | Owners, tests, artifacts, and operational seams |
| Can this command mutate, download, build, or leave files? | [Workflows](workflows.md) | [Generated boundaries](generated-boundaries.md) and owner leaf |
| How is the extension built, packaged, repaired, loaded, or published? | [Build/package/release/CI](build-package-release-ci.md) | Exact metadata, setup, workflow, or inspected artifact |
| How does pyghl depend on GRHayL? | [GRHayL integration](integration/grhayl-submodule.md) | Parent-pinned GRHayL object |
| Which conflict is confirmed and unresolved? | [Contradictions](contradictions.md) | Competing exact sources and smallest decision |
| How is the graph checked? | [KB checks](lint/CHECKS.md) | Checker source and regression tests |

Domain entry points are [bindings](bindings/index.md), [neural-network
tooling](nn/index.md), and [CLI/download workflows](cli/index.md). Unknown aliases
always start at the [catalog](catalog.md).

## Authority Is Question-Specific

No single file wins every question. Use the authority matching the claim.

| Question | Primary authority | Supporting or stronger proof |
| --- | --- | --- |
| Python import/export | `src/pyghl/__init__.py` and owning Python module | Exact import in a named environment; tests/examples only for what they run |
| Extension types/functions/errors | `csrc/pyghl_module.c` registration and wrapper code | Parent-pinned GRHayL declarations/definitions; exact wrapper execution |
| EOS resolution/lifetime | `src/pyghl/eos.py` and C EOS wrapper/deallocator | Parent-pinned EOS code; exact runtime lifecycle test |
| CLI spelling/dispatch | `src/pyghl/cli.py` and delegated parser/module | Exact CLI assertions or named execution |
| NN data/math/schema | Owning `src/pyghl/_nn_*` or `src/pyghl/nn_c2p/*` producer/consumer | Inspected named artifact and exact behavioral execution |
| Build/link/load | `setup.py` plus parent-pinned GRHayL inputs | Controlled named build, link inspection, then import |
| Package selection | `pyproject.toml`, `MANIFEST.in`, and `setup.py` | Contents of an exact named sdist or wheel |
| Test behavior | Exact test assertions | Executed result in a named environment |
| CI configuration | Exact `.github/workflows/*.yml` text | Exact hosted run; workflow presence is not run proof |
| Release | `pyproject.toml`, `PUBLISHING.md`, and publish workflow | Inspected immutable hosted release and published artifacts |
| GRHayL behavior | Source/header/build/test at parent-recorded gitlink revision | Upstream KB routes only; current checkout does not replace pin evidence |

README and PUBLISHING prose guide users and maintainers but do not override
source, metadata, workflows, artifacts, or results.

## Evidence States

Use exact states. Never replace them with unqualified “supported,” “works,”
“tested,” “CI covers,” “valid model,” or “compatible.”

- `source-present`: definition or file exists in inspected source.
- `declared`: public header, parser, metadata, or Python declaration exists.
- `implemented`: executable behavior exists in source.
- `registered`: CPython or CLI registration/dispatch exists.
- `exported`: source-visible Python facade/import route exists. This proves no
  import success, stability, package presence, or support promise.
- `selected`: generic build, test, or workflow metadata names an item/operation.
- `package-selected`: packaging metadata, manifest, or copy rule selects an item
  for a named mode. This is not artifact proof.
- `packaged`: contents of an exact named wheel or sdist were inspected and contain
  the item.
- `compiled`: an exact build completed for a named configuration.
- `linked`: an exact extension/consumer linked against named libraries.
- `imported`: an exact installed environment loaded package, extension, and
  needed shared libraries.
- `executed`: an exact test, API, example, or CLI ran with named prerequisites.
- `artifact-inspected`: an exact named artifact was examined against stated
  fields/schema. This proves neither loader compatibility nor numerical validity.
- `release-published`: exact immutable version/artifacts were observed published.
- `upstream-only`: parent-pinned GRHayL evidence exists without pyghl exposure.
- `coverage-gap`: no direct repo-local behavioral proof was found.

Selection, artifact contents, load, execution, and publication are separate
claims. Name the artifact/environment/run whenever using a state stronger than
source or configuration.

## Context and Hop Budget

Use only enough context to reach authority:

1. Root `AGENTS.md` supplies scope, safety, and first routes.
2. This index, catalog, or a cross-cutting map selects one owner.
3. A domain hub or owner leaf states the contract and routes exact source/proof.
4. Exact source, configuration, assertion, workflow, inspected artifact, or
   parent-pinned object settles the claim.

Common owner questions must reach an owner within two links after root.
Specialized questions may use three. Maps and hubs route; they do not copy owner
contracts. Catalog aliases have one canonical owner. Circular-only and
catalog-only pages are invalid.

## Page Contracts

Every page has one H1. Its first two to four sentences state purpose, scope, and
authority. Use relative documentation links and exact repo-root source paths or
symbols. Do not use workspace-absolute links, `file:` URIs, source line anchors,
copied source bodies, review badges, or maintenance history.

Router/map rows use:

| Query/task/path | Canonical owner | Primary authority | Proof/workflow route |
| --- | --- | --- | --- |

Leaves use only applicable sections in this order: purpose/scope; read first;
inputs/preconditions; flow/matrix; ownership/lifetime/mutation/artifacts;
errors/build gates; evidence/tests/gaps; change impact/related routes. Avoid empty
template headings.

Artifact inventories use:

| Artifact | Producer | Consumer | Location | Tracked/user-owned | Mutation/overwrite | Guard/cleanup | Strongest proof |
| --- | --- | --- | --- | --- | --- | --- | --- |

Contradictions use:

| Bounded proposition | Competing evidence | Impact/safe wording | Smallest decision | Owner |
| --- | --- | --- | --- | --- |

Only confirmed unresolved current conflicts belong in that table. Missing tests,
uninspected packages/models, narrower build matrices, live remote volatility, and
checkout drift are evidence gaps or boundaries, not contradictions.

## Freshness and Change Workflow

Freshness is dependency-driven. Do not store affirmative `last_verified`,
`source_digest`, `source_checksum`, `source_mtime`, or `kb_fingerprint` fields;
do not add review dates or maintenance logs. Product MD5/SHA-1 values,
`perm_sha1_16`, generated C-header guards, artifact timestamps, and provenance
remain valid product contracts.

For a change:

1. Preserve current root and nested state; collect committed, staged, unstaged,
   untracked, ignored, and submodule paths.
2. Route changed paths through [change impact](change-impact.md), then reopen
   exact implementation/configuration/tests and parent-pinned objects.
3. Update owner leaves first, then maps/catalog/root only if routing changed.
4. Run [KB checks](lint/CHECKS.md), targeted semantic review, and relevant tests.
5. Remove resolved contradiction rows; never turn the table into a backlog.

Git history supplies durable history. A remote GRHayL change alone does not make
this KB stale; a parent gitlink or integration behavior change does.

## Safe Operations and Proof Limits

- Default to read-only inspection. Do not build GRHayL/wheels, download EOS
  tables, train models, mutate HDF5, or publish merely to improve prose.
- Before an authorized operation, read [workflows](workflows.md) and [generated
  boundaries](generated-boundaries.md); name network, cost, mutation, output,
  overwrite guard, and cleanup.
- Preserve `extern/GRHayL`. Use its nested repository for `git show`, `git grep`,
  `git cat-file`, and `git archive` against the parent-recorded object; never
  reset/update/stage the checkout to make evidence convenient.
- Grep is candidate inventory, not semantic proof. Examples and module tools are
  manual consumers, not product tests. Upstream tests do not prove wrappers.
- A workflow file proves configured text only. Local YAML inspection does not
  prove platform acceptance or hosted success. Publishing configuration does not
  prove a version exists on PyPI.
- A packaging rule proves `package-selected`. Use `packaged` only after inspecting
  one exact archive. A tracked HDF5 file's presence proves no compatibility.

## External Ground Truth

- [Git submodule documentation](https://git-scm.com/docs/git-submodule) defines
  initialization/update behavior and distinguishes the superproject-recorded
  commit from a differing checked-out submodule commit.
