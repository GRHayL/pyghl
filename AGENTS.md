# pyghl Knowledge Base

This file routes repository work. Project source, configuration, tests, workflow
files, inspected artifacts, and the GRHayL revision recorded by this repository
remain authority; knowledge-base prose yields when they differ.

## Start Here

| Need | Route |
| --- | --- |
| Policy, evidence language, or task routing | [Knowledge-base index](wiki/index.md) |
| Unknown name, alias, command, artifact, or environment variable | [Catalog](wiki/catalog.md) |
| Changed path or downstream review set | [Change-impact map](wiki/change-impact.md) |
| Source owner or callable/export | [Source map](wiki/source-map.md) and [public API map](wiki/public-api-map.md) |
| Test or proof question | [Test map](wiki/test-map.md) |
| Generated or mutable artifact | [Generated boundaries](wiki/generated-boundaries.md) |
| Build, package, wheel, CI, or release | [Build/package/release/CI](wiki/build-package-release-ci.md) |
| Binding or tabulated-EOS work | [Bindings hub](wiki/bindings/index.md) |
| Neural-network work | [NN hub](wiki/nn/index.md) |
| CLI, catalog, or download work | [CLI hub](wiki/cli/index.md) |
| Confirmed current conflict | [Contradictions](wiki/contradictions.md) |

## Scope and Safety

- Read [workflows](wiki/workflows.md) before any build, network, training, HDF5,
  package, or release operation. Many commands create large files or mutate
  user-owned EOS tables, installed models, build trees, or external checkouts.
- Treat metadata as selection/configuration only. Claim `packaged`, `imported`,
  `executed`, hosted-CI success, or publication only after inspecting that exact
  artifact, environment, run, or release.
- Preserve staged, unstaged, untracked, ignored, generated, and nested-repository
  state. Never clean, reset, update, or stage `extern/GRHayL` as a side effect.
- Update owner leaves before routers. Use [change impact](wiki/change-impact.md)
  and run the [KB checks](wiki/lint/CHECKS.md) after documentation changes.

## GRHayL Delegation

Work under `extern/GRHayL/**` follows the nested
[GRHayL instructions](extern/GRHayL/AGENTS.md). Parent pyghl claims must use the
GRHayL commit recorded by the parent gitlink, inspected through nested-repository
Git object commands; the current nested checkout can differ and is not durable
authority. Parent integration ownership lives in the
[GRHayL integration page](wiki/integration/grhayl-submodule.md).
