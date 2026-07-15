# Confirmed Current Contradictions

This page contains only unresolved, triangulated conflicts among current
repository claims. Evidence gaps, remote volatility, checkout drift, narrower
selection matrices, duplicated code, and disproved audit leads are excluded.

## Current Conflicts

| Bounded proposition | Competing evidence | Impact/safe wording | Smallest decision | Owner |
| --- | --- | --- | --- | --- |
| How should a source user install dependencies after lazy NN helper imports fail? | `src/pyghl/nn.py:_load_training_api` says dependencies are optional and directs `python -m pip install -e ./python[nn]`; repository has no `python/` project or `[project.optional-dependencies].nn`, while `pyproject.toml` declares `numpy`, `torch`, and `h5py` as normal project dependencies and README installs from repository root with `python -m pip install -e .` | Current recovery command is not a valid path/extra in this repository. Safely state that current metadata declares NN libraries as normal dependencies and that editable installation uses repository root; do not claim an `nn` extra exists | Decide whether NN dependencies remain mandatory and correct the error to a root install, or add/document a real optional extra and align metadata/docs/tests | [NN public facade](nn/index.md) |

This row is a documentation of competing current source/configuration, not
authorization to edit product code or packaging metadata.

## Maintenance Rule

Before adding a row, identify one bounded proposition, two current competing
authorities, user/maintainer impact, safe wording, smallest decision, and owner.
Put missing tests in [test map](test-map.md), generated/mutable risk in
[generated boundaries](generated-boundaries.md), and release/remote uncertainty
in [build/package/release/CI](build-package-release-ci.md) or the owning leaf.
Delete a row when resolved; Git history retains the past.

## External Ground Truth

No external source is needed to establish the current conflict. Exact repository
source, metadata, paths, and workflow text are authority for this inventory.
