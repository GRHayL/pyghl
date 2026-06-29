# Publishing pyghl

This repository publishes `pyghl` to PyPI through GitHub Actions and PyPI
Trusted Publishing. The publishing workflow is `.github/workflows/publish.yml`
and runs when a GitHub release is published.

## Release Checklist

Use `X.Y.Z` for the package version and `vX.Y.Z` for the Git tag.

### 1. Update Version

Edit `pyproject.toml`:

```toml
[project]
version = "X.Y.Z"
```

This is the package version source of truth. `pyghl --version` reads the
installed package metadata generated from this value.

### 2. Run Local Checks

```bash
python -m compileall -q src setup.py
python -m pip wheel . -w /tmp/pyghl-wheel --no-deps --no-build-isolation
```

Optional local smoke test:

```bash
python -m venv /tmp/pyghl-smoke
/tmp/pyghl-smoke/bin/python -m pip install --no-deps /tmp/pyghl-wheel/pyghl-X.Y.Z-*.whl
/tmp/pyghl-smoke/bin/pyghl --version
```

### 3. Commit and Push

```bash
git status
git add pyproject.toml README.md PUBLISHING.md src setup.py .github
git commit -m "Release pyghl X.Y.Z"
git push
```

Adjust `git add` paths to match the files that actually changed.

### 4. Confirm Wheel CI

Wait for the `Wheels` workflow on `main` to pass. It builds wheels for:

- Linux x86_64
- macOS arm64

Download and test wheel artifacts if the release changes packaging, compiled
code, dependencies, or command-line behavior.

### 5. Create GitHub Release

In GitHub:

1. Go to `Releases`.
2. Choose `Draft a new release`.
3. Create or select tag `vX.Y.Z`.
4. Target the commit on `main` that contains `version = "X.Y.Z"`.
5. Title the release `pyghl X.Y.Z`.
6. Publish the release.

Publishing the release triggers `.github/workflows/publish.yml`.

## What the Publish Workflow Does

The publish workflow:

1. Checks out the repository with submodules.
2. Builds wheels with `cibuildwheel`.
3. Uploads wheel artifacts.
4. Publishes those artifacts to PyPI through Trusted Publishing.

No PyPI API token is needed when Trusted Publishing is configured correctly.

## Common Failure Modes

### Wrong Tag

If a release was published with the wrong tag, do not rerun the failed workflow.
GitHub reruns keep the original release event payload.

Fix:

1. Delete the bad release.
2. Delete the bad tag if needed.
3. Create a fresh release with the correct tag.

### Tag Blocked by Environment Rules

If GitHub reports that a tag is not allowed to deploy to environment `pypi`,
check repository settings:

```text
Settings -> Environments -> pypi -> Deployment branches and tags
```

Use `No restriction` while debugging, or `Selected branches and tags` with a
tag rule like:

```text
v*
```

### Version Already Exists on PyPI

PyPI does not allow replacing an uploaded file for the same project version.
If `X.Y.Z` was already published, bump to a new version and publish again.

### macOS Wheel Repair Fails

macOS wheels bundle Homebrew HDF5 dependencies through `delocate`. If delocate
reports a minimum macOS target mismatch, use a runner and
`MACOSX_DEPLOYMENT_TARGET` compatible with the bundled Homebrew libraries.

Current workflow uses:

```yaml
os: macos-14
MACOSX_DEPLOYMENT_TARGET=14.0
```
