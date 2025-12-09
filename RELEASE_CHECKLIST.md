# Release Checklist

Use this checklist when preparing a new release of TinyForgeAI.

## Pre-Release

- [ ] All tests pass: `pytest -q`
- [ ] Linting passes: `flake8 backend/ connectors/ cli/ inference_server/`
- [ ] Documentation is up to date
- [ ] CHANGELOG.md is updated with release notes
- [ ] Version number is bumped in `pyproject.toml`

## Build & Package

- [ ] Install build tools: `pip install build twine`
- [ ] Build distribution: `python -m build`
- [ ] Verify package contents: `tar -tzf dist/tinyforgeai-X.Y.Z.tar.gz`
- [ ] Test install in clean virtualenv

## Git & GitHub

- [ ] Commit version bump and changelog:
  ```bash
  git add pyproject.toml CHANGELOG.md
  git commit -m "chore(release): vX.Y.Z"
  ```

- [ ] Create annotated tag:
  ```bash
  git tag -a vX.Y.Z -m "Release vX.Y.Z"
  ```

- [ ] Push to remote:
  ```bash
  git push origin main --tags
  ```

- [ ] Create GitHub Release:
  1. Go to Releases page
  2. Click "Draft a new release"
  3. Select tag `vX.Y.Z`
  4. Use release notes from `releases/notes_*.md`
  5. Attach built artifacts (wheel, sdist)
  6. Publish release

## Post-Release

- [ ] Verify GitHub Actions CI passes on tag
- [ ] (Optional) Publish to PyPI:
  ```bash
  twine upload dist/*
  ```
- [ ] (Optional) Update Docker Hub image
- [ ] Announce release (Twitter, Discord, etc.)
- [ ] Update version in `pyproject.toml` to next dev version (e.g., `0.2.0.dev0`)

## Quick Commands

```bash
# Run the prepare_release.sh script (does everything except push)
bash releases/prepare_release.sh

# Or manually:
pytest -q
# Edit pyproject.toml version
python -m build
git add -A
git commit -m "chore(release): vX.Y.Z"
git tag -a vX.Y.Z -m "Release vX.Y.Z"
git push origin main --tags
```

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes to public API
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

Pre-release versions: `X.Y.Z-alpha.1`, `X.Y.Z-beta.1`, `X.Y.Z-rc.1`
