# Release Guide

This guide covers how to publish a new release of TinyForgeAI.

## Prerequisites

- Python 3.10+
- `build` package: `pip install build`
- `twine` package: `pip install twine`
- PyPI account with upload permissions (if publishing to PyPI)
- GitHub CLI (`gh`) for creating releases

## Release Process

### 1. Prepare the Release

Use the automated release preparation script:

```bash
./releases/prepare_release.sh 0.1.0
```

This script will:
- ✅ Verify working directory is clean
- ✅ Run the test suite
- ✅ Run linting (critical errors)
- ✅ Update version in `pyproject.toml`
- ✅ Update version in `tinyforgeai/__init__.py`
- ✅ Build sdist and wheel packages
- ✅ Verify packages with twine

### 2. Update CHANGELOG

Edit `CHANGELOG.md` to:
1. Move items from `[Unreleased]` to the new version section
2. Add the release date
3. Create a fresh `[Unreleased]` section

```markdown
## [Unreleased]

## [0.1.0] - 2025-01-15

### Added
- Initial release features...
```

### 3. Commit and Tag

```bash
# Stage version changes
git add pyproject.toml tinyforgeai/__init__.py CHANGELOG.md

# Commit
git commit -m "chore: bump version to 0.1.0"

# Create annotated tag
git tag -a v0.1.0 -m "Release v0.1.0"

# Push to remote
git push origin main
git push origin v0.1.0
```

### 4. Build Distribution

If not already built by the prep script:

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Build sdist and wheel
python -m build

# Verify
twine check dist/*
```

### 5. Upload to PyPI

#### Test PyPI (Recommended First)

```bash
twine upload --repository testpypi dist/*
```

Test installation:
```bash
pip install --index-url https://test.pypi.org/simple/ tinyforgeai
```

#### Production PyPI

```bash
twine upload dist/*
```

### 6. Create GitHub Release

Using GitHub CLI:

```bash
gh release create v0.1.0 dist/* \
  --title "v0.1.0" \
  --notes-file releases/notes_initial_release.md
```

Or manually:
1. Go to https://github.com/anthropics/TinyForgeAI/releases/new
2. Choose the tag `v0.1.0`
3. Set title: "v0.1.0"
4. Copy content from `releases/notes_initial_release.md`
5. Upload files from `dist/`
6. Publish release

## Version Numbering

TinyForgeAI follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0): Breaking API changes
- **MINOR** (0.1.0): New features, backward compatible
- **PATCH** (0.0.1): Bug fixes, backward compatible

### Pre-release Versions

For alpha/beta releases:
- `0.1.0a1` - Alpha 1
- `0.1.0b1` - Beta 1
- `0.1.0rc1` - Release Candidate 1

## Checklist

Before releasing:

- [ ] All tests pass (`pytest -q`)
- [ ] No critical linting errors
- [ ] CHANGELOG.md updated
- [ ] Version bumped in all locations
- [ ] Documentation updated
- [ ] Release notes written

After releasing:

- [ ] GitHub release created
- [ ] PyPI package uploaded
- [ ] Installation tested from PyPI
- [ ] Announcement posted (if applicable)

## Troubleshooting

### Build Fails

```bash
# Ensure build is installed
pip install --upgrade build

# Try verbose build
python -m build --verbose
```

### Twine Upload Fails

```bash
# Check credentials
cat ~/.pypirc

# Or use environment variables
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-xxx...
```

### Tag Already Exists

```bash
# Delete local tag
git tag -d v0.1.0

# Delete remote tag (use with caution!)
git push --delete origin v0.1.0
```

## Hotfix Releases

For urgent fixes:

1. Create hotfix branch from tag:
   ```bash
   git checkout -b hotfix/0.1.1 v0.1.0
   ```

2. Apply fix and test

3. Follow normal release process with patch version bump

## See Also

- [RELEASE_CHECKLIST.md](../RELEASE_CHECKLIST.md) - Quick reference checklist
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [releases/](../releases/) - Release scripts and notes
