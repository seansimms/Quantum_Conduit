# Release Process

This document outlines the release process for Quantum Conduit.

## Pre-Release Checklist

- [ ] All tests pass (`pytest`)
- [ ] Linting passes (`ruff check qconduit tests`)
- [ ] Type checking passes (`mypy qconduit`)
- [ ] Documentation updated (README.md, API.md)
- [ ] CHANGELOG.md updated with changes
- [ ] Version number updated in `pyproject.toml`
- [ ] All CI checks pass

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR** (x.0.0): Breaking changes
- **MINOR** (0.x.0): New features, backward compatible
- **PATCH** (0.0.x): Bug fixes, backward compatible

## Release Steps

### 1. Update Version

Update version in `pyproject.toml`:
```toml
version = "0.0.5"  # or appropriate version
```

### 2. Update CHANGELOG.md

Add a new section for the release:
```markdown
## [0.0.5] - 2025-XX-XX

### Added
- New feature X

### Changed
- Improved Y

### Fixed
- Bug fix Z
```

### 3. Commit Changes

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 0.0.5"
```

### 4. Create Tag

```bash
git tag -a v0.0.5 -m "Release v0.0.5"
git push origin main --tags
```

### 5. GitHub Release

1. Go to GitHub Releases
2. Create new release from tag `v0.0.5`
3. Copy CHANGELOG.md content to release notes
4. Publish release

### 6. Automated Publishing

The `.github/workflows/release.yml` workflow will:
- Build wheel and sdist
- Publish to TestPyPI (if `TEST_PYPI_API_TOKEN` is set)
- Publish to PyPI (if `PYPI_API_TOKEN` is set)

## Post-Release

- [ ] Verify package installs: `pip install qconduit==0.0.5`
- [ ] Verify package works: `python -c "import qconduit; print(qconduit.__version__)"`
- [ ] Announce release (if applicable)

## Rollback

If a release has critical issues:

1. **Immediate:** Create a new patch release with fixes
2. **If needed:** Yank the release from PyPI:
   ```bash
   twine yank qconduit==0.0.5 --reason "Critical bug"
   ```

## Release Notes Template

```markdown
## [VERSION] - YYYY-MM-DD

### Added
- 

### Changed
- 

### Fixed
- 

### Deprecated
- 

### Removed
- 

### Security
- 
```

