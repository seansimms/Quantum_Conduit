# Changelog

All notable changes to Quantum Conduit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Deterministic RNG fixtures for tests (`tests/conftest.py`)
- Pre-commit hooks configuration (`.pre-commit-config.yaml`)
- Enhanced CI workflows with coverage reporting
- Security scanning workflows (bandit, pip-audit, detect-secrets)
- Release workflow for automated PyPI publishing
- Documentation: CONTRIBUTING.md, API.md, RELEASE.md

### Fixed
- Fixed F821 undefined-name error in `qconduit/graphs/shortest.py` (missing `List` import)
- Fixed linting issues: unused imports, unused variables, line length
- Added physics notation exceptions in ruff config for DSP, graphs, probabilistic modules

### Changed
- Enhanced CI workflow with separate lint, typecheck, test, and build jobs
- Added caching for pip and pytest in CI
- Updated ruff configuration with per-file-ignores for physics notation

## [0.0.4] - 2025-11-XX

### Added
- Initial beta release
- Comprehensive quantum computing features
- Classical scientific computing modules
- PyTorch-native integration

[Unreleased]: https://github.com/seansimms/Quantum_Conduit/compare/v0.0.4...HEAD
[0.0.4]: https://github.com/seansimms/Quantum_Conduit/releases/tag/v0.0.4

