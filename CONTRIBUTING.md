# Contributing to Quantum Conduit

Thank you for your interest in contributing to Quantum Conduit! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/seansimms/Quantum_Conduit.git
   cd Quantum_Conduit
   ```

2. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks:**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

   This will automatically run linting and formatting checks before each commit.

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Write docstrings for all public functions and classes
- Keep functions focused and modular
- Maximum line length: 100 characters

### Physics Notation

We allow single-letter uppercase variables (H, X, Y, Z, T, etc.) in modules that use physics notation. These are configured in `pyproject.toml` via ruff's `per-file-ignores`.

## Pre-commit Hooks

We use pre-commit to ensure code quality. The hooks run:

- **ruff**: Linting and formatting
- **mypy**: Type checking (non-blocking)
- **pre-commit-hooks**: Trailing whitespace, end-of-file fixes, etc.
- **detect-secrets**: Secret scanning

To run hooks manually:
```bash
pre-commit run --all-files
```

## Testing

- Add tests for all new features
- Ensure all tests pass before submitting PR
- Aim for high test coverage (≥85% for new code)
- Use the `rng` and `torch_rng` fixtures from `tests/conftest.py` for deterministic tests

Run tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=qconduit --cov-report=html
```

## Pull Request Process

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Run tests and linting:**
   ```bash
   pytest
   ruff check qconduit tests
   ```

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

   Use conventional commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `test:` for tests
   - `style:` for formatting
   - `refactor:` for code refactoring

5. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** on GitHub

   - Include a clear description of your changes
   - Reference any related issues
   - Ensure CI checks pass

## CI/CD

Our CI pipeline runs:
- Linting (ruff)
- Type checking (mypy, non-blocking)
- Tests (pytest with coverage)
- Security scans (bandit, pip-audit, detect-secrets)
- Package building

All checks must pass before a PR can be merged.

## Documentation

- Update docstrings for any changed functions/classes
- Add examples to docstrings where helpful
- Update README.md if adding new features
- Add entries to CHANGELOG.md for user-facing changes

## Questions?

Feel free to open an issue for questions or discussions about contributions.

