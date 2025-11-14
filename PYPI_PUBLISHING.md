# Publishing to PyPI

This guide explains how to publish Quantum Conduit to PyPI.

## Prerequisites

1. **PyPI Account**: Create an account at https://pypi.org/account/register/
2. **TestPyPI Account** (recommended for testing): Create an account at https://test.pypi.org/account/register/
3. **API Token**: Generate an API token for authentication:
   - PyPI: https://pypi.org/manage/account/token/
   - TestPyPI: https://test.pypi.org/manage/account/token/
   - Create a token with scope: "Entire account" or "Project: qconduit"

## Installation

Install the required tools:

```bash
pip install build twine
```

Or install with dev dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Option 1: Using the Publishing Script (Recommended)

1. **Test on TestPyPI first** (recommended):
   ```bash
   python3 publish_to_pypi.py --test
   ```
   This will:
   - Clean previous builds
   - Build the package (source distribution + wheel)
   - Check the package
   - Upload to TestPyPI

2. **Test installation from TestPyPI**:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ qconduit
   ```

3. **Publish to PyPI** (production):
   ```bash
   python3 publish_to_pypi.py
   ```
   You'll be asked to confirm before uploading to production PyPI.

### Option 2: Manual Steps

1. **Clean previous builds**:
   ```bash
   rm -rf build dist *.egg-info
   ```

2. **Build the package**:
   ```bash
   python3 -m build
   ```
   This creates:
   - `dist/qconduit-0.0.1.tar.gz` (source distribution)
   - `dist/qconduit-0.0.1-py3-none-any.whl` (wheel)

3. **Check the package**:
   ```bash
   python3 -m twine check dist/*
   ```

4. **Upload to TestPyPI** (for testing):
   ```bash
   python3 -m twine upload --repository testpypi dist/*
   ```
   When prompted:
   - Username: `__token__`
   - Password: Your TestPyPI API token (starts with `pypi-`)

5. **Test installation**:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ qconduit
   ```

6. **Upload to PyPI** (production):
   ```bash
   python3 -m twine upload dist/*
   ```
   When prompted:
   - Username: `__token__`
   - Password: Your PyPI API token (starts with `pypi-`)

## Using Environment Variables

You can set credentials as environment variables to avoid manual entry:

```bash
export PYPI_USERNAME="__token__"
export PYPI_PASSWORD="pypi-your-actual-token-here"
```

Then run:
```bash
python3 publish_to_pypi.py
```

## Publishing a New Version

1. **Update version** in:
   - `pyproject.toml` → `version = "0.0.2"`
   - `qconduit/__init__.py` → `__version__ = "0.0.2"`

2. **Commit and tag**:
   ```bash
   git add pyproject.toml qconduit/__init__.py
   git commit -m "Bump version to 0.0.2"
   git tag v0.0.2
   git push origin main --tags
   ```

3. **Build and publish**:
   ```bash
   python3 publish_to_pypi.py
   ```

## Package Information

- **Package Name**: `qconduit`
- **PyPI URL**: https://pypi.org/project/qconduit/
- **Installation**: `pip install qconduit`

## Troubleshooting

### Package name already taken
If `qconduit` is already taken on PyPI, you'll need to choose a different name:
1. Update `name` in `pyproject.toml`
2. Consider alternatives like `quantum-conduit`, `qconduit-lib`, etc.

### Upload fails with "File already exists"
- Use `--skip-existing` flag: `python3 publish_to_pypi.py --skip-existing`
- Or increment the version number

### Build fails
- Make sure you have `build` installed: `pip install build`
- Check that `pyproject.toml` is valid
- Ensure all required files are included in `MANIFEST.in`

### Authentication fails
- Make sure you're using `__token__` as the username
- Verify your API token is correct
- Check that the token has the right scope/permissions

## Best Practices

1. **Always test on TestPyPI first** before publishing to production PyPI
2. **Use API tokens** instead of passwords for better security
3. **Keep versions in sync** across `pyproject.toml` and `__init__.py`
4. **Create a GitHub release** after publishing to PyPI
5. **Update changelog** with each new version

## Additional Resources

- [PyPI Documentation](https://packaging.python.org/en/latest/)
- [Python Packaging User Guide](https://packaging.python.org/)
- [Twine Documentation](https://twine.readthedocs.io/)

