# Quantum Conduit API Stability Policy

**Version:** 0.0.4 (Beta)  
**Last Updated:** November 2025

## Stability Guarantees

Quantum Conduit is currently in **beta** (v0.0.4). API stability is not guaranteed until v0.1.0.

### Current Status (v0.0.4)

- **Breaking changes may occur** without deprecation warnings
- **API may evolve** based on user feedback
- **Version pinning recommended** for production use

### Future Status (v0.1.0+)

- **Breaking changes** will be announced in CHANGELOG.md
- **Deprecation warnings** will be provided for at least one minor version before removal
- **Migration guides** will be provided for major breaking changes

## Deprecation Policy

When an API is deprecated:

1. **Deprecation warning** added with clear message
2. **Documentation** updated to mark as deprecated
3. **Alternative** API recommended in deprecation message
4. **Removal** after at least one minor version (e.g., deprecated in 0.1.0, removed in 0.2.0)

## Public API

The public API consists of:

- All symbols exported in `qconduit/__init__.py`
- All public functions and classes in submodules (not prefixed with `_`)
- Stable interfaces documented in README.md

### Internal API

- Functions/classes prefixed with `_` are internal and may change without notice
- Private modules are not part of the public API

## Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (x.0.0): Breaking changes
- **MINOR** (0.x.0): New features, backward compatible
- **PATCH** (0.0.x): Bug fixes, backward compatible

## Physics Notation Conventions

Single-letter uppercase variables (H, X, Y, Z, T, etc.) are allowed in modules using physics notation. These are intentionally exempt from standard naming conventions and are configured in `pyproject.toml`.

## Reporting Issues

If you encounter breaking changes or API issues:

1. Check CHANGELOG.md for announcements
2. Open an issue on GitHub
3. Include version information and minimal reproduction code

## Migration Guides

Migration guides for breaking changes will be provided in:
- CHANGELOG.md (for minor changes)
- MIGRATION_GUIDES.md (for major changes)

