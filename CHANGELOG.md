# Changelog

All notable changes to pfb-model-spec are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2026-06-02

### Added

- Vendor component-model spec library with synthetic tests

### CI

- Add LICENSE to build context
- Enforce conventional commits via pre-commit commit-msg hook
- Update dependabot config and add CODEOWNDERS file
- Install full extra so modelspec tests can import numpy/sympy

### Dependencies

- Update uv-build requirement
- **deps**: Add numpy, scipy, sympy, xarray to full extra

### Documentation

- Update CLAUDE.md and add copilot-instructions
- Add phase-1 modelspec implementation plan
- Add phase-1 modelspec extraction design

### Fixed

- **cli**: Regenerate onboard wrapper from cab definition

### Miscellaneous

- Add changelog tooling and .python-version for release parity
- Update _container_image name
- Initial project scaffold

### Testing

- Guard lightweight top-level import
- Tidy synthetic modelspec test bounds and comments


[0.0.1]: https://github.com/landmanbester/pfb-model-spec/releases/tag/v0.0.1

