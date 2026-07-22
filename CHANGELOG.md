# Changelog

All notable changes to pfb-model-spec are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.2] - 2026-07-22

### Added

- **model2comps**: Portable WSClean FITS -> .mds converter
- Keep models (X, Y) until spec if formally updated

### CI

- Bump actions/setup-python from 6 to 7 in the github-actions group
- Bump actions/cache from 5 to 6 in the github-actions group
- Bump actions/checkout from 6 to 7 in the github-actions group

### Dependencies

- Bump the python-minor-patch group across 1 directory with 3 updates
- Bump the python-minor-patch group across 1 directory with 2 updates

### Documentation

- Sync guidance with post-migration state

### Fixed

- **model2comps**: Respect --overwrite when writing the sanity FITS

### Miscellaneous

- Also remove roundtrip test for onboard
- Remove placeholder onboard command
- Merge in main
- Move modelspec into utils sub-module
- Update container tag

### Other

- Update hip-cargo dep


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


[0.0.2]: https://github.com/landmanbester/pfb-model-spec/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/landmanbester/pfb-model-spec/releases/tag/v0.0.1

