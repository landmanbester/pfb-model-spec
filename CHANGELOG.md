# Changelog

All notable changes to pfb-model-spec are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-05

### Added

- **degrid**: Multi-region degrid with per-row beam terms (#27)
- **degrid**: Separable bilinear interpolation of a coarse Mueller block
- **degrid**: Region crop helpers move here from pfb-imaging (#27)
- **degrid**: Pixel_lm, the sky coordinates of a model grid
- Band-average-preserving box/tent frequency fits, tent by default (#29)
- Fake-nodes basis so the model can be rendered between band centres (#29)
- Remap coefficients of linear parametrisations onto a new grid
- Minmod-limited tent kernel for the flux-conserving remap
- Flux-conserving box remap between rectilinear grids (#23)
- Optional sampling fields and flux_scale in .mds spec 0.1
- Add pfbspec convert to upgrade .mds datasets to the current spec
- Add pfbspec convert to upgrade .mds datasets to the current spec
- Model2comps writes spec 0.1 (Y, X) models with a Stokes axis
- Degrid (ny, nx) spec 0.1 models and upgrade older specs on read
- Write spec 0.1 .mds datasets from build_mds_dataset and model_to_ds
- Merge fit functions into a multi-Stokes (Y, X) fit_image_cube
- Add .mds spec registry with genesis -> 0.1 upgrade

### CI

- Bump ruff-pre-commit to v0.17.0 to match the locked ruff

### Changed

- **degrid**: Fold model_to_apparent_vis_for_region into the plural wrapper
- **degrid**: Interp_mueller keeps the beam block's precision

### Dependencies

- Bump the python-minor-patch group across 1 directory with 3 updates
- Bump ruff
- Bump ruff
- Update uv-build requirement
- Bump the python-minor-patch group across 1 directory with 3 updates
- Update uv-build requirement

### Documentation

- The degrid API has more than five functions now
- The multi-region beam API
- Bring agent docs up to date with the flux-conserving remap
- Implementation plan for flux-conserving remap (#23)
- Rename remap kernel to tent, note pfb-imaging coarsening guard
- Design for flux-conserving spatial remapping (#23)
- Document spec versioning, convert, and the 0.1 schema
- Implementation plan for spec versioning, convert and spec 0.1
- Design for .mds spec versioning, convert command and spec 0.1

### Fixed

- **degrid**: Require stokes_out when beam_terms are given
- **degrid**: Per-plane beam degrid, complex Mueller in the singular wrapper
- Anchor each tent piece at its band's freq, not the edge midpoint (#29)
- Conserve flux when rendering a model onto a different grid (#23)
- Refuse nested convert stores, fit zero-weight single samples, upgrade once in degrid wrapper
- Remote filesystems in full extra, clear genesis upgrade errors, model_to_ds shape checks, rollout notes

### Testing

- **degrid**: Restore the empty-mask crop assertions


## [0.0.3] - 2026-09-11

### Added

- **degrid**: Add the fused model-to-apparent-visibility entry point
- **degrid**: Apply a Stokes-basis Mueller beam to a model
- **degrid**: Render a component model onto an arbitrary grid
- **degrid**: Degrid Stokes images to visibilities with ducc0
- **degrid**: Add Stokes to correlation conversion

### CI

- Exclude Markdown from ruff formatting

### Dependencies

- Bump the python-minor-patch group with 3 updates

### Documentation

- Document the degrid kernel API
- **degrid**: Document the exceptions the fused entry point raises
- Add the degrid kernel implementation plan

### Fixed

- **degrid**: Let callers name the apparent Stokes products

### Other

- Revise component model function documentation

Co-authored-by: Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>


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


[0.1.0]: https://github.com/landmanbester/pfb-model-spec/compare/v0.0.3...v0.1.0
[0.0.3]: https://github.com/landmanbester/pfb-model-spec/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/landmanbester/pfb-model-spec/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/landmanbester/pfb-model-spec/releases/tag/v0.0.1

