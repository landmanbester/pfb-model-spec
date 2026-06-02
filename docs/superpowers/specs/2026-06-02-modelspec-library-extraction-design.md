# Design: Extract the component-model spec library into pfb-model-spec (phase 1)

**Date:** 2026-06-02
**Status:** Approved (design); pending spec review
**Scope:** Phase 1 — portable fit/render library + tests only.

## Background

`pfb-imaging` represents a sky model compactly as a **component model** stored in
an `.mds` ("model dataset") directory. Instead of a full image cube
(`time × freq × nx × ny`), an `.mds` stores Legendre/polynomial coefficients over
time and frequency, the pixel locations of non-zero components, a symbolic
`sympy` parametrisation (with time/frequency scaling expressions), and geometry
metadata. From this it can re-render an image at any frequency, time, and grid
resolution.

The portable heart of this is `pfb_imaging/utils/modelspec.py`, which contains:

- `fit_image_cube` — fit time+frequency axes of an image cube → coefficients.
- `fit_image_fscube` — fit the frequency axis of a (freq, corr, nx, ny) cube.
- `eval_coeffs_to_cube` — render coefficients back to a `(ntime, nfreq, nx, ny)` cube.
- `eval_coeffs_to_slice` — render coefficients to a single 2D slice, with
  padding + bilinear resampling to an arbitrary output grid.
- `model_from_mds` — open an `.mds` zarr and render it at original resolution.

This module imports only `numpy`, `sympy`, `scipy`, and `xarray` — **no
`pfb_imaging` references** — so it is fully portable.

In `pfb-imaging` these functions are used well beyond the `model2comps`
converter: `grid.py`, `degrid.py`, and `operators/gridder.py` consume `.mds`
files, and `deconv.py`/`kclean.py`/`sara.py`/`fluxtractor.py` write `.mds`
inline via `fit_image_cube`. The long-term intent is for `pfb-imaging` to
**depend on** `pfb-model-spec` for this functionality rather than duplicate it.

## Goals (phase 1)

1. Move the portable component-model **spec library** (`modelspec.py`) into
   `pfb-model-spec` with **identical public function signatures and the `.mds`
   schema preserved**, so `pfb-imaging` can later swap to importing it by
   changing import lines only — never call sites.
2. Provide a **self-contained, lightweight test** of the fit/render numerics
   (no measurement set, no `daskms`/`africanus`/`ducc0`).
3. Keep the lightweight install (`hip-cargo` + `typer`) intact, and adhere to
   hip-cargo design principles throughout.

## Non-goals (deferred to later phases)

- The `model2comps` CLI command and its generated cab.
- The `.dds` reading path (intrinsically coupled to `pfb-imaging`'s data format
  and its `daskms`/`ducc0` dependencies).
- FITS I/O (`save_fits` / `set_wcs`), `set_output_names` / `xds_from_url`,
  custom logging, and `set_envs` thread/TBB setup.
- Any change to `pfb-imaging` (it remains untouched in this phase).
- The optional `.mds` schema-guard round-trip test (see "Future additions").

## Architecture

### Module placement (Approach A — approved)

```
src/pfb_model_spec/
├── __init__.py          # stays light: only __version__. Does NOT import modelspec.
├── modelspec.py         # NEW — verbatim copy of pfb_imaging/utils/modelspec.py
├── cli/                 # unchanged (onboard only)
├── core/                # unchanged (onboard only)
└── cabs/                # unchanged (onboard only)
```

- Public API: `from pfb_model_spec.modelspec import fit_image_cube,
  fit_image_fscube, eval_coeffs_to_cube, eval_coeffs_to_slice, model_from_mds`.
- The whole module is copied (including the currently-unused-in-pfb-imaging
  `fit_image_fscube`) to remain a faithful drop-in.
- **Heavy imports stay out of the top-level `__init__.py`.** `import
  pfb_model_spec` must not pull `numpy`/`sympy`/`scipy`/`xarray`, so the
  lightweight install and cab generation are unaffected.

### Dependencies

Add to `[project.optional-dependencies].full` in `pyproject.toml`:

- `numpy`, `sympy`, `scipy`, `xarray`

No `daskms`, `ducc0`, `numba`, `astropy`, `casacore`, or `psutil`. The "slim
down" decision means we never pull those for the spec library.

The top-level `dependencies` list (`hip-cargo`) is unchanged.

### Testing

`tests/test_modelspec.py` ports the numerics of `pfb-imaging`'s
`tests/test_model2comps.py`, but replaces the MS-derived fixtures
(`ms_meta` / `image_geometry`) with **synthetic, hard-coded** inputs:

- `nchan` (small, e.g. 8), a linear `freq` array, `freq0 = freq.mean()`,
  a `cell_deg`, and `nx = ny` (even, modest size).
- Build a multi-Gaussian source model cube with a power-law spectrum.
- Round-trip: `fit_image_cube` → `eval_coeffs_to_cube`, assert the rendered
  cube matches the model on the source mask (`assert_allclose`, same tolerance).
- Spatial interpolation: shift the centre by an integer number of pixels via
  `eval_coeffs_to_slice` and assert pixel centres are preserved across a few
  output sizes (mirrors the existing test).

The two pure-numpy helpers the test needs — `gaussian2d` and `give_edges` —
are vendored into a **test-only** module `tests/_synth.py` (not part of the
public API). They are copied verbatim from `pfb_imaging/utils/misc.py`.

The test requires the `[full]` deps (`numpy` etc.) to be installed in the test
environment. The existing `tests/test_roundtrip.py` (onboard) and
`tests/test_install.py` are unaffected — no new CLI command means no new cab.

## Data flow (unchanged contract)

```
image cube (ntime, nband, nx, ny)
        │  fit_image_cube(time, freq, image, wgt, nbasisf, method, sigmasq)
        ▼
coeffs + (x_index, y_index) + expr + params + texpr + fexpr      ← the .mds payload
        │  eval_coeffs_to_cube / eval_coeffs_to_slice / model_from_mds
        ▼
rendered image (cube or single slice, arbitrary output grid)
```

The `.mds` schema (data_vars `coefficients`; coords `location_x`, `location_y`,
`params`, `times`, `freqs`; attrs incl. `parametrisation`, `texpr`, `fexpr`,
`cell_rad_x/y`, `npix_x/y`, `center_x/y`, `ra`, `dec`, flips, `stokes`, `spec`)
is **owned by the converter** (deferred), but `model_from_mds` reads it, so the
schema field names must not drift from `pfb-imaging`.

## Error handling

The library functions retain `pfb-imaging`'s behaviour: `fit_image_cube` /
`fit_image_fscube` raise `NotImplementedError` for unknown `method`, and
`numpy.linalg.LinAlgError` may propagate from `np.linalg.solve` on a singular
Hessian (callers handle this; the library does not swallow it). No new error
handling is introduced.

## Verification

- `uv run ruff format . && uv run ruff check . --fix` passes cleanly.
- `uv run pytest -v tests/` passes (new `test_modelspec.py` plus the existing
  roundtrip/install tests).
- `import pfb_model_spec` still works in a lightweight (non-`[full]`) env, i.e.
  it does not import `modelspec`.

## Future additions ("add what we need later")

- `.mds` schema-guard test: hand-build a minimal `.mds` `xr.Dataset`, write to a
  temp zarr, and round-trip through `model_from_mds` to pin the schema contract.
- A `model2comps` (or `render` / `comps2model`) CLI command + cab, including the
  portable FITS path, following the hip-cargo cli/core split and round-trip test.
- An adapter for the `.dds` path if/when `pfb-imaging` delegates to this package.
