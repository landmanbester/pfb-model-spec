# The Component Model (`.mds` spec)

Read this when working on `src/pfb_model_spec/modelspec.py`, the `.mds` format, or the fit/render
routines.

## What it is

The **component model** is a compact representation of a sky model stored as an `.mds`
("model dataset") directory. Instead of a full image cube (`time × freq × nx × ny`), it stores:

- **coefficients** of a Legendre/polynomial basis over time and frequency,
- the **pixel locations** (`location_x`, `location_y`) of the non-zero components,
- a symbolic **`sympy` parametrisation** plus time/frequency scaling expressions (`texpr`/`fexpr`),
- **geometry metadata** (cell size, npix, centre, ra/dec, flips, Stokes).

From this, the model can be re-rendered to an image at any time, frequency, and grid resolution.

## The library API (`modelspec.py`)

- `fit_image_cube(time, freq, image, wgt=None, nbasist=None, nbasisf=None, method="poly", sigmasq=0)`
  → `(coeffs, x_index, y_index, expr, params, texpr, fexpr)` — fit the time+freq axes of a cube.
- `fit_image_fscube(freq, image, wgt=None, nbasisf=None, method="Legendre", sigmasq=0)` — fit the
  frequency axis of a `(freq, corr, nx, ny)` cube. **Currently unused by callers**, kept verbatim
  for drop-in fidelity.
- `eval_coeffs_to_cube(time, freq, nx, ny, coeffs, x_index, y_index, expr, paramf, texpr, fexpr)`
  → render coefficients to a `(ntime, nfreq, nx, ny)` cube.
- `eval_coeffs_to_slice(...)` → render coefficients to a single 2D slice, with zero-padding +
  bilinear resampling onto an arbitrary output grid.
- `model_from_mds(mds_name, freqs=None)` → open an `.mds` zarr and render at original resolution.

## Drop-in fidelity rule (important)

`modelspec.py` is a **byte-identical vendored copy** of `pfb_imaging/utils/modelspec.py`. The goal
is for pfb-imaging to later import this module unchanged (changing only its import lines, never call
sites). Therefore:

- **Do not change public function signatures, return tuples, or the `.mds` schema** without
  coordinating with pfb-imaging.
- Cosmetic `ruff` formatting is fine; behavioural changes are not.
- To re-sync after an upstream change, re-copy the file verbatim rather than hand-editing.

## The `.mds` schema

Owned by the (deferred) converter; `model_from_mds` reads it, so the field names must not drift:

- **data_vars:** `coefficients` (dims `par`, `comps`)
- **coords:** `location_x`, `location_y`, `params`, `times`, `freqs`
- **attrs:** `parametrisation`, `texpr`, `fexpr`, `cell_rad_x`, `cell_rad_y`, `npix_x`, `npix_y`,
  `center_x`, `center_y`, `ra`, `dec`, `flip_u`, `flip_v`, `flip_w`, `stokes`, `spec`,
  `pfb-imaging-version`

## Deferred scope (not yet implemented)

These were intentionally left for later phases (see
`docs/superpowers/specs/2026-06-02-modelspec-library-extraction-design.md`):

- the **`model2comps` CLI converter** (pixelated image → `.mds`), including the portable
  WSClean-style FITS input path;
- the pfb-imaging **`.dds` reading path** (coupled to pfb-imaging's dataset format and its heavier
  deps — `daskms`, `ducc0`);
- **FITS I/O** (`save_fits` / `set_wcs`) and an optional `.mds` schema-guard test through
  `model_from_mds`.

## Testing

The spec library is tested with **synthetic, measurement-set-free** data (`tests/test_modelspec.py`
+ `tests/_synth.py`): a multi-Gaussian, power-law cube is fit and rendered back, asserting an exact
round-trip and integer-pixel-shift interpolation invariance. No MS / `daskms` / `africanus` needed.
Tests require the `full` extra (`uv run --extra full pytest`).
