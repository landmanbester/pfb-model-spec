# The Component Model (`.mds` spec)

Read this when working on `src/pfb_model_spec/utils/modelspec.py`, `src/pfb_model_spec/utils/io.py`,
the `.mds` format, or the fit/render routines.

## What it is

The **component model** is a compact representation of a sky model stored as an `.mds`
("model dataset") directory. Instead of a full image cube (`time × freq × nx × ny`), it stores:

- **coefficients** of a Legendre/polynomial basis over time and frequency,
- the **pixel locations** (`location_x`, `location_y`) of the non-zero components,
- a symbolic **`sympy` parametrisation** plus time/frequency scaling expressions (`texpr`/`fexpr`),
- **geometry metadata** (cell size, npix, centre, ra/dec, flips, Stokes).

From this, the model can be re-rendered to an image at any time, frequency, and grid resolution.

## Axis convention (x-major)

The current (`"genesis"`) `.mds` spec is **x-major**: `location_x`/`location_y` and every model
cube this library builds or consumes are `(nband, nx, ny)`-ordered. This is **not** pfb-imaging's
internal `(Y, X)` raster convention — pfb-model-spec deliberately does not transpose to match it.
Callers on a `(Y, X)` cube (e.g. pfb-imaging) must transpose to/from `(nband, nx, ny)` at their own
call site; do not add a transpose inside this library to paper over that mismatch. A future spec
revision is expected to flip the `.mds` format itself to `(Y, X)`, with conversion handled by the
planned `model2comps` converter (https://github.com/landmanbester/pfb-model-spec/issues/17) — this
doc and `utils/io.py`'s docstrings should be updated together when that lands.

## The library API (`utils/modelspec.py`)

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

## The I/O API (`utils/io.py`)

- `model_to_ds(time, freq, fsel, model, wgt, mds_name, cell_rad, nx, ny, x0, y0, flip_u, flip_v,
  flip_w, radec, stokes, version, nbasisf=None, method="Legendre", sigmasq=1e-6)` → fits
  `model[fsel]` via `fit_image_cube`, writes the coefficients to `mds_name` (zarr, `mode="w"`), then
  re-renders the fit at every band in `freq` via `eval_coeffs_to_slice` and returns the resulting
  `(nband, nx, ny)` cube (x-major — see "Axis convention" above; no internal transpose). This is
  what pfb-imaging's `deconv.py` calls each minor cycle to persist and re-evaluate the component
  model — geometry (`x0`/`y0`/flips) is a gridder concern (`wgridder_conventions`) and is passed in
  rather than computed, so `io.py` has no dependency on `pfb_imaging`.
- `build_mds_dataset(coeffs, x_index, y_index, expr, params, texpr, fexpr, time, freq, cell_rad,
  nx, ny, x0, y0, flip_u, flip_v, flip_w, radec, stokes, version)` → the **single owner of the
  `.mds` schema**: assembles the `xarray.Dataset` (data_vars/coords/attrs below) but does not write
  it. Both `model_to_ds` and the `model2comps` converter build through here, so the two write paths
  cannot drift from each other or from `model_from_mds`'s reader.

## The degrid API (`utils/degrid.py`)

Turns a component model into visibilities for one chunk of data. Pure numpy — no dask, no
Ray, no measurement-set handles, no `pfb_imaging` import. Data selection, chunking,
distribution and the MS write belong to the calling application (pfb-imaging's
`degrid-msv4`, ratt-ru/pfb-imaging#278; QuartiCal later).

- `model_geometry(model_ds)` → the `.mds` gridding attrs as a dict (`nx`, `ny`, `cell_rad`,
  `x0`, `y0`, `flip_u/v/w`, `stokes`). Callers pass these to `degrid_stokes` rather than
  reading attrs by hand; raises on non-square pixels.
- `render_model_region(model_ds, *, time, freq_out, nx=…, ny=…, cell_rad=…, x0=…, y0=…)` →
  `(nstokes, nx, ny)`. Wraps `eval_coeffs_to_slice`; the output grid defaults to the model's
  own. `freq_out` may lie between fitted bands — that continuity is what lets a consumer predict
  at finer spectral resolution than the imaging run used.
- `apply_mueller(stokes_image, mueller)` → `(nstokes_out, nx, ny)`. Pixelwise
  `apparent[i] = Σ_j mueller[i,j]·intrinsic[j]`. Never builds a beam and never folds the
  wgridder's `1/n` term; a caller that folds `1/n` into its beam must keep
  `divide_by_n=False` in `degrid_stokes`.
- `degrid_stokes(uvw, freq, stokes_image, *, cell_rad, x0, y0, flip_*, …, mask=None)` →
  `(nstokes, nrow, nchan)`. One `dirty2vis` per Stokes plane; empty planes are skipped.
  `mask` is `(nrow, nchan)` and exists because xarray-ms pads absent `(time, baseline)`
  cells with NaN UVW.
- `stokes_vis_to_corr(stokes_vis, stokes_in, corr_types)` → `(nrow, nchan, ncorr)`. Exact
  linear map; Stokes products absent from `stokes_in` are zero, so an I-only model gives
  `XX == YY == I`, `XY == YX == 0`. The coefficients match
  `africanus.model.coherency.convert` (verified elementwise) but are hard-coded, because
  tests here must not depend on africanus.
- `model_to_apparent_vis_for_region(…)` — the fused per-chunk wrapper over the above.

**Why five functions and not one.** Three consumers need pieces rather than the whole:
`--transfer-model-from` (ratt-ru/pfb-imaging#309) needs only `render_model_region`;
region-file degridding (ratt-ru/pfb-imaging#115) renders once and degrids N+1 times behind
different masks; chunks sharing a `(time, freq)` bin can reuse one rendering.

**Axis convention.** x-major throughout, like the rest of this library — and ducc0's
`dirty2vis` is also x-major, so there is no transpose anywhere in this module. Callers on a
`(Y, X)` raster (pfb-imaging) transpose at their own call site; note that the fused wrapper
returns *visibilities*, which have no image orientation, so a degridding consumer never
transposes at all.

**Representative time/frequency.** `time`/`freq_out` are the caller's choice for a chunk,
conventionally the **unweighted** means of its axes. Unweighted is deliberate: it is
reproducible across consumers regardless of their flagging, so two applications cannot
disagree about where the model was evaluated. (This differs from pfb-imaging's D28
weight-weighted effective frequency, which is an *imager* rule that applies where weights
are in hand.)

**Known limitation — resampling is not flux-conserving when coarsening.**
`eval_coeffs_to_slice` scales by `area_ratio = pix_area_out / pix_area_in`, correct for a
surface-brightness field but not for the point components a `.mds` stores. Measured with a
single unit component: same grid → 1.0 (exact); refine ×2 → 1.0 (conserved); coarsen ×2 →
4.0; coarsen ×4 → 16.0, i.e. inflated by exactly `area_ratio`. Degridding always renders on
the model's own grid and is unaffected, but `--transfer-model-from`
(ratt-ru/pfb-imaging#309) will hit it.

## The converter (`core/model2comps.py`, `utils/fits.py`)

- `model2comps(output_filename, from_fits, ...)` (core) — the portable **WSClean FITS → `.mds`**
  converter (`pfbspec model2comps`). `read_wsclean_model` reads a `{from_fits}-####-model.fits`
  cube (astropy, deferred import), transposing each row-major `(ny, nx)` plane to the spec's x-major
  `(nx, ny)`; the fit → `build_mds_dataset` → `to_zarr` path writes the `.mds`, and a sanity model
  FITS is rendered via `utils/fits.py`. It has **no** `.dds`/daskms/ducc0 coupling — the legacy
  `.dds`-input path from pfb-imaging was intentionally dropped (deconvolvers write `.mds` directly
  via `model_to_ds`; see ratt-ru/pfb-imaging#286).
- `utils/fits.py` (`save_fits`, `set_wcs`, `to4d`) — a minimal astropy-only FITS writer for model
  cubes. Deliberately excludes restoring-beam parametrisation, CASA beam tables, and MS-time
  handling (a component model has none of those). No dependency on pfb-imaging.

## Ownership (canonical, not vendored)

`utils/modelspec.py` **was** a byte-for-byte vendored copy of `pfb_imaging/utils/modelspec.py`.
That phase is over: pfb-imaging now imports the library from this package (and deleted its own
copy — ratt-ru/pfb-imaging#286), so **pfb-model-spec is the canonical owner**. There is no longer a
copy to keep in sync; the "re-copy verbatim to re-sync" rule is retired.

- **Public function signatures, return tuples, and the `.mds` schema are a cross-repo contract.**
  Changing any of them is a breaking change for pfb-imaging — coordinate, and treat an axis-order
  or schema change as a versioned spec revision (see "Axis convention", #17), never a silent edit.
- Behavioural changes to the numerics are contract changes; cosmetic `ruff` formatting is not.

## The `.mds` schema

Owned by the (deferred) converter; `model_from_mds` reads it, so the field names must not drift:

- **data_vars:** `coefficients` (dims `par`, `comps`)
- **coords:** `location_x`, `location_y`, `params`, `times`, `freqs`
- **attrs:** `parametrisation`, `texpr`, `fexpr`, `cell_rad_x`, `cell_rad_y`, `npix_x`, `npix_y`,
  `center_x`, `center_y`, `ra`, `dec`, `flip_u`, `flip_v`, `flip_w`, `stokes`, `spec`,
  `pfb-imaging-version`

## Deferred scope (not yet implemented)

- the pfb-imaging **`.dds` reading path** (coupled to pfb-imaging's dataset format and its heavier
  deps — `daskms`, `ducc0`); the `model2comps` converter migrated only the portable FITS-input path;
- ~~a shared **`.mds` reader** … for pfb-imaging's `degrid` and QuartiCal~~ — **built**, see
  "The degrid API" above (`model_geometry` + `render_model_region` replace the inline
  `parse_expr`/`lambdify` schema read).

## Testing

The library is tested with **synthetic, measurement-set-free** data. `tests/test_modelspec.py`
(+ `tests/_synth.py`): a multi-Gaussian, power-law cube is fit and rendered back, asserting an exact
round-trip and integer-pixel-shift interpolation invariance. `tests/test_io.py` covers `model_to_ds`
the same way, additionally asserting the written `.mds` zarr's attrs/coords. `tests/test_model2comps.py`
writes a synthetic cube out as WSClean `-####-model.fits` planes, runs the converter, and asserts the
`.mds` round-trips (exact for an `nbasisf == nband`, `sigmasq == 0` fit) plus the overwrite/no-image
guards. No MS / `daskms` / `africanus` needed. Tests require the `full` extra
(`uv run --extra full pytest`).
