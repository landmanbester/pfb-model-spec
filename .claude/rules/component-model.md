# The Component Model (`.mds` spec)

Read this when working on `src/pfb_model_spec/utils/{modelspec,io,spec}.py`,
the `.mds` format, or the fit/render routines.

## What it is

The **component model** is a compact representation of a sky model stored as an `.mds`
("model dataset") directory. Instead of a full image cube (`time × freq × stokes × ny × nx`), it stores:

- **coefficients** of a Legendre/polynomial basis over time and frequency,
- the **pixel locations** (`location_x`, `location_y`) of the non-zero components,
- a symbolic **`sympy` parametrisation** plus time/frequency scaling expressions (`texpr`/`fexpr`),
- **geometry metadata** (cell size, npix, centre, ra/dec, flips, Stokes).

From this, the model can be re-rendered to an image at any time, frequency, and grid resolution.

## Spec versions (`utils/spec.py`)

- A spec is named after the `major.minor` of the pfb-model-spec release that introduced it
  (`"0.1"`); `"genesis"` is the legacy name for `"0.0"`, and a missing `spec` attr means genesis.
- **A schema change requires a breaking version bump** (minor while on 0.x). A breaking bump
  that leaves the schema alone still adds a no-op step to `_UPGRADES`, so every spec name is
  defined. `tests/test_spec.py` fails if `SPEC_VERSION` falls behind the package's `major.minor`.
- Upgrades are one-way (older -> newer) pure `Dataset -> Dataset` steps chained by
  `upgrade()`. Every reader goes through `open_mds()`/`upgrade()`, so old `.mds` stores keep
  working; `pfbspec convert --input-mds IN --output-mds OUT` persists the same upgrade.
- To add a spec: bump `SPEC_VERSION`, add the step to `_UPGRADES`, change
  `build_mds_dataset`, and add a frozen fixture for the old spec next to `tests/_genesis.py`.

## Axis convention ((Y, X), spec 0.1)

Every model cube this library builds, consumes or renders is `(..., nstokes, ny, nx)` --
the FITS/astropy order, so callers (pfb-imaging, `model2comps`) no longer transpose.
`location_x` indexes FITS `NAXIS1` and `location_y` `NAXIS2`, in every spec. ducc0's
`dirty2vis` is x-major, so `degrid_stokes` transposes exactly once, before the gridder.

## The library API (`utils/modelspec.py`)

- `fit_image_cube(time, freq, image, wgt=None, nbasist=None, nbasisf=None, method="poly", sigmasq=0)`
  → `(coeffs, y_index, x_index, expr, params, texpr, fexpr)` — fit the time+freq axes of a
  multi-Stokes cube. `image` is `(ntime, nband, nstokes, ny, nx)`, `wgt` `(ntime, nband, nstokes)`,
  `coeffs` `(nstokes, npar, ncomps)`. All Stokes planes share one basis and the union of non-zero
  locations. This is the only fit function (`fit_image_fscube` was removed; use `ntime = 1`).
- `eval_coeffs_to_cube(time, freq, nx, ny, coeffs, y_index, x_index, expr, paramf, texpr, fexpr)`
  → render coefficients to a `(ntime, nfreq, nstokes, ny, nx)` cube.
- `eval_coeffs_to_slice(time, freq, coeffs, y_index, x_index, expr, paramf, texpr, fexpr, *, nxi,
  nyi, cellxi, cellyi, x0i, y0i, nxo, nyo, cellxo, cellyo, x0o, y0o)` → render to a
  `(nstokes, nyo, nxo)` slice, with zero-padding + bilinear resampling onto an arbitrary output
  grid. The geometry arguments are keyword-only.
- `model_from_mds(mds_name, freqs=None)` → open an `.mds` zarr (any known spec, upgraded on read) and render at original resolution.

## The I/O API (`utils/io.py`)

- `model_to_ds(time, freq, fsel, model, wgt, mds_name, cell_rad, nx, ny, x0, y0, flip_u, flip_v,
  flip_w, radec, stokes, writer_version, nbasisf=None, method="Legendre", sigmasq=1e-6)` → fits
  `model[fsel]` via `fit_image_cube`, writes the coefficients to `mds_name` (zarr, `mode="w"`), then
  re-renders the fit at every band in `freq` via `eval_coeffs_to_slice` and returns the resulting
  cube. `model` is `(nband, nstokes, ny, nx)`, `wgt` `(nband, nstokes)`, `stokes` a `list[str]`,
  and the return is `(nband, nstokes, ny, nx)` (see "Axis convention"; no transpose). This is
  what pfb-imaging's `deconv.py` calls each minor cycle to persist and re-evaluate the component
  model — geometry (`x0`/`y0`/flips) is a gridder concern (`wgridder_conventions`) and is passed in
  rather than computed, so `io.py` has no dependency on `pfb_imaging`.
- `build_mds_dataset(coeffs, y_index, x_index, expr, params, texpr, fexpr, time, freq, cell_rad,
  nx, ny, x0, y0, flip_u, flip_v, flip_w, radec, stokes, writer_version)` → the **single owner of the
  `.mds` schema**: assembles the `xarray.Dataset` (data_vars/coords/attrs below) but does not write
  it. Both `model_to_ds` and the `model2comps` converter build through here, so the two write paths
  cannot drift from each other or from `model_from_mds`'s reader.

## The degrid API (`utils/degrid.py`)

Turns a component model into visibilities for one chunk of data. Pure numpy — no dask, no
Ray, no measurement-set handles, no `pfb_imaging` import. Data selection, chunking,
distribution and the MS write belong to the calling application (pfb-imaging's
`degrid-msv4`, ratt-ru/pfb-imaging#278; QuartiCal later).

- `model_geometry(model_ds)` → the `.mds` gridding attrs as a dict (`nx`, `ny`, `cell_rad`,
  `x0`, `y0`, `flip_u/v/w`, `stokes`; `stokes` is a list). Callers pass these to `degrid_stokes` rather than
  reading attrs by hand; raises on non-square pixels.
- `render_model_region(model_ds, *, time, freq_out, nx=…, ny=…, cell_rad=…, x0=…, y0=…)` →
  `(nstokes, ny, nx)`. Wraps `eval_coeffs_to_slice`; the output grid defaults to the model's
  own. `freq_out` may lie between fitted bands — that continuity is what lets a consumer predict
  at finer spectral resolution than the imaging run used.
- `apply_mueller(stokes_image, mueller)` → `(nstokes_out, ny, nx)`. Pixelwise
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

**Specs and axes.** The readers (`model_geometry`, `render_model_region`) upgrade older specs on
read, so a genesis `.mds` degrids unchanged. Images are `(nstokes, ny, nx)`; `degrid_stokes`
transposes to x-major once, right before ducc0's `dirty2vis`. The fused wrapper returns
*visibilities*, which have no image orientation.

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
  cube (astropy, deferred import); the row-major `(ny, nx)` planes already match the spec's
  `(Y, X)` order, so there is no transpose on read. `product` must be a single Stokes parameter
  (I/Q/U/V) and a length-1 Stokes axis is written. The fit → `build_mds_dataset` → `to_zarr` path writes the `.mds`, and a sanity model
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
  or schema change as a versioned spec revision (see "Spec versions"), never a silent edit.
- Behavioural changes to the numerics are contract changes; cosmetic `ruff` formatting is not.

## The `.mds` schema (spec 0.1)

Owned by `build_mds_dataset`; `open_mds`/`model_from_mds` read it, so the field names must not drift:

| | `genesis` | `0.1` |
|---|---|---|
| `coefficients` | dims `(par, comps)` | dims `(stokes, par, comps)` |
| Stokes | `stokes` attr, a single product, e.g. `"I"` | `stokes` coord, dims `(stokes,)`, e.g. `["I", "Q", "U", "V"]`; no `stokes` attr |
| `location_x` | dims `(x,)` | dims `(comps,)` |
| `location_y` | dims `(y,)` | dims `(comps,)` |
| `params`, `times`, `freqs` | dims `(par,)`, `(t,)`, `(f,)` | unchanged |
| rendered cubes | x-major `(…, nx, ny)` | `(…, ny, nx)`, matching FITS/astropy |
| version attr | `pfb-imaging-version` | `writer-version` |
| `spec` attr | `"genesis"` (or missing) | `"0.1"` |

Unchanged attrs: `parametrisation`, `texpr`, `fexpr`, `cell_rad_x`, `cell_rad_y`, `npix_x`,
`npix_y`, `center_x`, `center_y`, `ra`, `dec`, `flip_u`, `flip_v`, `flip_w`.

`location_x`/`location_y` mean the same in both specs (`location_x` indexes FITS `NAXIS1`), so an
upgrade never changes stored values. Stokes planes share components and basis: locations are the
union of pixels non-zero in any plane, and `params`/`parametrisation`/`texpr`/`fexpr` are shared.

## Deferred scope (not yet implemented)

- the pfb-imaging **`.dds` reading path** (coupled to pfb-imaging's dataset format and its heavier
  deps — `daskms`, `ducc0`); the `model2comps` converter migrated only the portable FITS-input path;
- ~~a shared **`.mds` reader** … for pfb-imaging's `degrid` and QuartiCal~~ — **built**, see
  "The degrid API" above (`model_geometry` + `render_model_region` replace the inline
  `parse_expr`/`lambdify` schema read).

## Testing

The library is tested with **synthetic, measurement-set-free** data. `tests/test_modelspec.py`
(+ `tests/_synth.py`): a multi-Gaussian, power-law cube is fit and rendered back, asserting an exact
round-trip and integer-pixel-shift interpolation invariance. `tests/_genesis.py` is a frozen writer of the genesis `.mds` (never updated to follow the library);
`tests/test_spec.py` covers the upgrade registry (lossless genesis → 0.1, error messages, the
`SPEC_VERSION` guard) and `tests/test_convert.py` the `pfbspec convert` command. `tests/test_io.py` covers `model_to_ds`
the same way, additionally asserting the written `.mds` zarr's attrs/coords. `tests/test_model2comps.py`
writes a synthetic cube out as WSClean `-####-model.fits` planes, runs the converter, and asserts the
`.mds` round-trips (exact for an `nbasisf == nband`, `sigmasq == 0` fit) plus the overwrite/no-image
guards. No MS / `daskms` / `africanus` needed. Tests require the `full` extra
(`uv run --extra full pytest`).
