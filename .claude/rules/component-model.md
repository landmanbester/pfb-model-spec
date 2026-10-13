# The Component Model (`.mds` spec)

Read this when working on `src/pfb_model_spec/utils/{modelspec,io,spec,remap}.py`,
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
- Optional fields may be added to a spec before its release is tagged; after that, any schema change is a bump.

## Axis convention ((Y, X), spec 0.1)

Every model cube this library builds, consumes or renders is `(..., nstokes, ny, nx)` --
the FITS/astropy order, so callers (pfb-imaging, `model2comps`) no longer transpose.
`location_x` indexes FITS `NAXIS1` and `location_y` `NAXIS2`, in every spec. ducc0's
`dirty2vis` is x-major, so `degrid_stokes` transposes exactly once, before the gridder.

## The library API (`utils/modelspec.py`)

- `fit_image_cube(time, freq, image, wgt=None, nbasist=None, nbasisf=None, method="poly", sigmasq=0,
  *, time_bounds=None, freq_bounds=None)`
  → `(coeffs, y_index, x_index, expr, params, texpr, fexpr)` — fit the time+freq axes of a
  multi-Stokes cube. `image` is `(ntime, nband, nstokes, ny, nx)`, `wgt` `(ntime, nband, nstokes)`,
  `coeffs` `(nstokes, npar, ncomps)`. All Stokes planes share one basis and the union of non-zero
  locations. This is the only fit function (`fit_image_fscube` was removed; use `ntime = 1`).
- `method` is a basis -- `"poly"`, `"Legendre"`, `"fakenodes"` -- or a histopolation, `"box"` /
  `"tent"` (#29). The polynomial bases ring between band centres (Runge) when `nbasisf ≈ nband`.
  `"fakenodes"` fits Chebyshev polynomials in `S(x) = -cos(pi (x - lo) / (hi - lo))` over the
  outer bin edges (the map lives in `texpr`/`fexpr`): stable, but with an O(1/nband) bias
  near the outer edges, where the map is flat.
- **Histopolation** (`"box"`/`"tent"`) treats each band value as the band's *weighted* average
  and `freq` as its weighted frequency (pfb-imaging's D28 `freq_out`, not the edge midpoint).
  The `parametrisation` is a sympy `Piecewise` in `f`, linear in per-band values `m<b>` and
  (tent) slopes `s<b>` (value per half band width), each piece anchored at its band's `freq`,
  with the band edges baked in (`freq_bounds`, else midway between centres), so the schema is
  unchanged and `remap_coefficients` applies. Tent slopes are minmod-limited between
  neighbouring anchors and sign-clipped at the outer edges: rendering at `freq` returns the
  band value exactly (for a linear piece, the weighted average under the weights that define
  `freq`), the error is O(1/nband²), and the sign of the data is kept (non-negative in,
  non-negative out). Needs `ntime == 1` and each `freq` inside its band; ignores
  `nbasist`/`sigmasq`, and `nbasisf` with a warning when it is not `nband`; drops bands with
  zero weight in every plane; holds the end values beyond the outer edges and splits gaps at
  their midpoint.
- `eval_coeffs_to_cube(time, freq, nx, ny, coeffs, y_index, x_index, expr, paramf, texpr, fexpr)`
  → render coefficients to a `(ntime, nfreq, nstokes, ny, nx)` cube.
- `eval_coeffs_to_slice(time, freq, coeffs, y_index, x_index, expr, paramf, texpr, fexpr, *, nxi,
  nyi, cellxi, cellyi, x0i, y0i, nxo, nyo, cellxo, cellyo, x0o, y0o, kernel="box")` → render to a
  `(nstokes, nyo, nxo)` slice on the model's grid, then remap it flux-conservingly onto an
  arbitrary output grid via `remap_image` (see "The remap API"). The geometry arguments are
  keyword-only.
- `model_from_mds(mds_name, freqs=None)` → open an `.mds` zarr (any known spec, upgraded on read) and render at original resolution.

## The I/O API (`utils/io.py`)

- `model_to_ds(time, freq, fsel, model, wgt, mds_name, cell_rad, nx, ny, x0, y0, flip_u, flip_v,
  flip_w, radec, stokes, writer_version, nbasisf=None, method="tent", sigmasq=1e-6, *, freq_bounds=None, time_bounds=None, flux_scale=None)` → fits
  `model[fsel]` via `fit_image_cube`, writes the coefficients to `mds_name` (zarr, `mode="w"`), then
  re-renders the fit at every band in `freq` via `eval_coeffs_to_slice` and returns the resulting
  cube. `model` is `(nband, nstokes, ny, nx)`, `wgt` `(nband, nstokes)`, `stokes` a `list[str]`,
  and the return is `(nband, nstokes, ny, nx)` (see "Axis convention"; no transpose). This is
  what pfb-imaging's `deconv.py` calls each minor cycle to persist and re-evaluate the component
  model — geometry (`x0`/`y0`/flips) is a gridder concern (`wgridder_conventions`) and is passed in
  rather than computed, so `io.py` has no dependency on `pfb_imaging`. The keyword-only
  `freq_bounds`/`time_bounds`/`flux_scale` are forwarded to `build_mds_dataset`
  (`freq_bounds[fsel]` also gives the fit its band edges); `weights` is always
  written from `wgt`, zero outside `fsel`. An all-zero model writes an empty (`comps == 0`) `.mds`
  that renders to zeros.
- `build_mds_dataset(coeffs, y_index, x_index, expr, params, texpr, fexpr, time, freq, cell_rad,
  nx, ny, x0, y0, flip_u, flip_v, flip_w, radec, stokes, writer_version, *, freq_bounds=None, time_bounds=None, weights=None, flux_scale=None)` → the **single owner of the
  `.mds` schema**: assembles the `xarray.Dataset` (data_vars/coords/attrs below) but does not write
  it. Both `model_to_ds` and the `model2comps` converter build through here, so the two write paths
  cannot drift from each other or from `model_from_mds`'s reader. The keyword-only sampling fields
  are validated (`freq_bounds` `(nfreq, 2)`, `time_bounds` `(ntime, 2)`, `weights`
  `(nstokes, ntime, nfreq)`, `flux_scale` in `{"intrinsic", "apparent"}`) and omitted when `None`.

## The degrid API (`utils/degrid.py`)

Turns a component model into visibilities for one chunk of data. Pure numpy — no dask, no
Ray, no measurement-set handles, no `pfb_imaging` import. Data selection, chunking,
distribution and the MS write belong to the calling application (pfb-imaging's
`degrid-msv4`, ratt-ru/pfb-imaging#278; QuartiCal later).

- `model_geometry(model_ds)` → the `.mds` gridding attrs as a dict (`nx`, `ny`, `cell_rad`,
  `x0`, `y0`, `flip_u/v/w`, `stokes`; `stokes` is a list). Callers pass these to `degrid_stokes` rather than
  reading attrs by hand; raises on non-square pixels.
- `model_sampling(model_ds)` → the optional sampling fields as a dict (`freq_bounds`, `time_bounds`,
  `weights`, `flux_scale`), each `None` when absent (always the case for an upgraded genesis `.mds`).
  Consumers derive defaults from these, e.g. a degrid chunk spanning one imaging band.
- `render_model_region(model_ds, *, time, freq_out, nx=…, ny=…, cell_rad=…, x0=…, y0=…, kernel="box")` →
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
- `pixel_lm(geom, *, i0, j0, shape)` → `(l, m)` axes of the model grid or a crop of it. The
  convention is the wgridder's `V = Σ I exp(+2πi(ul + vm + w(n-1))ν/c)`, `l` East and `m` North,
  measured against ducc0 for all four `(flip_u, flip_v)`. Evaluate a beam on these, or an
  asymmetric beam comes out mirrored.
- `interp_mueller(mueller, l_axis, m_axis, l, m)` — separable bilinear interpolation of a coarse
  Mueller block onto the pixel axes (no full-resolution mesh); points outside the coarse axes are 0.
- `crop_bbox(mask)` → `RegionMask(mask, i0, j0)` and `crop_phase_centre(…)` — region cropping and
  the shifted phase centre. They live here since #27 (moved from ratt-ru/pfb-imaging#331).
- `BeamTerm(mueller, l_axis, m_axis, rows=None, conj_rows=None)` — one bare primary beam
  `(nso, nsi, nmb, nlb)`, real or complex, plus the rows it owns. `conj_rows` (a subset of
  `rows`) take `conj(mueller)`, for baseline orderings reversed relative to the beam's. The
  caller builds the beam; this package never depends on a beam library.
- `model_to_apparent_vis_for_regions(…, region_masks, beam_terms, stokes_out, divide_by_n=False)`
  → one `(nrow, nchan, ncorr)` array per region. Renders the model once, crops each region to
  its bounding box with a shifted phase centre, and applies each `BeamTerm` on its rows. A
  complex beam is degridded as Re + i·Im. Rows covered by no term predict zero. **`divide_by_n`
  defaults to `False` and passes straight through: a caller with a bare beam must pass `True`**
  so ducc folds in `1/n`; only a beam that already contains `1/n` keeps `False`.

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

## The remap API (`utils/remap.py`)

Flux-conserving remapping between axis-aligned rectilinear grids (#23). Pixel values are flux
per pixel, so remapping redistributes flux; it does not sample a field. Separable sparse 1-D
overlap weights are applied along x, then y. Flux outside the output footprint is dropped,
never wrapped. NUFFT/sinc resampling was rejected because it rings (negative sidelobes) and is
periodic.

- `remap_image(image, *, kernel="box", nxi, nyi, cellxi, cellyi, x0i, y0i, nxo, nyo, cellxo,
  cellyo, x0o, y0o)` → `(..., nyo, nxo)`. `kernel` is one of `KERNELS`:
  - `"box"`: uniform brightness per input pixel; integer coarsening equals block sums.
  - `"tent"`: piecewise-linear brightness with a minmod-limited slope. It is the identity on the
    same grid. An isolated component is never smeared, and smooth emission
    refines smoothly. The limiter makes it nonlinear, hence the name "tent" and not "linear".
  - Both kernels conserve flux, and map non-negative input to non-negative output (Stokes
    Q/U/V may be negative; that is fine); identical geometry returns a copy.
- `remap_coefficients(coeffs, y_index, x_index, expr, params, *, <geometry>)` →
  `(coeffs_out, y_index_out, x_index_out)`. It remaps `.mds` coefficients directly (box only),
  equivalent to box-remapping every rendering. It applies only to parametrisations that are
  linear and homogeneous in their coefficients, and raises `ValueError` otherwise (e.g. power-law
  models, possibly #29). **The image-space path must always remain available**: nothing may
  depend on the coefficient shortcut.

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
| `freq_bounds` | absent | optional, absent in genesis; coord, dims `(f, bound)`, `[lo, hi]` Hz per band |
| `time_bounds` | absent | optional, absent in genesis; coord, dims `(t, bound)` |
| `weights` | absent | optional, absent in genesis; dims `(stokes, t, f)`, fit weights, 0 = no data |
| `flux_scale` attr | absent | optional, absent in genesis; `"intrinsic"` or `"apparent"` |

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
guards. `tests/test_remap.py` covers the remap: flux conservation (refine/coarsen/non-integer ratios,
offsets, non-square cells), same-grid identity, exact block sums and integer shifts, footprint
drop/straddle, sparsity, the tent kernel's ramp exactness and positivity, and
`remap_coefficients` ≡ box-remapping the rendering; `test_modelspec.py`/`test_degrid.py` assert a unit
component keeps unit flux through `eval_coeffs_to_slice`/`render_model_region`. No MS / `daskms` / `africanus` needed. Tests require the `full` extra
(`uv run --extra full pytest`).
