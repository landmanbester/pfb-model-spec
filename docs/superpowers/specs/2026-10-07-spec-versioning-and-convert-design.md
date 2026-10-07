# Design: `.mds` spec versioning, `pfbspec convert`, and the `0.1` spec

**Date:** 2026-10-07
**Status:** Approved (design); pending spec review
**Issues:** #17 (convert mechanism), #19 (Stokes axis), #20 (`(Y, X)` ordering)
**Release:** pfb-model-spec **0.1.0** (breaking for pfb-imaging)

## Background

Every `.mds` records its schema in `attrs["spec"]`; today the only value is `"genesis"`,
written by `utils/io.py::build_mds_dataset` (the single schema owner). The only reader that
checks it is `utils/degrid.py` (`_SUPPORTED_SPECS = {"genesis"}`). `utils/modelspec.py` carries
a dead `specs = ["genesis", "exodus"]` list.

`genesis` has two known limitations: it holds a single Stokes product (an attr, #19), and it
renders model cubes x-major `(nx, ny)`, so every FITS/astropy/pfb-imaging caller has to
transpose (#20). Fixing either is a schema change, so we first need a mechanism to move old
`.mds` datasets onto a new spec (#17). The new spec doubles as the end-to-end test of that
mechanism.

## Goals

1. A spec-versioning policy and an upgrade registry that converts any older spec to the current
   one (one-way; newer specs may not be expressible in older ones).
2. A `pfbspec convert --input-mds IN --output-mds OUT [--overwrite]` command that detects the input spec from `attrs["spec"]`.
3. Transparent in-memory upgrade on read, so old `.mds` datasets keep working in every reader.
4. Spec `"0.1"`: a Stokes axis and `(Y, X)`-ordered rendered cubes.
5. Merge `fit_image_cube` and `fit_image_fscube` into one function.

## Non-goals

- Downgrading (newer → older spec).
- In-place conversion (`convert` requires distinct input/output paths).
- Multi-Stokes WSClean FITS input to `model2comps` (it writes a length-1 Stokes I axis for now).
- The pfb-imaging adoption PR (listed under "Rollout" for coordination only).
- Fixing #23 (`area_ratio` flux inflation) or #27/#29; they are unaffected by this work.

## Versioning policy

- **The spec name is the `major.minor` of the pfb-model-spec release that introduced it.** A
  spec `"0.1"` dataset is read and written by every `0.1.x` release.
- **A schema change requires a breaking version bump** (the minor number while we are on `0.x`,
  the major number after 1.0).
- **`"genesis"` is the legacy name for `"0.0"`.** A dataset with no `spec` attr is treated as
  `"genesis"`.
- **The spec name is a constant, `SPEC_VERSION`**, not derived from `__version__`. A breaking
  bump that does *not* change the schema (e.g. an API rename → 0.2) must still add a spec step:
  a no-op upgrade `"0.1" → "0.2"`. That keeps every spec name explicitly defined in the registry.
- **A guard test asserts `SPEC_VERSION >= major.minor(__version__)`**, comparing as integer
  tuples. It tolerates the spec running ahead of the package on a development branch (`"0.1"`
  while `pyproject.toml` is still `0.0.3`, until `tbump 0.1.0`). It fails if the package is
  bumped to a new minor without adding a spec step.
- The exact writing version is recorded separately in the `writer-version` attr. This replaces
  the misnamed `pfb-imaging-version` attr.

## The `"0.1"` schema

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
`npix_y`, `center_x`, `center_y`, `ra`, `dec`, `flip_u`, `flip_v`, `flip_w`. They name their
axis explicitly, so axis order does not affect them.

**The meaning of `location_x`/`location_y` does not change.** In both specs `location_x` indexes
the FITS `NAXIS1` axis (`model2comps` transposes FITS `(ny, nx)` planes to `(nx, ny)` today
before fitting). The `(Y, X)` change therefore alters only the axis order of *rendered* cubes and
of cubes passed to the fit, never the stored values. This is what makes the conversion lossless.

**Stokes planes share components and basis.** Locations are the union of pixels that are
non-zero in *any* Stokes plane (a plane may store zeros at a location). All planes use one
basis, so `params`, `parametrisation`, `texpr` and `fexpr` stay shared and `params` stays 1-D.

## Architecture

### `utils/spec.py` (new): registry, upgrade, open

```python
SPEC_VERSION = "0.1"
LEGACY_SPEC = "genesis"

# from-spec -> (to-spec, pure Dataset -> Dataset step)
_UPGRADES: dict[str, tuple[str, Callable[[xr.Dataset], xr.Dataset]]] = {
    "genesis": ("0.1", _genesis_to_0_1),
}

def spec_of(ds: xr.Dataset) -> str: ...      # attrs.get("spec", "genesis")
def upgrade(ds: xr.Dataset) -> xr.Dataset: ...
def open_mds(path: str | os.PathLike) -> xr.Dataset: ...  # xr.open_zarr(path, chunks=None) + upgrade
```

`upgrade(ds)`:
- returns `ds` unchanged if `spec_of(ds) == SPEC_VERSION`;
- otherwise follows `_UPGRADES` step by step until it reaches `SPEC_VERSION`;
- raises `ValueError` if a spec is not in the registry. The message depends on the spec:
  - a version string newer than `SPEC_VERSION` → "this `.mds` was written with spec X and needs
    pfb-model-spec >= X";
  - anything else → "unknown `.mds` spec X; known specs: …".

The order of steps is defined by the registry chain itself; version comparison is used only to
choose the error message.

`_genesis_to_0_1(ds)` is a pure function:
- reads `ds.attrs["stokes"]`; raises `ValueError` unless it names exactly one Stokes product;
- `coefficients` → dims `(stokes, par, comps)` (values `coefficients[None]`);
- adds a `stokes` coord, e.g. `["I"]`;
- moves `location_x` and `location_y` onto the `comps` dim (values unchanged);
- drops the `stokes` attr;
- renames `pfb-imaging-version` → `writer-version`;
- sets `spec = "0.1"`;
- leaves every other attr untouched.

`utils/spec.py` imports xarray at module top. It lives in `utils/`, so that is allowed; `cli/`
imports it only lazily.

### `pfbspec convert`

- `cli/convert.py`: a `@stimela_cab` Typer wrapper. `input` and `output` are `Directory`/`URI`
  path params parsed with `parse_upath`. It also takes `--overwrite`, plus the auto-generated
  `--backend` and `--always-pull-images`. It is registered in `cli/__init__.py`, and its cab is
  generated by the pre-commit hook.
- `core/convert.py::convert(input, output, overwrite=False)`:
  1. Refuse if `input` and `output` resolve to the same path.
  2. Refuse if `output` exists and `overwrite` is not set.
  3. Open `input` with `open_mds`, which upgrades it in memory.
  4. Write the result with `to_zarr(output, mode="w")`.
  5. Log the source spec → target spec. If the input was already at `SPEC_VERSION`, log that and
     write it anyway, so scripts don't need to branch.

### Library changes for `0.1`

**`utils/modelspec.py`**
- `fit_image_cube(time, freq, image, wgt=None, nbasist=None, nbasisf=None, method="poly", sigmasq=0)`
  - `image` is `(ntime, nband, nstokes, ny, nx)`; `wgt` is `(ntime, nband, nstokes)`.
  - Returns `(coeffs, y_index, x_index, expr, params, texpr, fexpr)`. `coeffs` is
    `(nstokes, npar, ncomps)`; the indices are returned in array-axis order.
  - Builds the design matrix once and solves one normal-equations system per Stokes plane.
  - Fixes the latent bug where `tfunc`/`ffunc` are undefined (`ntime == nband == 1`; poly with
    `nband == 1`): the identity expressions `t` and `f` are used in those cases.
- **`fit_image_fscube` is deleted.** The merged function covers it with `ntime = 1`.
- `eval_coeffs_to_cube(...)` returns `(ntime, nfreq, nstokes, ny, nx)` and takes `nx`/`ny` as
  before. `coeffs` is `(nstokes, npar, ncomps)`.
- `eval_coeffs_to_slice(time, freq, coeffs, y_index, x_index, expr, paramf, texpr, fexpr, *, nxi, nyi, cellxi, cellyi, x0i, y0i, nxo, nyo, cellxo, cellyo, x0o, y0o)`
  - Returns `(nstokes, nyo, nxo)`.
  - **The geometry arguments become keyword-only.** Old positional calls then fail loudly
    instead of silently getting swapped axes.
  - The resampling internals still work in x-major order and transpose once at the end.
- `model_from_mds(mds_name, freqs=None)` reads through `open_mds` and returns
  `(ntime, nfreq, nstokes, ny, nx)`.
- The dead `specs` list is removed.

**`utils/io.py`**
- `build_mds_dataset(coeffs, y_index, x_index, expr, params, texpr, fexpr, time, freq, cell_rad, nx, ny, x0, y0, flip_u, flip_v, flip_w, radec, stokes: list[str], writer_version: str)`
  writes the `0.1` schema, with `spec = SPEC_VERSION` taken from `utils/spec.py`.
- `model_to_ds(...)`:
  - `model` is `(nband, nstokes, ny, nx)` and `wgt` is `(nband, nstokes)`; the returned cube is
    `(nband, nstokes, ny, nx)`;
  - `stokes: list[str]`, and `writer_version` replaces `version`.
- The module docstring's axis-convention note is updated to `(Y, X)`.

**`utils/degrid.py`**
- `_SUPPORTED_SPECS` is removed. `model_geometry` and `render_model_region` accept a dataset at
  any known spec and run `upgrade()` on it.
- `model_geometry` reads `stokes` from the coord and returns a list.
- `render_model_region` returns `(nstokes, ny, nx)`.
- `degrid_stokes` and `model_to_apparent_vis_for_region`:
  - accept `(nstokes, ny, nx)`;
  - transpose to x-major in exactly one place, immediately before ducc0's `dirty2vis`, which is
    x-major;
  - keep `apply_mueller` axis-agnostic (pixelwise).
- Visibilities have no orientation, so the fused wrapper's output is unchanged.

**`core/model2comps.py`**
- Drops the FITS `(ny, nx) → (nx, ny)` transpose.
- Passes `(ntime=1, nband, 1, ny, nx)` to the fit and writes `stokes=["I"]`.
- Renders the sanity FITS from `(nband, 1, ny, nx)` (Stokes I plane).

### Error handling

- An unknown or newer spec raises `ValueError` from `upgrade` (and so from `open_mds`, readers and
  `convert`), with an actionable message.
- A genesis dataset whose `stokes` attr names more than one product raises `ValueError` in the
  upgrade step.
- `core.convert` raises `ValueError` when the input and output paths are the same (compared after
  normalisation: relative vs absolute, trailing slash, remote protocol), or when the output exists
  without `--overwrite`. It raises from `core`, the same way `model2comps` does, because the
  generated `cli/` wrapper must round-trip byte-identically and so carries no custom code.
- Shape mismatches in `fit_image_cube` (`wgt` vs `image`) raise `ValueError`.

## Testing

All synthetic, with no MS; they run with `uv run --extra full pytest`.

- **`tests/_genesis.py`** (new, frozen): writes a genesis `.mds` exactly as `build_mds_dataset`
  did at 0.0.3. Fixture only; never refactored to track the library.
- **`tests/test_spec.py`** (new):
  - upgrading genesis → 0.1 is lossless: the 0.1 render of the upgraded dataset equals the
    transposed genesis render (genesis rendered by a frozen reference computation in the test);
  - the upgraded dataset has the 0.1 schema (dims, coords, attrs, `spec`);
  - a missing `spec` attr is treated as genesis;
  - a multi-Stokes genesis `stokes` attr raises;
  - an unknown spec raises the "unknown" message, and a newer spec (e.g. `"9.9"`) raises the
    "needs pfb-model-spec >=" message;
  - `upgrade` on a current-spec dataset is a no-op;
  - guard test: `SPEC_VERSION >= major.minor(__version__)`.
- **`tests/test_convert.py`** (new):
  - the converted output reopens at `SPEC_VERSION` and renders identically;
  - output exists without `--overwrite` → refused;
  - same input and output path → refused;
  - an already-current input is written through.
- **`tests/test_modelspec.py`** (updated):
  - exact round-trip for a multi-Stokes `(ntime, nband, nstokes, ny, nx)` cube when
    `nbasis == size` and `sigmasq == 0`;
  - the existing integer-shift interpolation invariance, now on `(nstokes, ny, nx)` slices;
  - the `ntime == nband == 1` and poly-with-one-band cases (the fixed bug);
  - Stokes planes with disjoint supports share the union of locations.
- **`tests/test_io.py`, `tests/test_degrid.py`, `tests/test_model2comps.py`** (updated): new
  shapes and the 0.1 schema; degrid gives the same visibilities from a genesis input and from its
  0.1 conversion.
- **`tests/test_roundtrip.py`**: add a `convert` case.
- **`tests/_synth.py`**: unchanged. Its helpers (`gaussian2d`, `give_edges`) have no axis
  semantics; tests build x-major Gaussians as before and transpose to `(Y, X)` explicitly.

## Docs

- `.claude/rules/component-model.md`: the versioning policy, the 0.1 schema (replacing "Axis
  convention (x-major)"), the upgrade registry and `open_mds`, updated API signatures, and the
  removal of `fit_image_fscube`.
- `CLAUDE.md`: current status (convert command, spec 0.1) and the rule "a schema change requires
  a breaking version bump plus a registry step".
- `.claude/rules/testing-and-ci.md`: the `SPEC_VERSION` guard test, and the release note that
  `tbump` to a new minor requires a spec step.

## Rollout

1. On a feature branch, implement the above; `CONTAINER_IMAGE` gets the branch tag per
   `architecture.md`.
2. Merge. **Before `tbump 0.1.0`**, either release a pfb-imaging patch that caps
   `pfb-model-spec<0.1`, or land the pfb-imaging adoption PR first: pfb-imaging currently pins
   `pfb-model-spec[full]>=0.0.3` with no upper bound, so publishing 0.1.0 would break fresh
   installs (deconv's `model_to_ds` call, degrid's `geom["stokes"].upper()`). Then `tbump 0.1.0`.
3. Follow-up PR in pfb-imaging (not part of this plan's implementation):
   - remove the `(Y, X) ↔ (X, Y)` transposes around `model_to_ds` in `deconv`;
   - add the Stokes axis at the `model_to_ds` and degrid call sites;
   - rename `version=` → `writer_version=`;
   - pin `pfb-model-spec>=0.1,<0.2`.
