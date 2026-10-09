# `.mds` Spec Versioning, `pfbspec convert`, and Spec 0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-way `.mds` spec upgrade registry with a `pfbspec convert` command, and use it to introduce spec `"0.1"` (Stokes axis + `(Y, X)`-ordered cubes, merged fit function).

**Architecture:**
- A new `utils/spec.py` owns `SPEC_VERSION`, a registry of pure `Dataset → Dataset` upgrade steps, `upgrade()` and `open_mds()`.
- Every reader upgrades in memory. `pfbspec convert` is "open, upgrade, write".
- The library (`modelspec`, `io`, `degrid`, `model2comps`) moves to the 0.1 schema:
  - coefficients `(stokes, par, comps)`;
  - rendered cubes `(…, nstokes, ny, nx)`.

**Tech Stack:** Python ≥3.10, numpy, sympy, scipy, xarray + zarr, ducc0, astropy, universal-pathlib (`upath`, via hip-cargo), Typer + hip-cargo, pytest.

**Spec:** `docs/superpowers/specs/2026-10-07-spec-versioning-and-convert-design.md` (read it first).

## Global Constraints

- **Spec name:** `SPEC_VERSION = "0.1"` is a constant string in `src/pfb_model_spec/utils/spec.py`. The legacy spec is named `"genesis"`, and a missing `spec` attr means `"genesis"`.
- **Version guard:** `SPEC_VERSION >= major.minor(__version__)`, compared as integer tuples.
- **0.1 schema:**
  - `coefficients` has dims `("stokes", "par", "comps")`;
  - coords are `stokes` `("stokes",)`, `location_x` `("comps",)`, `location_y` `("comps",)`, `params` `("par",)`, `times` `("t",)`, `freqs` `("f",)`;
  - attr `writer-version` replaces `pfb-imaging-version`;
  - there is no `stokes` attr;
  - `spec == "0.1"`.
- **`location_x` indexes FITS `NAXIS1`** in both specs; stored values never change in an upgrade.
- **Rendered cubes are `(…, nstokes, ny, nx)`.** ducc0's `dirty2vis` stays x-major, so the transpose happens exactly once, inside `degrid_stokes`.
- **`eval_coeffs_to_slice` geometry arguments are keyword-only.**
- **`fit_image_fscube` is deleted.**
- **Tests:**
  - run with `uv run --extra full pytest -v`;
  - after every code change, run `uv run ruff format . && uv run ruff check . --fix`;
  - tests are synthetic, with no MS, daskms or africanus.
- **`cli/` modules import `core/` lazily.** `src/pfb_model_spec/__init__.py` must not import `utils.*`.
- **Never hand-edit `src/pfb_model_spec/cabs/*.yml`.** The `generate-cabs` pre-commit hook regenerates them.
- **Commit with the venv active:** `source .venv/bin/activate && git add … && git commit -m "…"`.
  - Messages must be Conventional Commits.
  - End every message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - If the hook rewrites a cab, re-stage it and commit again.
- **Work on branch `spec-convert`.** It already exists and holds the spec commit.

## Review Focus

These are the failure modes most likely to bite a user that the obvious tests would miss. Each one is pinned by a test in the task named in brackets.

1. **Non-square grids (`nx != ny`).** A swapped axis passes silently on square grids. Every fit/render/degrid test uses `nx != ny`. [Tasks 2, 3, 4, 5]
2. **Remote URIs in `convert`** (e.g. `s3://`, exercised with fsspec `memory://`): the existence check, same-path check and write must all work without `os.path`. [Task 6]
3. **The same store named two ways** (`a.mds` vs `./a.mds/`): `convert` must refuse it as in-place. [Task 6]
4. **Genesis `.mds` files missing optional attrs** (no `spec`, no `pfb-imaging-version`, e.g. from early pfb-imaging) must still upgrade. [Task 1]
5. **Orientation of the sanity FITS `model2comps` writes**: it must match the input WSClean planes pixel for pixel, not transposed. [Task 5]

## Red window

Task 2 changes `fit_image_cube`/`eval_coeffs_to_slice`, which `io`, `degrid` and `model2comps` call. Their test files fail from Task 2 until Tasks 3, 4 and 5 respectively land.
- Each task runs its **own** test file to verify.
- The full suite must be green at the end of Task 5 and stay green afterwards.
- Do not "fix" a later module early to quiet a red test.

## File map

| File | Change | Responsibility |
|---|---|---|
| `src/pfb_model_spec/_container_image.py` | modify | image tag `:spec-convert` for the feature branch |
| `src/pfb_model_spec/utils/spec.py` | create | `SPEC_VERSION`, registry, `spec_of`, `upgrade`, `open_mds` |
| `tests/_genesis.py` | create | frozen genesis fixture + hand-computed reference render |
| `tests/test_spec.py` | create | registry/upgrade/version-guard tests |
| `src/pfb_model_spec/utils/modelspec.py` | modify | merged multi-Stokes `fit_image_cube`, `(Y, X)` renders, keyword-only slice geometry |
| `tests/test_modelspec.py` | rewrite | multi-Stokes, non-square, time×freq, edge-case fits |
| `src/pfb_model_spec/utils/io.py` | modify | 0.1 `build_mds_dataset`, `model_to_ds` |
| `tests/test_io.py` | rewrite | 0.1 round trip + schema parity with upgraded genesis |
| `src/pfb_model_spec/utils/degrid.py` | modify | upgrade-on-read, `(ny, nx)` images, single transpose |
| `tests/test_degrid.py` | modify | `(ny, nx)` fixtures + genesis orientation pin |
| `src/pfb_model_spec/core/model2comps.py` | modify | no FITS transpose, length-1 Stokes axis |
| `tests/test_model2comps.py` | modify | 0.1 schema + sanity-FITS orientation |
| `src/pfb_model_spec/core/convert.py` | create | `convert(input_mds, output_mds, overwrite=False)` |
| `src/pfb_model_spec/cli/convert.py` | create | generated-shape Typer wrapper |
| `src/pfb_model_spec/cli/__init__.py` | modify | register `convert` |
| `tests/test_convert.py` | create | convert behaviour incl. remote + aliasing |
| `tests/test_roundtrip.py` | modify | `convert` round-trip case |
| `.claude/rules/component-model.md`, `.claude/rules/testing-and-ci.md`, `CLAUDE.md` | modify | docs |

---

### Task 1: Spec registry, upgrade, and genesis fixture

**Files:**
- Modify: `src/pfb_model_spec/_container_image.py`
- Create: `src/pfb_model_spec/utils/spec.py`
- Create: `tests/_genesis.py`
- Test: `tests/test_spec.py`

**Interfaces:**
- Produces (`pfb_model_spec.utils.spec`):
  - `SPEC_VERSION: str = "0.1"`, `LEGACY_SPEC: str = "genesis"`;
  - `spec_of(ds: xr.Dataset) -> str`;
  - `upgrade(ds: xr.Dataset) -> xr.Dataset`;
  - `open_mds(path: str | os.PathLike) -> xr.Dataset`.
- Produces (`tests._genesis`):
  - constants `NX = 32`, `NY = 24`, `FREQS`, `TIMES`, `CELL_RAD`, `LOC_X`, `LOC_Y`, `COEFFS`;
  - `genesis_dataset(**attr_overrides) -> xr.Dataset` (an override value of `None` deletes that attr);
  - `genesis_cube(freqs: np.ndarray = FREQS) -> np.ndarray`, x-major `(nfreq, NX, NY)`.

- [ ] **Step 1: Point the container image at the branch tag**

Replace the contents of `src/pfb_model_spec/_container_image.py` with:

```python
CONTAINER_IMAGE = "ghcr.io/landmanbester/pfb-model-spec:spec-convert"
```

- [ ] **Step 2: Write the frozen genesis fixture**

Create `tests/_genesis.py`:

```python
"""Frozen writer for `genesis`-spec `.mds` fixtures.

The library no longer writes genesis, so these fixtures pin what a genesis `.mds` looked
like at pfb-model-spec 0.0.3. Do NOT update this module to track library changes -- its
whole value is that it does not move.

The model is written by hand rather than fitted: ``expr = t0 + f1 * fs`` with
``fs = (f - 1.1e9) / 1e8``, so the reference render below is exact and independent of
the library's fit/render code.
"""

import numpy as np
import xarray as xr

NX = 32  # non-square on purpose: a swapped axis must not pass silently
NY = 24
FREQS = np.array([1.0e9, 1.1e9, 1.2e9])
TIMES = np.array([0.0])
CELL_RAD = float(np.deg2rad(2.0 / 3600.0))
LOC_X = np.array([4, 27, 15])
LOC_Y = np.array([3, 20, 11])
# (par, comps): rows are the t0 and f1 coefficients
COEFFS = np.array([[1.0, -2.0, 0.5], [0.3, 0.1, -0.4]])
PARAMS = ["t0", "f1"]
EXPR = "f*f1 + t0"
TEXPR = "t"
FEXPR = "(f - 1100000000.0)/100000000.0"


def genesis_dataset(**attr_overrides) -> xr.Dataset:
    """A genesis `.mds` dataset; an override value of ``None`` deletes that attr."""
    attrs = {
        "pfb-imaging-version": "0.0.3",
        "spec": "genesis",
        "cell_rad_x": CELL_RAD,
        "cell_rad_y": CELL_RAD,
        "npix_x": NX,
        "npix_y": NY,
        "texpr": TEXPR,
        "fexpr": FEXPR,
        "center_x": 0.0,
        "center_y": 0.0,
        "flip_u": False,
        "flip_v": True,
        "flip_w": False,
        "ra": 0.1,
        "dec": -0.5,
        "stokes": "I",
        "parametrisation": EXPR,
    }
    for key, value in attr_overrides.items():
        if value is None:
            attrs.pop(key, None)
        else:
            attrs[key] = value
    return xr.Dataset(
        data_vars={"coefficients": (("par", "comps"), COEFFS)},
        coords={
            "location_x": (("x",), LOC_X),
            "location_y": (("y",), LOC_Y),
            "params": (("par",), PARAMS),
            "times": (("t",), TIMES),
            "freqs": (("f",), FREQS),
        },
        attrs=attrs,
    )


def genesis_cube(freqs: np.ndarray = FREQS) -> np.ndarray:
    """Exact reference render of ``genesis_dataset()``, x-major ``(nfreq, NX, NY)``."""
    cube = np.zeros((freqs.size, NX, NY))
    fs = (freqs - 1.1e9) / 1.0e8
    cube[:, LOC_X, LOC_Y] = COEFFS[0][None, :] + COEFFS[1][None, :] * fs[:, None]
    return cube
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_spec.py`:

```python
"""Tests for the .mds spec registry and one-way upgrade path."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pfb_model_spec import __version__
from pfb_model_spec.utils.spec import LEGACY_SPEC, SPEC_VERSION, open_mds, spec_of, upgrade
from tests._genesis import COEFFS, LOC_X, LOC_Y, genesis_dataset


def _minor(version: str) -> tuple[int, int]:
    major, minor = version.split(".")[:2]
    return int(major), int(minor)


def test_spec_version_is_not_behind_the_package():
    """Bumping the package to a new minor needs a new spec step (even a no-op one)."""
    assert _minor(SPEC_VERSION) >= _minor(__version__)


def test_upgrade_genesis_gives_the_current_schema():
    up = upgrade(genesis_dataset())
    assert spec_of(up) == SPEC_VERSION == "0.1"
    assert up.coefficients.dims == ("stokes", "par", "comps")
    assert_array_equal(up.coefficients.values, COEFFS[None])
    assert list(up.stokes.values) == ["I"]
    assert up.location_x.dims == ("comps",) and up.location_y.dims == ("comps",)
    # stored pixel indices are untouched: the (Y, X) change is render-order only
    assert_array_equal(up.location_x.values, LOC_X)
    assert_array_equal(up.location_y.values, LOC_Y)
    assert up.attrs["writer-version"] == "0.0.3"
    assert "pfb-imaging-version" not in up.attrs
    assert "stokes" not in up.attrs


def test_upgrade_keeps_every_other_attr():
    g = genesis_dataset()
    up = upgrade(g)
    moved = {"spec", "stokes", "pfb-imaging-version"}
    for key, value in g.attrs.items():
        if key not in moved:
            assert up.attrs[key] == value, key


def test_upgrade_does_not_mutate_its_input():
    g = genesis_dataset()
    before = dict(g.attrs)
    upgrade(g)
    assert g.attrs == before
    assert g.coefficients.dims == ("par", "comps")


def test_missing_spec_attr_means_genesis():
    g = genesis_dataset(spec=None)
    assert spec_of(g) == LEGACY_SPEC
    assert spec_of(upgrade(g)) == SPEC_VERSION


def test_missing_version_attr_still_upgrades():
    up = upgrade(genesis_dataset(**{"pfb-imaging-version": None}))
    assert up.attrs["writer-version"] == "unknown"


def test_multi_stokes_genesis_is_rejected():
    with pytest.raises(ValueError, match="single Stokes"):
        upgrade(genesis_dataset(stokes="IQ"))


def test_unknown_spec_is_rejected():
    with pytest.raises(ValueError, match="Unknown .mds spec 'exodus'"):
        upgrade(genesis_dataset(spec="exodus"))


def test_newer_spec_asks_for_a_newer_package():
    with pytest.raises(ValueError, match=r"needs pfb-model-spec >= 9\.9"):
        upgrade(genesis_dataset(spec="9.9"))


def test_upgrade_of_current_spec_is_a_no_op():
    up = upgrade(genesis_dataset())
    assert upgrade(up) is up


def test_open_mds_upgrades_on_read(tmp_path):
    path = tmp_path / "g.mds"
    genesis_dataset().to_zarr(path)
    ds = open_mds(path)
    assert spec_of(ds) == SPEC_VERSION
    assert np.array_equal(ds.coefficients.values, COEFFS[None])
```

- [ ] **Step 4: Run them to verify they fail**

Run: `uv run --extra full pytest tests/test_spec.py -v`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'pfb_model_spec.utils.spec'`.

- [ ] **Step 5: Implement `utils/spec.py`**

Create `src/pfb_model_spec/utils/spec.py`:

```python
"""`.mds` spec versions and the one-way upgrade path between them.

Policy (see `.claude/rules/component-model.md`): a spec is named after the ``major.minor``
of the pfb-model-spec release that introduced it, and any schema change requires a breaking
version bump. ``"genesis"`` is the legacy (pre-versioning) name for ``"0.0"``. Older specs
always upgrade to newer ones; there is no downgrade path, because a newer spec may hold
things an older one cannot express.

Every step in ``_UPGRADES`` is a pure ``Dataset -> Dataset`` function. Readers call
:func:`open_mds` (or :func:`upgrade` on an already-open dataset), so a `.mds` written under
any known spec keeps working; ``pfbspec convert`` persists the same upgrade to disk.
"""

import os
from collections.abc import Callable

import xarray as xr

SPEC_VERSION = "0.1"
LEGACY_SPEC = "genesis"

_STOKES = ("I", "Q", "U", "V")


def spec_of(ds: xr.Dataset) -> str:
    """Return the spec a `.mds` dataset declares; a missing attr means ``"genesis"``."""
    return str(ds.attrs.get("spec", LEGACY_SPEC))


def _genesis_to_0_1(ds: xr.Dataset) -> xr.Dataset:
    """genesis -> 0.1: add a length-1 Stokes axis and put the locations on ``comps``.

    Lossless: no values change. ``location_x`` indexes FITS ``NAXIS1`` in both specs; the
    ``(Y, X)`` change in 0.1 affects only the axis order of rendered cubes.
    """
    attrs = dict(ds.attrs)
    stokes = str(attrs.pop("stokes"))
    if stokes not in _STOKES:
        raise ValueError(f"A genesis .mds carries a single Stokes product, got {stokes!r}")
    attrs["writer-version"] = attrs.pop("pfb-imaging-version", "unknown")
    attrs["spec"] = "0.1"
    # rebuilt from .values so no zarr encoding from the old layout leaks into a later write
    return xr.Dataset(
        data_vars={"coefficients": (("stokes", "par", "comps"), ds.coefficients.values[None])},
        coords={
            "stokes": (("stokes",), [stokes]),
            "location_x": (("comps",), ds.location_x.values),
            "location_y": (("comps",), ds.location_y.values),
            "params": (("par",), ds.params.values),
            "times": (("t",), ds.times.values),
            "freqs": (("f",), ds.freqs.values),
        },
        attrs=attrs,
    )


# from-spec -> (to-spec, step). A breaking release that leaves the schema alone still adds
# a no-op step here, so that every spec name is explicitly defined.
_UPGRADES: dict[str, tuple[str, Callable[[xr.Dataset], xr.Dataset]]] = {
    LEGACY_SPEC: ("0.1", _genesis_to_0_1),
}


def _as_version(spec: str) -> tuple[int, int] | None:
    parts = spec.split(".")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1])


def upgrade(ds: xr.Dataset) -> xr.Dataset:
    """Upgrade a `.mds` dataset to ``SPEC_VERSION``; a current dataset is returned as is.

    Args:
        ds: An opened `.mds` dataset at any known spec.

    Returns:
        The dataset at ``SPEC_VERSION``. The input is never mutated.

    Raises:
        ValueError: If the spec is newer than this package understands, unknown, or the
            dataset cannot be expressed in the newer spec.
    """
    spec = spec_of(ds)
    while spec != SPEC_VERSION:
        if spec not in _UPGRADES:
            version = _as_version(spec)
            if version is not None and version > _as_version(SPEC_VERSION):
                raise ValueError(
                    f"This .mds uses spec {spec!r}, which needs pfb-model-spec >= {spec}; "
                    f"this installation reads spec {SPEC_VERSION!r}"
                )
            known = sorted([*_UPGRADES, SPEC_VERSION])
            raise ValueError(f"Unknown .mds spec {spec!r}; known specs: {known}")
        spec, step = _UPGRADES[spec]
        ds = step(ds)
    return ds


def open_mds(path: str | os.PathLike) -> xr.Dataset:
    """Open a `.mds` zarr store (local path or remote URI) and upgrade it to ``SPEC_VERSION``."""
    return upgrade(xr.open_zarr(str(path), chunks=None))
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run --extra full pytest tests/test_spec.py tests/test_lightweight_import.py -v`
Expected: all PASS. The guard test passes because `(0, 1) >= (0, 0)`.

- [ ] **Step 7: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
source .venv/bin/activate && git add src/pfb_model_spec/_container_image.py src/pfb_model_spec/utils/spec.py tests/_genesis.py tests/test_spec.py src/pfb_model_spec/cabs && git commit -m "feat: add .mds spec registry with genesis -> 0.1 upgrade

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

If the hook rewrites `cabs/model2comps.yml` (the image tag changed), re-run the same `git add … && git commit` line.

### Task 2: Merged multi-Stokes fit and `(Y, X)` rendering in `modelspec.py`

**Files:**
- Modify: `src/pfb_model_spec/utils/modelspec.py` (rewrite everything except the pad/resample body of `eval_coeffs_to_slice`)
- Test: `tests/test_modelspec.py` (rewrite)

**Interfaces:**
- Consumes: `open_mds` from Task 1. Also `tests._genesis.{genesis_dataset, genesis_cube, FREQS, NX, NY}`.
- Produces (`pfb_model_spec.utils.modelspec`):
  - `fit_image_cube(time, freq, image, wgt=None, nbasist=None, nbasisf=None, method="poly", sigmasq=0) -> (coeffs, y_index, x_index, expr: str, params: list[str], texpr: str, fexpr: str)`
    - `image` is `(ntime, nband, nstokes, ny, nx)` and `wgt` is `(ntime, nband, nstokes)`;
    - `coeffs` is `(nstokes, npar, ncomps)`.
  - `eval_coeffs_to_cube(time, freq, nx, ny, coeffs, y_index, x_index, expr, paramf, texpr, fexpr) -> (ntime, nfreq, nstokes, ny, nx)`
  - `eval_coeffs_to_slice(time, freq, coeffs, y_index, x_index, expr, paramf, texpr, fexpr, *, nxi, nyi, cellxi, cellyi, x0i, y0i, nxo, nyo, cellxo, cellyo, x0o, y0o) -> (nstokes, nyo, nxo)`
  - `model_from_mds(mds_name, freqs=None) -> (ntime, nfreq, nstokes, ny, nx)`
  - `fit_image_fscube` and the `specs` list are **removed**.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_modelspec.py` entirely with:

```python
"""Synthetic tests of the component-model fit/render library (spec 0.1).

Cubes are (..., nstokes, ny, nx). Grids are deliberately non-square so that a swapped
axis cannot pass silently.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pfb_model_spec.utils.modelspec import (
    eval_coeffs_to_cube,
    eval_coeffs_to_slice,
    fit_image_cube,
    model_from_mds,
)
from tests._genesis import FREQS, NX, NY, genesis_cube, genesis_dataset
from tests._synth import gaussian2d, give_edges


def _gaussian_cube(freq, nx, ny, nsource=10, seed=420):
    """Multi-Gaussian power-law cube, x-major (nchan, nx, ny) -- callers transpose."""
    rng = np.random.default_rng(seed)
    freq0 = float(freq.mean())
    model = np.zeros((freq.size, nx, ny))
    border = max(int(0.15 * nx), int(0.15 * ny))
    xs = rng.integers(border, nx - border, nsource)
    ys = rng.integers(border, ny - border, nsource)
    alpha = -0.7 + 0.1 * rng.standard_normal(nsource)
    ref_flux = 1.0 + np.exp(rng.standard_normal(nsource))
    ex = rng.integers(3, int(0.1 * nx) + 3, nsource)
    ey = rng.integers(3, int(0.1 * ny) + 3, nsource)
    pas = rng.random(nsource) * 180
    xin, yin = np.meshgrid(-(nx / 2) + np.arange(nx), -(ny / 2) + np.arange(ny), indexing="ij")
    for i in range(nsource):
        gauss = gaussian2d(xin, yin, gausspar=(max(ex[i], ey[i]), min(ex[i], ey[i]), pas[i]))
        mx, my, gx, gy = give_edges(xs[i], ys[i], nx, ny, nx, ny)
        spectrum = ref_flux[i] * (freq / freq0) ** alpha[i]
        model[:, mx, my] += spectrum[:, None, None] * gauss[None, gx, gy]
    return model


def _geometry(nx, ny, cell, nxo=None, nyo=None, x0o=0.0, y0o=0.0):
    return dict(
        nxi=nx,
        nyi=ny,
        cellxi=cell,
        cellyi=cell,
        x0i=0.0,
        y0i=0.0,
        nxo=nx if nxo is None else nxo,
        nyo=ny if nyo is None else nyo,
        cellxo=cell,
        cellyo=cell,
        x0o=x0o,
        y0o=y0o,
    )


def test_multi_stokes_frequency_roundtrip_is_exact():
    nchan, nx, ny = 8, 40, 32
    freq = np.linspace(1.0e9, 2.0e9, nchan)
    time = np.array([0.0])
    model_yx = _gaussian_cube(freq, nx, ny).transpose(0, 2, 1)
    cube = np.stack([model_yx, -0.25 * model_yx], axis=1)  # (nchan, 2, ny, nx)

    coeffs, yi, xi, expr, params, texpr, fexpr = fit_image_cube(
        time, freq, cube[None], nbasisf=nchan, method="Legendre", sigmasq=0.0
    )
    assert coeffs.shape == (2, nchan, yi.size)

    image = eval_coeffs_to_cube(time, freq, nx, ny, coeffs, yi, xi, expr, params, texpr, fexpr)
    assert image.shape == (1, nchan, 2, ny, nx)
    mask = cube != 0
    assert_allclose(image[0][mask], cube[mask], atol=1e-10)


@pytest.mark.parametrize("method", ["Legendre", "poly"])
def test_time_and_frequency_fit_is_exact(method):
    """ntime > 1 and nband > 1 together: pins the (time, band) row order of the design matrix."""
    time = np.array([10.0, 20.0, 30.0])
    freq = np.linspace(1.0e9, 1.3e9, 4)
    ny, nx = 6, 10
    rng = np.random.default_rng(3)
    ts = (time - 20.0) / 10.0
    fs = (freq - 1.15e9) / 1.5e8
    a = rng.standard_normal((2, 3))
    b = rng.standard_normal((2, 3))
    cube = np.zeros((time.size, freq.size, 2, ny, nx))
    for s in range(2):
        # quadratic in time plus quadratic in frequency: inside the additive basis's span
        val = np.polyval(a[s], ts)[:, None] + np.polyval(b[s], fs)[None, :]
        cube[:, :, s, 2, 7] = val
        cube[:, :, s, 4, 1] = 2.0 * val

    coeffs, yi, xi, expr, params, texpr, fexpr = fit_image_cube(
        time, freq, cube, nbasist=3, nbasisf=3, method=method, sigmasq=0.0
    )
    image = eval_coeffs_to_cube(time, freq, nx, ny, coeffs, yi, xi, expr, params, texpr, fexpr)
    assert_allclose(image, cube, atol=1e-8)


@pytest.mark.parametrize("method", ["Legendre", "poly"])
def test_single_time_single_band_returns_the_data(method):
    """Nothing to fit: the pixel values are the coefficients, even with sigmasq set."""
    cube = np.zeros((1, 1, 2, 4, 6))
    cube[0, 0, 0, 1, 5] = 3.0
    cube[0, 0, 1, 3, 2] = -1.5
    time, freq = np.array([0.0]), np.array([1.4e9])
    coeffs, yi, xi, expr, params, texpr, fexpr = fit_image_cube(time, freq, cube, method=method, sigmasq=1e-3)
    assert (texpr, fexpr) == ("t", "f")
    image = eval_coeffs_to_cube(time, freq, 6, 4, coeffs, yi, xi, expr, params, texpr, fexpr)
    assert_allclose(image, cube, rtol=1e-12)


def test_poly_fit_with_a_single_band():
    """Used to raise UnboundLocalError: ffunc was never defined for poly with one band."""
    time, freq = np.array([1.0, 2.0]), np.array([1.4e9])
    cube = np.zeros((2, 1, 1, 3, 5))
    cube[:, 0, 0, 1, 4] = [1.0, 3.0]
    coeffs, yi, xi, expr, params, texpr, fexpr = fit_image_cube(time, freq, cube, method="poly", sigmasq=0.0)
    assert fexpr == "f"
    image = eval_coeffs_to_cube(time, freq, 5, 3, coeffs, yi, xi, expr, params, texpr, fexpr)
    assert_allclose(image, cube, atol=1e-12)


def test_stokes_planes_share_the_union_of_locations():
    cube = np.zeros((1, 1, 2, 5, 7))
    cube[0, 0, 0, 2, 3] = 1.0
    cube[0, 0, 1, 4, 6] = 2.0
    coeffs, yi, xi, *_ = fit_image_cube(np.array([0.0]), np.array([1.0e9]), cube)
    assert list(yi) == [2, 4] and list(xi) == [3, 6]
    assert_allclose(coeffs[0, 0], [1.0, 0.0])
    assert_allclose(coeffs[1, 0], [0.0, 2.0])


def test_fit_rejects_mismatched_weights():
    cube = np.ones((1, 2, 2, 3, 3))
    with pytest.raises(ValueError, match="wgt"):
        fit_image_cube(np.array([0.0]), np.array([1.0e9, 1.1e9]), cube, wgt=np.ones((1, 2, 1)))


def test_slice_matches_cube_and_is_shift_invariant():
    nchan, nx, ny = 4, 64, 48
    cell = 2.5 / 3600.0
    freq = np.linspace(1.0e9, 2.0e9, nchan)
    time = np.array([0.0])
    model_xy = _gaussian_cube(freq, nx, ny)
    cube = model_xy.transpose(0, 2, 1)[:, None]  # (nchan, 1, ny, nx)
    coeffs, yi, xi, expr, params, texpr, fexpr = fit_image_cube(
        time, freq, cube[None], nbasisf=nchan, method="Legendre", sigmasq=0.0
    )
    args = (time[0], freq[0], coeffs, yi, xi, expr, params, texpr, fexpr)

    same = eval_coeffs_to_slice(*args, **_geometry(nx, ny, cell))
    assert same.shape == (1, ny, nx)
    assert_allclose(same[0], cube[0, 0], atol=1e-10)

    # an integer-pixel shift of the output centre must leave pixel values unchanged
    xshift, yshift = 25, -10
    for npix_out in (40, nx, 2 * nx):
        imout = eval_coeffs_to_slice(
            *args, **_geometry(nx, ny, cell, nxo=npix_out, nyo=npix_out, x0o=cell * xshift, y0o=cell * yshift)
        )
        assert imout.shape == (1, npix_out, npix_out)
        mx, my, gx, gy = give_edges(npix_out // 2 - xshift, npix_out // 2 - yshift, npix_out, npix_out, nx, ny)
        assert_allclose(1.0 + imout[0].T[mx, my], 1.0 + model_xy[0, gx, gy])


def test_slice_geometry_is_keyword_only():
    """Old positional calls must fail loudly rather than silently swap axes."""
    coeffs = np.ones((1, 1, 1))
    with pytest.raises(TypeError):
        eval_coeffs_to_slice(
            0.0, 1.0e9, coeffs, [0], [0], "t0", ["t0"], "t", "f", 4, 4, 1.0, 1.0, 0.0, 0.0, 4, 4, 1.0, 1.0, 0.0, 0.0
        )


def test_model_from_mds_renders_an_upgraded_genesis_losslessly(tmp_path):
    path = tmp_path / "g.mds"
    genesis_dataset().to_zarr(path)
    got = model_from_mds(str(path))
    assert got.shape == (1, FREQS.size, 1, NY, NX)
    assert_allclose(got[0, :, 0], genesis_cube().transpose(0, 2, 1), atol=1e-12)

    one = model_from_mds(str(path), freqs=FREQS[2])
    assert_allclose(one[0, 0, 0], genesis_cube(FREQS[2:])[0].T, atol=1e-12)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --extra full pytest tests/test_modelspec.py -v`
Expected: most FAIL, typically with `ValueError: not enough values to unpack` from the old 4-D `fit_image_cube`, or with shape assertions.

- [ ] **Step 3: Rewrite the fit and render functions**

In `src/pfb_model_spec/utils/modelspec.py`:

1. Delete `specs = ["genesis", "exodus"]`.
2. Add `from pfb_model_spec.utils.spec import open_mds` to the imports.
3. Replace `fit_image_cube` **and** `fit_image_fscube` with the code below.

```python
def _scale(x, sym, method):
    """Map an axis onto the fit domain, returning ``(scaled values, sympy scaling expr)``.

    A length-1 axis is left unscaled: only the constant basis function can be fitted to
    it, so any scaling is irrelevant -- and ``x / x[0]`` would divide by zero at t=0.
    """
    if x.size == 1:
        return x.astype(float), sym
    if method == "poly":
        return x / x[0], sym / x[0]
    # Legendre: scale onto [-1, 1] for stability
    centre = (x.max() + x.min()) / 2
    half = (x - centre).max()
    return (x - centre) / half, (sym - centre) / half


def fit_image_cube(time, freq, image, wgt=None, nbasist=None, nbasisf=None, method="poly", sigmasq=0):
    """Fit the time and frequency axes of a multi-Stokes image cube.

    The basis is additive: functions of time (including the constant) plus functions of
    frequency (excluding the constant). Every Stokes plane is fitted separately against the
    same basis, and all planes share one set of component locations: the pixels that are
    non-zero in any plane.

    Args:
        time: Time axis, shape ``(ntime,)``.
        freq: Frequency axis, shape ``(nband,)``.
        image: Pixelated cube, shape ``(ntime, nband, nstokes, ny, nx)``.
        wgt: Optional weights, shape ``(ntime, nband, nstokes)``.
        nbasist: Number of time basis functions (default ``ntime``).
        nbasisf: Number of frequency basis functions, counting the shared constant
            (default ``nband``).
        method: ``"poly"`` (monomials) or ``"Legendre"``.
        sigmasq: Ridge term added to the Hessian; ignored when ``ntime == nband == 1``
            (nothing to fit, the data are the coefficients).

    Returns:
        ``(coeffs, y_index, x_index, expr, params, texpr, fexpr)``:
        - ``coeffs`` has shape ``(nstokes, npar, ncomps)``;
        - ``y_index``/``x_index`` are the component pixel locations, in array-axis order;
        - ``expr`` is the stringified sympy model in ``t``, ``f`` and ``params``;
        - ``texpr``/``fexpr`` map raw time/frequency onto the fit domain.

    Raises:
        ValueError: On inconsistent shapes or basis sizes.
        NotImplementedError: For an unknown ``method``.
    """
    from sympy.abc import f, t

    if image.ndim != 5:
        raise ValueError(f"image must be (ntime, nband, nstokes, ny, nx), got shape {image.shape}")
    ntime, nband, nstokes, ny, nx = image.shape
    if time.size != ntime or freq.size != nband:
        raise ValueError(f"time/freq sizes ({time.size}, {freq.size}) do not match image {image.shape}")
    if wgt is None:
        wgt = np.ones((ntime, nband, nstokes), dtype=float)
    elif wgt.shape != (ntime, nband, nstokes):
        raise ValueError(f"wgt must have shape {(ntime, nband, nstokes)}, got {wgt.shape}")
    nbasist = ntime if nbasist is None else nbasist
    nbasisf = nband if nbasisf is None else nbasisf
    if not 1 <= nbasist <= ntime or not 1 <= nbasisf <= nband:
        raise ValueError(f"need 1 <= nbasist <= {ntime} and 1 <= nbasisf <= {nband}")
    if ntime == 1 and nband == 1:
        sigmasq = 0

    if method == "poly":

        def basis(i, w, sym):
            return w**i, sym**i

    elif method == "Legendre":

        def basis(i, w, sym):
            return np.polynomial.Legendre.basis(i)(w), sm.polys.orthopolys.legendre_poly(i, sym)

    else:
        raise NotImplementedError(f"Method {method} not implemented")

    wt, tfunc = _scale(time, t, method)
    wf, ffunc = _scale(freq, f, method)
    tparams = sm.symbols(f"t(0:{nbasist})")
    fparams = sm.symbols(f"f(1:{nbasisf})") if nbasisf > 1 else ()

    expr = sm.Integer(0)
    xt = np.zeros((ntime, nbasist))
    for i, p in enumerate(tparams):
        xt[:, i], term = basis(i, wt, t)
        expr += term * p
    xf = np.zeros((nband, nbasisf - 1))
    for i, p in enumerate(fparams, start=1):
        xf[:, i - 1], term = basis(i, wf, f)
        expr += term * p
    # rows ordered (time, band), matching the C-order reshape of beta below
    xfit = np.hstack((np.repeat(xt, nband, axis=0), np.tile(xf, (ntime, 1))))

    mask = np.any(image, axis=(0, 1, 2))
    y_index, x_index = np.where(mask)
    beta = image[:, :, :, y_index, x_index].reshape(ntime * nband, nstokes, y_index.size)
    wgt = wgt.reshape(ntime * nband, nstokes)

    coeffs = np.zeros((nstokes, xfit.shape[1], y_index.size), dtype=np.result_type(beta.dtype, float))
    for s in range(nstokes):
        w = wgt[:, s : s + 1]
        hess = xfit.T.dot(w * xfit)
        if sigmasq:
            hess += sigmasq * np.eye(hess.shape[0])
        coeffs[s] = np.linalg.solve(hess, xfit.T.dot(w * beta[:, s]))

    params = [*tparams, *fparams]
    return coeffs, y_index, x_index, str(expr), [str(p) for p in params], str(tfunc), str(ffunc)
```

- [ ] **Step 4: Replace the render functions**

Replace `eval_coeffs_to_cube`, `eval_coeffs_to_slice` and `model_from_mds` with the code below.

`_resample_xmajor`'s body is the **existing** body of `eval_coeffs_to_slice`, moved without changes. It starts at `pix_area_in = cellxi * cellyi` and runs through the final `return image_in`. Keep its numerics exactly; #23 tracks the known `area_ratio` limitation.

```python
def _model_functions(expr, paramf, texpr, fexpr):
    """Lambdify a stored parametrisation into ``(modelf, tfunc, ffunc)``."""
    params = sm.symbols(("t", "f"))
    params += sm.symbols(tuple(paramf))
    modelf = lambdify(params, parse_expr(expr))
    tfunc = lambdify(params[0], parse_expr(texpr))
    ffunc = lambdify(params[1], parse_expr(fexpr))
    return modelf, tfunc, ffunc


def eval_coeffs_to_cube(time, freq, nx, ny, coeffs, y_index, x_index, expr, paramf, texpr, fexpr):
    """Render ``(nstokes, npar, ncomps)`` coefficients to a ``(ntime, nfreq, nstokes, ny, nx)`` cube."""
    modelf, tfunc, ffunc = _model_functions(expr, paramf, texpr, fexpr)
    nstokes = coeffs.shape[0]
    image = np.zeros((time.size, freq.size, nstokes, ny, nx), dtype=float)
    for i, tval in enumerate(time):
        for j, fval in enumerate(freq):
            for s in range(nstokes):
                image[i, j, s, y_index, x_index] = modelf(tfunc(tval), ffunc(fval), *coeffs[s])
    return image


def eval_coeffs_to_slice(
    time,
    freq,
    coeffs,
    y_index,
    x_index,
    expr,
    paramf,
    texpr,
    fexpr,
    *,
    nxi,
    nyi,
    cellxi,
    cellyi,
    x0i,
    y0i,
    nxo,
    nyo,
    cellxo,
    cellyo,
    x0o,
    y0o,
):
    """Render coefficients at one (time, freq) onto an arbitrary output grid.

    Returns ``(nstokes, nyo, nxo)``. The geometry is keyword-only so that a call written
    against the old x-major positional signature fails loudly instead of swapping axes.
    """
    modelf, tfunc, ffunc = _model_functions(expr, paramf, texpr, fexpr)
    tval, fval = tfunc(time), ffunc(freq)
    out = np.zeros((coeffs.shape[0], nyo, nxo), dtype=float)
    for s in range(coeffs.shape[0]):
        image_in = np.zeros((nxi, nyi), dtype=float)
        image_in[x_index, y_index] = modelf(tval, fval, *coeffs[s])
        # resampling works x-major internally; transpose once on the way out
        out[s] = _resample_xmajor(image_in, nxi, nyi, cellxi, cellyi, x0i, y0i, nxo, nyo, cellxo, cellyo, x0o, y0o).T
    return out


def _resample_xmajor(image_in, nxi, nyi, cellxi, cellyi, x0i, y0i, nxo, nyo, cellxo, cellyo, x0o, y0o):
    """Zero-pad and bilinearly resample an x-major ``(nxi, nyi)`` slice onto ``(nxo, nyo)``."""
    # <-- the existing body of eval_coeffs_to_slice, from `pix_area_in = cellxi * cellyi`
    #     through `return image_in`, moved here unchanged


def model_from_mds(mds_name, freqs=None):
    """Render a `.mds` at any known spec at its own resolution: ``(ntime, nfreq, nstokes, ny, nx)``."""
    mds = open_mds(mds_name)
    if freqs is None:
        freqs = mds.freqs.values
    else:
        freqs = np.atleast_1d(freqs)
    return eval_coeffs_to_cube(
        mds.times.values,
        freqs,
        mds.npix_x,
        mds.npix_y,
        mds.coefficients.values,
        mds.location_y.values,
        mds.location_x.values,
        mds.parametrisation,
        mds.params.values,
        mds.texpr,
        mds.fexpr,
    )
```

The comment inside `_resample_xmajor` marks code that is **moved, not rewritten**. Paste the existing lines there and delete the comment.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra full pytest tests/test_modelspec.py tests/test_spec.py tests/test_lightweight_import.py -v`
Expected: all PASS.

`test_io.py`, `test_degrid.py` and `test_model2comps.py` are now expected to fail; see "Red window". If the shift-invariance test fails, debug it with superpowers:systematic-debugging. Do not loosen its tolerance: it passed bit-for-bit on the old x-major layout.

- [ ] **Step 6: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
source .venv/bin/activate && git add src/pfb_model_spec/utils/modelspec.py tests/test_modelspec.py && git commit -m "feat!: merge fit functions into a multi-Stokes (Y, X) fit_image_cube

fit_image_fscube is removed; eval_coeffs_to_slice geometry is keyword-only.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

If the `commit-msg` hook rejects the `!` marker, use `feat: …` and add a `BREAKING CHANGE:` footer line instead.

### Task 3: 0.1 writer in `io.py`

**Files:**
- Modify: `src/pfb_model_spec/utils/io.py`
- Test: `tests/test_io.py` (rewrite)

**Interfaces:**
- Consumes:
  - `SPEC_VERSION` and `upgrade` from Task 1;
  - `fit_image_cube` and `eval_coeffs_to_slice` (keyword geometry) from Task 2.
- Produces (`pfb_model_spec.utils.io`):
  - `build_mds_dataset(coeffs, y_index, x_index, expr, params, texpr, fexpr, time, freq, cell_rad, nx, ny, x0, y0, flip_u, flip_v, flip_w, radec, stokes: list[str], writer_version: str) -> xr.Dataset`
  - `model_to_ds(time, freq, fsel, model, wgt, mds_name, cell_rad, nx, ny, x0, y0, flip_u, flip_v, flip_w, radec, stokes: list[str], writer_version: str, nbasisf=None, method="Legendre", sigmasq=1e-6) -> np.ndarray`
    - `model` is `(nband, nstokes, ny, nx)` and `wgt` is `(nband, nstokes)`;
    - it returns `(nband, nstokes, ny, nx)`.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_io.py` entirely with:

```python
"""Synthetic, self-contained tests of pfb_model_spec.utils.io (spec 0.1)."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pfb_model_spec.utils.io import build_mds_dataset, model_to_ds
from pfb_model_spec.utils.spec import SPEC_VERSION, upgrade
from tests._genesis import genesis_dataset
from tests._synth import gaussian2d, give_edges


def _schema(ds: xr.Dataset):
    return (
        {k: v.dims for k, v in ds.data_vars.items()},
        {k: v.dims for k, v in ds.coords.items()},
        sorted(ds.attrs),
    )


def _cube_yx(freq, nx, ny, nsource=10, seed=420):
    """Multi-Gaussian power-law cube, built x-major and returned (nchan, ny, nx)."""
    rng = np.random.default_rng(seed)
    freq0 = float(freq.mean())
    model = np.zeros((freq.size, nx, ny))
    border = max(int(0.15 * nx), int(0.15 * ny))
    xs = rng.integers(border, nx - border, nsource)
    ys = rng.integers(border, ny - border, nsource)
    alpha = -0.7 + 0.1 * rng.standard_normal(nsource)
    ref_flux = 1.0 + np.exp(rng.standard_normal(nsource))
    ex = rng.integers(3, int(0.1 * nx) + 3, nsource)
    ey = rng.integers(3, int(0.1 * ny) + 3, nsource)
    pas = rng.random(nsource) * 180
    xin, yin = np.meshgrid(-(nx / 2) + np.arange(nx), -(ny / 2) + np.arange(ny), indexing="ij")
    for i in range(nsource):
        gauss = gaussian2d(xin, yin, gausspar=(max(ex[i], ey[i]), min(ex[i], ey[i]), pas[i]))
        mx, my, gx, gy = give_edges(xs[i], ys[i], nx, ny, nx, ny)
        model[:, mx, my] += (ref_flux[i] * (freq / freq0) ** alpha[i])[:, None, None] * gauss[None, gx, gy]
    return model.transpose(0, 2, 1)


def test_model_to_ds_roundtrip(tmp_path):
    nchan, nx, ny = 8, 64, 48
    freq = np.linspace(1.0e9, 2.0e9, nchan)
    cell_rad = np.deg2rad(2.5 / 3600.0)
    model_i = _cube_yx(freq, nx, ny)
    model = np.stack([model_i, 0.1 * model_i], axis=1)  # (nchan, 2, ny, nx)
    time = np.array([0.0])
    mds_name = str(tmp_path / "test.mds")

    # nbasisf == nband and ntime == 1 make the Legendre fit exact
    new_model = model_to_ds(
        time,
        freq,
        np.ones(nchan, dtype=bool),
        model,
        np.ones((nchan, 2)),
        mds_name,
        cell_rad,
        nx,
        ny,
        0.0,
        0.0,
        False,
        False,
        False,
        (0.1, -0.2),
        ["I", "V"],
        "test-version",
        nbasisf=nchan,
        sigmasq=0.0,
    )
    assert new_model.shape == model.shape
    mask = model != 0
    assert_allclose(new_model[mask], model[mask], atol=1e-8)

    mds = xr.open_zarr(mds_name, chunks=None)
    assert mds.attrs["spec"] == SPEC_VERSION
    assert mds.coefficients.dims == ("stokes", "par", "comps")
    assert list(mds.stokes.values) == ["I", "V"]
    assert "stokes" not in mds.attrs
    assert mds.attrs["writer-version"] == "test-version"
    assert mds.attrs["npix_x"] == nx and mds.attrs["npix_y"] == ny
    assert mds.attrs["cell_rad_x"] == cell_rad
    assert_allclose(mds.freqs.values, freq)
    assert_allclose(mds.times.values, time)


def test_model_to_ds_rerenders_unfitted_bands(tmp_path):
    """Bands outside fsel are still rendered from the fit."""
    nchan, nx, ny = 4, 32, 24
    freq = np.linspace(1.0e9, 1.3e9, nchan)
    model = np.zeros((nchan, 1, ny, nx))
    model[:, 0, 5, 20] = 1.0  # flat spectrum
    fsel = np.array([True, True, False, True])
    out = model_to_ds(
        np.array([0.0]),
        freq,
        fsel,
        model,
        np.ones((nchan, 1)),
        str(tmp_path / "m.mds"),
        1e-5,
        nx,
        ny,
        0.0,
        0.0,
        False,
        False,
        False,
        (0.0, 0.0),
        ["I"],
        "v",
        nbasisf=2,
        sigmasq=0.0,
    )
    assert_allclose(out[:, 0, 5, 20], 1.0, atol=1e-10)


def test_build_mds_dataset_matches_the_upgraded_genesis_schema():
    """The writer and the genesis upgrade step must produce the same schema."""
    up = upgrade(genesis_dataset())
    built = build_mds_dataset(
        up.coefficients.values,
        up.location_y.values,
        up.location_x.values,
        up.attrs["parametrisation"],
        list(up.params.values),
        up.attrs["texpr"],
        up.attrs["fexpr"],
        up.times.values,
        up.freqs.values,
        up.attrs["cell_rad_x"],
        up.attrs["npix_x"],
        up.attrs["npix_y"],
        up.attrs["center_x"],
        up.attrs["center_y"],
        up.attrs["flip_u"],
        up.attrs["flip_v"],
        up.attrs["flip_w"],
        (up.attrs["ra"], up.attrs["dec"]),
        list(up.stokes.values),
        "0.0.3",
    )
    assert _schema(built) == _schema(up)
    assert built.attrs == up.attrs
    assert_allclose(built.coefficients.values, up.coefficients.values)


def test_build_mds_dataset_rejects_a_stokes_length_mismatch():
    with pytest.raises(ValueError, match="stokes"):
        build_mds_dataset(
            np.zeros((2, 1, 1)),
            np.array([0]),
            np.array([0]),
            "t0",
            ["t0"],
            "t",
            "f",
            np.array([0.0]),
            np.array([1.0e9]),
            1e-5,
            4,
            4,
            0.0,
            0.0,
            False,
            False,
            False,
            (0.0, 0.0),
            ["I"],
            "v",
        )
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --extra full pytest tests/test_io.py -v`
Expected: FAIL. The old `build_mds_dataset` writes `spec="genesis"` and 2-D coefficients, and the old `model_to_ds` cannot fit a 4-D model.

- [ ] **Step 3: Implement**

In `src/pfb_model_spec/utils/io.py`:

1. Replace the module docstring's axis-convention paragraph with:

```
The current (`"0.1"`) `.mds` spec stores coefficients as `(stokes, par, comps)`, and every
model cube passed in or returned here is `(nband, nstokes, ny, nx)` -- the FITS/astropy
`(Y, X)` order, so callers no longer transpose. Older `.mds` datasets are upgraded on read
(see `pfb_model_spec.utils.spec`).
```

2. Add `from pfb_model_spec.utils.spec import SPEC_VERSION` to the imports.

3. Replace `build_mds_dataset` with the code below. Keep the existing Args docstring entries for the unchanged arguments.

```python
def build_mds_dataset(
    coeffs: np.ndarray,
    y_index: np.ndarray,
    x_index: np.ndarray,
    expr: str,
    params: list,
    texpr: str,
    fexpr: str,
    time: np.ndarray,
    freq: np.ndarray,
    cell_rad: float,
    nx: int,
    ny: int,
    x0: float,
    y0: float,
    flip_u: bool,
    flip_v: bool,
    flip_w: bool,
    radec: tuple[float, float],
    stokes: list[str],
    writer_version: str,
) -> xr.Dataset:
    """Assemble a `.mds` dataset at the current spec (``SPEC_VERSION``) from fitted coefficients.

    Single owner of the `.mds` schema: ``model_to_ds`` and the ``model2comps`` converter
    both build through here. The ``genesis -> 0.1`` step in ``utils/spec.py`` must produce
    the same schema; ``tests/test_io.py`` checks that.

    Args:
        coeffs: Fitted coefficients, shape ``(nstokes, npar, ncomps)``.
        y_index: Component row (y) pixel locations, shape ``(ncomps,)``.
        x_index: Component column (x, FITS ``NAXIS1``) pixel locations, shape ``(ncomps,)``.
        stokes: Stokes product per ``coeffs`` plane, e.g. ``["I", "V"]``.
        writer_version: pfb-model-spec (or caller) version recorded as ``writer-version``.

    Returns:
        The `.mds` ``xarray.Dataset`` (not yet written to disk).

    Raises:
        ValueError: If ``stokes`` does not match ``coeffs``'s leading axis.
    """
    stokes = list(stokes)
    if len(stokes) != coeffs.shape[0]:
        raise ValueError(f"stokes has {len(stokes)} entries but coeffs has {coeffs.shape[0]} planes")
    return xr.Dataset(
        data_vars={"coefficients": (("stokes", "par", "comps"), coeffs)},
        coords={
            "stokes": (("stokes",), stokes),
            "location_x": (("comps",), x_index),
            "location_y": (("comps",), y_index),
            "params": (("par",), params),
            "times": (("t",), time),
            "freqs": (("f",), freq),
        },
        attrs={
            "writer-version": writer_version,
            "spec": SPEC_VERSION,
            "cell_rad_x": cell_rad,
            "cell_rad_y": cell_rad,
            "npix_x": nx,
            "npix_y": ny,
            "texpr": texpr,
            "fexpr": fexpr,
            "center_x": x0,
            "center_y": y0,
            "flip_u": flip_u,
            "flip_v": flip_v,
            "flip_w": flip_w,
            "ra": radec[0],
            "dec": radec[1],
            "parametrisation": expr,
        },
    )
```

4. In `model_to_ds`:
   - rename the parameter `version` to `writer_version`;
   - change `stokes: str` to `stokes: list[str]`;
   - update the docstring shapes: `model` and the return are `(nband, nstokes, ny, nx)`, and `wgt` is `(nband, nstokes)`;
   - replace the body with the code below.

```python
    coeffs, y_index, x_index, expr, params, texpr, fexpr = fit_image_cube(
        time,
        freq[fsel],
        model[None, fsel],
        wgt=wgt[None, fsel],
        nbasisf=nbasisf,
        method=method,
        sigmasq=sigmasq,
    )

    coeff_dataset = build_mds_dataset(
        coeffs,
        y_index,
        x_index,
        expr,
        params,
        texpr,
        fexpr,
        time,
        freq,
        cell_rad,
        nx,
        ny,
        x0,
        y0,
        flip_u,
        flip_v,
        flip_w,
        radec,
        stokes,
        writer_version,
    )
    coeff_dataset.to_zarr(mds_name, mode="w")

    nband, nstokes = model.shape[:2]
    new_model = np.zeros((nband, nstokes, ny, nx), dtype=model.dtype)
    for b in range(nband):
        new_model[b] = eval_coeffs_to_slice(
            time[0],
            freq[b],
            coeffs,
            y_index,
            x_index,
            expr,
            params,
            texpr,
            fexpr,
            nxi=nx,
            nyi=ny,
            cellxi=cell_rad,
            cellyi=cell_rad,
            x0i=x0,
            y0i=y0,
            nxo=nx,
            nyo=ny,
            cellxo=cell_rad,
            cellyo=cell_rad,
            x0o=x0,
            y0o=y0,
        )
    return new_model
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra full pytest tests/test_io.py tests/test_spec.py tests/test_modelspec.py -v`
Expected: all PASS.

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
source .venv/bin/activate && git add src/pfb_model_spec/utils/io.py tests/test_io.py && git commit -m "feat!: write spec 0.1 .mds datasets from build_mds_dataset and model_to_ds

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 4: Upgrade-on-read and `(ny, nx)` images in `degrid.py`

**Files:**
- Modify: `src/pfb_model_spec/utils/degrid.py`
- Test: `tests/test_degrid.py`

**Interfaces:**
- Consumes:
  - `upgrade` (Task 1);
  - `fit_image_cube`/`eval_coeffs_to_slice` (Task 2);
  - `build_mds_dataset` (Task 3);
  - `tests._genesis`.
- Produces (`pfb_model_spec.utils.degrid`):
  - `model_geometry(ds)["stokes"]` is now `list[str]` (was `str`);
  - `render_model_region(...) -> (nstokes, ny, nx)`;
  - `degrid_stokes` takes `stokes_image` `(nstokes, ny, nx)`;
  - `apply_mueller`: `mueller` is `(nso, nsi, ny, nx)`;
  - `model_to_apparent_vis_for_region`: `region_mask` is `(ny, nx)`;
  - all signatures are otherwise unchanged.

- [ ] **Step 1: Update the existing tests to the 0.1 layout**

Edit `tests/test_degrid.py`:

1. Add these imports:
   ```python
   from tests._genesis import FREQS, genesis_cube, genesis_dataset
   ```
2. In `test_degrid_stokes_geometry_round_trip`, `img[0, src[0], src[1]] = 1.0` now means `y=37, x=91`. `vis2dirty` returns an x-major `(nx, ny)` image, so change the final assertion to:
   ```python
   assert np.unravel_index(np.argmax(back), back.shape) == (src[1], src[0])
   ```
3. Replace the body of `_synthetic_mds` (the cube is now `(nband, 1, ny, nx)`, with the same two components at `(x=20, y=31)` and `(x=44, y=12)`):
   ```python
   freq = np.linspace(1.0e9, 1.2e9, nband)
   time = np.array([0.0])
   cell = np.deg2rad(2.0 / 3600.0)
   cube = np.zeros((nband, 1, ny, nx))
   cube[:, 0, 31, 20] = (freq / freq[0]) ** -0.7
   cube[:, 0, 12, 44] = 2.0 * (freq / freq[0]) ** -1.1
   coeffs, yi, xi, expr, params, texpr, fexpr = fit_image_cube(
       time, freq, cube[None], nbasisf=nband, method="Legendre", sigmasq=0
   )
   ds = build_mds_dataset(
       coeffs, yi, xi, expr, params, texpr, fexpr, time, freq, cell, nx, ny,
       0.0, 0.0, False, True, False, (0.1, -0.5), ["I"], "test",
   )
   return ds, cube[:, 0], time, freq, cell
   ```
   It returns `cube[:, 0]`, shape `(nband, ny, nx)`, so callers index `cube[band]` as before. After pasting, let `ruff format` reflow the call.
4. Do the same in `_single_component_mds`:
   ```python
   cube = np.zeros((2, 1, ny, nx))
   cube[:, 0, ny // 2, nx // 2] = 1.0
   coeffs, yi, xi, expr, params, texpr, fexpr = fit_image_cube(
       time, freq, cube[None], nbasisf=2, method="Legendre", sigmasq=0
   )
   ```
   Pass `yi, xi` and `["I"]` to `build_mds_dataset` exactly as in item 3.
5. `test_render_model_region_matches_the_fitted_cube`: the shape assertion becomes `assert got.shape == (1, ds.attrs["npix_y"], ds.attrs["npix_x"])`.
6. `test_render_model_region_interpolates_between_fitted_bands`: index `(y=31, x=20)`:
   ```python
   lo, hi = cube[0, 31, 20], cube[1, 31, 20]
   assert min(lo, hi) < got[31, 20] < max(lo, hi)
   ```
7. `test_model_geometry_reads_the_mds_attrs`: `assert geom["stokes"] == ["I"]`.
8. `test_render_model_region_rejects_unknown_spec`: keep it as it is. The message now comes from `upgrade` ("Unknown .mds spec 'some-future-spec'") and still matches.
9. `test_fused_wrapper_region_mask_selects_components`: the mask is `(ny, nx)`:
   ```python
   inside = np.zeros((ny, nx))
   inside[31, 20] = 1.0  # the first component only
   ```
10. In `test_fused_wrapper_identity_mueller_matches_no_beam` and `test_stokes_out_relabels_a_non_prefix_mueller`, build Muellers on `(ny, nx)`: use `_mueller(1, 1, ny, nx)` and `np.zeros((2, 1, ny, nx))`.
11. Append the orientation pin and the newer-spec test:

```python
def test_genesis_model_degrids_like_its_x_major_render():
    """Pins orientation across the upgrade: a genesis .mds (x-major, non-square) must predict
    exactly what degridding its own x-major render with ducc0 directly predicts."""
    ds = genesis_dataset()
    rng = np.random.default_rng(40)
    uvw, chan = _uvw_freq(rng, nrow=300, nchan=2)
    got = model_to_apparent_vis_for_region(
        ds, uvw=uvw, freq=chan, corr_types=("XX",), time=0.0, freq_out=FREQS[2]
    )
    want = dirty2vis(
        uvw=uvw,
        freq=chan,
        dirty=genesis_cube(FREQS[2:])[0],
        pixsize_x=ds.attrs["cell_rad_x"],
        pixsize_y=ds.attrs["cell_rad_y"],
        center_x=0.0,
        center_y=0.0,
        flip_u=False,
        flip_v=True,
        flip_w=False,
        epsilon=1e-7,
        do_wgridding=True,
        divide_by_n=False,
        nthreads=1,
    )
    assert_allclose(got[..., 0], want, atol=1e-8)


def test_model_geometry_rejects_a_newer_spec():
    with pytest.raises(ValueError, match="needs pfb-model-spec"):
        model_geometry(genesis_dataset(spec="9.9"))
```

Also add `from ducc0.wgridder.experimental import dirty2vis` at the top of the test module.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --extra full pytest tests/test_degrid.py -v`
Expected: FAIL in the render/geometry/fused tests (shape, `stokes`, positional `eval_coeffs_to_slice` TypeError). The pure `stokes_vis_to_corr`/`apply_mueller` tests still pass.

- [ ] **Step 3: Implement**

In `src/pfb_model_spec/utils/degrid.py`:

1. Replace the module docstring's axis paragraph with:

```
Axis convention: images are `(nstokes, ny, nx)`, the spec 0.1 / FITS order. ducc0's
`dirty2vis` is x-major, so :func:`degrid_stokes` transposes exactly once, right before the
gridder call; nothing else here transposes. Visibilities have no orientation, so a caller
of the fused wrapper never transposes at all.
```

2. Delete `_SUPPORTED_SPECS` and its comment. Add `from pfb_model_spec.utils.spec import upgrade`.

3. In `degrid_stokes`:
   - pass `dirty=np.ascontiguousarray(stokes_image[s].T),  # (ny, nx) -> ducc0's x-major (nx, ny)`;
   - change the docstring's `stokes_image` shape to `(nstokes, ny, nx)`.

4. In `model_geometry`, replace the spec check with an upgrade and read Stokes from the coord:

```python
    ds = upgrade(model_ds)
    a = ds.attrs
    cell_x = float(a["cell_rad_x"])
    ...  # unchanged square-pixel check and fields
        "stokes": [str(s) for s in ds.stokes.values],
```

   Update its docstring:
   - `Raises` now says "unknown or newer `.mds` spec (from `upgrade`)";
   - `stokes` is a list.

5. In `render_model_region`:
   - replace the spec check with `ds = upgrade(model_ds)` and `a = ds.attrs`;
   - read the coefficients and locations from `ds`;
   - return the slice directly (no more `[None]`).

```python
    return eval_coeffs_to_slice(
        time,
        freq_out,
        ds.coefficients.values,
        ds.location_y.values,
        ds.location_x.values,
        a["parametrisation"],
        ds.params.values,
        a["texpr"],
        a["fexpr"],
        nxi=nxi,
        nyi=nyi,
        cellxi=cellxi,
        cellyi=cellyi,
        x0i=x0i,
        y0i=y0i,
        nxo=nxo,
        nyo=nyo,
        cellxo=cellxo,
        cellyo=cellyo,
        x0o=x0o,
        y0o=y0o,
    )
```

   Docstring changes:
   - the return is `(nstokes, ny, nx)` with one plane per entry of the `.mds` `stokes` coord;
   - drop the "1 for genesis" sentence;
   - `Raises`: unknown or newer spec.

6. Change the shape text in the remaining docstrings; the code does not change:
   - `apply_mueller`: `(nstokes_in, ny, nx)`, `(nstokes_out, nstokes_in, ny, nx)`, `(nstokes_out, ny, nx)`. Its einsum `"ijxy,jxy->ixy"` is axis-agnostic, so leave it.
   - `model_to_apparent_vis_for_region`: `region_mask` is `(ny, nx)`, `mueller` is `(nstokes_out, nstokes_in, ny, nx)`, and the `stokes_out` default is "the model's `stokes` coord".

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra full pytest tests/test_degrid.py tests/test_io.py tests/test_modelspec.py tests/test_spec.py -v`
Expected: all PASS.

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
source .venv/bin/activate && git add src/pfb_model_spec/utils/degrid.py tests/test_degrid.py && git commit -m "feat!: degrid (ny, nx) spec 0.1 models and upgrade older specs on read

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 5: `model2comps` writes spec 0.1 without transposing

**Files:**
- Modify: `src/pfb_model_spec/core/model2comps.py`
- Test: `tests/test_model2comps.py`

**Interfaces:**
- Consumes:
  - `fit_image_cube`/`eval_coeffs_to_slice`/`model_from_mds` (Task 2);
  - `build_mds_dataset(..., stokes: list[str], writer_version)` (Task 3);
  - `save_fits(..., yx_order=True)` (existing in `utils/fits.py`).
- Produces:
  - `read_wsclean_model(...)["model"]` is now `(nband, ny, nx)`;
  - `model2comps(...)` has an unchanged signature; `product` must be one of `I`, `Q`, `U`, `V`.

- [ ] **Step 1: Update the tests**

Edit `tests/test_model2comps.py`:

1. Add `from pfb_model_spec import __version__` to the imports.
2. In `test_model2comps_fits_roundtrip`, replace everything from the `# schema` comment through `assert (tmp_path / "out.fits").exists()` with:

```python
    # schema
    mds = xr.open_zarr(coeff_name, chunks=None)
    assert mds.attrs["spec"] == "0.1"
    assert mds.attrs["writer-version"] == __version__
    assert list(mds.stokes.values) == ["I"]
    assert "stokes" not in mds.attrs
    assert mds.attrs["npix_x"] == nx
    assert mds.attrs["npix_y"] == ny
    assert_allclose(mds.attrs["cell_rad_x"], np.deg2rad(cell_deg))
    assert mds.attrs["flip_v"] is True
    assert mds.attrs["flip_u"] is False
    assert_allclose(mds.attrs["ra"], np.deg2rad(ra_deg))
    assert_allclose(mds.attrs["dec"], np.deg2rad(dec_deg))
    assert_allclose(mds.freqs.values, freq)

    # numerical round-trip: exact fit reproduces the input on the source mask
    model_yx = model.transpose(0, 2, 1)  # the FITS planes as written: (nchan, ny, nx)
    rendered = model_from_mds(coeff_name)[0, :, 0]  # (nchan, ny, nx)
    assert rendered.shape == model_yx.shape
    mask = model_yx > 0
    assert_allclose(rendered[mask], model_yx[mask], atol=1e-8)

    # the sanity FITS must match the input planes pixel for pixel, not transposed
    sanity = fits.getdata(tmp_path / "out.fits")
    assert sanity.shape == (1, nchan, ny, nx)
    assert_allclose(sanity[0], model_yx, atol=1e-6)
```

3. Append:

```python
def test_model2comps_rejects_a_non_stokes_product(tmp_path):
    nchan, nx, ny = 2, 24, 16
    freq = np.linspace(1.0e9, 1.5e9, nchan)
    model = _synth_cube(nchan, nx, ny, freq, float(freq.mean()))
    prefix = str(tmp_path / "wsclean")
    _write_wsclean_fits(prefix, model, freq, 2.5 / 3600.0, 0.0, -30.0, np.ones(nchan))
    with pytest.raises(ValueError, match="product"):
        model2comps(str(tmp_path / "out"), from_fits=prefix, nbasisf=nchan, product="IQ")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --extra full pytest tests/test_model2comps.py -v`
Expected: FAIL. `fit_image_cube` rejects the 4-D model, or the schema assertions fail.

- [ ] **Step 3: Implement**

In `src/pfb_model_spec/core/model2comps.py`:

1. Replace the module docstring's last paragraph with:

```
Axis convention: spec 0.1 is `(Y, X)` like FITS, so WSClean planes are used as read --
`(nband, ny, nx)` -- with a length-1 Stokes axis added for the fit. No transpose.
```

2. In `read_wsclean_model`:
   - change `planes.append(hdu[0].data.squeeze().T)  # ...` to `planes.append(hdu[0].data.squeeze())  # (ny, nx), FITS row-major`;
   - in the docstring, ``model`` becomes `` `(nband, ny, nx)` ``.

3. In `model2comps`:
   - Directly after the logging setup, add:
     ```python
     if product not in ("I", "Q", "U", "V"):
         raise ValueError(f"product must be a single Stokes parameter (I, Q, U or V), got {product!r}")
     ```
   - Change the comment `# (nband, nx, ny)` on `model = cube["model"]` to `# (nband, ny, nx)`.
   - Replace the fit call with:
     ```python
     coeffs, y_index, x_index, expr, params, texpr, fexpr = fit_image_cube(
         time,
         mfreqs[fsel],
         model[None, fsel, None],
         wgt=wsums[None, fsel, None],
         nbasisf=nbasisf,
         method=fit_mode,
         sigmasq=sigmasq,
     )
     ```
   - In the `build_mds_dataset(...)` call, pass `y_index, x_index` (in that order) where `x_index, y_index` were, and pass `[product]` instead of `product`.
   - In both `_render(...)` call sites, pass `y_index, x_index` in place of `x_index, y_index`.
   - The interpolation-error block compares `modelo[:, 0]` with `model`:
     ```python
     log.info(f"Fractional interpolation error is {np.linalg.norm((modelo[:, 0] - model).ravel()) / denom:.3e}")
     ```
   - Write the sanity FITS from the `(nband, 1, ny, nx)` stack as-is:
     ```python
     save_fits(modelo, fits_name, hdr, overwrite=overwrite, yx_order=True)
     ```

4. Replace `_render`:

```python
def _render(coeffs, y_index, x_index, expr, params, texpr, fexpr, t, f, nx, ny, cell_rad, x0, y0):
    """Render coefficients to a ``(nstokes, ny, nx)`` slice at time ``t``, frequency ``f``.

    The output grid matches the fit grid (same npix/cell/centre), so this is the identity
    resample used for the interpolation-error and sanity-FITS renders.
    """
    geometry = dict(nxi=nx, nyi=ny, cellxi=cell_rad, cellyi=cell_rad, x0i=x0, y0i=y0)
    return eval_coeffs_to_slice(
        t,
        f,
        coeffs,
        y_index,
        x_index,
        expr,
        params,
        texpr,
        fexpr,
        **geometry,
        nxo=nx,
        nyo=ny,
        cellxo=cell_rad,
        cellyo=cell_rad,
        x0o=x0,
        y0o=y0,
    )
```

- [ ] **Step 4: Run the full suite (the red window closes here)**

Run: `uv run --extra full pytest -v`
Expected: **all** tests PASS.

- [ ] **Step 5: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
source .venv/bin/activate && git add src/pfb_model_spec/core/model2comps.py tests/test_model2comps.py && git commit -m "feat!: model2comps writes spec 0.1 (Y, X) models with a Stokes axis

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 6: `pfbspec convert`

**Files:**
- Create: `src/pfb_model_spec/core/convert.py`
- Create: `src/pfb_model_spec/cli/convert.py`
- Modify: `src/pfb_model_spec/cli/__init__.py`
- Test: `tests/test_convert.py`, `tests/test_roundtrip.py`
- Generated by the hook (never hand-written): `src/pfb_model_spec/cabs/convert.yml`

**Interfaces:**
- Consumes: `SPEC_VERSION`, `spec_of`, `upgrade` (Task 1); `model_from_mds` (Task 2); `tests._genesis`.
- Produces:
  - `pfb_model_spec.core.convert.convert(input_mds, output_mds, overwrite: bool = False) -> None`;
  - the CLI command `pfbspec convert --input-mds IN --output-mds OUT [--overwrite]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_convert.py`:

```python
"""Tests for the `.mds` spec converter (core implementation)."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pfb_model_spec.core.convert import convert
from pfb_model_spec.utils.modelspec import model_from_mds
from pfb_model_spec.utils.spec import SPEC_VERSION, upgrade
from tests._genesis import genesis_cube, genesis_dataset


def test_convert_upgrades_genesis_on_disk(tmp_path):
    src, dst = tmp_path / "g.mds", tmp_path / "new.mds"
    genesis_dataset().to_zarr(src)
    convert(str(src), str(dst))

    raw = xr.open_zarr(dst, chunks=None)  # read without upgrading: the store itself is 0.1
    assert raw.attrs["spec"] == SPEC_VERSION
    assert list(raw.stokes.values) == ["I"]
    assert_allclose(model_from_mds(str(dst))[0, :, 0], genesis_cube().transpose(0, 2, 1), atol=1e-12)
    # the source is untouched
    assert xr.open_zarr(src, chunks=None).attrs["spec"] == "genesis"


def test_convert_writes_a_current_spec_input_through(tmp_path):
    src, dst = tmp_path / "cur.mds", tmp_path / "copy.mds"
    upgrade(genesis_dataset()).to_zarr(src)
    convert(str(src), str(dst))
    xr.testing.assert_identical(xr.open_zarr(dst, chunks=None), xr.open_zarr(src, chunks=None))


def test_convert_refuses_to_overwrite_without_the_flag(tmp_path):
    src, dst = tmp_path / "g.mds", tmp_path / "new.mds"
    genesis_dataset().to_zarr(src)
    convert(str(src), str(dst))
    with pytest.raises(ValueError, match="exists"):
        convert(str(src), str(dst))
    convert(str(src), str(dst), overwrite=True)


@pytest.mark.parametrize("alias", ["{p}", "{p}/", "./{rel}", "file://{p}"])
def test_convert_refuses_in_place(tmp_path, monkeypatch, alias):
    """The same store named differently is still in-place, even with --overwrite."""
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "g.mds"
    genesis_dataset().to_zarr(src)
    with pytest.raises(ValueError, match="in-place"):
        convert(str(src), alias.format(p=src, rel="g.mds"), overwrite=True)


def test_convert_works_on_remote_uris():
    """fsspec's in-memory filesystem stands in for s3:// / gs:// here."""
    src, dst = "memory://convert-test/g.mds", "memory://convert-test/new.mds"
    genesis_dataset().to_zarr(src, mode="w")
    convert(src, dst)
    assert xr.open_zarr(dst, chunks=None).attrs["spec"] == SPEC_VERSION
    with pytest.raises(ValueError, match="exists"):
        convert(src, dst)
    with pytest.raises(ValueError, match="in-place"):
        convert(src, src + "/", overwrite=True)


def test_convert_rejects_a_newer_spec(tmp_path):
    src = tmp_path / "future.mds"
    genesis_dataset(spec="9.9").to_zarr(src)
    with pytest.raises(ValueError, match="needs pfb-model-spec"):
        convert(str(src), str(tmp_path / "out.mds"))
    assert not (tmp_path / "out.mds").exists()


def test_convert_accepts_path_objects(tmp_path):
    """The CLI hands core UPath objects, not strings."""
    from upath import UPath

    src = tmp_path / "g.mds"
    genesis_dataset().to_zarr(src)
    convert(UPath(src), UPath(tmp_path / "new.mds"))
    assert np.array_equal(xr.open_zarr(tmp_path / "new.mds", chunks=None).location_x.values, [4, 27, 15])
```

Append to `tests/test_roundtrip.py`:

```python
def test_roundtrip_convert() -> None:
    """The convert command must round-trip cleanly through a cab."""
    _assert_roundtrip("convert")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --extra full pytest tests/test_convert.py tests/test_roundtrip.py -v`
Expected: `ModuleNotFoundError: pfb_model_spec.core.convert`, and an AssertionError `Missing CLI module`.

- [ ] **Step 3: Implement the core**

Create `src/pfb_model_spec/core/convert.py`:

```python
"""Upgrade a `.mds` component model to the current spec (``pfbspec convert``).

Conversion is one-way (older -> newer) and is exactly the in-memory upgrade every reader
already applies (`pfb_model_spec.utils.spec.upgrade`), persisted to a new store. In-place
conversion is not supported: the source is opened lazily, so writing over it would destroy
the data being read.
"""

import logging
import os

import xarray as xr
from upath import UPath

from pfb_model_spec.utils.spec import SPEC_VERSION, spec_of, upgrade

log = logging.getLogger("pfb_model_spec.convert")


def _canonical(path: str | os.PathLike) -> str:
    """Normalise a local path or remote URI so two spellings of one store compare equal."""
    up = UPath(path)
    if up.protocol in ("", "file", "local"):
        return os.path.realpath(up.path)
    return f"{up.protocol}://{up.path.rstrip('/')}"


def convert(input_mds: str | os.PathLike, output_mds: str | os.PathLike, overwrite: bool = False) -> None:
    """Upgrade ``input_mds`` to ``SPEC_VERSION`` and write the result to ``output_mds``.

    Args:
        input_mds: Source `.mds` (local path or remote URI) at any known spec.
        output_mds: Destination `.mds`; must not be the source.
        overwrite: Replace ``output_mds`` if it exists.

    Raises:
        ValueError: If input and output are the same store, if the output exists and
            ``overwrite`` is False, or if the input's spec is unknown or newer than this
            package understands.
    """
    if not log.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    if _canonical(input_mds) == _canonical(output_mds):
        raise ValueError(f"Input and output are the same store ({input_mds}); in-place conversion is not supported")
    if UPath(output_mds).exists() and not overwrite:
        raise ValueError(f"{output_mds} exists. Set overwrite=True to replace it.")

    raw = xr.open_zarr(str(input_mds), chunks=None)
    source = spec_of(raw)
    ds = upgrade(raw)  # raises before anything is written
    if source == SPEC_VERSION:
        log.info(f"{input_mds} is already at spec {SPEC_VERSION}; writing it through unchanged")
    else:
        log.info(f"Upgrading {input_mds} from spec {source} to {SPEC_VERSION}")
    ds.to_zarr(str(output_mds), mode="w")
    log.info(f"Wrote {output_mds}")
```

- [ ] **Step 4: Add the CLI wrapper**

Create `src/pfb_model_spec/cli/convert.py` with exactly this content. It is the canonical `hip-cargo generate-function` output, verified to round-trip when this plan was written:

```python
from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import StimelaMeta, parse_upath, stimela_cab, stimela_output

Directory = NewType("Directory", Path)


@stimela_cab(
    name="convert",
    info="Upgrade a component model (.mds) to the current spec.",
)
@stimela_output(
    dtype="Directory",
    name="output-mds",
    info="Path to write the upgraded component model (.mds) to.",
    required=True,
    must_exist=False,
    metadata={"rich_help_panel": "Output"},
)
def convert(
    input_mds: Annotated[
        Directory,
        typer.Option(
            ...,
            parser=parse_upath,
            help="Component model (.mds) to upgrade. Its spec is detected from the dataset attrs.",
            rich_help_panel="Input",
        ),
        StimelaMeta(
            must_exist=True,
        ),
    ],
    output_mds: Annotated[
        Directory,
        typer.Option(
            ...,
            parser=parse_upath,
            help="Path to write the upgraded component model (.mds) to.",
            rich_help_panel="Output",
        ),
        StimelaMeta(
            must_exist=False,
        ),
    ],
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Allow overwrite of an existing output.",
            rich_help_panel="Control",
        ),
    ] = False,
    backend: Annotated[
        Literal["auto", "native", "apptainer", "singularity", "docker", "podman"],
        typer.Option(
            help="Execution backend.",
        ),
        StimelaMeta(
            skip=True,
        ),
    ] = "auto",
    always_pull_images: Annotated[
        bool,
        typer.Option(
            help="Always pull container images, even if cached locally.",
        ),
        StimelaMeta(
            skip=True,
        ),
    ] = False,
):
    """
    Upgrade a component model (.mds) to the current spec.
    """
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                convert,
                dict(
                    input_mds=input_mds,
                    overwrite=overwrite,
                    output_mds=output_mds,
                ),
            )

            # Lazy import the core implementation
            from pfb_model_spec.core.convert import convert as convert_core  # noqa: E402

            # Call the core function with all parameters
            convert_core(
                input_mds,
                overwrite=overwrite,
                output_mds=output_mds,
            )
            return
        except ImportError:
            if backend == "native":
                raise

    # Resolve container image from installed package metadata
    from hip_cargo.utils.config import get_container_image  # noqa: E402
    from hip_cargo.utils.runner import run_in_container  # noqa: E402

    image = get_container_image("pfb-model-spec")
    if image is None:
        raise RuntimeError("No Container URL in pfb-model-spec metadata.")

    run_in_container(
        convert,
        dict(
            input_mds=input_mds,
            overwrite=overwrite,
            output_mds=output_mds,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
```

Register it in `src/pfb_model_spec/cli/__init__.py` after the `model2comps` lines:

```python
from pfb_model_spec.cli.convert import convert  # noqa: E402

app.command(name="convert")(convert)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra full pytest tests/test_convert.py tests/test_roundtrip.py tests/test_lightweight_import.py -v`
Expected: all PASS.

If `test_roundtrip_convert` reports a differing line, do not hand-tune the file. Regenerate it canonically, copy the result over the CLI module, and re-run:

```bash
uv run --extra full hip-cargo generate-cabs --module src/pfb_model_spec/cli/convert.py --output-dir /tmp/convert-cab
uv run --extra full hip-cargo generate-function --cab-file /tmp/convert-cab/convert.yml --output-file src/pfb_model_spec/cli/convert.py --config-file pyproject.toml
```

- [ ] **Step 6: Smoke-test the real CLI**

```bash
uv run --extra full python -c "from tests._genesis import genesis_dataset; genesis_dataset().to_zarr('/tmp/g.mds', mode='w')"
uv run --extra full pfbspec convert --input-mds /tmp/g.mds --output-mds /tmp/g01.mds --backend native
uv run --extra full python -c "import xarray as xr; print(xr.open_zarr('/tmp/g01.mds').attrs['spec'])"
```

Expected: a log line `Upgrading /tmp/g.mds from spec genesis to 0.1`, then `0.1`. Afterwards `rm -rf /tmp/g.mds /tmp/g01.mds`.

- [ ] **Step 7: Format, lint, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
source .venv/bin/activate && git add src/pfb_model_spec/core/convert.py src/pfb_model_spec/cli/convert.py src/pfb_model_spec/cli/__init__.py tests/test_convert.py tests/test_roundtrip.py src/pfb_model_spec/cabs && git commit -m "feat: add pfbspec convert to upgrade .mds datasets to the current spec

Closes #17

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

The hook generates `cabs/convert.yml`. If the first commit "fails" because of that, `git add src/pfb_model_spec/cabs` and re-run the commit.

### Task 7: Docs and whole-branch verification

**Files:**
- Modify: `.claude/rules/component-model.md`, `.claude/rules/testing-and-ci.md`, `CLAUDE.md`

**Interfaces:** documentation only. It describes the names produced in Tasks 1–6 exactly as defined there.

- [ ] **Step 1: Update `.claude/rules/component-model.md`**

1. **Replace the "Axis convention (x-major)" section** with a "Spec versions" section (next item) plus this axis text:

```markdown
## Axis convention ((Y, X), spec 0.1)

Every model cube this library builds, consumes or renders is `(..., nstokes, ny, nx)` --
the FITS/astropy order, so callers (pfb-imaging, `model2comps`) no longer transpose.
`location_x` indexes FITS `NAXIS1` and `location_y` `NAXIS2`, in every spec. ducc0's
`dirty2vis` is x-major, so `degrid_stokes` transposes exactly once, before the gridder.
```

2. **Add the "Spec versions" section:**

```markdown
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
```

3. **Update "The library API":**
   - new `fit_image_cube` signature and return, as in Task 2's Interfaces;
   - delete the `fit_image_fscube` bullet;
   - `eval_coeffs_to_cube` returns `(ntime, nfreq, nstokes, ny, nx)`;
   - `eval_coeffs_to_slice` returns `(nstokes, nyo, nxo)`, with keyword-only geometry;
   - `model_from_mds` reads any known spec.
4. **Update "The I/O API":** the new `build_mds_dataset`/`model_to_ds` signatures and shapes, as in Task 3's Interfaces.
5. **Update "The degrid API":**
   - shapes become `(nstokes, ny, nx)`;
   - `model_geometry` returns `stokes` as a list;
   - readers upgrade older specs;
   - remove the old "Axis convention" paragraph that says there is no transpose in the module.
6. **Update "The converter":** there is no transpose on read, and `product` must be a single Stokes parameter.
7. **Replace "The `.mds` schema"** with the spec 0.1 table from the design doc (dims, coords, attrs, `writer-version`).
8. **Update "Testing":** mention `tests/_genesis.py` (frozen), `test_spec.py` and `test_convert.py`.

- [ ] **Step 2: Update `.claude/rules/testing-and-ci.md`**

1. Under "1. Testing", add:
   - `tests/test_spec.py` guards that `SPEC_VERSION` is not behind the package `major.minor`;
   - `tests/_genesis.py` is a frozen fixture and must never be updated to follow the library.
2. Under "5. Releases", add:
   - `tbump` to a new minor version fails CI until a spec step (possibly a no-op) is added to `utils/spec.py`;
   - schema changes ship only in a new minor release.

- [ ] **Step 3: Update `CLAUDE.md`**

In "Current status":
- add a bullet: **Spec versioning + `pfbspec convert`** (`utils/spec.py`, `core/convert.py`) — the current spec is `"0.1"` (Stokes axis, `(Y, X)` cubes); older `.mds` stores upgrade on read; a schema change needs a breaking version bump plus a registry step;
- remove the mention of x-major;
- in "Project structure", add `spec.py` to the `utils/` comment.

- [ ] **Step 4: Verify the whole branch**

```bash
uv run ruff format . && uv run ruff check . --fix
uv run --extra full pytest -v
grep -rn "fit_image_fscube\|_SUPPORTED_SPECS\|pfb-imaging-version\|x-major" src/ .claude/ CLAUDE.md
```

Expected:
- the tests all PASS;
- `grep` finds `pfb-imaging-version` only in `utils/spec.py` (the genesis step);
- `x-major` appears only where it describes ducc0 or the internal `_resample_xmajor`/`genesis` behaviour.

Fix anything else it finds.

- [ ] **Step 5: Commit**

```bash
source .venv/bin/activate && git add .claude/rules/component-model.md .claude/rules/testing-and-ci.md CLAUDE.md && git commit -m "docs: document spec versioning, convert, and the 0.1 schema

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## After the plan (not part of execution)

- **Open the PR.** It closes #17, #19 and #20. The PR body must list the breaking changes for pfb-imaging:
  - `model_to_ds` takes `(nband, nstokes, ny, nx)` and `wgt` `(nband, nstokes)`, with `stokes: list[str]` and `writer_version=`;
  - `model_geometry()["stokes"]` is a list;
  - degrid images are `(nstokes, ny, nx)`.
- **After merge:** wait for `update-cabs.yml`, then `git pull` and `uv run tbump 0.1.0`.
- **Follow-up PR in ratt-ru/pfb-imaging:**
  - drop the transposes around `model_to_ds` in `deconv`;
  - add the Stokes axis;
  - rename `version=` → `writer_version=`;
  - pin `pfb-model-spec>=0.1,<0.2`.
