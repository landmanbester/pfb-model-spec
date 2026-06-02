# Component-Model Spec Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the portable component-model fit/render library (`pfb_imaging/utils/modelspec.py`) into `pfb-model-spec` as a top-level `modelspec.py` module, with self-contained synthetic-data tests and no changes to pfb-imaging.

**Architecture:** Copy the self-contained `modelspec.py` (numpy/sympy/scipy/xarray only, zero pfb-imaging imports) verbatim into `src/pfb_model_spec/modelspec.py`, preserving every public signature and the `.mds` schema so pfb-imaging can later swap to importing it by changing import lines only. Heavy deps go in the `[full]` extra and are kept out of the top-level `__init__.py` so the lightweight install stays intact. Tests replace pfb-imaging's measurement-set fixtures with synthetic constants and vendor two pure-numpy helpers.

**Tech Stack:** Python 3.10+, `uv` (use `uv run --extra full ...` for all execution), `numpy`, `sympy`, `scipy`, `xarray`, `pytest`, `ruff`, hip-cargo packaging.

---

## Pre-flight notes (read once)

- **Branch:** Work is on `modelspec-library`. `_container_image.py` is already set to `ghcr.io/landmanbester/pfb-model-spec:modelspec-library` (the hip-cargo feature-branch tag), and `cabs/onboard.yml` already matches. Do not touch either.
- **Execution:** Always run via `uv run --extra full ...` so the scientific stack (added in Task 1) is present. Plain `uv run` may not include the `full` extra.
- **Pre-commit:** Every `git commit` triggers hooks: `ruff format`, `ruff check`, and `generate-cabs`. Because this phase adds **no** `cli/*.py` command, `generate-cabs` must produce **no** change to `cabs/onboard.yml`. If a commit fails because the hook modified `onboard.yml`, STOP — it means the container image isn't resolving (re-check `uv sync --extra full` / editable install) — do not commit a stripped/changed `onboard.yml`. If the hook only reformats your *new* files, re-stage (`git add -u`) and re-commit; that is normal.
- **Source of truth for the copy:** `/home/bester/software/pfb-imaging/src/pfb_imaging/utils/modelspec.py`.
- **Out of scope (do not add):** any `model2comps` CLI/command, the `.dds` path, FITS I/O, logging, `set_envs`. No edits to pfb-imaging.

---

## File Structure

- **Create** `src/pfb_model_spec/modelspec.py` — the spec library (verbatim copy). Responsibility: fit an image cube to coefficients and render coefficients back to images. Public API: `fit_image_cube`, `fit_image_fscube`, `eval_coeffs_to_cube`, `eval_coeffs_to_slice`, `model_from_mds`.
- **Modify** `pyproject.toml` — add scientific deps to `[project.optional-dependencies].full`.
- **Create** `tests/_synth.py` — test-only pure-numpy helpers (`gaussian2d`, `give_edges`) used to build synthetic model cubes. Not part of the public API.
- **Create** `tests/test_modelspec.py` — synthetic fit→render round-trip + spatial-interpolation test.
- **Create** `tests/test_lightweight_import.py` — guards that `import pfb_model_spec` does not pull in `modelspec` (keeps lightweight install intact).
- **Leave untouched:** `src/pfb_model_spec/__init__.py`, everything under `cli/`, `core/`, `cabs/`, `tests/test_roundtrip.py`, `tests/test_install.py`, `_container_image.py`.

---

### Task 1: Add the scientific stack to the `full` extra

**Files:**
- Modify: `pyproject.toml:30-31`

- [ ] **Step 1: Add the deps to the `full` extra**

In `pyproject.toml`, replace the empty `full` extra:

```toml
[project.optional-dependencies]
full = []
```

with:

```toml
[project.optional-dependencies]
full = [
    "numpy",
    "scipy",
    "sympy",
    "xarray",
]
```

- [ ] **Step 2: Sync the environment with the full extra**

Run: `uv sync --extra full`
Expected: resolves and installs numpy, scipy, sympy, xarray (plus existing dev tools) with no errors.

- [ ] **Step 3: Verify the stack imports**

Run: `uv run --extra full python -c "import numpy, scipy, sympy, xarray; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Lint**

Run: `uv run ruff format . && uv run ruff check . --fix`
Expected: passes (pyproject.toml is not Python; no code changes expected).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(deps): add numpy, scipy, sympy, xarray to full extra"
```
Expected: pre-commit hooks pass; `cabs/onboard.yml` is NOT modified.

---

### Task 2: Write the synthetic test (failing) to drive the module into existence

**Files:**
- Create: `tests/_synth.py`
- Create: `tests/test_modelspec.py`

- [ ] **Step 1: Create the vendored test helpers**

Create `tests/_synth.py` (copied verbatim from `pfb_imaging/utils/misc.py`; pure numpy):

```python
"""Pure-numpy helpers vendored from pfb_imaging.utils.misc for building
synthetic model cubes in tests. Not part of the public API."""

import numpy as np


def give_edges(p, q, nx, ny, nx_psf, ny_psf):
    nx0 = nx_psf // 2
    ny0 = ny_psf // 2

    # image overlap edges
    # left edge for x coordinate
    dxl = p - nx0
    xl = np.maximum(dxl, 0)

    # right edge for x coordinate
    dxu = p + nx0
    xu = np.minimum(dxu, nx)
    # left edge for y coordinate
    dyl = q - ny0
    yl = np.maximum(dyl, 0)
    # right edge for y coordinate
    dyu = q + ny0
    yu = np.minimum(dyu, ny)

    # PSF overlap edges
    xlpsf = np.maximum(nx0 - p, 0)
    xupsf = np.minimum(nx0 + nx - p, nx_psf)
    ylpsf = np.maximum(ny0 - q, 0)
    yupsf = np.minimum(ny0 + ny - q, ny_psf)

    return slice(xl, xu), slice(yl, yu), slice(xlpsf, xupsf), slice(ylpsf, yupsf)


def gaussian2d(xin, yin, gausspar=(1.0, 1.0, 0.0), normalise=True, nsigma=5):
    """
    xin         - grid of x coordinates
    yin         - grid of y coordinates
    gausspar    - (emaj, emin, pa) with emaj/emin as FWHM in units of xin/yin and pa in radians.
    normalise   - normalise kernel to have volume 1
    nsigma      - compute kernel out to this many standard deviations of the major axis
    """
    smaj, smin, pa = gausspar
    fwhm_conv = 2 * np.sqrt(2 * np.log(2))
    amat = np.array([[1.0 / smaj**2, 0], [0, 1.0 / smin**2]])
    # this parametrisation is equivalent to a standard rotation matrix with
    # t = np.pi/2 + pa; used for compatibility with fits
    rmat = np.array([[-np.sin(pa), -np.cos(pa)], [np.cos(pa), -np.sin(pa)]])
    amat = np.dot(np.dot(rmat, amat), rmat.T)
    sout = xin.shape
    # only compute the result out to nsigma standard deviations
    sigma_maj = smaj / fwhm_conv
    extent = (nsigma * sigma_maj) ** 2
    xflat = xin.squeeze()
    yflat = yin.squeeze()
    idx, idy = np.where(xflat**2 + yflat**2 <= extent)
    x = np.array([xflat[idx, idy].ravel(), yflat[idx, idy].ravel()])
    rmat = np.einsum("nb,bc,cn->n", x.T, amat, x)
    # adjust for the fact that gausspar corresponds to FWHM
    tmp = np.exp(-0.5 * fwhm_conv**2 * rmat)
    gausskern = np.zeros(xflat.shape, dtype=np.float64)
    gausskern[idx, idy] = tmp

    if normalise:
        gausskern /= np.sum(gausskern)
    return np.ascontiguousarray(gausskern.reshape(sout), dtype=np.float64)
```

- [ ] **Step 2: Create the synthetic round-trip test**

Create `tests/test_modelspec.py`:

```python
"""Synthetic, self-contained test of the component-model spec library.

Ports the numerics of pfb-imaging's tests/test_model2comps.py but replaces the
measurement-set-derived fixtures (ms_meta / image_geometry) with hard-coded
synthetic values, so no MS download or daskms/africanus is needed.
"""

import numpy as np
from numpy.testing import assert_allclose

from pfb_model_spec.modelspec import (
    eval_coeffs_to_cube,
    eval_coeffs_to_slice,
    fit_image_cube,
)
from tests._synth import gaussian2d, give_edges


def test_modelspec_roundtrip():
    np.random.seed(420)

    # synthetic spectral axis (replaces the MS-derived fixture)
    nchan = 8
    freq = np.linspace(1.0e9, 2.0e9, nchan)
    freq0 = float(freq.mean())

    # synthetic image geometry (replaces the image_geometry fixture).
    # cell_deg value is arbitrary but must be positive and consistent.
    cell_deg = 2.5 / 3600.0
    nx = 256
    ny = 256
    npix = nx

    # build a cube of Gaussian sources with power-law spectra
    model = np.zeros((nchan, nx, ny), dtype=np.float64)
    nsource = 25
    border = np.maximum(int(0.15 * nx), int(0.15 * ny))
    x_index = np.random.randint(border, npix - border, nsource)
    y_index = np.random.randint(border, npix - border, nsource)
    alpha = -0.7 + 0.1 * np.random.randn(nsource)
    ref_flux = 1.0 + np.exp(np.random.randn(nsource))
    extentx = np.random.randint(3, int(0.1 * nx), nsource)
    extenty = np.random.randint(3, int(0.1 * nx), nsource)
    pas = np.random.random(nsource) * 180
    x = -(nx / 2) + np.arange(nx)
    y = -(nx / 2) + np.arange(ny)
    xin, yin = np.meshgrid(x, y, indexing="ij")
    for i in range(nsource):
        emaj = np.maximum(extentx[i], extenty[i])
        emin = np.minimum(extentx[i], extenty[i])
        gauss = gaussian2d(xin, yin, gausspar=(emaj, emin, pas[i]))
        mx, my, gx, gy = give_edges(x_index[i], y_index[i], nx, ny, nx, ny)
        spectrum = ref_flux[i] * (freq / freq0) ** alpha[i]
        model[:, mx, my] += spectrum[:, None, None] * gauss[None, gx, gy]

    mfreqs = freq
    mtimes = np.array([0.0])  # single arbitrary time (ntime == 1)

    # fit then render back to a cube; with nbasisf == nband and ntime == 1 the
    # Legendre fit is exact, so the round-trip is accurate to ~machine precision.
    coeffs, x_idx, y_idx, expr, params, tfunc, ffunc = fit_image_cube(
        mtimes, mfreqs, model[None, :, :, :], nbasisf=nchan, sigmasq=0.0, method="Legendre"
    )
    image = eval_coeffs_to_cube(mtimes, mfreqs, nx, ny, coeffs, x_idx, y_idx, expr, params, tfunc, ffunc)

    image = image[0]  # drop the length-1 time axis
    mask = model > 0
    assert_allclose(image[mask], model[mask], atol=1e-10)

    # spatial interpolation: shifting the centre by an integer number of pixels
    # must leave the pixel centres unchanged.
    xshift = 25
    x0 = cell_deg * xshift
    yshift = -10
    y0 = cell_deg * yshift
    for npix_out in [100, nx, 2 * nx]:
        imout = eval_coeffs_to_slice(
            mtimes[0],
            mfreqs[0],
            coeffs,
            x_idx,
            y_idx,
            expr,
            params,
            tfunc,
            ffunc,
            nx,
            ny,
            cell_deg,
            cell_deg,
            0.0,
            0.0,
            npix_out,
            npix_out,
            cell_deg,
            cell_deg,
            x0,
            y0,
        )

        mx, my, gx, gy = give_edges(npix_out // 2 - xshift, npix_out // 2 - yshift, npix_out, npix_out, nx, ny)
        assert_allclose(1.0 + imout[mx, my], 1.0 + model[0, gx, gy])
```

- [ ] **Step 3: Run the test to verify it fails for the right reason**

Run: `uv run --extra full pytest tests/test_modelspec.py -v`
Expected: FAIL during collection with `ModuleNotFoundError: No module named 'pfb_model_spec.modelspec'`.

(Do not commit yet — the implementation does not exist.)

---

### Task 3: Vendor `modelspec.py` to make the test pass

**Files:**
- Create: `src/pfb_model_spec/modelspec.py`

- [ ] **Step 1: Copy the module verbatim from pfb-imaging**

Run:
```bash
cp /home/bester/software/pfb-imaging/src/pfb_imaging/utils/modelspec.py \
   /home/bester/software/pfb-model-spec/src/pfb_model_spec/modelspec.py
```

- [ ] **Step 2: Confirm the copy is byte-identical to the source**

Run:
```bash
diff /home/bester/software/pfb-imaging/src/pfb_imaging/utils/modelspec.py \
     src/pfb_model_spec/modelspec.py && echo "identical"
```
Expected: prints `identical` (no diff output). This module has zero `pfb_imaging` imports, so no edits are needed.

- [ ] **Step 3: Lint (ruff may apply cosmetic reformatting only)**

Run: `uv run ruff format . && uv run ruff check . --fix`
Expected: passes. Any change ruff makes to `modelspec.py` must be cosmetic (formatting / import order) — it must NOT alter function names, signatures, return tuples, or logic. If `ruff check` reports a non-auto-fixable error, fix it minimally without changing behavior.

- [ ] **Step 4: Run the synthetic test to verify it passes**

Run: `uv run --extra full pytest tests/test_modelspec.py -v`
Expected: PASS (`test_modelspec_roundtrip`).

- [ ] **Step 5: Commit**

```bash
git add src/pfb_model_spec/modelspec.py tests/_synth.py tests/test_modelspec.py
git commit -m "feat: vendor component-model spec library with synthetic tests"
```
Expected: pre-commit hooks pass; `cabs/onboard.yml` is NOT modified. If the hook reformats a new file, run `git add -u && git commit` to include it.

---

### Task 4: Guard the lightweight top-level import

**Files:**
- Create: `tests/test_lightweight_import.py`

- [ ] **Step 1: Write the guard test**

Create `tests/test_lightweight_import.py`:

```python
"""Guard: importing the top-level package must NOT pull in the heavy spec
library, so the lightweight install (hip-cargo + typer) and cab generation
stay fast and dependency-free."""

import sys


def test_top_level_import_does_not_load_modelspec():
    # Drop any prior import so the assertion reflects a fresh top-level import.
    for mod in list(sys.modules):
        if mod == "pfb_model_spec" or mod.startswith("pfb_model_spec."):
            del sys.modules[mod]

    import pfb_model_spec  # noqa: F401

    assert "pfb_model_spec.modelspec" not in sys.modules
```

- [ ] **Step 2: Run the guard test to verify it passes**

Run: `uv run --extra full pytest tests/test_lightweight_import.py -v`
Expected: PASS (the top-level `__init__.py` only defines `__version__`, so it does not import `modelspec`).

- [ ] **Step 3: Lint**

Run: `uv run ruff format . && uv run ruff check . --fix`
Expected: passes.

- [ ] **Step 4: Commit**

```bash
git add tests/test_lightweight_import.py
git commit -m "test: guard lightweight top-level import"
```
Expected: pre-commit hooks pass; `cabs/onboard.yml` is NOT modified.

---

### Task 5: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Lint the whole tree**

Run: `uv run ruff format --check . && uv run ruff check .`
Expected: passes with no changes needed.

- [ ] **Step 2: Run the full test suite**

Run: `uv run --extra full pytest -v`
Expected: PASS — `test_modelspec_roundtrip`, `test_top_level_import_does_not_load_modelspec`, the existing `test_roundtrip_onboard`, and `tests/test_install.py`.

- [ ] **Step 3: Confirm a clean tree and consistent cab**

Run: `git status --short && git diff --stat`
Expected: empty (everything committed; `cabs/onboard.yml` unchanged).

---

## Self-Review

**Spec coverage:**
- Spec goal 1 (move spec library, identical signatures + `.mds` schema) → Task 3 (verbatim copy + byte-identical diff check).
- Spec goal 2 (self-contained lightweight test, no MS/daskms/africanus) → Tasks 2 (synthetic test + vendored helpers).
- Spec goal 3 (lightweight install intact, hip-cargo principles) → Task 1 (`full` extra only), Task 4 (lightweight-import guard), pre-flight notes (cab no-op, `_container_image.py` untouched).
- Spec "Dependencies" (numpy/scipy/sympy/xarray; nothing heavier) → Task 1.
- Spec "Testing" (port numerics, synthetic fixtures, vendor `gaussian2d`/`give_edges` into `tests/_synth.py`) → Task 2.
- Spec non-goals (no converter/CLI/.dds/FITS/logging, no pfb-imaging edits) → pre-flight notes + file structure (none of those files are touched).
- Spec "Future additions" (schema-guard test, CLI command) → intentionally excluded from this plan.

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to". All code steps contain full code or exact commands.

**Type/name consistency:** `fit_image_cube` returns `(coeffs, x_index, y_index, expr, params, texpr, fexpr)` in `modelspec.py`; the test binds them as `coeffs, x_idx, y_idx, expr, params, tfunc, ffunc` (positional, names are local and consistent within the test) and passes them positionally to `eval_coeffs_to_cube` / `eval_coeffs_to_slice` in the exact argument order those functions declare. Helper names (`gaussian2d`, `give_edges`) match between `tests/_synth.py` and the import in `tests/test_modelspec.py`.
