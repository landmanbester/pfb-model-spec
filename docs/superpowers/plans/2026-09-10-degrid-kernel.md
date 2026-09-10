# Degrid Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure-numpy library API to `pfb-model-spec` that renders a component model onto a grid, optionally attenuates it with a Mueller beam, degrids it to visibilities and converts Stokes visibilities to correlations — so that `pfb-imaging` (and later QuartiCal) stop re-implementing model evaluation and degridding inline.

**Architecture:** Four small pure functions in a new `utils/degrid.py`, plus one fused convenience wrapper. Every argument is a plain numpy array, a scalar, or an already-opened `xarray.Dataset`. No dask, no Ray, no MS handles, no file paths, no `pfb_imaging` import. Distribution and I/O stay in the calling applications.

**Tech Stack:** numpy, ducc0 (new dependency), xarray, sympy/scipy (already present via `utils/modelspec.py`).

**Spec:** `../../../pfb-imaging/docs/superpowers/specs/2026-09-10-degrid-msv4-design.md` §9 (the consuming design). Issue: landmanbester/pfb-model-spec#22. Consumer issues: ratt-ru/pfb-imaging#278 (degrid-msv4), ratt-ru/pfb-imaging#309 (`--transfer-model-from`).

## Global Constraints

- **Branch:** `issue22`. Already checked out.
- **x-major throughout.** The `genesis` spec is `(nx, ny)`-ordered and `.claude/rules/component-model.md` says: *"do not add a transpose inside this library to paper over that mismatch"*. Obey it. ducc0's `dirty2vis` also wants `(npix_x, npix_y)`, so an x-major library needs **no transpose anywhere**. Callers wanting `(Y, X)` transpose at their own call site.
- **No `africanus`, no `daskms`, no MS in tests.** `.claude/rules/testing-and-ci.md` §1: tests are synthetic and self-contained.
- **Lint after every change:** `uv run ruff format . && uv run ruff check . --fix`
- **Test command:** `uv run --extra full pytest -v` (the `full` extra is REQUIRED).
- **Commit with the venv active, inline, in the same shell** (`.claude/rules/testing-and-ci.md` §2):
  `source /home/bester/software/pfb-model-spec/.venv/bin/activate && git add <files> && git commit -m "..."`
- **Conventional Commits** enforced by a `commit-msg` hook. Types: `feat fix refactor perf docs deps chore ci style test build`.
- **Type hints on every signature; Google-style docstrings** (`.claude/rules/python-standards.md`).
- **`src/pfb_model_spec/__init__.py` must not import the new module** (lightweight-import guard test).
- **No CLI command, no cab.** This is library API only, so `tests/test_roundtrip.py` needs no new case.
- **When a task says "append to `tests/test_degrid.py`", merge its `from pfb_model_spec…`
  import into the existing import block at the top of the file** rather than leaving it
  mid-file — ruff's E402 rejects module-level imports below other code.
- **Validation:** this plan's code was assembled and run end to end before it was written
  down — 24 new tests pass, and the full suite stays green at 33 passed. The code blocks
  below are transcriptions of what ran, not sketches.
- **Do not restyle `utils/modelspec.py` or `utils/io.py`** — their signatures are a cross-repo contract.

---

## File Structure

| File | Responsibility |
|---|---|
| Create: `src/pfb_model_spec/utils/degrid.py` | The whole kernel: `render_model_region`, `apply_mueller`, `degrid_stokes`, `stokes_vis_to_corr`, `model_to_apparent_vis_for_region` |
| Create: `tests/test_degrid.py` | Synthetic tests for all five |
| Modify: `pyproject.toml` | add `ducc0` to the `full` extra |
| Modify: `.claude/rules/component-model.md` | document the new API + the resampling caveat |
| Modify: `CLAUDE.md` | "Current status" — the deferred shared-reader item is now built |

One module, because the five functions are one cohesive unit that changes together, and the repo already keeps `modelspec.py` and `io.py` as single-responsibility library files.

---

## Task 1: `stokes_vis_to_corr`

Pure numpy, no new dependency. Done first so the module exists and the cheapest function pins the polarisation convention before anything depends on it.

**Files:**
- Create: `src/pfb_model_spec/utils/degrid.py`
- Test: `tests/test_degrid.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `stokes_vis_to_corr(stokes_vis: np.ndarray, stokes_in: str | Sequence[str], corr_types: Sequence[str]) -> np.ndarray`, mapping `(nstokes, nrow, nchan)` complex → `(nrow, nchan, ncorr)` complex.

The correlation expressions below were derived empirically from `africanus.model.coherency.convert` and verified bit-identical (max error 0.0). They are hard-coded rather than imported because tests here must not depend on africanus.

- [ ] **Step 1: Write the failing test**

```python
"""Synthetic, self-contained tests for the degrid kernel."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pfb_model_spec.utils.degrid import stokes_vis_to_corr


def test_stokes_vis_to_corr_linear():
    rng = np.random.default_rng(1)
    sv = rng.normal(size=(4, 5, 3)) + 1j * rng.normal(size=(4, 5, 3))
    i, q, u, v = sv
    out = stokes_vis_to_corr(sv, "IQUV", ("XX", "XY", "YX", "YY"))
    assert out.shape == (5, 3, 4)
    assert_allclose(out[..., 0], i + q)
    assert_allclose(out[..., 1], u + 1j * v)
    assert_allclose(out[..., 2], u - 1j * v)
    assert_allclose(out[..., 3], i - q)


def test_stokes_vis_to_corr_circular():
    rng = np.random.default_rng(2)
    sv = rng.normal(size=(4, 5, 3)) + 1j * rng.normal(size=(4, 5, 3))
    i, q, u, v = sv
    out = stokes_vis_to_corr(sv, "IQUV", ("RR", "RL", "LR", "LL"))
    assert_allclose(out[..., 0], i + v)
    assert_allclose(out[..., 1], q + 1j * u)
    assert_allclose(out[..., 2], q - 1j * u)
    assert_allclose(out[..., 3], i - v)


def test_stokes_vis_to_corr_stokes_i_only():
    """The v1 degrid path: an I-only model must give XX == YY == I, XY == YX == 0."""
    rng = np.random.default_rng(3)
    sv = (rng.normal(size=(1, 5, 3)) + 0j)
    out = stokes_vis_to_corr(sv, "I", ("XX", "XY", "YX", "YY"))
    assert_allclose(out[..., 0], sv[0])
    assert_allclose(out[..., 3], sv[0])
    assert np.abs(out[..., 1]).max() == 0.0
    assert np.abs(out[..., 2]).max() == 0.0


def test_stokes_vis_to_corr_two_correlations():
    rng = np.random.default_rng(4)
    sv = rng.normal(size=(2, 5, 3)) + 0j
    out = stokes_vis_to_corr(sv, "IQ", ("XX", "YY"))
    assert out.shape == (5, 3, 2)
    assert_allclose(out[..., 0], sv[0] + sv[1])
    assert_allclose(out[..., 1], sv[0] - sv[1])


def test_stokes_vis_to_corr_rejects_unknown_correlation():
    sv = np.zeros((1, 2, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="ZZ"):
        stokes_vis_to_corr(sv, "I", ("ZZ",))


def test_stokes_vis_to_corr_rejects_length_mismatch():
    sv = np.zeros((2, 2, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="stokes_in"):
        stokes_vis_to_corr(sv, "IQU", ("XX",))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra full pytest tests/test_degrid.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pfb_model_spec.utils.degrid'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/pfb_model_spec/utils/degrid.py`:

```python
"""Render a component model to visibilities for a chunk of data.

Pure numpy: no dask, no Ray, no measurement-set handles. Data selection,
chunking, distribution and I/O belong to the calling application
(pfb-imaging's `degrid-msv4`, QuartiCal's predict); this module only turns a
model plus a chunk's `(uvw, freq)` into visibilities.

Axis convention: **x-major**, `(nx, ny)`, matching the `genesis` `.mds` spec and
ducc0's `dirty2vis`. See `.claude/rules/component-model.md` -> "Axis
convention": this library deliberately does not transpose to pfb-imaging's
`(Y, X)` raster convention, so no transpose appears anywhere below.
"""

from typing import Sequence

import numpy as np

# Stokes -> correlation expressions, matching africanus.model.coherency.convert
# (verified elementwise; africanus is deliberately not a dependency here because
# tests must stay self-contained -- see .claude/rules/testing-and-ci.md).
_STOKES_TO_CORR: dict[str, dict[str, complex]] = {
    "XX": {"I": 1.0 + 0j, "Q": 1.0 + 0j},
    "XY": {"U": 1.0 + 0j, "V": 1.0j},
    "YX": {"U": 1.0 + 0j, "V": -1.0j},
    "YY": {"I": 1.0 + 0j, "Q": -1.0 + 0j},
    "RR": {"I": 1.0 + 0j, "V": 1.0 + 0j},
    "RL": {"Q": 1.0 + 0j, "U": 1.0j},
    "LR": {"Q": 1.0 + 0j, "U": -1.0j},
    "LL": {"I": 1.0 + 0j, "V": -1.0 + 0j},
}


def stokes_vis_to_corr(
    stokes_vis: np.ndarray,
    stokes_in: str | Sequence[str],
    corr_types: Sequence[str],
) -> np.ndarray:
    """Convert Stokes visibilities to correlations.

    An exact linear map. Stokes products absent from ``stokes_in`` are treated
    as zero, so an I-only model yields ``XX == YY == I`` and ``XY == YX == 0``.

    Args:
        stokes_vis: Complex visibilities, shape ``(nstokes, nrow, nchan)``.
        stokes_in: Stokes products present, in the order of ``stokes_vis``'s
            first axis, e.g. ``"I"`` or ``"IQUV"``.
        corr_types: Output correlations, e.g. ``("XX", "XY", "YX", "YY")``.

    Returns:
        Complex visibilities, shape ``(nrow, nchan, ncorr)``.

    Raises:
        ValueError: If ``stokes_in`` does not match ``stokes_vis``'s first axis,
            or a correlation is not one of XX/XY/YX/YY/RR/RL/LR/LL.
    """
    stokes_in = tuple(stokes_in)
    if len(stokes_in) != stokes_vis.shape[0]:
        raise ValueError(
            f"stokes_in has {len(stokes_in)} entries but stokes_vis has "
            f"{stokes_vis.shape[0]} planes"
        )
    nrow, nchan = stokes_vis.shape[1:]
    out = np.zeros((nrow, nchan, len(corr_types)), dtype=stokes_vis.dtype)
    index = {s: i for i, s in enumerate(stokes_in)}
    for c, corr in enumerate(corr_types):
        try:
            terms = _STOKES_TO_CORR[corr]
        except KeyError:
            raise ValueError(f"Unsupported correlation {corr!r}") from None
        for stokes, coeff in terms.items():
            if stokes in index:
                out[:, :, c] += coeff * stokes_vis[index[stokes]]
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra full pytest tests/test_degrid.py -v`
Expected: 6 passed

- [ ] **Step 5: Lint**

Run: `uv run ruff format . && uv run ruff check . --fix`
Expected: no remaining errors

- [ ] **Step 6: Commit**

```bash
source /home/bester/software/pfb-model-spec/.venv/bin/activate && \
git add src/pfb_model_spec/utils/degrid.py tests/test_degrid.py && \
git commit -m "feat(degrid): add Stokes to correlation conversion"
```

---

## Task 2: `degrid_stokes` (and the `ducc0` dependency)

**Files:**
- Modify: `pyproject.toml` (the `full` extra)
- Modify: `src/pfb_model_spec/utils/degrid.py`
- Test: `tests/test_degrid.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `degrid_stokes(uvw, freq, stokes_image, *, cell_rad, x0, y0, flip_u, flip_v, flip_w, epsilon=1e-7, do_wgridding=True, divide_by_n=False, nthreads=1, mask=None) -> np.ndarray` mapping `(nstokes, nx, ny)` real → `(nstokes, nrow, nchan)` complex.

Three properties are worth testing and each catches a different class of bug: a source at the phase centre pins **flux scaling** (convention-free — the visibility is the flux on every baseline); linearity pins **superposition**; and a degrid → `vis2dirty` round trip pins **geometry**, because a sign error in `flip_u`/`flip_v` mirrors the image and moves the recovered peak to the mirrored pixel.

`mask` exists because `xarray-ms` lays data on a regular `(time, baseline)` grid and fills absent cells with NaN UVW. pfb-imaging documents an unmasked `dirty2vis` deriving a NaN w-extent from those rows and failing with "too many w planes" (`operators/gridder.residual_from_partitions`). On ducc0 0.41.0 that hard failure did not reproduce — NaN rows were tolerated and good rows stayed correct — but masking is correct by intent (absent cells must not contribute), costs nothing, and yields exact zeros for those rows.

- [ ] **Step 1: Add the dependency**

Edit `pyproject.toml`, adding `ducc0` to the `full` extra (keep the list alphabetical):

```toml
full = [
    "astropy",
    "ducc0",
    "numpy",
    "scipy",
    "sympy",
    "xarray",
    "zarr",
]
```

Then: `uv sync --extra full`

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_degrid.py`:

```python
from pfb_model_spec.utils.degrid import degrid_stokes

# ducc0 wgridder conventions used throughout these tests; see
# https://github.com/mreineck/ducc/issues/34
_CONV = dict(flip_u=False, flip_v=True, flip_w=False, x0=0.0, y0=0.0)


def _uvw_freq(rng, nrow=2000, nchan=4):
    uvw = rng.normal(0.0, 3000.0, (nrow, 3))
    uvw[:, 2] *= 0.05
    return uvw, np.linspace(1.0e9, 1.1e9, nchan)


def test_degrid_stokes_centre_source_is_flat():
    """A point source at the phase centre gives V == flux on every baseline."""
    rng = np.random.default_rng(10)
    uvw, freq = _uvw_freq(rng)
    npix, cell = 128, np.deg2rad(2.0 / 3600.0)
    img = np.zeros((1, npix, npix))
    img[0, npix // 2, npix // 2] = 3.0
    vis = degrid_stokes(uvw, freq, img, cell_rad=cell, **_CONV)
    assert vis.shape == (1, uvw.shape[0], freq.size)
    assert_allclose(vis, 3.0, atol=1e-6)


def test_degrid_stokes_is_linear():
    rng = np.random.default_rng(11)
    uvw, freq = _uvw_freq(rng)
    npix, cell = 128, np.deg2rad(2.0 / 3600.0)
    a = np.zeros((1, npix, npix))
    a[0, 40, 70] = 1.5
    b = np.zeros((1, npix, npix))
    b[0, 90, 33] = -0.7
    kw = dict(cell_rad=cell, **_CONV)
    va = degrid_stokes(uvw, freq, a, **kw)
    vb = degrid_stokes(uvw, freq, b, **kw)
    vab = degrid_stokes(uvw, freq, a + b, **kw)
    assert_allclose(vab, va + vb, atol=1e-10)


def test_degrid_stokes_geometry_round_trip():
    """Degrid then grid back: the peak must land on the source pixel.

    A sign error in flip_u/flip_v mirrors the image, moving the peak to the
    mirrored pixel, so this pins the geometric conventions end to end.
    """
    from ducc0.wgridder.experimental import vis2dirty

    rng = np.random.default_rng(12)
    uvw, freq = _uvw_freq(rng)
    npix, cell = 128, np.deg2rad(2.0 / 3600.0)
    src = (37, 91)
    img = np.zeros((1, npix, npix))
    img[0, src[0], src[1]] = 1.0
    vis = degrid_stokes(uvw, freq, img, cell_rad=cell, **_CONV)
    back = vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=np.ascontiguousarray(vis[0]),
        wgt=np.ones((uvw.shape[0], freq.size)),
        npix_x=npix,
        npix_y=npix,
        pixsize_x=cell,
        pixsize_y=cell,
        center_x=_CONV["x0"],
        center_y=_CONV["y0"],
        flip_u=_CONV["flip_u"],
        flip_v=_CONV["flip_v"],
        flip_w=_CONV["flip_w"],
        epsilon=1e-7,
        do_wgridding=True,
        divide_by_n=False,
        nthreads=1,
    )
    assert np.unravel_index(np.argmax(back), back.shape) == src


def test_degrid_stokes_masks_absent_rows():
    """xarray-ms pads absent (time, baseline) cells with NaN UVW."""
    rng = np.random.default_rng(13)
    uvw, freq = _uvw_freq(rng, nrow=500)
    uvw[10:20, :] = np.nan
    npix, cell = 64, np.deg2rad(2.0 / 3600.0)
    img = np.zeros((1, npix, npix))
    img[0, npix // 2, npix // 2] = 1.0
    mask = np.isfinite(uvw).all(axis=1)[:, None] & np.ones(freq.size, bool)[None, :]
    vis = degrid_stokes(
        uvw, freq, img, cell_rad=cell, mask=mask.astype(np.uint8), **_CONV
    )
    assert np.abs(vis[0][10:20]).max() == 0.0
    assert_allclose(vis[0][mask[:, 0]], 1.0, atol=1e-6)


def test_degrid_stokes_skips_empty_planes():
    """An all-zero Stokes plane costs no gridder call and returns zeros."""
    rng = np.random.default_rng(14)
    uvw, freq = _uvw_freq(rng, nrow=200)
    npix, cell = 64, np.deg2rad(2.0 / 3600.0)
    img = np.zeros((2, npix, npix))
    img[0, npix // 2, npix // 2] = 1.0
    vis = degrid_stokes(uvw, freq, img, cell_rad=cell, **_CONV)
    assert_allclose(vis[0], 1.0, atol=1e-6)
    assert np.abs(vis[1]).max() == 0.0
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run --extra full pytest tests/test_degrid.py -k degrid_stokes -v`
Expected: FAIL — `ImportError: cannot import name 'degrid_stokes'`

- [ ] **Step 4: Write the implementation**

Add to the imports at the top of `src/pfb_model_spec/utils/degrid.py`:

```python
from ducc0.wgridder.experimental import dirty2vis
```

Append the function:

```python
def degrid_stokes(
    uvw: np.ndarray,
    freq: np.ndarray,
    stokes_image: np.ndarray,
    *,
    cell_rad: float,
    x0: float,
    y0: float,
    flip_u: bool,
    flip_v: bool,
    flip_w: bool,
    epsilon: float = 1e-7,
    do_wgridding: bool = True,
    divide_by_n: bool = False,
    nthreads: int = 1,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Degrid one Stokes image per plane to visibilities.

    One ``dirty2vis`` call per Stokes plane. Degridding in Stokes and
    converting to correlations afterwards (:func:`stokes_vis_to_corr`) is
    cheaper than degridding the real and imaginary parts of the brightness
    matrix, which is what the wgridder would otherwise require.

    The geometry arguments (``x0``/``y0``/``flip_*``) must come from the
    ``.mds`` attrs, never be recomputed, so that a model degrids under the
    same conventions it was gridded with.

    Args:
        uvw: Baseline coordinates in metres, shape ``(nrow, 3)``. Rows may be
            NaN where the caller's data layout pads absent samples; pass
            ``mask`` to exclude them.
        freq: Channel frequencies in Hz, shape ``(nchan,)``.
        stokes_image: Model image per Stokes plane, shape ``(nstokes, nx, ny)``,
            x-major (see the module docstring).
        cell_rad: Pixel size in radians (square pixels).
        x0: Phase-centre x offset from ``wgridder_conventions``.
        y0: Phase-centre y offset from ``wgridder_conventions``.
        flip_u: U-axis flip convention.
        flip_v: V-axis flip convention.
        flip_w: W-axis flip convention.
        epsilon: Gridder accuracy.
        do_wgridding: Perform w-correction via improved w-stacking.
        divide_by_n: Divide by the geometric n-term. Left to the caller: a
            caller that folds ``1/n`` into its beam must keep this ``False``.
        nthreads: Threads for the gridder.
        mask: Optional ``(nrow, nchan)`` mask; zero entries are not degridded
            and come back as exactly zero.

    Returns:
        Complex visibilities, shape ``(nstokes, nrow, nchan)``.
    """
    nstokes = stokes_image.shape[0]
    dtype = np.result_type(stokes_image.dtype, np.complex64)
    out = np.zeros((nstokes, uvw.shape[0], freq.size), dtype=dtype)
    for s in range(nstokes):
        # an empty plane contributes nothing; skip the gridder call entirely
        if not stokes_image[s].any():
            continue
        out[s] = dirty2vis(
            uvw=uvw,
            freq=freq,
            dirty=stokes_image[s],
            mask=mask,
            pixsize_x=cell_rad,
            pixsize_y=cell_rad,
            center_x=x0,
            center_y=y0,
            flip_u=flip_u,
            flip_v=flip_v,
            flip_w=flip_w,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            divide_by_n=divide_by_n,
            nthreads=nthreads,
        )
    return out
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra full pytest tests/test_degrid.py -v`
Expected: 11 passed

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
source /home/bester/software/pfb-model-spec/.venv/bin/activate && \
git add pyproject.toml uv.lock src/pfb_model_spec/utils/degrid.py tests/test_degrid.py && \
git commit -m "feat(degrid): degrid Stokes images to visibilities with ducc0"
```

---

## Task 3: `render_model_region`

**Files:**
- Modify: `src/pfb_model_spec/utils/degrid.py`
- Test: `tests/test_degrid.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-2.
- Produces: `render_model_region(model_ds, *, time, freq, nx=None, ny=None, cell_rad=None, x0=None, y0=None) -> np.ndarray` returning `(nstokes, nx, ny)`, and `model_geometry(model_ds) -> dict` returning the `.mds` geometry attrs (`cell_rad`, `x0`, `y0`, `flip_u`, `flip_v`, `flip_w`, `nx`, `ny`, `stokes`) that Task 5 and pfb-imaging pass to `degrid_stokes`.

This wraps `eval_coeffs_to_slice`, replacing the inline
`image[x_index, y_index] = modelf(tout, fout, *comps)` that pfb-imaging's
`operators/gridder._comps2vis_impl` does today. The wrapper is more capable than the
inline version, which assumes the `.mds` grid *is* the output grid.

`model_geometry` exists so callers stop reading `.mds` attrs by hand — that hand-reading is
where pfb-imaging picked up `celly = mds.cell_rad_x` (a typo for `cell_rad_y`, masked only
because `build_mds_dataset` always writes square cells).

**Known limitation to document, not fix here.** `eval_coeffs_to_slice` scales by
`area_ratio = pix_area_out / pix_area_in`, which is correct for a surface-brightness field
but not for the point components a `.mds` stores. Measured with a single unit component:
same grid → sum 1.0 (exact); refine ×2 → sum 1.0 (conserved); **coarsen ×2 → sum 4.0,
coarsen ×4 → sum 16.0** (inflated by exactly `area_ratio`). Degrid always renders on the
model's own grid, so v1 is unaffected, but `--transfer-model-from` (ratt-ru/pfb-imaging#309)
will hit it. Do not write a flux-conservation test — it would assert something false.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_degrid.py`:

```python
from pfb_model_spec.utils.degrid import model_geometry, render_model_region
from pfb_model_spec.utils.io import build_mds_dataset
from pfb_model_spec.utils.modelspec import fit_image_cube


def _synthetic_mds(nx=64, ny=48, nband=4):
    """A two-component power-law model on a deliberately NON-square grid."""
    freq = np.linspace(1.0e9, 1.2e9, nband)
    time = np.array([0.0])
    cell = np.deg2rad(2.0 / 3600.0)
    cube = np.zeros((nband, nx, ny))
    cube[:, 20, 31] = (freq / freq[0]) ** -0.7
    cube[:, 44, 12] = 2.0 * (freq / freq[0]) ** -1.1
    coeffs, xi, yi, expr, params, texpr, fexpr = fit_image_cube(
        time, freq, cube[None], nbasisf=nband, method="Legendre", sigmasq=0
    )
    ds = build_mds_dataset(
        coeffs, xi, yi, expr, params, texpr, fexpr, time, freq, cell,
        nx, ny, 0.0, 0.0, False, True, False, (0.1, -0.5), "I", "test",
    )
    return ds, cube, time, freq, cell


def test_render_model_region_matches_the_fitted_cube():
    ds, cube, time, freq, _ = _synthetic_mds()
    for band in (0, freq.size - 1):
        got = render_model_region(ds, time=time[0], freq=freq[band])
        assert got.shape == (1, ds.attrs["npix_x"], ds.attrs["npix_y"])
        assert_allclose(got[0], cube[band], atol=1e-10)


def test_render_model_region_interpolates_between_fitted_bands():
    """The model is continuous in frequency -- this is what upsampling relies on."""
    ds, cube, time, freq, _ = _synthetic_mds()
    mid = 0.5 * (freq[0] + freq[1])
    got = render_model_region(ds, time=time[0], freq=mid)[0]
    lo, hi = cube[0, 20, 31], cube[1, 20, 31]
    assert min(lo, hi) < got[20, 31] < max(lo, hi)


def test_model_geometry_reads_the_mds_attrs():
    ds, _, _, _, cell = _synthetic_mds()
    geom = model_geometry(ds)
    assert geom["nx"] == 64 and geom["ny"] == 48
    assert geom["cell_rad"] == pytest.approx(cell)
    assert geom["x0"] == 0.0 and geom["y0"] == 0.0
    assert geom["flip_u"] is False and geom["flip_v"] is True
    assert geom["stokes"] == "I"


def test_model_geometry_rejects_non_square_pixels():
    """cell_rad_x != cell_rad_y has no single cell_rad to hand the gridder."""
    ds, _, _, _, cell = _synthetic_mds()
    ds.attrs["cell_rad_y"] = 2.0 * cell
    with pytest.raises(ValueError, match="square"):
        model_geometry(ds)


def test_render_model_region_rejects_unknown_spec():
    ds, _, time, freq, _ = _synthetic_mds()
    ds.attrs["spec"] = "some-future-spec"
    with pytest.raises(ValueError, match="some-future-spec"):
        render_model_region(ds, time=time[0], freq=freq[0])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra full pytest tests/test_degrid.py -k "render_model or model_geometry" -v`
Expected: FAIL — `ImportError: cannot import name 'model_geometry'`

- [ ] **Step 3: Write the implementation**

Add to the imports at the top of `src/pfb_model_spec/utils/degrid.py`:

```python
import xarray as xr

from pfb_model_spec.utils.modelspec import eval_coeffs_to_slice
```

Add the module constant below `_STOKES_TO_CORR`:

```python
# .mds spec versions this module understands. A new spec (e.g. the (Y, X) +
# stokes-axis revision, issues #19/#20) must be added here deliberately, not
# silently accepted -- the axis order and the stokes axis both change meaning.
_SUPPORTED_SPECS = frozenset({"genesis"})
```

Append the functions:

```python
def model_geometry(model_ds: xr.Dataset) -> dict:
    """Extract the gridding geometry a `.mds` records.

    Callers pass these straight to :func:`degrid_stokes`. Reading them here
    rather than at each call site keeps the convention in one place.

    Args:
        model_ds: An opened `.mds` dataset.

    Returns:
        ``{"nx", "ny", "cell_rad", "x0", "y0", "flip_u", "flip_v", "flip_w",
        "stokes"}``.

    Raises:
        ValueError: If the pixels are not square, since ``degrid_stokes`` takes
            a single ``cell_rad``.
    """
    a = model_ds.attrs
    cell_x = float(a["cell_rad_x"])
    cell_y = float(a["cell_rad_y"])
    if cell_x != cell_y:
        raise ValueError(
            f"Non-square pixels (cell_rad_x={cell_x}, cell_rad_y={cell_y}) are "
            "not supported; degridding takes a single cell size"
        )
    return {
        "nx": int(a["npix_x"]),
        "ny": int(a["npix_y"]),
        "cell_rad": cell_x,
        "x0": float(a["center_x"]),
        "y0": float(a["center_y"]),
        "flip_u": bool(a["flip_u"]),
        "flip_v": bool(a["flip_v"]),
        "flip_w": bool(a["flip_w"]),
        "stokes": str(a["stokes"]),
    }


def render_model_region(
    model_ds: xr.Dataset,
    *,
    time: float,
    freq: float,
    nx: int | None = None,
    ny: int | None = None,
    cell_rad: float | None = None,
    x0: float | None = None,
    y0: float | None = None,
) -> np.ndarray:
    """Render a component model to an image at one time and frequency.

    The model is a continuous function of time and frequency, so ``freq`` may
    lie between the bands the model was fitted at -- that is what lets a
    consumer predict at finer spectral resolution than the imaging run used.

    The output grid defaults to the model's own. Supplying a different one
    resamples via ``eval_coeffs_to_slice``.

    Note:
        Resampling to a **coarser** grid is not flux-conserving for the point
        components a `.mds` stores: ``eval_coeffs_to_slice`` applies an
        ``area_ratio`` factor appropriate to a surface-brightness field, which
        inflates a point component's flux by exactly that factor (measured:
        coarsen x2 -> x4, coarsen x4 -> x16; refining and same-grid are exact).
        Degridding always uses the model's own grid and so is unaffected.

    Args:
        model_ds: An opened `.mds` dataset.
        time: Time to evaluate at, in the model's own time units.
        freq: Frequency to evaluate at, in Hz.
        nx: Output pixels along x. Defaults to the model's ``npix_x``.
        ny: Output pixels along y. Defaults to the model's ``npix_y``.
        cell_rad: Output pixel size in radians. Defaults to the model's.
        x0: Output phase-centre x offset. Defaults to the model's ``center_x``.
        y0: Output phase-centre y offset. Defaults to the model's ``center_y``.

    Returns:
        The rendered image, shape ``(nstokes, nx, ny)``, x-major. ``nstokes``
        is 1 for the ``genesis`` spec, which carries a single Stokes product;
        the axis is present so that adding a Stokes axis (#19) changes values
        rather than shapes.

    Raises:
        ValueError: If the `.mds` declares a spec this module does not know.
    """
    a = model_ds.attrs
    spec = str(a.get("spec", "genesis"))
    if spec not in _SUPPORTED_SPECS:
        raise ValueError(
            f"Unsupported .mds spec {spec!r}; this module understands "
            f"{sorted(_SUPPORTED_SPECS)}"
        )

    nxi, nyi = int(a["npix_x"]), int(a["npix_y"])
    cellxi, cellyi = float(a["cell_rad_x"]), float(a["cell_rad_y"])
    x0i, y0i = float(a["center_x"]), float(a["center_y"])

    nxo = nxi if nx is None else int(nx)
    nyo = nyi if ny is None else int(ny)
    cellxo = cellxi if cell_rad is None else float(cell_rad)
    cellyo = cellyi if cell_rad is None else float(cell_rad)
    x0o = x0i if x0 is None else float(x0)
    y0o = y0i if y0 is None else float(y0)

    image = eval_coeffs_to_slice(
        time,
        freq,
        model_ds.coefficients.values,
        model_ds.location_x.values,
        model_ds.location_y.values,
        a["parametrisation"],
        model_ds.params.values,
        a["texpr"],
        a["fexpr"],
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
    )
    # leading stokes axis: 1 for genesis, which stores a single product
    return image[None]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra full pytest tests/test_degrid.py -v`
Expected: 16 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
source /home/bester/software/pfb-model-spec/.venv/bin/activate && \
git add src/pfb_model_spec/utils/degrid.py tests/test_degrid.py && \
git commit -m "feat(degrid): render a component model onto an arbitrary grid"
```

---

## Task 4: `apply_mueller`

**Files:**
- Modify: `src/pfb_model_spec/utils/degrid.py`
- Test: `tests/test_degrid.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-3.
- Produces: `apply_mueller(stokes_image: np.ndarray, mueller: np.ndarray) -> np.ndarray`, mapping `(nstokes_in, nx, ny)` and `(nstokes_out, nstokes_in, nx, ny)` → `(nstokes_out, nx, ny)`.

Unused by the first `degrid-msv4` release, which ships without beams
(ratt-ru/pfb-imaging#278). It is defined now so the API does not change shape when beams
arrive, and because it is the one place the attenuation convention should be documented.

The function never folds the geometric `1/n` term and never constructs a beam: what it
receives is whatever the caller decided a beam is. pfb-imaging folds `1/n` into its stored
beam and keeps `divide_by_n=False` throughout; that is a pfb-imaging gridding convention and
must not leak in here.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_degrid.py`:

```python
from pfb_model_spec.utils.degrid import apply_mueller


def _mueller(nso, nsi, nx, ny):
    m = np.zeros((nso, nsi, nx, ny))
    for i in range(min(nso, nsi)):
        m[i, i] = 1.0
    return m


def test_apply_mueller_identity_is_a_no_op():
    rng = np.random.default_rng(20)
    img = rng.normal(size=(4, 8, 6))
    out = apply_mueller(img, _mueller(4, 4, 8, 6))
    assert_allclose(out, img)


def test_apply_mueller_diagonal_scales_each_plane():
    rng = np.random.default_rng(21)
    img = rng.normal(size=(2, 8, 6))
    m = _mueller(2, 2, 8, 6)
    m[0, 0] *= 0.5
    m[1, 1] *= 3.0
    out = apply_mueller(img, m)
    assert_allclose(out[0], 0.5 * img[0])
    assert_allclose(out[1], 3.0 * img[1])


def test_apply_mueller_predicts_leakage_from_stokes_i():
    """The first Mueller column turns an I-only sky into apparent I, Q, U, V."""
    rng = np.random.default_rng(22)
    img = rng.normal(size=(1, 8, 6))
    m = rng.normal(size=(4, 1, 8, 6))
    out = apply_mueller(img, m)
    assert out.shape == (4, 8, 6)
    for i in range(4):
        assert_allclose(out[i], m[i, 0] * img[0])


def test_apply_mueller_rejects_shape_mismatch():
    img = np.zeros((2, 8, 6))
    with pytest.raises(ValueError, match="does not match"):
        apply_mueller(img, _mueller(4, 3, 8, 6))


def test_apply_mueller_rejects_grid_mismatch():
    img = np.zeros((1, 8, 6))
    with pytest.raises(ValueError, match="grid"):
        apply_mueller(img, _mueller(1, 1, 8, 7))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra full pytest tests/test_degrid.py -k mueller -v`
Expected: FAIL — `ImportError: cannot import name 'apply_mueller'`

- [ ] **Step 3: Write the implementation**

Append to `src/pfb_model_spec/utils/degrid.py`:

```python
def apply_mueller(stokes_image: np.ndarray, mueller: np.ndarray) -> np.ndarray:
    """Attenuate a Stokes image with a Stokes-basis Mueller beam.

    ``apparent[i] = sum_j mueller[i, j] * intrinsic[j]``, pixel by pixel.

    The beam is whatever the caller supplies; this function neither builds one
    nor folds the wgridder's geometric ``1/n`` term into it. A caller that
    folds ``1/n`` into its beam (as pfb-imaging does) must correspondingly
    leave ``divide_by_n=False`` in :func:`degrid_stokes`.

    Args:
        stokes_image: Intrinsic model, shape ``(nstokes_in, nx, ny)``.
        mueller: Real Mueller block, shape ``(nstokes_out, nstokes_in, nx, ny)``
            on the same grid as ``stokes_image``.

    Returns:
        The apparent model, shape ``(nstokes_out, nx, ny)``.

    Raises:
        ValueError: If the Stokes axes or the image grids disagree.
    """
    if mueller.ndim != 4:
        raise ValueError(f"mueller must be 4-D (nso, nsi, nx, ny), got {mueller.shape}")
    if mueller.shape[1] != stokes_image.shape[0]:
        raise ValueError(
            f"mueller input axis {mueller.shape[1]} does not match the model's "
            f"{stokes_image.shape[0]} Stokes planes"
        )
    if mueller.shape[2:] != stokes_image.shape[1:]:
        raise ValueError(
            f"mueller grid {mueller.shape[2:]} does not match the model grid "
            f"{stokes_image.shape[1:]}"
        )
    return np.einsum("ijxy,jxy->ixy", mueller, stokes_image)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra full pytest tests/test_degrid.py -v`
Expected: 21 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
source /home/bester/software/pfb-model-spec/.venv/bin/activate && \
git add src/pfb_model_spec/utils/degrid.py tests/test_degrid.py && \
git commit -m "feat(degrid): apply a Stokes-basis Mueller beam to a model"
```

---

## Task 5: `model_to_apparent_vis_for_region`

**Files:**
- Modify: `src/pfb_model_spec/utils/degrid.py`
- Test: `tests/test_degrid.py`

**Interfaces:**
- Consumes: `render_model_region`, `model_geometry` (Task 3), `apply_mueller` (Task 4), `degrid_stokes` (Task 2), `stokes_vis_to_corr` (Task 1).
- Produces: `model_to_apparent_vis_for_region(model_ds, *, uvw, freq, corr_types, time, freq_out, mueller=None, mask=None, region_mask=None, epsilon=1e-7, do_wgridding=True, divide_by_n=False, nthreads=1) -> np.ndarray` returning `(nrow, nchan, ncorr)` complex.

The one call an application makes per chunk. It exists as a convenience over the four
primitives, which stay separately callable because three consumers need pieces rather than
the whole: `--transfer-model-from` (ratt-ru/pfb-imaging#309) needs only `render_model_region`;
region-file degridding (ratt-ru/pfb-imaging#115) renders once and degrids N+1 times behind
different masks; and chunks sharing a `(time, freq)` bin can reuse one rendering.

`time`/`freq_out` are the caller's choice of representative values for the chunk,
conventionally the **unweighted** means of its time and frequency axes — unweighted so that
two consumers with different flagging cannot disagree about where the model was evaluated.
The per-channel phase is exact regardless, since `dirty2vis` receives the full `freq` array;
the only approximation is the model's spectral variation across one chunk, which the caller
controls by choosing the chunk width.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_degrid.py`:

```python
from pfb_model_spec.utils.degrid import model_to_apparent_vis_for_region


def test_fused_wrapper_equals_the_composition():
    ds, _, time, freq, cell = _synthetic_mds()
    rng = np.random.default_rng(30)
    uvw, chan = _uvw_freq(rng, nrow=400, nchan=3)
    corr = ("XX", "XY", "YX", "YY")

    fused = model_to_apparent_vis_for_region(
        ds, uvw=uvw, freq=chan, corr_types=corr, time=time[0], freq_out=freq[0]
    )

    geom = model_geometry(ds)
    img = render_model_region(ds, time=time[0], freq=freq[0])
    sv = degrid_stokes(
        uvw, chan, img,
        cell_rad=geom["cell_rad"], x0=geom["x0"], y0=geom["y0"],
        flip_u=geom["flip_u"], flip_v=geom["flip_v"], flip_w=geom["flip_w"],
    )
    manual = stokes_vis_to_corr(sv, geom["stokes"], corr)

    assert fused.shape == (uvw.shape[0], chan.size, 4)
    assert_allclose(fused, manual)


def test_fused_wrapper_region_mask_selects_components():
    """Masking splits the model: the parts must sum back to the whole."""
    ds, _, time, freq, _ = _synthetic_mds()
    rng = np.random.default_rng(31)
    uvw, chan = _uvw_freq(rng, nrow=300, nchan=2)
    corr = ("XX", "YY")
    nx, ny = ds.attrs["npix_x"], ds.attrs["npix_y"]

    inside = np.zeros((nx, ny))
    inside[20, 31] = 1.0                      # the first component only
    outside = 1.0 - inside

    kw = dict(uvw=uvw, freq=chan, corr_types=corr, time=time[0], freq_out=freq[0])
    whole = model_to_apparent_vis_for_region(ds, **kw)
    part_a = model_to_apparent_vis_for_region(ds, region_mask=inside, **kw)
    part_b = model_to_apparent_vis_for_region(ds, region_mask=outside, **kw)

    assert_allclose(part_a + part_b, whole, atol=1e-10)
    assert np.abs(part_a).max() > 0.0
    assert np.abs(part_b).max() > 0.0


def test_fused_wrapper_identity_mueller_matches_no_beam():
    ds, _, time, freq, _ = _synthetic_mds()
    rng = np.random.default_rng(32)
    uvw, chan = _uvw_freq(rng, nrow=200, nchan=2)
    corr = ("XX", "YY")
    nx, ny = ds.attrs["npix_x"], ds.attrs["npix_y"]
    kw = dict(uvw=uvw, freq=chan, corr_types=corr, time=time[0], freq_out=freq[0])
    no_beam = model_to_apparent_vis_for_region(ds, **kw)
    with_beam = model_to_apparent_vis_for_region(ds, mueller=_mueller(1, 1, nx, ny), **kw)
    assert_allclose(with_beam, no_beam, atol=1e-10)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra full pytest tests/test_degrid.py -k fused -v`
Expected: FAIL — `ImportError: cannot import name 'model_to_apparent_vis_for_region'`

- [ ] **Step 3: Write the implementation**

Append to `src/pfb_model_spec/utils/degrid.py`:

```python
def model_to_apparent_vis_for_region(
    model_ds: xr.Dataset,
    *,
    uvw: np.ndarray,
    freq: np.ndarray,
    corr_types: Sequence[str],
    time: float,
    freq_out: float,
    mueller: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    region_mask: np.ndarray | None = None,
    epsilon: float = 1e-7,
    do_wgridding: bool = True,
    divide_by_n: bool = False,
    nthreads: int = 1,
) -> np.ndarray:
    """Render a model and degrid it to apparent correlated visibilities.

    Composes :func:`render_model_region`, :func:`apply_mueller`,
    :func:`degrid_stokes` and :func:`stokes_vis_to_corr` for one chunk of data.
    The primitives remain separately callable for consumers that need only part
    of the chain.

    The model is always rendered on its own grid, taken from the `.mds` attrs,
    so no resampling occurs (see :func:`render_model_region` on why resampling
    to a coarser grid would not conserve flux).

    Args:
        model_ds: An opened `.mds` dataset.
        uvw: Baseline coordinates in metres, shape ``(nrow, 3)``.
        freq: Channel frequencies in Hz, shape ``(nchan,)``.
        corr_types: Output correlations, e.g. ``("XX", "XY", "YX", "YY")``.
        time: Representative time for the chunk, conventionally the unweighted
            mean of its time axis.
        freq_out: Representative frequency in Hz for the chunk, conventionally
            the unweighted mean of its frequency axis. May lie between the
            bands the model was fitted at.
        mueller: Optional Stokes-basis Mueller block on the model grid, shape
            ``(nstokes_out, nstokes_in, nx, ny)``.
        mask: Optional ``(nrow, nchan)`` visibility mask; zero entries are not
            degridded and come back as exactly zero.
        region_mask: Optional ``(nx, ny)`` image mask applied to the rendered
            model, for degridding a sub-region into its own column.
        epsilon: Gridder accuracy.
        do_wgridding: Perform w-correction via improved w-stacking.
        divide_by_n: Divide by the geometric n-term; see :func:`apply_mueller`.
        nthreads: Threads for the gridder.

    Returns:
        Complex visibilities, shape ``(nrow, nchan, len(corr_types))``.
    """
    geom = model_geometry(model_ds)
    image = render_model_region(model_ds, time=time, freq=freq_out)
    stokes_in = geom["stokes"]

    if region_mask is not None:
        if region_mask.shape != image.shape[1:]:
            raise ValueError(
                f"region_mask grid {region_mask.shape} does not match the model "
                f"grid {image.shape[1:]}"
            )
        image = image * region_mask[None]

    if mueller is not None:
        image = apply_mueller(image, mueller)
        # a Mueller may map a single intrinsic Stokes product onto several
        # apparent ones, so the labels follow its output axis, not the model's
        stokes_in = "IQUV"[: image.shape[0]]

    stokes_vis = degrid_stokes(
        uvw,
        freq,
        image,
        cell_rad=geom["cell_rad"],
        x0=geom["x0"],
        y0=geom["y0"],
        flip_u=geom["flip_u"],
        flip_v=geom["flip_v"],
        flip_w=geom["flip_w"],
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        divide_by_n=divide_by_n,
        nthreads=nthreads,
        mask=mask,
    )
    return stokes_vis_to_corr(stokes_vis, stokes_in, corr_types)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra full pytest tests/test_degrid.py -v`
Expected: 24 passed

- [ ] **Step 5: Run the whole suite (nothing else may regress)**

Run: `uv run --extra full pytest -v`
Expected: all pass, including `test_lightweight_import.py`

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
source /home/bester/software/pfb-model-spec/.venv/bin/activate && \
git add src/pfb_model_spec/utils/degrid.py tests/test_degrid.py && \
git commit -m "feat(degrid): add the fused model-to-apparent-visibility entry point"
```

---

## Task 6: Documentation

The rules files are how the next agent finds this API; without them the module is
invisible. Split from Task 5 because a reviewer can accept the code and reject the prose.

**Files:**
- Modify: `.claude/rules/component-model.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add the degrid API section to `.claude/rules/component-model.md`**

Insert after the "The I/O API (`utils/io.py`)" section:

```markdown
## The degrid API (`utils/degrid.py`)

Turns a component model into visibilities for one chunk of data. Pure numpy — no dask, no
Ray, no measurement-set handles, no `pfb_imaging` import. Data selection, chunking,
distribution and the MS write belong to the calling application (pfb-imaging's
`degrid-msv4`, ratt-ru/pfb-imaging#278; QuartiCal later).

- `model_geometry(model_ds)` → the `.mds` gridding attrs as a dict (`nx`, `ny`, `cell_rad`,
  `x0`, `y0`, `flip_u/v/w`, `stokes`). Callers pass these to `degrid_stokes` rather than
  reading attrs by hand; raises on non-square pixels.
- `render_model_region(model_ds, *, time, freq, nx=…, ny=…, cell_rad=…, x0=…, y0=…)` →
  `(nstokes, nx, ny)`. Wraps `eval_coeffs_to_slice`; the output grid defaults to the model's
  own. `freq` may lie between fitted bands — that continuity is what lets a consumer predict
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
```

- [ ] **Step 2: Update the "Deferred scope" list in the same file**

The shared-reader bullet is now built. Replace it:

```markdown
- ~~a shared **`.mds` reader** … for pfb-imaging's `degrid` and QuartiCal~~ — **built**, see
  "The degrid API" above (`model_geometry` + `render_model_region` replace the inline
  `parse_expr`/`lambdify` schema read).
```

- [ ] **Step 3: Update `CLAUDE.md` "Current status"**

Add a bullet after the `model2comps` one:

```markdown
- **Degrid kernel** (`utils/degrid.py`): renders a `.mds` to an image, optionally attenuates
  it with a Mueller beam, degrids to visibilities (ducc0) and converts Stokes to
  correlations. Pure numpy — the calling application owns chunking, distribution and I/O.
  Consumed by pfb-imaging's `degrid-msv4` (ratt-ru/pfb-imaging#278). See
  `.claude/rules/component-model.md` → "The degrid API".
```

And remove the shared-`.mds`-reader item from the "Deferred (not yet built)" bullet, leaving
the `.dds` reading path.

- [ ] **Step 4: Verify the docs match the code**

Run: `uv run --extra full python -c "
from pfb_model_spec.utils import degrid
names = ['model_geometry','render_model_region','apply_mueller','degrid_stokes','stokes_vis_to_corr','model_to_apparent_vis_for_region']
missing = [n for n in names if not hasattr(degrid, n)]
assert not missing, missing
print('all documented names exist')"`
Expected: `all documented names exist`

- [ ] **Step 5: Commit**

```bash
source /home/bester/software/pfb-model-spec/.venv/bin/activate && \
git add CLAUDE.md .claude/rules/component-model.md && \
git commit -m "docs: document the degrid kernel API"
```

---

## Definition of done

- [ ] `uv run --extra full pytest -v` — all green, 24 new tests in `tests/test_degrid.py`
- [ ] `uv run ruff format --check . && uv run ruff check .` — clean
- [ ] `ducc0` in the `full` extra; `uv.lock` updated and committed
- [ ] `import pfb_model_spec` still does not import `utils.degrid` (guard test passes)
- [ ] No `africanus`, `daskms`, `dask`, `ray` or `pfb_imaging` import anywhere in the module
- [ ] `.claude/rules/component-model.md` and `CLAUDE.md` updated
- [ ] A release cut (`uv run tbump <version>`) so pfb-imaging can pin it — see
      `.claude/rules/testing-and-ci.md` §5, including the release-ordering caveat

## Follow-ups deliberately not in this plan

- Flux-conserving resampling for coarsening (blocks ratt-ru/pfb-imaging#309) — needs its own
  issue; the current behaviour is documented, not fixed.
- The Stokes axis and `(Y, X)` spec revision (#19, #20) and the converter (#17).
- Recording imaging metadata in the `.mds` attrs (ratt-ru/pfb-imaging#327).
