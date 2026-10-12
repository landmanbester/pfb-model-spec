# Flux-Conserving Remap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flux-inflating bilinear resampler in `eval_coeffs_to_slice` with a flux-conserving, positivity-preserving remap (`"box"` or `"tent"` kernel), and add a coefficient-space remap for linear parametrisations (#23).

**Architecture:** A new pure numpy/scipy.sparse module, `utils/remap.py`, builds separable 1-D overlap weights between two rectilinear grids. `remap_image` applies them along x, then along y. `"tent"` adds a minmod-limited first-moment term. `remap_coefficients` reuses the box weights on sparse coefficient planes. `eval_coeffs_to_slice` renders on the input grid and calls `remap_image`; `render_model_region` passes `kernel` through.

**Tech Stack:** numpy, scipy.sparse (`csr_array`, `diags_array`; scipy ≥ 1.11, locked at 1.17), sympy (linearity check only), pytest.

**Spec:** `docs/superpowers/specs/2026-10-12-flux-conserving-remap-design.md`

## Global Constraints

- Pixel `k` of an `n`-pixel axis with cell `c` and offset `x0` is centred at `(k - n//2)·c + x0` and spans `±c/2`. Pixel values are flux per pixel.
- Images are `(..., ny, nx)`: x is the last axis (`NAXIS1`).
- Kernels are exactly `"box"` (default) and `"tent"`. Any other value raises `ValueError`.
- Flux outside the output footprint is dropped, never wrapped.
- Same geometry returns a bit-exact copy.
- `remap_coefficients` is box only and raises `ValueError` unless `expr` is linear and homogeneous in `params`.
- No `.mds` schema change, no `SPEC_VERSION` change, no CLI command.
- Every new argument is keyword-only and defaulted; existing calls must keep working.
- `src/pfb_model_spec/__init__.py` must not import `utils/remap.py` (lightweight-import guard).
- After every code change: `uv run ruff format . && uv run ruff check . --fix`. Tests: `uv run --extra full pytest -v`.
- Commit with the venv active: `source .venv/bin/activate && git add … && git commit -m "…"`. Use Conventional Commits and end each message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Floating-point slivers at aligned edges.** An integer-pixel shift or exact cell ratio must give exact weights, not `1e-16` leakage into neighbours. This is pinned by the block-sum and integer-shift tests (Task 1).
2. **Input pixels straddling the output edge.** They should contribute only their overlapping fraction. Pinned by `test_straddling_pixel_contributes_its_overlap` (Task 1).
3. **Large grids.** Weights must stay sparse (no dense `n_out × n_in`). Pinned by `test_axis_weights_are_sparse` (Task 1).
4. **Empty models (`ncomps == 0`).** They must remap to an empty model, not crash. Pinned by `test_remap_coefficients_empty_model` (Task 3).
5. **Extreme coarsening, where the whole input lands in one output pixel.** Total flux must be preserved. Pinned by `test_extreme_coarsening_conserves_flux` (Task 1).

---

## File Structure

- **Create `src/pfb_model_spec/utils/remap.py`:** grid weights, the box and tent kernels, `remap_image` and `remap_coefficients`. It knows nothing about `.mds` or sympy rendering; sympy is imported lazily for the linearity check only.
- **Create `tests/test_remap.py`:** unit tests for `remap.py`.
- **Modify `src/pfb_model_spec/utils/modelspec.py`:** `eval_coeffs_to_slice` delegates to `remap_image`. Delete `_resample_xmajor` and the scipy interpolator import.
- **Modify `src/pfb_model_spec/utils/degrid.py`:** `render_model_region` gains `kernel`, and its docstrings are updated.
- **Modify `tests/test_modelspec.py` and `tests/test_degrid.py`:** add flux-conservation tests and replace the test that pins the inflation.
- **Modify `.claude/rules/component-model.md` and `CLAUDE.md`:** documentation.

---

### Task 1: Box remap (`remap_image`)

**Files:**
- Create: `src/pfb_model_spec/utils/remap.py`
- Test: `tests/test_remap.py`

**Interfaces:**
- Produces:
  - `KERNELS: tuple[str, ...]`, which is `("box",)` after this task.
  - `_axis_weights(n_in, cell_in, x0_in, n_out, cell_out, x0_out) -> tuple[csr_array, csr_array]`, returning `(W, M)`, each of shape `(n_out, n_in)`.
  - `_check_geometry(nxi, nyi, cellxi, cellyi, nxo, nyo, cellxo, cellyo) -> None`.
  - `remap_image(image, *, kernel="box", nxi, nyi, cellxi, cellyi, x0i, y0i, nxo, nyo, cellxo, cellyo, x0o, y0o) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_remap.py`:

```python
"""Synthetic tests of the flux-conserving remap (utils/remap.py).

Grids are deliberately non-square so that a swapped axis cannot pass silently.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pfb_model_spec.utils.remap import KERNELS, _axis_weights, remap_image

NX, NY = 32, 24

# output cell / input cell: refine x4, x2, x1.7; coarsen x1.7, x2, x4
FACTORS = [0.25, 0.5, 1 / 1.7, 1.7, 2.0, 4.0]


def _grid(nxi, nyi, celli, nxo, nyo, cello, *, x0i=0.0, y0i=0.0, x0o=0.0, y0o=0.0, cellyi=None, cellyo=None):
    return dict(
        nxi=nxi,
        nyi=nyi,
        cellxi=celli,
        cellyi=celli if cellyi is None else cellyi,
        x0i=x0i,
        y0i=y0i,
        nxo=nxo,
        nyo=nyo,
        cellxo=cello,
        cellyo=cello if cellyo is None else cellyo,
        x0o=x0o,
        y0o=y0o,
    )


def _sparse_image(shape, seed=0):
    """Non-negative, about half the pixels zero: compact and extended structure mixed."""
    rng = np.random.default_rng(seed)
    return rng.random(shape) * (rng.random(shape) > 0.5)


@pytest.mark.parametrize("kernel", KERNELS)
def test_same_grid_is_identity(kernel):
    image = _sparse_image((2, NY, NX))
    out = remap_image(image, kernel=kernel, **_grid(NX, NY, 1.0, NX, NY, 1.0, x0i=0.3, x0o=0.3))
    assert_array_equal(out, image)
    assert out is not image


def test_integer_input_is_returned_as_float():
    out = remap_image(np.ones((NY, NX), dtype=int), **_grid(NX, NY, 1.0, NX, NY, 1.0))
    assert out.dtype == np.float64


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("factor", FACTORS)
def test_conserves_flux_when_contained(kernel, factor):
    image = _sparse_image((2, NY, NX), seed=1)
    # a margin of 4 output pixels a side keeps the input inside the output footprint
    nxo = int(np.ceil(NX / factor)) + 8
    nyo = int(np.ceil(NY / factor)) + 8
    geometry = _grid(NX, NY, 1.0, nxo, nyo, factor, x0o=0.37 * factor, y0o=-0.21 * factor)
    out = remap_image(image, kernel=kernel, **geometry)
    assert out.shape == (2, nyo, nxo)
    assert_allclose(out.sum(axis=(-2, -1)), image.sum(axis=(-2, -1)), rtol=1e-12)
    assert out.min() >= -1e-12 * image.max()


@pytest.mark.parametrize("kernel", KERNELS)
def test_conserves_flux_with_non_square_cells(kernel):
    image = _sparse_image((NY, NX), seed=2)
    geometry = _grid(NX, NY, 1.0, 30, 50, 1.7, cellyo=0.6, x0o=0.11, y0o=0.07)
    out = remap_image(image, kernel=kernel, **geometry)
    assert out.shape == (50, 30)
    assert out.sum() == pytest.approx(image.sum(), rel=1e-12)


def test_box_coarsening_by_integer_factor_is_block_sums():
    image = np.random.default_rng(3).random((6, 8))
    # offsets chosen so that the output pixel edges coincide with input pixel edges
    out = remap_image(image, kernel="box", **_grid(8, 6, 1.0, 4, 3, 2.0, x0o=0.5, y0o=-0.5))
    assert_array_equal(out, image.reshape(3, 2, 4, 2).sum(axis=(1, 3)))


def test_box_integer_shift_moves_values_exactly():
    image = _sparse_image((NY, NX), seed=4)
    out = remap_image(image, kernel="box", **_grid(NX, NY, 1.0, NX, NY, 1.0, x0o=3.0, y0o=-2.0))
    expected = np.zeros_like(image)
    expected[2:, : NX - 3] = image[: NY - 2, 3:]
    assert_array_equal(out, expected)


@pytest.mark.parametrize("kernel", KERNELS)
def test_flux_outside_output_is_dropped_not_wrapped(kernel):
    image = np.zeros((NY, NX))
    image[5, NX - 1] = 1.0
    # the output covers only the left half of the input
    out = remap_image(image, kernel=kernel, **_grid(NX, NY, 1.0, NX // 2, NY, 1.0, x0o=-NX / 4))
    assert out.sum() == 0.0


@pytest.mark.parametrize("kernel", KERNELS)
def test_straddling_pixel_contributes_its_overlap(kernel):
    image = np.zeros((4, 4))
    image[1, 3] = 1.0
    # shifting the output half a pixel left puts its right edge through the centre of x = 3
    out = remap_image(image, kernel=kernel, **_grid(4, 4, 1.0, 4, 4, 1.0, x0o=-0.5))
    assert out.sum() == pytest.approx(0.5, abs=1e-15)


@pytest.mark.parametrize("kernel", KERNELS)
def test_extreme_coarsening_conserves_flux(kernel):
    image = _sparse_image((NY, NX), seed=5)
    out = remap_image(image, kernel=kernel, **_grid(NX, NY, 1.0, 1, 1, 100.0))
    assert out.shape == (1, 1)
    assert out[0, 0] == pytest.approx(image.sum(), rel=1e-12)


def test_axis_weights_are_sparse():
    w, m = _axis_weights(10_000, 1.0, 0.0, 7_000, 1.43, 0.3)
    assert w.shape == m.shape == (7_000, 10_000)
    # each input pixel overlaps at most ceil(cell_in / cell_out) + 1 = 2 output pixels
    assert w.nnz <= 2 * 10_000
    # every input pixel inside the output footprint is fully redistributed
    assert_allclose(w.sum(axis=0)[100:-100], 1.0, rtol=1e-12)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"kernel": "bilinear"}, "kernel"),
        ({"cellxo": 0.0}, "cell"),
        ({"nyo": 0}, "pixel count"),
    ],
)
def test_rejects_invalid_arguments(override, match):
    kwargs = {"kernel": "box", **_grid(NX, NY, 1.0, NX, NY, 2.0)}
    kwargs.update(override)
    with pytest.raises(ValueError, match=match):
        remap_image(np.zeros((NY, NX)), **kwargs)


def test_rejects_mismatched_image_shape():
    with pytest.raises(ValueError, match="trailing shape"):
        remap_image(np.zeros((NX, NY)), **_grid(NX, NY, 1.0, NX, NY, 2.0))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra full pytest tests/test_remap.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'pfb_model_spec.utils.remap'`.

- [ ] **Step 3: Write the implementation**

Create `src/pfb_model_spec/utils/remap.py`:

```python
"""Flux-conserving remapping of pixelated images between axis-aligned rectilinear grids.

Pixel values are flux per pixel (Jy/pixel), so remapping redistributes flux rather than
sampling a field. Along each axis pixel ``k`` of an ``n``-pixel grid with cell ``c`` and centre
offset ``x0`` covers ``[x_k - c/2, x_k + c/2]`` with ``x_k = (k - n//2)·c + x0``. A 2-D remap is
a 1-D remap along x (the last axis) followed by one along y.

Kernels:
    ``"box"``: brightness is uniform within each input pixel (first-order conservative).

Flux from input pixels outside the output footprint is dropped, never wrapped.
"""

import numpy as np
from scipy import sparse

KERNELS = ("box",)

# pixel edges this close to an integer (in output-pixel units) are snapped onto it, so that
# integer shifts and exact cell ratios produce exact weights instead of 1e-16 slivers
_SNAP = 1e-9


def _snap(p: np.ndarray) -> np.ndarray:
    nearest = np.rint(p)
    return np.where(np.abs(p - nearest) < _SNAP, nearest, p)


def _axis_weights(
    n_in: int, cell_in: float, x0_in: float, n_out: int, cell_out: float, x0_out: float
) -> tuple[sparse.csr_array, sparse.csr_array]:
    """1-D overlap weights ``(W, M)``, each a ``(n_out, n_in)`` sparse array.

    ``W[o, i]`` is the fraction of input pixel ``i`` lying in output pixel ``o``; ``M[o, i]`` is
    the first moment of that overlap, ``∫ q dq`` with ``q`` the position within pixel ``i`` in
    input-pixel units (``q`` in ``[-1/2, 1/2]``).
    """
    # input pixel edges in output-pixel units, where output pixel o spans [o, o + 1]
    origin = (-(n_out // 2) - 0.5) * cell_out + x0_out
    lo = _snap(((np.arange(n_in) - n_in // 2 - 0.5) * cell_in + x0_in - origin) / cell_out)
    hi = _snap(lo + cell_in / cell_out)
    width = hi - lo
    centre = (lo + hi) / 2

    # each input pixel overlaps at most nspan consecutive output pixels, starting at `first`
    first = np.clip(np.floor(lo).astype(np.int64), 0, n_out - 1)
    nspan = int(np.ceil(cell_in / cell_out)) + 1
    o = first[:, None] + np.arange(nspan)[None, :]
    a = np.maximum(lo[:, None], o)
    b = np.minimum(hi[:, None], o + 1)
    keep = (b > a) & (o < n_out)

    cols = np.broadcast_to(np.arange(n_in)[:, None], o.shape)[keep]
    rows = o[keep]
    qa = ((a - centre[:, None]) / width[:, None])[keep]
    qb = ((b - centre[:, None]) / width[:, None])[keep]
    w = sparse.csr_array((qb - qa, (rows, cols)), shape=(n_out, n_in))
    m = sparse.csr_array(((qb**2 - qa**2) / 2, (rows, cols)), shape=(n_out, n_in))
    return w, m


def _remap_axis(image: np.ndarray, axis: int, w: sparse.csr_array) -> np.ndarray:
    moved = np.moveaxis(image, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    out = w @ flat
    return np.moveaxis(out.reshape((w.shape[0],) + moved.shape[1:]), 0, axis)


def _check_grid(name: str, n: int, cell: float) -> None:
    if int(n) != n or n < 1:
        raise ValueError(f"{name}: pixel count must be a positive integer, got {n}")
    if not cell > 0:
        raise ValueError(f"{name}: cell size must be positive, got {cell}")


def _check_geometry(
    nxi: int, nyi: int, cellxi: float, cellyi: float, nxo: int, nyo: int, cellxo: float, cellyo: float
) -> None:
    _check_grid("x input", nxi, cellxi)
    _check_grid("y input", nyi, cellyi)
    _check_grid("x output", nxo, cellxo)
    _check_grid("y output", nyo, cellyo)


def remap_image(
    image: np.ndarray,
    *,
    kernel: str = "box",
    nxi: int,
    nyi: int,
    cellxi: float,
    cellyi: float,
    x0i: float,
    y0i: float,
    nxo: int,
    nyo: int,
    cellxo: float,
    cellyo: float,
    x0o: float,
    y0o: float,
) -> np.ndarray:
    """Remap flux-per-pixel images onto another grid, conserving flux.

    Args:
        image: Shape ``(..., nyi, nxi)``; leading axes are remapped independently.
        kernel: One of ``KERNELS`` (see the module docstring).
        nxi, nyi, cellxi, cellyi, x0i, y0i: Input grid (pixels, cell sizes, centre offsets).
        nxo, nyo, cellxo, cellyo, x0o, y0o: Output grid.

    Returns:
        A float array of shape ``(..., nyo, nxo)``. Identical geometries return a copy.

    Raises:
        ValueError: On an unknown kernel, an invalid grid, or a mismatched ``image`` shape.
    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
    _check_geometry(nxi, nyi, cellxi, cellyi, nxo, nyo, cellxo, cellyo)
    image = np.asarray(image, dtype=float)
    if image.ndim < 2 or image.shape[-2:] != (nyi, nxi):
        raise ValueError(f"image must have trailing shape (nyi, nxi) = {(nyi, nxi)}, got {image.shape}")
    if (nxi, nyi, cellxi, cellyi, x0i, y0i) == (nxo, nyo, cellxo, cellyo, x0o, y0o):
        return image.copy()
    wx, _ = _axis_weights(nxi, cellxi, x0i, nxo, cellxo, x0o)
    wy, _ = _axis_weights(nyi, cellyi, y0i, nyo, cellyo, y0o)
    return _remap_axis(_remap_axis(image, -1, wx), -2, wy)
```

`M` is computed now but used only by the tent kernel in Task 2. It costs one extra sparse array.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra full pytest tests/test_remap.py -v`
Expected: all PASS. With `KERNELS == ("box",)`, the kernel-parametrised tests run once each.

- [ ] **Step 5: Lint, run the full suite, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
uv run --extra full pytest -q
source .venv/bin/activate && git add src/pfb_model_spec/utils/remap.py tests/test_remap.py && git commit -m "feat: flux-conserving box remap between rectilinear grids (#23)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Expected: the full suite passes; `tests/test_lightweight_import.py` still passes because nothing imports `remap.py` yet.

---

### Task 2: Tent kernel

**Files:**
- Modify: `src/pfb_model_spec/utils/remap.py`
- Test: `tests/test_remap.py`

**Interfaces:**
- Consumes: `_axis_weights` and `remap_image` from Task 1.
- Produces: `KERNELS == ("box", "tent")`; `remap_image(..., kernel="tent")`; `_minmod_slopes(u) -> np.ndarray`.

Adding `"tent"` to `KERNELS` automatically runs every `KERNELS`-parametrised Task 1 test for tent: identity, conservation, positivity, footprint, straddle and extreme coarsening.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_remap.py`:

```python
def test_kernels_are_box_and_tent():
    assert KERNELS == ("box", "tent")


def test_tent_leaves_an_isolated_component_unsmeared():
    """Neighbour differences of a lone pixel have opposite signs, so minmod gives zero slope."""
    image = np.zeros((10, 12))
    image[4, 6] = 1.0
    geometry = _grid(12, 10, 1.0, 48, 40, 0.25, x0o=0.1, y0o=0.03)
    box = remap_image(image, kernel="box", **geometry)
    tent = remap_image(image, kernel="tent", **geometry)
    assert_allclose(tent, box, atol=1e-15)
    assert tent.sum() == pytest.approx(1.0, abs=1e-14)


def test_tent_reproduces_a_linear_ramp_exactly():
    """Brightness rising linearly in x, constant in y, refined x2 onto aligned half-pixels."""
    n, ny = 8, 6
    image = np.tile(1.0 + np.arange(n), (ny, 1))
    # offsets of -1/4 input pixel align the output halves with the input pixels
    out = remap_image(image, kernel="tent", **_grid(n, ny, 1.0, 2 * n, 2 * ny, 0.5, x0o=-0.25, y0o=-0.25))
    # within pixel i the flux density is u_i + q (q in [-1/2, 1/2]); each half spans 1/2 in y too
    expected = np.zeros((2 * ny, 2 * n))
    expected[:, 0::2] = (image[0] / 2 - 1 / 8) / 2
    expected[:, 1::2] = (image[0] / 2 + 1 / 8) / 2
    # the last input column borders zero flux, so the limiter flattens it there
    assert_allclose(out[:, :-2], expected[:, :-2], rtol=1e-13)
    box = remap_image(image, kernel="box", **_grid(n, ny, 1.0, 2 * n, 2 * ny, 0.5, x0o=-0.25, y0o=-0.25))
    assert not np.allclose(box[:, :-2], expected[:, :-2])


def test_tent_refines_smooth_emission_better_than_box():
    """Block-sum a fine Gaussian onto a x4 coarser grid, then remap it back to the fine grid."""
    xf = np.arange(64) - 32
    xx, yy = np.meshgrid(xf, xf)
    fine = np.exp(-((xx - 2.3) ** 2 + (yy + 1.1) ** 2) / (2 * 6.0**2))
    coarse = fine.reshape(16, 4, 16, 4).sum(axis=(1, 3))
    # x0i = 1.5 aligns the coarse pixel edges with every fourth fine pixel edge
    geometry = _grid(16, 16, 4.0, 64, 64, 1.0, x0i=1.5, y0i=1.5)
    err = {k: np.abs(remap_image(coarse, kernel=k, **geometry) - fine).sum() / fine.sum() for k in ("box", "tent")}
    assert err["tent"] < 0.5 * err["box"]


def test_tent_stays_positive_on_steep_emission():
    image = np.random.default_rng(6).random((NY, NX)) ** 8
    out = remap_image(image, kernel="tent", **_grid(NX, NY, 1.0, 80, 61, 0.43, x0o=0.17, y0o=-0.29))
    assert out.min() >= -1e-12 * image.max()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra full pytest tests/test_remap.py -v`
Expected: the new tests FAIL. `test_kernels_are_box_and_tent` fails on the assertion; the others fail with `ValueError: kernel must be one of ('box',), got 'tent'`.

- [ ] **Step 3: Write the implementation**

In `src/pfb_model_spec/utils/remap.py`:

1. Replace the `Kernels:` block of the module docstring with:

```python
Kernels:
    ``"box"``: brightness is uniform within each input pixel (first-order conservative).
    ``"tent"``: brightness is linear within each input pixel, with a minmod-limited slope
        (second-order conservative). It is positive and the identity on the same grid; an
        isolated component gets zero slope, so it is never smeared. The limiter makes it
        nonlinear in the image, and the x-then-y order makes it very slightly anisotropic.
```

2. Change `KERNELS = ("box",)` to `KERNELS = ("box", "tent")`.

3. Add `_minmod_slopes` above `_remap_axis`, and replace `_remap_axis` with:

```python
def _minmod_slopes(u: np.ndarray) -> np.ndarray:
    """Minmod-limited slopes along axis 0 of ``u``, with zero flux beyond both ends."""
    padded = np.pad(u, ((1, 1), (0, 0)))
    fwd = padded[2:] - padded[1:-1]
    bwd = padded[1:-1] - padded[:-2]
    return np.where(fwd * bwd > 0, np.sign(fwd) * np.minimum(np.abs(fwd), np.abs(bwd)), 0.0)


def _remap_axis(
    image: np.ndarray, axis: int, w: sparse.csr_array, m: sparse.csr_array, kernel: str
) -> np.ndarray:
    moved = np.moveaxis(image, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    out = w @ flat
    if kernel == "tent":
        # within input pixel i the flux density is u_i + s_i q, so the overlap gets W u + M s
        out = out + m @ _minmod_slopes(flat)
    return np.moveaxis(out.reshape((w.shape[0],) + moved.shape[1:]), 0, axis)
```

4. Replace the last three lines of `remap_image` with:

```python
    wx, mx = _axis_weights(nxi, cellxi, x0i, nxo, cellxo, x0o)
    wy, my = _axis_weights(nyi, cellyi, y0i, nyo, cellyo, y0o)
    out = _remap_axis(image, -1, wx, mx, kernel)
    return _remap_axis(out, -2, wy, my, kernel)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra full pytest tests/test_remap.py -v`
Expected: all PASS, with the Task 1 kernel-parametrised tests now running for both `box` and `tent`.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
source .venv/bin/activate && git add src/pfb_model_spec/utils/remap.py tests/test_remap.py && git commit -m "feat: minmod-limited tent kernel for the flux-conserving remap

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Coefficient-space remap (`remap_coefficients`)

**Files:**
- Modify: `src/pfb_model_spec/utils/remap.py`
- Test: `tests/test_remap.py`

**Interfaces:**
- Consumes: `_axis_weights` and `_check_geometry` from Task 1. In the tests, `fit_image_cube` and `eval_coeffs_to_cube` from `utils/modelspec.py` (unchanged signatures).
- Produces: `remap_coefficients(coeffs, y_index, x_index, expr, params, *, nxi, nyi, cellxi, cellyi, x0i, y0i, nxo, nyo, cellxo, cellyo, x0o, y0o) -> tuple[np.ndarray, np.ndarray, np.ndarray]`, returning `(coeffs_out (nstokes, npar, ncomps_out), y_index_out, x_index_out)`. Also `_check_linear(expr, params) -> None`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_remap.py`, extend the import block:

```python
from pfb_model_spec.utils.modelspec import eval_coeffs_to_cube, fit_image_cube
from pfb_model_spec.utils.remap import KERNELS, _axis_weights, remap_coefficients, remap_image
```

Then append:

```python
COEFF_GRID = _grid(30, 20, 1.0, 22, 17, 1.7, cellyo=1.3, x0o=0.37, y0o=-0.2)


def _fitted_components(seed=3):
    """Twelve random components, two Stokes planes, four bands, a 3-term Legendre fit."""
    rng = np.random.default_rng(seed)
    freq = np.linspace(1.0e9, 1.6e9, 4)
    cube = np.zeros((1, freq.size, 2, 20, 30))
    ys, xs = rng.integers(0, 20, 12), rng.integers(0, 30, 12)
    cube[0, :, :, ys, xs] = rng.random((12, freq.size, 2))
    return fit_image_cube(np.array([0.0]), freq, cube, nbasisf=3, method="Legendre")


def test_remap_coefficients_matches_remapping_the_rendered_image():
    """Linearity: remap-then-render equals render-then-remap at any (time, freq)."""
    coeffs, yi, xi, expr, params, texpr, fexpr = _fitted_components()
    c2, y2, x2 = remap_coefficients(coeffs, yi, xi, expr, params, **COEFF_GRID)
    assert c2.shape[:2] == coeffs.shape[:2]
    assert y2.shape == x2.shape == (c2.shape[2],)
    time = np.array([0.0])
    freqs = np.array([1.0e9, 1.13e9, 1.6e9])  # includes a frequency between fitted bands
    src = eval_coeffs_to_cube(time, freqs, 30, 20, coeffs, yi, xi, expr, params, texpr, fexpr)
    dst = eval_coeffs_to_cube(time, freqs, 22, 17, c2, y2, x2, expr, params, texpr, fexpr)
    assert_allclose(dst, remap_image(src, kernel="box", **COEFF_GRID), atol=1e-12)


def test_remap_coefficients_orders_components_like_np_where():
    coeffs, yi, xi, expr, params, *_ = _fitted_components()
    _, y2, x2 = remap_coefficients(coeffs, yi, xi, expr, params, **COEFF_GRID)
    order = y2 * COEFF_GRID["nxo"] + x2
    assert np.all(np.diff(order) > 0)


def test_remap_coefficients_empty_model():
    coeffs = np.zeros((1, 3, 0))
    empty = np.array([], dtype=np.int64)
    params = ["t0", "f1", "f2"]
    c2, y2, x2 = remap_coefficients(coeffs, empty, empty, "t0 + f*f1 + f**2*f2", params, **COEFF_GRID)
    assert c2.shape == (1, 3, 0)
    assert y2.size == x2.size == 0


@pytest.mark.parametrize("expr", ["t0*exp(f1*f)", "t0 + f1*f + 1", "t0*f1"])
def test_remap_coefficients_rejects_nonlinear_parametrisations(expr):
    coeffs = np.ones((1, 2, 1))
    with pytest.raises(ValueError, match="remap_image"):
        remap_coefficients(coeffs, np.array([3]), np.array([4]), expr, ["t0", "f1"], **COEFF_GRID)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra full pytest tests/test_remap.py -v`
Expected: collection error, `ImportError: cannot import name 'remap_coefficients'`.

- [ ] **Step 3: Write the implementation**

Append to `src/pfb_model_spec/utils/remap.py`:

```python
def _check_linear(expr: str, params: list[str]) -> None:
    """Raise unless ``expr`` is linear and homogeneous in ``params``."""
    import sympy as sm
    from sympy.parsing.sympy_parser import parse_expr

    parsed = parse_expr(expr)
    symbols = sm.symbols(tuple(params))
    homogeneous = sm.simplify(parsed.subs({p: 0 for p in symbols})) == 0
    linear = all(sm.simplify(sm.diff(parsed, p, q)) == 0 for p in symbols for q in symbols)
    if not (homogeneous and linear):
        raise ValueError(
            f"parametrisation {expr!r} is not linear and homogeneous in its coefficients, so the "
            "coefficients cannot be remapped directly; render it and use remap_image instead"
        )


def remap_coefficients(
    coeffs: np.ndarray,
    y_index: np.ndarray,
    x_index: np.ndarray,
    expr: str,
    params: list[str],
    *,
    nxi: int,
    nyi: int,
    cellxi: float,
    cellyi: float,
    x0i: float,
    y0i: float,
    nxo: int,
    nyo: int,
    cellxo: float,
    cellyo: float,
    x0o: float,
    y0o: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remap the coefficients of a linear parametrisation onto another grid (box kernel).

    Rendering the result on the output grid equals ``remap_image(kernel="box")`` of the
    rendering on the input grid, at every time and frequency. No dense cube is built. The
    tent kernel is not offered: its limiter is nonlinear in the image.

    Args:
        coeffs: Shape ``(nstokes, npar, ncomps)``.
        y_index, x_index: Component pixel locations on the input grid, shape ``(ncomps,)``.
        expr: The stored sympy parametrisation.
        params: Its coefficient names.
        nxi, nyi, cellxi, cellyi, x0i, y0i: Input grid.
        nxo, nyo, cellxo, cellyo, x0o, y0o: Output grid.

    Returns:
        ``(coeffs_out, y_index_out, x_index_out)``. The output components are the output pixels
        non-zero in any ``(stokes, par)`` plane, in row-major (``np.where``) order.

    Raises:
        ValueError: If ``expr`` is not linear and homogeneous in ``params``, or on an invalid grid.
    """
    _check_linear(expr, params)
    _check_geometry(nxi, nyi, cellxi, cellyi, nxo, nyo, cellxo, cellyo)
    nstokes, npar, _ = coeffs.shape
    wx, _ = _axis_weights(nxi, cellxi, x0i, nxo, cellxo, x0o)
    wy, _ = _axis_weights(nyi, cellyi, y0i, nyo, cellyo, y0o)
    # columns of the weights at the component locations: (nyo, ncomps) and (nxo, ncomps)
    wyc = wy.tocsc()[:, np.asarray(y_index)]
    wxc = wx.tocsc()[:, np.asarray(x_index)]

    # each (stokes, par) plane as sparse (flat output index, value) pairs
    planes = []
    for s in range(nstokes):
        for p in range(npar):
            plane = (wyc @ sparse.diags_array(coeffs[s, p]) @ wxc.T).tocoo()
            nz = plane.data != 0
            planes.append((plane.row[nz].astype(np.int64) * nxo + plane.col[nz], plane.data[nz]))
    union = np.unique(np.concatenate([lin for lin, _ in planes]))
    coeffs_out = np.zeros((nstokes, npar, union.size), dtype=float)
    for k, (lin, data) in enumerate(planes):
        coeffs_out[k // npar, k % npar, np.searchsorted(union, lin)] = data
    y_out, x_out = np.divmod(union, nxo)
    return coeffs_out, y_out, x_out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra full pytest tests/test_remap.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint, run the lightweight-import guard, commit**

```bash
uv run ruff format . && uv run ruff check . --fix
uv run --extra full pytest tests/test_lightweight_import.py tests/test_remap.py -q
source .venv/bin/activate && git add src/pfb_model_spec/utils/remap.py tests/test_remap.py && git commit -m "feat: remap coefficients of linear parametrisations onto a new grid

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Wire the remap into rendering; retire `area_ratio`; docs

**Files:**
- Modify: `src/pfb_model_spec/utils/modelspec.py` (imports; `eval_coeffs_to_slice`; delete `_resample_xmajor`)
- Modify: `src/pfb_model_spec/utils/degrid.py` (`render_model_region`; `model_to_apparent_vis_for_region` docstring)
- Modify: `tests/test_modelspec.py`, `tests/test_degrid.py`
- Modify: `.claude/rules/component-model.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: `remap_image` and `KERNELS` from Tasks 1–2.
- Produces: `eval_coeffs_to_slice(..., *, <geometry>, kernel="box")` and `render_model_region(model_ds, *, time, freq_out, nx=None, ny=None, cell_rad=None, x0=None, y0=None, kernel="box")`. All existing calls are unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_modelspec.py`:

```python
@pytest.mark.parametrize("kernel", ["box", "tent"])
@pytest.mark.parametrize(("factor", "npix"), [(1.0, 64), (0.5, 128), (2.0, 32), (4.0, 16)])
def test_slice_conserves_single_component_flux(kernel, factor, npix):
    """#23: a unit component keeps unit flux whether the output grid is refined or coarsened
    (the old resampler gave 4.0 for x2 coarsening and 16.0 for x4)."""
    cube = np.zeros((1, 1, 1, 64, 64))
    cube[0, 0, 0, 32, 32] = 1.0
    coeffs, yi, xi, expr, params, texpr, fexpr = fit_image_cube(np.array([0.0]), np.array([1.0e9]), cube)
    cell = np.deg2rad(2.0 / 3600.0)
    out = eval_coeffs_to_slice(
        0.0,
        1.0e9,
        coeffs,
        yi,
        xi,
        expr,
        params,
        texpr,
        fexpr,
        nxi=64,
        nyi=64,
        cellxi=cell,
        cellyi=cell,
        x0i=0.0,
        y0i=0.0,
        nxo=npix,
        nyo=npix,
        cellxo=cell * factor,
        cellyo=cell * factor,
        x0o=0.0,
        y0o=0.0,
        kernel=kernel,
    )
    assert out.shape == (1, npix, npix)
    assert out.sum() == pytest.approx(1.0, abs=1e-12)
```

In `tests/test_degrid.py`, replace the whole of `test_render_model_region_coarsen_inflates_flux_by_area_ratio` (docstring and body) with:

```python
@pytest.mark.parametrize("kernel", ["box", "tent"])
def test_render_model_region_coarsen_conserves_flux(kernel):
    """#23: coarsening used to inflate flux by `area_ratio` (x4 for this x2 coarsening)."""
    ds, time, freq, cell = _single_component_mds()
    nxi, nyi = ds.attrs["npix_x"], ds.attrs["npix_y"]
    got = render_model_region(
        ds,
        time=time[0],
        freq_out=freq[0],
        nx=nxi // 2,
        ny=nyi // 2,
        cell_rad=cell * 2,
        kernel=kernel,
    )
    assert got.sum() == pytest.approx(1.0, abs=1e-8)
```

In `test_render_model_region_refine_conserves_flux`, replace this docstring:

```python
    """Refining the grid (finer cell, more pixels) is exact -- see the module
    docstring's Note. This exercises the non-default-grid path, previously
    untested."""
```

with:

```python
    """Refining the grid (finer cell, more pixels) conserves flux."""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra full pytest tests/test_modelspec.py tests/test_degrid.py -v -k "conserve"`
Expected: FAIL. The new tests fail with `TypeError: ... unexpected keyword argument 'kernel'`.

- [ ] **Step 3: Rewire `eval_coeffs_to_slice`**

In `src/pfb_model_spec/utils/modelspec.py`:

1. Delete the line `from scipy.interpolate import RegularGridInterpolator`.
2. Add `from pfb_model_spec.utils.remap import remap_image` directly above `from pfb_model_spec.utils.spec import open_mds`.
3. Replace everything from `def eval_coeffs_to_slice(` up to (not including) `def model_from_mds(`, which includes the whole of `_resample_xmajor`, with:

```python
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
    kernel="box",
):
    """Render coefficients at one (time, freq) onto an arbitrary output grid.

    The components are rendered on their own grid, then remapped onto the output grid with
    :func:`pfb_model_spec.utils.remap.remap_image`, which conserves flux (``kernel`` is
    ``"box"`` or ``"tent"``). Returns ``(nstokes, nyo, nxo)``. The geometry is keyword-only so
    that a call written against the old x-major positional signature fails loudly instead of
    swapping axes.
    """
    modelf, tfunc, ffunc = _model_functions(expr, paramf, texpr, fexpr)
    tval, fval = tfunc(time), ffunc(freq)
    image_in = np.zeros((coeffs.shape[0], nyi, nxi), dtype=float)
    for s in range(coeffs.shape[0]):
        image_in[s, y_index, x_index] = modelf(tval, fval, *coeffs[s])
    return remap_image(
        image_in,
        kernel=kernel,
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

- [ ] **Step 4: Pass `kernel` through `render_model_region`**

In `src/pfb_model_spec/utils/degrid.py`:

1. In the `render_model_region` signature, add `kernel: str = "box",` after `y0: float | None = None,`.
2. Replace this docstring text:

```
    The output grid defaults to the model's own. Supplying a different one
    resamples via ``eval_coeffs_to_slice``.

    Note:
        Resampling to a **coarser** grid is not flux-conserving for the point
        components a `.mds` stores: ``eval_coeffs_to_slice`` applies an
        ``area_ratio`` factor appropriate to a surface-brightness field, which
        inflates a point component's flux by exactly that factor (measured:
        coarsen x2 -> x4, coarsen x4 -> x16; refining and same-grid are exact).
        Degridding always uses the model's own grid and so is unaffected.
```

with:

```
    The output grid defaults to the model's own. Supplying a different one
    remaps via ``eval_coeffs_to_slice``, which conserves flux whether the grid
    is refined, coarsened or shifted; flux outside the output grid is dropped.
```

3. In the Args section, after the `y0:` entry, add:

```
        kernel: Remap kernel for a different output grid: ``"box"``
            (piecewise-constant) or ``"tent"`` (minmod-limited piecewise-linear,
            smoother when refining extended emission).
```

4. In the `eval_coeffs_to_slice(...)` call at the end of `render_model_region`, add `kernel=kernel,` after `y0o=y0o,`.
5. In the `model_to_apparent_vis_for_region` docstring, replace:

```
    The model is always rendered on its own grid, taken from the `.mds` attrs,
    so no resampling occurs (see :func:`render_model_region` on why resampling
    to a coarser grid would not conserve flux).
```

with:

```
    The model is always rendered on its own grid, taken from the `.mds` attrs,
    so no resampling occurs.
```

- [ ] **Step 5: Run the full suite**

Run: `uv run ruff format . && uv run ruff check . --fix && uv run --extra full pytest -q`
Expected: all PASS. This includes the unchanged `test_slice_matches_cube_and_is_shift_invariant` (box moves integer shifts exactly) and `tests/test_lightweight_import.py`. `grep -rn "area_ratio\|RegularGridInterpolator\|_resample_xmajor" src` should return nothing.

- [ ] **Step 6: Update the docs**

In `.claude/rules/component-model.md`:

1. Change `Read this when working on \`src/pfb_model_spec/utils/{modelspec,io,spec}.py\`,` to `Read this when working on \`src/pfb_model_spec/utils/{modelspec,io,spec,remap}.py\`,`.
2. Replace the `eval_coeffs_to_slice` bullet (from `- \`eval_coeffs_to_slice(time, freq,` through `grid. The geometry arguments are keyword-only.`) with:

```markdown
- `eval_coeffs_to_slice(time, freq, coeffs, y_index, x_index, expr, paramf, texpr, fexpr, *, nxi,
  nyi, cellxi, cellyi, x0i, y0i, nxo, nyo, cellxo, cellyo, x0o, y0o, kernel="box")` → render to a
  `(nstokes, nyo, nxo)` slice on the model's grid, then remap it flux-conservingly onto an
  arbitrary output grid via `remap_image` (see "The remap API"). The geometry arguments are
  keyword-only.
```

3. Replace the whole paragraph that starts `**Known limitation — resampling is not flux-conserving when coarsening.**` and ends `(ratt-ru/pfb-imaging#309) will hit it.` with:

```markdown
## The remap API (`utils/remap.py`)

Flux-conserving remapping between axis-aligned rectilinear grids (#23). Pixel values are flux
per pixel, so remapping redistributes flux; it does not sample a field. Separable sparse 1-D
overlap weights are applied along x, then y. Flux outside the output footprint is dropped,
never wrapped. NUFFT/sinc resampling was rejected because it rings (negative sidelobes) and is
periodic.

- `remap_image(image, *, kernel="box", nxi, nyi, cellxi, cellyi, x0i, y0i, nxo, nyo, cellxo,
  cellyo, x0o, y0o)` → `(..., nyo, nxo)`. `kernel` is one of `KERNELS`:
  - `"box"`: uniform brightness per input pixel; integer coarsening equals block sums.
  - `"tent"`: piecewise-linear brightness with a minmod-limited slope. It is positive and the
    identity on the same grid. An isolated component is never smeared, and smooth emission
    refines smoothly. The limiter makes it nonlinear, hence the name "tent" and not "linear".
  - Both kernels conserve flux and preserve positivity; identical geometry returns a copy.
- `remap_coefficients(coeffs, y_index, x_index, expr, params, *, <geometry>)` →
  `(coeffs_out, y_index_out, x_index_out)`. It remaps `.mds` coefficients directly (box only),
  equivalent to box-remapping every rendering. It applies only to parametrisations that are
  linear and homogeneous in their coefficients, and raises `ValueError` otherwise (e.g. power-law
  models, possibly #29). **The image-space path must always remain available**: nothing may
  depend on the coefficient shortcut.
```

4. In "The degrid API", in the `render_model_region(...)` bullet, add `kernel="box"` after `y0=…` in the signature.

In `CLAUDE.md`, change the `utils/` line of the project structure to:

```
├── utils/                 # spec.py (versions/upgrade) + modelspec.py (fit/render) + remap.py (flux-conserving regrid) + io.py (.mds write/re-render) + fits.py (portable FITS I/O)
```

- [ ] **Step 7: Commit**

```bash
uv run ruff format . && uv run ruff check . --fix
uv run --extra full pytest -q
source .venv/bin/activate && git add src/pfb_model_spec/utils/modelspec.py src/pfb_model_spec/utils/degrid.py tests/test_modelspec.py tests/test_degrid.py .claude/rules/component-model.md CLAUDE.md && git commit -m "fix: conserve flux when rendering a model onto a different grid (#23)

eval_coeffs_to_slice and render_model_region now remap via remap_image
(kernel box|tent) instead of bilinear interpolation scaled by area_ratio,
which inflated flux by the area ratio when coarsening.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Expected: the full suite passes and the commit hooks pass (cabs are unchanged; no CLI was touched).

---

## Follow-ups (not in this plan)

- pfb-imaging: lift the coarsening refusal in `check_transfer_geometry` (`pfb_imaging/core/imager.py`) once a release containing this change is pinned, and consider exposing `kernel`.
- Close #23 from the PR.
