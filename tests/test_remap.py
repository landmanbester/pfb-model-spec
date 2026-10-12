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
