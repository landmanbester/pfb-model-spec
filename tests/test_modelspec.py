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
