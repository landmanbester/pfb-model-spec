"""Synthetic tests of the component-model fit/render library (spec 0.1).

Cubes are (..., nstokes, ny, nx). Grids are deliberately non-square so that a swapped
axis cannot pass silently.
"""

import numpy as np
import pytest
import sympy as sm
from numpy.testing import assert_allclose
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.lambdify import lambdify

from pfb_model_spec.utils.modelspec import (
    eval_coeffs_to_cube,
    eval_coeffs_to_slice,
    fit_image_cube,
    model_from_mds,
)
from pfb_model_spec.utils.remap import remap_coefficients
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


@pytest.mark.parametrize("method", ["Legendre", "poly", "fakenodes", "box", "tent"])
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


def test_single_sample_fit_ignores_a_zero_weight_plane():
    """A zero-weight Stokes plane must not make the trivial (1 time, 1 band) fit singular."""
    cube = np.zeros((1, 1, 2, 4, 6))
    cube[0, 0, 0, 1, 5] = 3.0
    cube[0, 0, 1, 3, 2] = -1.5
    time, freq = np.array([0.0]), np.array([1.4e9])
    wgt = np.array([[[1.0, 0.0]]])
    coeffs, yi, xi, expr, params, texpr, fexpr = fit_image_cube(time, freq, cube, wgt=wgt)
    image = eval_coeffs_to_cube(time, freq, 6, 4, coeffs, yi, xi, expr, params, texpr, fexpr)
    assert_allclose(image, cube, rtol=1e-12)


def _channel_render(freq, cube, chan, **fit_kwargs):
    """Fit ``(nband, ny, nx)`` and render the fit at the channel frequencies ``chan``."""
    time = np.array([0.0])
    coeffs, yi, xi, expr, params, texpr, fexpr = fit_image_cube(
        time, freq, cube[None, :, None], sigmasq=0.0, **fit_kwargs
    )
    ny, nx = cube.shape[-2:]
    return eval_coeffs_to_cube(time, chan, nx, ny, coeffs, yi, xi, expr, params, texpr, fexpr)[0, :, 0]


def test_fakenodes_does_not_ring_between_band_centres():
    """#29: a 0.1% perturbation of the band values stays small everywhere in the outer band edges.

    An interpolating polynomial amplifies it exponentially in nband (Runge); fake nodes keeps
    the gain at the Chebyshev nodes' ~log(nband). The price is an O(1/nband) bias near the
    outer edges, where the cosine map has zero slope -- hence the 1% tolerance.
    """
    nband, nsub = 32, 16
    edges = np.linspace(0.856e9, 1.712e9, nband + 1)
    freq = (edges[:-1] + edges[1:]) / 2
    chan = np.linspace(edges[0], edges[-1], nband * nsub)
    perturbation = 1 + 1e-3 * np.random.default_rng(29).standard_normal(nband)
    cube = np.zeros((nband, 3, 4))
    cube[:, 1, 2] = (freq / 1.284e9) ** -0.7 * perturbation
    truth = (chan / 1.284e9) ** -0.7

    fake = _channel_render(freq, cube, chan, method="fakenodes")[:, 1, 2]
    assert_allclose(fake, truth, rtol=1e-2)
    legendre = _channel_render(freq, cube, chan, method="Legendre")[:, 1, 2]
    assert np.abs(legendre - truth).max() > 1.0


def test_fakenodes_interpolates_without_ringing():
    """nbasisf == nband reproduces every band; a peaked spectrum barely undershoots between them."""
    nband = 12
    edges = np.linspace(0.856e9, 1.712e9, nband + 1)
    freq = (edges[:-1] + edges[1:]) / 2
    chan = np.linspace(edges[0], edges[-1], 50 * nband)
    cube = np.zeros((nband, 3, 4))
    cube[:, 2, 1] = np.exp(-(((freq - 1.1e9) / 1.5e8) ** 2))
    assert_allclose(_channel_render(freq, cube, freq, method="fakenodes"), cube, atol=1e-12)
    assert _channel_render(freq, cube, chan, method="fakenodes")[:, 2, 1].min() > -1e-3
    assert _channel_render(freq, cube, chan, method="Legendre")[:, 2, 1].min() < -0.1


def test_fakenodes_maps_band_centres_onto_chebyshev_nodes():
    """The domain is the outer band edges, from freq_bounds or half a spacing past the ends."""
    nband = 7
    edges = np.linspace(1.0e9, 1.7e9, nband + 1)
    freq = (edges[:-1] + edges[1:]) / 2
    cube = np.ones((1, nband, 1, 2, 2))
    chebyshev = -np.cos(np.pi * (2 * np.arange(nband) + 1) / (2 * nband))
    bounds = np.stack([edges[:-1], edges[1:]], axis=1)
    for freq_bounds in (None, bounds):
        *_, fexpr = fit_image_cube(np.array([0.0]), freq, cube, method="fakenodes", freq_bounds=freq_bounds)
        ffunc = lambdify(sm.Symbol("f"), parse_expr(fexpr))
        assert_allclose(ffunc(freq), chebyshev, atol=1e-12)
        assert_allclose(ffunc(edges[[0, -1]]), [-1.0, 1.0], atol=1e-12)


def test_fakenodes_rejects_misshapen_bounds():
    cube = np.ones((1, 3, 1, 2, 2))
    freq = np.array([1.0e9, 1.1e9, 1.2e9])
    with pytest.raises(ValueError, match="bounds"):
        fit_image_cube(np.array([0.0]), freq, cube, method="fakenodes", freq_bounds=np.ones((2, 2)))


def _sub_band(edges, nsub):
    """``nsub`` equal sub-cell midpoints per band: their mean of a linear piece is its band average."""
    frac = (np.arange(nsub) + 0.5) / nsub
    return (edges[:-1, None] + frac[None, :] * np.diff(edges)[:, None]).ravel()


def _histo_render(edges, cube, nsub=64, **fit_kwargs):
    """Fit band averages ``(nband, ny, nx)`` and render ``nsub`` sub-bands per band."""
    freq = (edges[:-1] + edges[1:]) / 2
    rendered = _channel_render(freq, cube, _sub_band(edges, nsub), **fit_kwargs)
    return rendered.reshape(freq.size, nsub, *cube.shape[1:])


def test_tent_reproduces_band_averages_and_keeps_their_sign():
    """Sparse, spiky, non-negative band values: exact band averages and no negative excursions."""
    nband = 10
    edges = np.linspace(0.856e9, 1.712e9, nband + 1)
    rng = np.random.default_rng(7)
    values = rng.random((nband, 2, 3)) * (rng.random((nband, 2, 3)) > 0.4)
    cube = np.concatenate([values, -values], axis=1)  # a non-negative and a non-positive plane
    sub = _histo_render(edges, cube, method="tent")
    assert_allclose(sub.mean(axis=1), cube, atol=1e-12)
    assert sub[:, :, :2].min() >= 0 and sub[:, :, 2:].max() <= 0


def test_tent_is_second_order_on_a_smooth_spectrum():
    """No edge bias to speak of: unlike fake nodes, the outer half-bands are as good as the rest."""
    nband, nsub = 16, 256
    edges = np.linspace(0.856e9, 1.712e9, nband + 1)
    chan = _sub_band(edges, nsub)
    truth = (chan / 1.284e9) ** (-0.7 - 0.3 * np.log(chan / 1.284e9))
    cube = np.zeros((nband, 2, 2))
    cube[:, 1, 0] = truth.reshape(nband, nsub).mean(axis=1)
    tent = _histo_render(edges, cube, nsub=nsub, method="tent")[:, :, 1, 0].ravel()
    fake = _histo_render(edges, cube, nsub=nsub, method="fakenodes")[:, :, 1, 0].ravel()
    assert np.abs(tent - truth).max() < 5e-4 * truth.max()
    assert np.abs(fake - truth).max() > 10 * np.abs(tent - truth).max()


def test_box_holds_each_band_value():
    edges = np.array([1.0e9, 1.1e9, 1.3e9, 1.4e9])
    cube = np.zeros((3, 2, 2))
    cube[:, 0, 1] = [1.0, 3.0, 2.0]
    bounds = np.stack([edges[:-1], edges[1:]], axis=1)
    sub = _histo_render(edges, cube, method="box", freq_bounds=bounds)
    assert_allclose(sub, np.broadcast_to(cube[:, None], sub.shape), atol=1e-12)


def test_tent_does_not_smear_an_isolated_component():
    edges = np.linspace(1.0e9, 1.8e9, 9)
    cube = np.zeros((8, 2, 2))
    cube[3, 1, 1] = 1.0
    tent = _histo_render(edges, cube, method="tent")
    assert_allclose(tent, _histo_render(edges, cube, method="box"), atol=1e-12)


def test_tent_honours_non_uniform_bounds():
    edges = np.array([0.9e9, 1.0e9, 1.25e9, 1.3e9, 1.6e9])
    cube = np.zeros((4, 2, 2))
    cube[:, 0, 0] = [4.0, 3.0, 2.5, 1.0]
    bounds = np.stack([edges[:-1], edges[1:]], axis=1)
    sub = _histo_render(edges, cube, method="tent", freq_bounds=bounds)
    assert_allclose(sub.mean(axis=1), cube, atol=1e-12)
    assert sub.min() >= 0


def test_histopolation_drops_bands_without_weight():
    """A band with zero weight in every Stokes plane is ignored; its neighbours span the gap."""
    nband = 5
    freq = np.linspace(1.0e9, 1.4e9, nband)
    cube = np.zeros((1, nband, 1, 2, 2))
    cube[0, :, 0, 1, 1] = [1.0, 1.0, 100.0, 1.0, 1.0]
    wgt = np.ones((1, nband, 1))
    wgt[0, 2] = 0.0
    out = fit_image_cube(np.array([0.0]), freq, cube, wgt=wgt, method="tent")
    chan = np.linspace(0.9e9, 1.5e9, 61)
    image = eval_coeffs_to_cube(np.array([0.0]), chan, 2, 2, *out)
    assert_allclose(image[0, :, 0, 1, 1], 1.0, atol=1e-12)


def test_histopolation_needs_a_single_time():
    cube = np.ones((2, 3, 1, 2, 2))
    with pytest.raises(NotImplementedError, match="ntime"):
        fit_image_cube(np.array([0.0, 1.0]), np.array([1.0e9, 1.1e9, 1.2e9]), cube, method="tent")


def test_tent_coefficients_can_be_remapped():
    """The parametrisation is linear in its coefficients, so the coefficient shortcut applies."""
    freq = np.linspace(1.0e9, 1.3e9, 4)
    cube = np.zeros((1, 4, 1, 4, 6))
    cube[0, :, 0, 1, 2] = [1.0, 2.0, 2.5, 2.0]
    coeffs, yi, xi, expr, params, *_ = fit_image_cube(np.array([0.0]), freq, cube, method="tent")
    out, yo, xo = remap_coefficients(coeffs, yi, xi, expr, params, **_geometry(6, 4, 1e-5))
    assert_allclose(out, coeffs)
    assert list(yo) == list(yi) and list(xo) == list(xi)


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
