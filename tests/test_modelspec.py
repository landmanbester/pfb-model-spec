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

    # build a cube of Gaussian sources with power-law spectra
    model = np.zeros((nchan, nx, ny), dtype=np.float64)
    nsource = 25
    border = np.maximum(int(0.15 * nx), int(0.15 * ny))
    x_index = np.random.randint(border, nx - border, nsource)
    y_index = np.random.randint(border, ny - border, nsource)
    alpha = -0.7 + 0.1 * np.random.randn(nsource)
    ref_flux = 1.0 + np.exp(np.random.randn(nsource))
    extentx = np.random.randint(3, int(0.1 * nx), nsource)
    extenty = np.random.randint(3, int(0.1 * nx), nsource)
    pas = (
        np.random.random(nsource) * 180
    )  # nominal degrees; gaussian2d treats as radians (matches upstream test convention)
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
    # output grid smaller than, equal to, and larger than the input grid
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
