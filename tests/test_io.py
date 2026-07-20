"""Synthetic, self-contained test of pfb_model_spec.utils.io.model_to_ds."""

import numpy as np
import xarray as xr
from numpy.testing import assert_allclose

from pfb_model_spec.utils.io import model_to_ds
from tests._synth import gaussian2d, give_edges


def test_model_to_ds_roundtrip(tmp_path):
    np.random.seed(420)

    nchan = 8
    freq = np.linspace(1.0e9, 2.0e9, nchan)
    freq0 = float(freq.mean())

    cell_rad = np.deg2rad(2.5 / 3600.0)
    nx = 64
    ny = 64

    # x-major (nchan, nx, ny), matching model_to_ds's .mds convention directly --
    # no transpose needed here (that's the caller's job on a (Y, X) cube).
    model = np.zeros((nchan, nx, ny), dtype=np.float64)
    nsource = 10
    border = np.maximum(int(0.15 * nx), int(0.15 * ny))
    x_index = np.random.randint(border, nx - border, nsource)
    y_index = np.random.randint(border, ny - border, nsource)
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

    time = np.array([0.0])
    fsel = np.ones(nchan, dtype=bool)
    wgt = np.ones(nchan)
    mds_name = str(tmp_path / "test.mds")

    # nbasisf == nband and ntime == 1 make the Legendre fit exact, so the
    # rendered model must match the input to ~machine precision.
    new_model = model_to_ds(
        time,
        freq,
        fsel,
        model,
        wgt,
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
        "I",
        "test-version",
        nbasisf=nchan,
        sigmasq=0.0,
    )

    assert new_model.shape == model.shape
    mask = model > 0
    assert_allclose(new_model[mask], model[mask], atol=1e-8)

    mds = xr.open_zarr(mds_name, chunks=None)
    assert mds.attrs["npix_x"] == nx
    assert mds.attrs["npix_y"] == ny
    assert mds.attrs["cell_rad_x"] == cell_rad
    assert mds.attrs["stokes"] == "I"
    assert mds.attrs["pfb-imaging-version"] == "test-version"
    assert_allclose(mds.freqs.values, freq)
    assert_allclose(mds.times.values, time)
