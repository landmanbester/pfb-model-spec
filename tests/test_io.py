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


def test_model_to_ds_rejects_a_transposed_model(tmp_path):
    nband, nx, ny = 2, 16, 12
    freq = np.linspace(1.0e9, 2.0e9, nband)
    wrong = np.zeros((nband, 1, nx, ny))  # (nband, nstokes, nx, ny): transposed, non-square
    with pytest.raises(ValueError, match=r"\(nband, nstokes, ny, nx\)"):
        model_to_ds(
            np.array([0.0]),
            freq,
            np.ones(nband, dtype=bool),
            wrong,
            np.ones((nband, 1)),
            str(tmp_path / "x.mds"),
            1e-5,
            nx,
            ny,
            0.0,
            0.0,
            False,
            True,
            False,
            (0.0, 0.0),
            ["I"],
            "test",
        )
