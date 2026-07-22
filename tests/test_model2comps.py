"""Synthetic, self-contained test of the model2comps FITS -> .mds converter.

Builds a multi-Gaussian, power-law model cube, writes it out as WSClean-style
per-band `-####-model.fits` images, runs the converter, and asserts the fitted
`.mds` round-trips back to the input (exact for an nbasisf==nband, sigmasq==0
fit) with the expected schema. No measurement set / daskms / astropy WCS beyond
what the converter itself uses.
"""

import numpy as np
import pytest
import xarray as xr
from astropy.io import fits
from numpy.testing import assert_allclose

from pfb_model_spec.core.model2comps import model2comps
from pfb_model_spec.utils.modelspec import model_from_mds
from tests._synth import gaussian2d, give_edges


def _synth_cube(nchan, nx, ny, freq, freq0, nsource=8, seed=42):
    """Multi-Gaussian, power-law x-major (nchan, nx, ny) model cube."""
    rng = np.random.default_rng(seed)
    model = np.zeros((nchan, nx, ny), dtype=np.float64)
    border = np.maximum(int(0.2 * nx), int(0.2 * ny))
    x_index = rng.integers(border, nx - border, nsource)
    y_index = rng.integers(border, ny - border, nsource)
    alpha = -0.7 + 0.1 * rng.standard_normal(nsource)
    ref_flux = 1.0 + np.exp(rng.standard_normal(nsource))
    extentx = rng.integers(3, int(0.1 * nx) + 3, nsource)
    extenty = rng.integers(3, int(0.1 * ny) + 3, nsource)
    pas = rng.random(nsource) * 180
    x = -(nx / 2) + np.arange(nx)
    y = -(ny / 2) + np.arange(ny)
    xin, yin = np.meshgrid(x, y, indexing="ij")
    for i in range(nsource):
        emaj = np.maximum(extentx[i], extenty[i])
        emin = np.minimum(extentx[i], extenty[i])
        gauss = gaussian2d(xin, yin, gausspar=(emaj, emin, pas[i]))
        mx, my, gx, gy = give_edges(x_index[i], y_index[i], nx, ny, nx, ny)
        spectrum = ref_flux[i] * (freq / freq0) ** alpha[i]
        model[:, mx, my] += spectrum[:, None, None] * gauss[None, gx, gy]
    return model


def _write_wsclean_fits(prefix, model, freq, cell_deg, ra_deg, dec_deg, wsums):
    """Write an x-major (nchan, nx, ny) cube as WSClean `-####-model.fits` planes."""
    nchan, nx, ny = model.shape
    for b in range(nchan):
        # FITS data is row-major (ny, nx); the converter transposes back on read.
        hdu = fits.PrimaryHDU(data=model[b].T.astype(np.float32))
        hdr = hdu.header
        hdr["CRVAL1"] = ra_deg
        hdr["CRVAL2"] = dec_deg
        hdr["CRVAL3"] = float(freq[b])
        hdr["CDELT1"] = cell_deg
        hdr["CDELT2"] = -cell_deg  # WSClean RA increment is negative; reader takes abs
        hdr["WSCVWSUM"] = float(wsums[b])
        hdu.writeto(f"{prefix}-{b:04d}-model.fits", overwrite=True)


def test_model2comps_fits_roundtrip(tmp_path):
    nchan = 6
    nx, ny = 40, 32
    freq = np.linspace(1.0e9, 2.0e9, nchan)
    freq0 = float(freq.mean())
    cell_deg = 2.5 / 3600.0
    ra_deg, dec_deg = 10.0, -30.0
    wsums = np.linspace(1.0, 2.0, nchan)

    model = _synth_cube(nchan, nx, ny, freq, freq0)
    prefix = str(tmp_path / "wsclean")
    _write_wsclean_fits(prefix, model, freq, cell_deg, ra_deg, dec_deg, wsums)

    coeff_name = str(tmp_path / "out.mds")
    # nbasisf == nchan and use_wsum off with sigmasq 0 make the Legendre fit exact.
    model2comps(
        str(tmp_path / "out"),
        from_fits=prefix,
        model_out=coeff_name,
        nbasisf=nchan,
        sigmasq=0.0,
        use_wsum=False,
        product="I",
    )

    # schema
    mds = xr.open_zarr(coeff_name, chunks=None)
    assert mds.attrs["spec"] == "genesis"
    assert mds.attrs["npix_x"] == nx
    assert mds.attrs["npix_y"] == ny
    assert_allclose(mds.attrs["cell_rad_x"], np.deg2rad(cell_deg))
    assert mds.attrs["stokes"] == "I"
    assert mds.attrs["flip_v"] is True
    assert mds.attrs["flip_u"] is False
    assert_allclose(mds.attrs["ra"], np.deg2rad(ra_deg))
    assert_allclose(mds.attrs["dec"], np.deg2rad(dec_deg))
    assert_allclose(mds.freqs.values, freq)

    # numerical round-trip: exact fit reproduces the input on the source mask
    rendered = model_from_mds(coeff_name)[0]  # (nchan, nx, ny)
    assert rendered.shape == model.shape
    mask = model > 0
    assert_allclose(rendered[mask], model[mask], atol=1e-8)

    # the sanity-render FITS was written at the mds location
    assert (tmp_path / "out.fits").exists()


def test_model2comps_overwrite_guard(tmp_path):
    nchan = 4
    nx, ny = 24, 24
    freq = np.linspace(1.0e9, 1.5e9, nchan)
    model = _synth_cube(nchan, nx, ny, freq, float(freq.mean()))
    prefix = str(tmp_path / "wsclean")
    _write_wsclean_fits(prefix, model, freq, 2.5 / 3600.0, 0.0, -30.0, np.ones(nchan))
    coeff_name = str(tmp_path / "out.mds")

    model2comps(str(tmp_path / "out"), from_fits=prefix, model_out=coeff_name, nbasisf=nchan, sigmasq=0.0)
    # a second run without overwrite must refuse
    with pytest.raises(ValueError, match="exists"):
        model2comps(str(tmp_path / "out"), from_fits=prefix, model_out=coeff_name, nbasisf=nchan, sigmasq=0.0)
    # ... and succeed with overwrite=True
    model2comps(
        str(tmp_path / "out"), from_fits=prefix, model_out=coeff_name, nbasisf=nchan, sigmasq=0.0, overwrite=True
    )


def test_model2comps_no_images(tmp_path):
    with pytest.raises(ValueError, match="No images"):
        model2comps(str(tmp_path / "out"), from_fits=str(tmp_path / "does-not-exist"))
