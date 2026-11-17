"""Portable FITS I/O for component-model outputs.

A minimal, self-contained subset (astropy + numpy only) of the FITS helpers
pfb-imaging carries -- just enough to write a rendered model cube to a FITS
file with a valid celestial + spectral WCS. Deliberately excludes restoring-beam
parametrisation, CASA beam tables, and observation-time handling: a component
model has no restoring beam and no meaningful DATE-OBS. This module has no
dependency on pfb-imaging.

Axis convention: ``save_fits`` takes `(..., ny, nx)` cubes, the `.mds` spec 0.1 / FITS
row-major order, so the spatial axes are never transposed on write.
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from pfb_model_spec import __version__


def to4d(data: np.ndarray) -> np.ndarray:
    """Broadcast an array of ndim <= 4 to 4D by prepending singleton axes."""
    if data.ndim == 4:
        return data
    elif data.ndim == 2:
        return data[None, None]
    elif data.ndim == 3:
        return data[None]
    elif data.ndim == 1:
        return data[None, None, None]
    else:
        raise ValueError("Only arrays with ndim <= 4 can be broadcast to 4D.")


def save_fits(data, name, hdr, overwrite=True, dtype=np.float32):
    """Write ``data`` to a FITS image with header ``hdr``.

    Args:
        data: Model cube `(..., ny, nx)`, FITS row-major. Broadcast to 4D `(nband, nstokes, ny, nx)`.
        name: Output FITS path.
        hdr: FITS header (e.g. from ``set_wcs``).
        overwrite: Overwrite an existing file.
        dtype: On-disk data type.
    """
    hdu = fits.PrimaryHDU(header=hdr)
    # (nband, nstokes, ny, nx) -> FITS (STOKES, FREQ, DEC, RA) axis order
    data = np.transpose(to4d(data), axes=(1, 0, 2, 3))
    hdu.data = np.require(data, dtype=dtype, requirements="F")
    hdu.writeto(name, overwrite=overwrite)


def set_wcs(cell_x, cell_y, nx, ny, radec, freq, unit="Jy/pixel", ncorr=1):
    """Build a FITS header for a model cube.

    Args:
        cell_x: Pixel size along x (RA) in degrees.
        cell_y: Pixel size along y (Dec) in degrees.
        nx: Number of x pixels.
        ny: Number of y pixels.
        radec: `(ra, dec)` of the phase centre in radians.
        freq: Frequency (Hz) scalar or array of channel frequencies.
        unit: BUNIT value.
        ncorr: Number of correlation/Stokes planes (NAXIS4).

    Returns:
        A ``fits.Header``.
    """
    w = WCS(naxis=4)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ", "STOKES"]
    w.wcs.cdelt[0] = -cell_x
    w.wcs.cdelt[1] = cell_y
    w.wcs.cdelt[3] = 1
    w.wcs.cunit[0] = "deg"
    w.wcs.cunit[1] = "deg"
    w.wcs.cunit[2] = "Hz"
    w.wcs.cunit[3] = ""
    if np.size(freq) > 1:
        nchan = freq.size
        crpix3 = nchan // 2 + 1
        ref_freq = freq[crpix3 - 1]  # zero-based indexing
        w.wcs.cdelt[2] = freq[1] - freq[0]
    else:
        ref_freq = freq[0] if (isinstance(freq, np.ndarray) and freq.size == 1) else freq
        nchan = 1
        crpix3 = 1
    w.wcs.crval = [radec[0] * 180.0 / np.pi, radec[1] * 180.0 / np.pi, ref_freq, 1]
    w.wcs.crpix = [1 + nx // 2, 1 + ny // 2, crpix3, 1]
    w.wcs.equinox = 2000.0

    hdr = fits.Header()
    hdr["SIMPLE"] = (True, "conforms to FITS standard")
    hdr["BITPIX"] = (-32, "array data type")
    hdr["NAXIS"] = (4, "number of array dimensions")
    for i, size in enumerate((nx, ny, nchan, ncorr), 1):
        hdr[f"NAXIS{i}"] = (size, f"length of data axis {i}")
    hdr["EXTEND"] = True
    hdr["BSCALE"] = 1.0
    hdr["BZERO"] = 0.0
    hdr["BUNIT"] = unit
    hdr["EQUINOX"] = 2000.0
    hdr["BTYPE"] = "Intensity"
    hdr.update(w.to_header())
    hdr["RESTFRQ"] = ref_freq
    hdr["ORIGIN"] = f"pfb-model-spec: v{__version__}"
    hdr["SPECSYS"] = "TOPOCENT"
    return hdr
