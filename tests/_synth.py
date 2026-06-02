"""Pure-numpy helpers vendored from pfb_imaging.utils.misc for building
synthetic model cubes in tests. Not part of the public API."""

import numpy as np


def give_edges(p, q, nx, ny, nx_psf, ny_psf):
    nx0 = nx_psf // 2
    ny0 = ny_psf // 2

    # image overlap edges
    # left edge for x coordinate
    dxl = p - nx0
    xl = np.maximum(dxl, 0)

    # right edge for x coordinate
    dxu = p + nx0
    xu = np.minimum(dxu, nx)
    # left edge for y coordinate
    dyl = q - ny0
    yl = np.maximum(dyl, 0)
    # right edge for y coordinate
    dyu = q + ny0
    yu = np.minimum(dyu, ny)

    # PSF overlap edges
    xlpsf = np.maximum(nx0 - p, 0)
    xupsf = np.minimum(nx0 + nx - p, nx_psf)
    ylpsf = np.maximum(ny0 - q, 0)
    yupsf = np.minimum(ny0 + ny - q, ny_psf)

    return slice(xl, xu), slice(yl, yu), slice(xlpsf, xupsf), slice(ylpsf, yupsf)


def gaussian2d(xin, yin, gausspar=(1.0, 1.0, 0.0), normalise=True, nsigma=5):
    """
    xin         - grid of x coordinates
    yin         - grid of y coordinates
    gausspar    - (emaj, emin, pa) with emaj/emin as FWHM in units of xin/yin and pa in radians.
    normalise   - normalise kernel to have volume 1
    nsigma      - compute kernel out to this many standard deviations of the major axis
    """
    smaj, smin, pa = gausspar
    fwhm_conv = 2 * np.sqrt(2 * np.log(2))
    amat = np.array([[1.0 / smaj**2, 0], [0, 1.0 / smin**2]])
    # this parametrisation is equivalent to a standard rotation matrix with
    # t = np.pi/2 + pa; used for compatibility with fits
    rmat = np.array([[-np.sin(pa), -np.cos(pa)], [np.cos(pa), -np.sin(pa)]])
    amat = np.dot(np.dot(rmat, amat), rmat.T)
    sout = xin.shape
    # only compute the result out to nsigma standard deviations
    sigma_maj = smaj / fwhm_conv
    extent = (nsigma * sigma_maj) ** 2
    xflat = xin.squeeze()
    yflat = yin.squeeze()
    idx, idy = np.where(xflat**2 + yflat**2 <= extent)
    x = np.array([xflat[idx, idy].ravel(), yflat[idx, idy].ravel()])
    rmat = np.einsum("nb,bc,cn->n", x.T, amat, x)
    # adjust for the fact that gausspar corresponds to FWHM
    tmp = np.exp(-0.5 * fwhm_conv**2 * rmat)
    gausskern = np.zeros(xflat.shape, dtype=np.float64)
    gausskern[idx, idy] = tmp

    if normalise:
        gausskern /= np.sum(gausskern)
    return np.ascontiguousarray(gausskern.reshape(sout), dtype=np.float64)
