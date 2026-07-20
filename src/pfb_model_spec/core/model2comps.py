"""Convert a pixelated WSClean-style FITS model cube into a `.mds` component model.

Portable image -> `.mds` converter: reads a per-band WSClean `*-model.fits` cube, fits
the Legendre/polynomial component model over frequency (via ``fit_image_cube``), writes
the coefficients to a `.mds` zarr, re-renders the fit to report the interpolation error,
and optionally writes a rendered model FITS cube. Depends only on the portable scientific
stack (numpy / xarray / astropy) plus this package's own spec library -- it has **no**
dependency on pfb-imaging and no `.dds`/daskms coupling (the legacy `.dds`-input path was
dropped in the migration; deconvolvers write `.mds` directly via ``io.model_to_ds``).

Axis convention: the `.mds` spec is x-major `(nband, nx, ny)`. WSClean FITS data is
row-major `(ny, nx)`, so each plane is transposed on read.
"""

import logging
import os
import shutil
from glob import glob

import numpy as np

from pfb_model_spec import __version__
from pfb_model_spec.utils.fits import save_fits, set_wcs
from pfb_model_spec.utils.io import build_mds_dataset
from pfb_model_spec.utils.modelspec import eval_coeffs_to_slice, fit_image_cube

log = logging.getLogger("pfb_model_spec.model2comps")


def read_wsclean_model(from_fits: str) -> dict:
    """Read a WSClean-style per-band model cube (`{from_fits}-####-model.fits`).

    Args:
        from_fits: Prefix of the WSClean model images; the glob
            ``{from_fits}-[0-9][0-9][0-9][0-9]-model.fits`` selects the per-band planes.

    Returns:
        Dict with ``model`` (x-major `(nband, nx, ny)`), ``freqs`` (`(nband,)` Hz),
        ``wsums`` (`(nband,)`), ``cell_deg``, ``nx``, ``ny``, ``ra``/``dec`` (radians).

    Raises:
        ValueError: No matching images, or inconsistent geometry across bands.
    """
    from astropy.io import fits  # deferred: keeps astropy off the import path of lightweight consumers

    images = sorted(glob(f"{from_fits}-[0-9][0-9][0-9][0-9]-model.fits"), key=os.path.getctime)
    if not images:
        raise ValueError(f"No images found matching {from_fits}-####-model.fits")

    planes, freqs, wsums = [], [], []
    cellx = celly = nx = ny = ra = dec = None
    for image in images:
        log.info(f"Loading {image}")
        with fits.open(image) as hdu:
            hdr = hdu[0].header
            planes.append(hdu[0].data.squeeze().T)  # (ny, nx) row-major -> (nx, ny) x-major
            freqs.append(hdr["CRVAL3"])
            if "WSCVWSUM" in hdr:
                wsums.append(hdr["WSCVWSUM"])
            elif "WSCIMGWG" in hdr:
                wsums.append(hdr["WSCIMGWG"])
            else:
                wsums.append(1.0)
            geom = (
                np.abs(hdr["CDELT2"]),
                np.abs(hdr["CDELT1"]),
                hdr["NAXIS1"],
                hdr["NAXIS2"],
                hdr["CRVAL1"],
                hdr["CRVAL2"],
            )
            if cellx is None:
                cellx, celly, nx, ny, ra, dec = geom
            elif geom != (cellx, celly, nx, ny, ra, dec):
                raise ValueError(f"Geometry of {image} does not match the first band")

    # isclose, not ==: a FITS card stores CDELT2 with one fewer significant digit
    # when it carries WSClean's negative RA sign, so exact equality spuriously fails.
    if not np.isclose(cellx, celly):
        raise ValueError(f"Non-square pixels are not supported (CDELT1={celly}, CDELT2={cellx})")

    freqs = np.array(freqs, dtype=np.float64)
    order = np.argsort(freqs)
    model = np.stack([planes[i] for i in order]).astype(np.float64)
    return {
        "model": model,
        "freqs": freqs[order],
        "wsums": np.array(wsums, dtype=np.float64)[order],
        "cell_deg": float(cellx),
        "nx": int(nx),
        "ny": int(ny),
        "ra": np.deg2rad(ra),
        "dec": np.deg2rad(dec),
    }


def model2comps(
    output_filename: str,
    from_fits: str,
    overwrite: bool = False,
    nbasisf: int | None = None,
    fit_mode: str = "Legendre",
    min_val: float | None = None,
    suffix: str = "main",
    model_name: str = "MODEL",
    use_wsum: bool = True,
    sigmasq: float = 1e-10,
    model_out: str | None = None,
    out_freqs: str | None = None,
    product: str = "I",
    fits_output_folder: str | None = None,
):
    """Fit a WSClean FITS model cube to a `.mds` component model.

    Args:
        output_filename: Basename for derived output names.
        from_fits: WSClean model prefix (see ``read_wsclean_model``).
        overwrite: Overwrite an existing `.mds`.
        nbasisf: Frequency basis order; ``nband - 1`` by default.
        fit_mode: Basis for the frequency fit (e.g. ``"Legendre"``).
        min_val: Zero out model pixels below this flux before fitting.
        suffix: Naming suffix for the derived `.mds`/FITS names.
        model_name: Model label used in the derived names.
        use_wsum: Weight the fit by the per-band WSClean weights.
        sigmasq: Tikhonov regularisation added to the fit Hessian.
        model_out: Explicit `.mds` path, overriding the derived name.
        out_freqs: ``flow:fhigh:step`` (Hz) to render the model FITS onto; renders
            at the input band frequencies when omitted.
        product: Stokes/correlation product recorded in the `.mds`.
        fits_output_folder: Directory for the rendered model FITS (cwd if omitted).

    Raises:
        ValueError: Empty model, or the `.mds` exists and ``overwrite`` is False.
        numpy.linalg.LinAlgError: Singular fit Hessian (e.g. empty sub-bands).
    """
    if not log.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    coeff_name = model_out or f"{output_filename}_{product}_{suffix}_{model_name.lower()}.mds"
    fits_dir = fits_output_folder or os.path.dirname(coeff_name) or "."
    fits_name = os.path.join(fits_dir, os.path.basename(coeff_name).removesuffix(".mds") + ".fits")

    if os.path.exists(coeff_name):
        if overwrite:
            log.info(f"Overwriting {coeff_name}")
            shutil.rmtree(coeff_name)
        else:
            raise ValueError(f"{coeff_name} exists. Set overwrite=True to replace it.")

    cube = read_wsclean_model(from_fits)
    model = cube["model"]  # (nband, nx, ny)
    mfreqs = cube["freqs"]
    wsums = cube["wsums"]
    nx, ny = cube["nx"], cube["ny"]
    cell_deg = cube["cell_deg"]
    cell_rad = np.deg2rad(cell_deg)
    radec = (cube["ra"], cube["dec"])
    x0 = y0 = 0.0
    # WSClean image-plane flip convention (matches pfb-imaging's FITS-input path)
    flip_u, flip_v, flip_w = False, True, False
    nband = mfreqs.size
    time = np.ones((1,), dtype=np.float64)

    if not use_wsum:
        wsums = np.ones_like(wsums)
    else:
        # normalise so the ridge parameter has a more intuitive meaning
        wsums = wsums / wsums.max()

    if min_val is not None:
        model = np.where(model > min_val, model, 0.0)
    if not np.any(model):
        raise ValueError("Model is empty" if min_val is None else f"Model has no components above {min_val}")

    if nbasisf is None:
        nbasisf = nband - 1
    fsel = np.ones(nband, dtype=bool)

    log.info(f"Fitting {nband} bands with {nbasisf} basis functions")
    try:
        coeffs, x_index, y_index, expr, params, texpr, fexpr = fit_image_cube(
            time,
            mfreqs[fsel],
            model[None, fsel],
            wgt=wsums[None, fsel],
            nbasisf=nbasisf,
            method=fit_mode,
            sigmasq=sigmasq,
        )
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError(f"Fit failed ({e}). Empty sub-bands? Try decreasing nbasisf.") from e

    coeff_dataset = build_mds_dataset(
        coeffs,
        x_index,
        y_index,
        expr,
        params,
        texpr,
        fexpr,
        time,
        mfreqs,
        cell_rad,
        nx,
        ny,
        x0,
        y0,
        flip_u,
        flip_v,
        flip_w,
        radec,
        product,
        __version__,
    )
    log.info(f"Writing component model to {coeff_name}")
    coeff_dataset.to_zarr(coeff_name, mode="w")

    # re-render at the input bands to report the fit's interpolation error
    modelo = np.stack(
        [
            _render(coeffs, x_index, y_index, expr, params, texpr, fexpr, time[0], mfreqs[b], nx, ny, cell_rad, x0, y0)
            for b in range(nband)
        ]
    )
    denom = np.linalg.norm(model.ravel())
    if denom > 0:
        log.info(f"Fractional interpolation error is {np.linalg.norm((modelo - model).ravel()) / denom:.3e}")

    # optional rendered model FITS (a sanity check on the fit)
    if out_freqs is not None:
        flow, fhigh, step = map(float, out_freqs.split(":"))
        freq_out = np.linspace(flow, fhigh, int((fhigh - flow) / step))
        log.info(f"Rendering model cube to {freq_out.size} output bands")
        modelo = np.stack(
            [
                _render(coeffs, x_index, y_index, expr, params, texpr, fexpr, time[0], f, nx, ny, cell_rad, x0, y0)
                for f in freq_out
            ]
        )
    else:
        freq_out = mfreqs

    hdr = set_wcs(cell_deg, cell_deg, nx, ny, radec, freq_out, unit="Jy/pixel")
    os.makedirs(fits_dir, exist_ok=True)
    log.info(f"Writing rendered model to {fits_name}")
    save_fits(modelo[:, None, :, :], fits_name, hdr, overwrite=True)


def _render(coeffs, x_index, y_index, expr, params, texpr, fexpr, t, f, nx, ny, cell_rad, x0, y0):
    """Render coefficients to a single `(nx, ny)` slice at time ``t``, frequency ``f``.

    Output grid matches the fit grid (same npix/cell/centre), so this is the
    identity resample used for the interpolation-error and sanity-FITS renders.
    """
    return eval_coeffs_to_slice(
        t,
        f,
        coeffs,
        x_index,
        y_index,
        expr,
        params,
        texpr,
        fexpr,
        nx,
        ny,
        cell_rad,
        cell_rad,
        x0,
        y0,
        nx,
        ny,
        cell_rad,
        cell_rad,
        x0,
        y0,
    )
