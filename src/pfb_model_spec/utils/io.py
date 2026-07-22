"""Component-model I/O: fit a model cube, write it to a `.mds`, and re-render it.

Geometry conventions (flips, phase-centre offsets) are a pfb-imaging/gridder concern
(`wgridder_conventions`) and are deliberately taken as arguments here rather than
computed, so this module has no dependency on pfb-imaging.

The current (`"genesis"`) `.mds` spec is **x-major**: model cubes passed in and returned
here are `(nband, nx, ny)`-ordered, matching `location_x`/`location_y`. This is *not*
pfb-imaging's internal `(Y, X)` raster convention -- callers on a `(Y, X)` cube must
transpose at the call site. A future spec revision is expected to make the `.mds` format
itself `(Y, X)`-ordered, with conversion handled by the planned `model2comps` converter
(see https://github.com/landmanbester/pfb-model-spec/issues/17); this module will be
updated to match when that lands.
"""

import numpy as np
import xarray as xr

from pfb_model_spec.utils.modelspec import eval_coeffs_to_slice, fit_image_cube


def build_mds_dataset(
    coeffs: np.ndarray,
    x_index: np.ndarray,
    y_index: np.ndarray,
    expr: str,
    params: list,
    texpr: str,
    fexpr: str,
    time: np.ndarray,
    freq: np.ndarray,
    cell_rad: float,
    nx: int,
    ny: int,
    x0: float,
    y0: float,
    flip_u: bool,
    flip_v: bool,
    flip_w: bool,
    radec: tuple[float, float],
    stokes: str,
    version: str,
) -> xr.Dataset:
    """Assemble a `.mds` (``"genesis"`` spec) dataset from fitted coefficients.

    Single owner of the `.mds` schema: both ``model_to_ds`` and the ``model2comps``
    converter build their datasets through here so the coord/attr field names cannot
    drift between the two write paths (``model_from_mds`` reads the same schema).

    Args:
        coeffs: Fitted coefficients, dims `(par, comps)`.
        x_index: Component x pixel locations.
        y_index: Component y pixel locations.
        expr: Symbolic parametrisation (already stringified).
        params: Parameter names for the coefficient axis.
        texpr: Time scaling expression.
        fexpr: Frequency scaling expression.
        time: Time axis, shape `(ntime,)`.
        freq: Frequency axis, shape `(nband,)`.
        cell_rad: Pixel size in radians (assumed square).
        nx: Number of pixels along x.
        ny: Number of pixels along y.
        x0: Phase-centre x offset (`wgridder_conventions`).
        y0: Phase-centre y offset (`wgridder_conventions`).
        flip_u: U-axis flip convention.
        flip_v: V-axis flip convention.
        flip_w: W-axis flip convention.
        radec: `(ra, dec)` in radians.
        stokes: Stokes/correlation product, e.g. `"I"`.
        version: pfb-imaging/model-spec version to record in the attrs.

    Returns:
        The `.mds` ``xarray.Dataset`` (not yet written to disk).
    """
    return xr.Dataset(
        data_vars={"coefficients": (("par", "comps"), coeffs)},
        coords={
            "location_x": (("x",), x_index),
            "location_y": (("y",), y_index),
            "params": (("par",), params),
            "times": (("t",), time),
            "freqs": (("f",), freq),
        },
        attrs={
            "pfb-imaging-version": version,
            "spec": "genesis",
            "cell_rad_x": cell_rad,
            "cell_rad_y": cell_rad,
            "npix_x": nx,
            "npix_y": ny,
            "texpr": texpr,
            "fexpr": fexpr,
            "center_x": x0,
            "center_y": y0,
            "flip_u": flip_u,
            "flip_v": flip_v,
            "flip_w": flip_w,
            "ra": radec[0],
            "dec": radec[1],
            "stokes": stokes,
            "parametrisation": expr,
        },
    )


def model_to_ds(
    time: np.ndarray,
    freq: np.ndarray,
    fsel: np.ndarray,
    model: np.ndarray,
    wgt: np.ndarray,
    mds_name: str,
    cell_rad: float,
    nx: int,
    ny: int,
    x0: float,
    y0: float,
    flip_u: bool,
    flip_v: bool,
    flip_w: bool,
    radec: tuple[float, float],
    stokes: str,
    version: str,
    nbasisf: int | None = None,
    method: str = "Legendre",
    sigmasq: float = 1e-6,
) -> np.ndarray:
    """Fit a model cube to the component model, write it to a `.mds`, and re-render it.

    Fits `model[fsel]` over time and frequency, writes the resulting coefficients to
    `mds_name` (zarr, overwriting any existing dataset), then re-evaluates the fit at
    every frequency in `freq` (not just the fitted `fsel` subset) to produce a model
    cube consistent with the stored component model.

    `model` (in) and the returned cube (out) both follow the current `.mds` spec's
    x-major `(nband, nx, ny)` axis order -- see the module docstring. Callers whose
    model cube is `(nband, ny, nx)`-ordered (e.g. pfb-imaging's internal convention)
    must transpose to/from `(nband, nx, ny)` themselves before/after calling this.

    Args:
        time: Time axis, shape `(ntime,)`.
        freq: Full frequency axis, shape `(nband,)`.
        fsel: Boolean mask over `freq`/`model` selecting the bands to fit.
        model: Model cube, shape `(nband, nx, ny)`.
        wgt: Per-band fit weight, shape `(nband,)`; only `wgt[fsel]` is used.
        mds_name: Output `.mds` (zarr) path.
        cell_rad: Pixel size in radians (assumed square).
        nx: Number of pixels along x.
        ny: Number of pixels along y.
        x0: Phase-centre x offset (`wgridder_conventions`).
        y0: Phase-centre y offset (`wgridder_conventions`).
        flip_u: U-axis flip convention.
        flip_v: V-axis flip convention.
        flip_w: W-axis flip convention.
        radec: `(ra, dec)` in radians.
        stokes: Stokes/correlation product, e.g. `"I"`.
        version: pfb-imaging version to record in the `.mds` attrs.
        nbasisf: Number of frequency basis functions; defaults to `fit_image_cube`'s
            own default (number of fitted bands) when `None`.
        method: Basis for the frequency fit, forwarded to `fit_image_cube`.
        sigmasq: Regularisation, forwarded to `fit_image_cube`.

    Returns:
        The model cube re-rendered from the fitted coefficients, shape
        `(nband, nx, ny)`.
    """
    coeffs, x_index, y_index, expr, params, texpr, fexpr = fit_image_cube(
        time,
        freq[fsel],
        model[None, fsel, :, :],
        wgt=wgt[None, fsel],
        nbasisf=nbasisf,
        method=method,
        sigmasq=sigmasq,
    )

    coeff_dataset = build_mds_dataset(
        coeffs,
        x_index,
        y_index,
        expr,
        params,
        texpr,
        fexpr,
        time,
        freq,
        cell_rad,
        nx,
        ny,
        x0,
        y0,
        flip_u,
        flip_v,
        flip_w,
        radec,
        stokes,
        version,
    )
    coeff_dataset.to_zarr(mds_name, mode="w")

    nband = freq.size
    new_model = np.zeros((nband, nx, ny), dtype=model.dtype)
    for b in range(nband):
        new_model[b] = eval_coeffs_to_slice(
            time[0],
            freq[b],
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
    return new_model
