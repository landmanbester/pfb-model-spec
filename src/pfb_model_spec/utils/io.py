"""Component-model I/O: fit a model cube, write it to a `.mds`, and re-render it.

Geometry conventions (flips, phase-centre offsets) are a pfb-imaging/gridder concern
(`wgridder_conventions`) and are deliberately taken as arguments here rather than
computed, so this module has no dependency on pfb-imaging.

The current (`"0.1"`) `.mds` spec stores coefficients as `(stokes, par, comps)`, and every
model cube passed in or returned here is `(nband, nstokes, ny, nx)` -- the FITS/astropy
`(Y, X)` order, so callers no longer transpose. Older `.mds` datasets are upgraded on read
(see `pfb_model_spec.utils.spec`).
"""

import numpy as np
import xarray as xr

from pfb_model_spec.utils.modelspec import eval_coeffs_to_slice, fit_image_cube
from pfb_model_spec.utils.spec import SPEC_VERSION


def build_mds_dataset(
    coeffs: np.ndarray,
    y_index: np.ndarray,
    x_index: np.ndarray,
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
    stokes: list[str],
    writer_version: str,
    *,
    freq_bounds: np.ndarray | None = None,
    time_bounds: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    flux_scale: str | None = None,
) -> xr.Dataset:
    """Assemble a `.mds` dataset at the current spec (``SPEC_VERSION``) from fitted coefficients.

    Single owner of the `.mds` schema: ``model_to_ds`` and the ``model2comps`` converter
    both build through here. The ``genesis -> 0.1`` step in ``utils/spec.py`` must produce
    the same schema; ``tests/test_io.py`` checks that.

    Args:
        coeffs: Fitted coefficients, shape ``(nstokes, npar, ncomps)``.
        y_index: Component row (y) pixel locations, shape ``(ncomps,)``.
        x_index: Component column (x, FITS ``NAXIS1``) pixel locations, shape ``(ncomps,)``.
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
        stokes: Stokes product per ``coeffs`` plane, e.g. ``["I", "V"]``.
        writer_version: pfb-model-spec (or caller) version recorded as ``writer-version``.
        freq_bounds: Optional ``(nfreq, 2)`` ``[lo, hi]`` in Hz that each band represents.
        time_bounds: Optional ``(ntime, 2)`` ``[lo, hi]`` that each time represents.
        weights: Optional ``(nstokes, ntime, nfreq)`` fit weights; 0 means no data.
        flux_scale: Optional ``"intrinsic"`` or ``"apparent"``.

    Returns:
        The `.mds` ``xarray.Dataset`` (not yet written to disk).

    Raises:
        ValueError: If ``stokes`` does not match ``coeffs``'s leading axis, if
            ``freq_bounds``, ``time_bounds`` or ``weights`` have the wrong shape, or if
            ``flux_scale`` is not ``"intrinsic"`` or ``"apparent"``.
    """
    stokes = list(stokes)
    if len(stokes) != coeffs.shape[0]:
        raise ValueError(f"stokes has {len(stokes)} entries but coeffs has {coeffs.shape[0]} planes")
    data_vars = {"coefficients": (("stokes", "par", "comps"), coeffs)}
    coords = {
        "stokes": (("stokes",), stokes),
        "location_x": (("comps",), x_index),
        "location_y": (("comps",), y_index),
        "params": (("par",), params),
        "times": (("t",), time),
        "freqs": (("f",), freq),
    }
    attrs = {
        "writer-version": writer_version,
        "spec": SPEC_VERSION,
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
        "parametrisation": expr,
    }
    nt, nf, ns = len(time), len(freq), len(stokes)
    if freq_bounds is not None:
        freq_bounds = np.asarray(freq_bounds, dtype=np.float64)
        if freq_bounds.shape != (nf, 2):
            raise ValueError(f"freq_bounds must have shape (nfreq, 2) = ({nf}, 2), got {freq_bounds.shape}")
        coords["freq_bounds"] = (("f", "bound"), freq_bounds)
    if time_bounds is not None:
        time_bounds = np.asarray(time_bounds, dtype=np.float64)
        if time_bounds.shape != (nt, 2):
            raise ValueError(f"time_bounds must have shape (ntime, 2) = ({nt}, 2), got {time_bounds.shape}")
        coords["time_bounds"] = (("t", "bound"), time_bounds)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (ns, nt, nf):
            raise ValueError(
                f"weights must have shape (nstokes, ntime, nfreq) = ({ns}, {nt}, {nf}), got {weights.shape}"
            )
        data_vars["weights"] = (("stokes", "t", "f"), weights)
    if flux_scale is not None:
        if flux_scale not in ("intrinsic", "apparent"):
            raise ValueError(f"flux_scale must be 'intrinsic' or 'apparent', got {flux_scale!r}")
        attrs["flux_scale"] = flux_scale
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


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
    stokes: list[str],
    writer_version: str,
    nbasisf: int | None = None,
    method: str = "Legendre",
    sigmasq: float = 1e-6,
    *,
    freq_bounds: np.ndarray | None = None,
    time_bounds: np.ndarray | None = None,
    flux_scale: str | None = None,
) -> np.ndarray:
    """Fit a model cube to the component model, write it to a `.mds`, and re-render it.

    Fits `model[fsel]` over time and frequency, writes the resulting coefficients to
    `mds_name` (zarr, overwriting any existing dataset), then re-evaluates the fit at
    every frequency in `freq` (not just the fitted `fsel` subset) to produce a model
    cube consistent with the stored component model.

    `model` (in) and the returned cube (out) both follow the spec 0.1
    `(nband, nstokes, ny, nx)` axis order (FITS/astropy `(Y, X)`) -- see the module
    docstring.

    Args:
        time: Time axis, shape `(ntime,)`.
        freq: Full frequency axis, shape `(nband,)`.
        fsel: Boolean mask over `freq`/`model` selecting the bands to fit.
        model: Model cube, shape `(nband, nstokes, ny, nx)`.
        wgt: Per-band, per-Stokes fit weight, shape `(nband, nstokes)`; only `wgt[fsel]` is used.
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
        stokes: Stokes product list, e.g. ``["I", "V"]``.
        writer_version: pfb-model-spec (or caller) version to record in the `.mds` attrs.
        nbasisf: Number of frequency basis functions; defaults to `fit_image_cube`'s
            own default (number of fitted bands) when `None`.
        method: Basis for the frequency fit, forwarded to `fit_image_cube`.
        sigmasq: Regularisation, forwarded to `fit_image_cube`.
        freq_bounds: Optional ``(nband, 2)`` band edges in Hz, forwarded to `build_mds_dataset`.
        time_bounds: Optional ``(ntime, 2)`` time edges, forwarded to `build_mds_dataset`.
        flux_scale: Optional ``"intrinsic"`` or ``"apparent"``, forwarded to `build_mds_dataset`.

    The `.mds` always records ``weights`` from `wgt`, shape ``(nstokes, ntime, nband)``,
    zero outside `fsel`.

    Returns:
        The model cube re-rendered from the fitted coefficients, shape
        `(nband, nstokes, ny, nx)`.
    """
    if model.ndim != 4 or model.shape[-2:] != (ny, nx):
        raise ValueError(
            f"model must have shape (nband, nstokes, ny, nx) = (nband, nstokes, {ny}, {nx}), got {model.shape}"
        )
    if len(stokes) != model.shape[1]:
        raise ValueError(
            f"len(stokes) = {len(stokes)} does not match model.shape[1] = {model.shape[1]} "
            "(expected layout (nband, nstokes, ny, nx))"
        )
    coeffs, y_index, x_index, expr, params, texpr, fexpr = fit_image_cube(
        time,
        freq[fsel],
        model[None, fsel],
        wgt=wgt[None, fsel],
        nbasisf=nbasisf,
        method=method,
        sigmasq=sigmasq,
    )

    nstokes = model.shape[1]
    # the fit's own weights over every band (zero where not fitted), (stokes, t, f)
    weights = np.zeros((nstokes, time.size, freq.size))
    weights[:, :, fsel] = np.asarray(wgt)[fsel].T[:, None, :]
    coeff_dataset = build_mds_dataset(
        coeffs,
        y_index,
        x_index,
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
        writer_version,
        freq_bounds=freq_bounds,
        time_bounds=time_bounds,
        weights=weights,
        flux_scale=flux_scale,
    )
    coeff_dataset.to_zarr(mds_name, mode="w")

    nband, nstokes = model.shape[:2]
    new_model = np.zeros((nband, nstokes, ny, nx), dtype=model.dtype)
    for b in range(nband):
        new_model[b] = eval_coeffs_to_slice(
            time[0],
            freq[b],
            coeffs,
            y_index,
            x_index,
            expr,
            params,
            texpr,
            fexpr,
            nxi=nx,
            nyi=ny,
            cellxi=cell_rad,
            cellyi=cell_rad,
            x0i=x0,
            y0i=y0,
            nxo=nx,
            nyo=ny,
            cellxo=cell_rad,
            cellyo=cell_rad,
            x0o=x0,
            y0o=y0,
        )
    return new_model
