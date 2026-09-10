"""Render a component model to visibilities for a chunk of data.

Pure numpy: no dask, no Ray, no measurement-set handles. Data selection,
chunking, distribution and I/O belong to the calling application
(pfb-imaging's `degrid-msv4`, QuartiCal's predict); this module only turns a
model plus a chunk's `(uvw, freq)` into visibilities.

Axis convention: **x-major**, `(nx, ny)`, matching the `genesis` `.mds` spec and
ducc0's `dirty2vis`. See `.claude/rules/component-model.md` -> "Axis
convention": this library deliberately does not transpose to pfb-imaging's
`(Y, X)` raster convention, so no transpose appears anywhere below.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr
from ducc0.wgridder.experimental import dirty2vis

from pfb_model_spec.utils.modelspec import eval_coeffs_to_slice

# Stokes -> correlation expressions, matching africanus.model.coherency.convert
# (verified elementwise; africanus is deliberately not a dependency here because
# tests must stay self-contained -- see .claude/rules/testing-and-ci.md).
_STOKES_TO_CORR: dict[str, dict[str, complex]] = {
    "XX": {"I": 1.0 + 0j, "Q": 1.0 + 0j},
    "XY": {"U": 1.0 + 0j, "V": 1.0j},
    "YX": {"U": 1.0 + 0j, "V": -1.0j},
    "YY": {"I": 1.0 + 0j, "Q": -1.0 + 0j},
    "RR": {"I": 1.0 + 0j, "V": 1.0 + 0j},
    "RL": {"Q": 1.0 + 0j, "U": 1.0j},
    "LR": {"Q": 1.0 + 0j, "U": -1.0j},
    "LL": {"I": 1.0 + 0j, "V": -1.0 + 0j},
}

# .mds spec versions this module understands. A new spec (e.g. the (Y, X) +
# stokes-axis revision, issues #19/#20) must be added here deliberately, not
# silently accepted -- the axis order and the stokes axis both change meaning.
_SUPPORTED_SPECS = frozenset({"genesis"})


def stokes_vis_to_corr(
    stokes_vis: np.ndarray,
    stokes_in: str | Sequence[str],
    corr_types: Sequence[str],
) -> np.ndarray:
    """Convert Stokes visibilities to correlations.

    An exact linear map. Stokes products absent from ``stokes_in`` are treated
    as zero, so an I-only model yields ``XX == YY == I`` and ``XY == YX == 0``.

    Args:
        stokes_vis: Complex visibilities, shape ``(nstokes, nrow, nchan)``.
        stokes_in: Stokes products present, in the order of ``stokes_vis``'s
            first axis, e.g. ``"I"`` or ``"IQUV"``.
        corr_types: Output correlations, e.g. ``("XX", "XY", "YX", "YY")``.

    Returns:
        Complex visibilities, shape ``(nrow, nchan, ncorr)``.

    Raises:
        ValueError: If ``stokes_in`` does not match ``stokes_vis``'s first axis,
            or a correlation is not one of XX/XY/YX/YY/RR/RL/LR/LL.
    """
    stokes_in = tuple(stokes_in)
    if len(stokes_in) != stokes_vis.shape[0]:
        raise ValueError(f"stokes_in has {len(stokes_in)} entries but stokes_vis has {stokes_vis.shape[0]} planes")
    nrow, nchan = stokes_vis.shape[1:]
    dtype = np.result_type(stokes_vis.dtype, np.complex64)
    out = np.zeros((nrow, nchan, len(corr_types)), dtype=dtype)
    index = {s: i for i, s in enumerate(stokes_in)}
    for c, corr in enumerate(corr_types):
        try:
            terms = _STOKES_TO_CORR[corr]
        except KeyError:
            raise ValueError(f"Unsupported correlation {corr!r}") from None
        for stokes, coeff in terms.items():
            if stokes in index:
                out[:, :, c] += coeff * stokes_vis[index[stokes]]
    return out


def degrid_stokes(
    uvw: np.ndarray,
    freq: np.ndarray,
    stokes_image: np.ndarray,
    *,
    cell_rad: float,
    x0: float,
    y0: float,
    flip_u: bool,
    flip_v: bool,
    flip_w: bool,
    epsilon: float = 1e-7,
    do_wgridding: bool = True,
    divide_by_n: bool = False,
    nthreads: int = 1,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Degrid one Stokes image per plane to visibilities.

    One ``dirty2vis`` call per Stokes plane. Degridding in Stokes and
    converting to correlations afterwards (:func:`stokes_vis_to_corr`) is
    cheaper than degridding the real and imaginary parts of the brightness
    matrix, which is what the wgridder would otherwise require.

    The geometry arguments (``x0``/``y0``/``flip_*``) must come from the
    ``.mds`` attrs, never be recomputed, so that a model degrids under the
    same conventions it was gridded with.

    Args:
        uvw: Baseline coordinates in metres, shape ``(nrow, 3)``. Rows may be
            NaN where the caller's data layout pads absent samples; pass
            ``mask`` to exclude them.
        freq: Channel frequencies in Hz, shape ``(nchan,)``.
        stokes_image: Model image per Stokes plane, shape ``(nstokes, nx, ny)``,
            x-major (see the module docstring).
        cell_rad: Pixel size in radians (square pixels).
        x0: Phase-centre x offset from ``wgridder_conventions``.
        y0: Phase-centre y offset from ``wgridder_conventions``.
        flip_u: U-axis flip convention.
        flip_v: V-axis flip convention.
        flip_w: W-axis flip convention.
        epsilon: Gridder accuracy.
        do_wgridding: Perform w-correction via improved w-stacking.
        divide_by_n: Divide by the geometric n-term. Left to the caller: a
            caller that folds ``1/n`` into its beam must keep this ``False``.
        nthreads: Threads for the gridder.
        mask: Optional ``(nrow, nchan)`` mask; zero entries are not degridded
            and come back as exactly zero.

    Returns:
        Complex visibilities, shape ``(nstokes, nrow, nchan)``.
    """
    nstokes = stokes_image.shape[0]
    dtype = np.result_type(stokes_image.dtype, np.complex64)
    out = np.zeros((nstokes, uvw.shape[0], freq.size), dtype=dtype)
    for s in range(nstokes):
        # an empty plane contributes nothing; skip the gridder call entirely
        if not stokes_image[s].any():
            continue
        out[s] = dirty2vis(
            uvw=uvw,
            freq=freq,
            dirty=stokes_image[s],
            mask=mask,
            pixsize_x=cell_rad,
            pixsize_y=cell_rad,
            center_x=x0,
            center_y=y0,
            flip_u=flip_u,
            flip_v=flip_v,
            flip_w=flip_w,
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            divide_by_n=divide_by_n,
            nthreads=nthreads,
        )
    return out


def model_geometry(model_ds: xr.Dataset) -> dict[str, Any]:
    """Extract the gridding geometry a `.mds` records.

    Callers pass these straight to :func:`degrid_stokes`. Reading them here
    rather than at each call site keeps the convention in one place.

    Args:
        model_ds: An opened `.mds` dataset.

    Returns:
        ``{"nx", "ny", "cell_rad", "x0", "y0", "flip_u", "flip_v", "flip_w",
        "stokes"}``.

    Raises:
        ValueError: If the `.mds` declares a spec this module does not know,
            or if the pixels are not square, since ``degrid_stokes`` takes a
            single ``cell_rad``.
    """
    a = model_ds.attrs
    spec = str(a.get("spec", "genesis"))
    if spec not in _SUPPORTED_SPECS:
        raise ValueError(f"Unsupported .mds spec {spec!r}; this module understands {sorted(_SUPPORTED_SPECS)}")

    cell_x = float(a["cell_rad_x"])
    cell_y = float(a["cell_rad_y"])
    if cell_x != cell_y:
        raise ValueError(
            f"Non-square pixels (cell_rad_x={cell_x}, cell_rad_y={cell_y}) are "
            "not supported; degridding takes a single cell size"
        )
    return {
        "nx": int(a["npix_x"]),
        "ny": int(a["npix_y"]),
        "cell_rad": cell_x,
        "x0": float(a["center_x"]),
        "y0": float(a["center_y"]),
        "flip_u": bool(a["flip_u"]),
        "flip_v": bool(a["flip_v"]),
        "flip_w": bool(a["flip_w"]),
        "stokes": str(a["stokes"]),
    }


def render_model_region(
    model_ds: xr.Dataset,
    *,
    time: float,
    freq_out: float,
    nx: int | None = None,
    ny: int | None = None,
    cell_rad: float | None = None,
    x0: float | None = None,
    y0: float | None = None,
) -> np.ndarray:
    """Render a component model to an image at one time and frequency.

    The model is a continuous function of time and frequency, so ``freq_out``
    may lie between the bands the model was fitted at -- that is what lets a
    consumer predict at finer spectral resolution than the imaging run used.

    The output grid defaults to the model's own. Supplying a different one
    resamples via ``eval_coeffs_to_slice``.

    Note:
        Resampling to a **coarser** grid is not flux-conserving for the point
        components a `.mds` stores: ``eval_coeffs_to_slice`` applies an
        ``area_ratio`` factor appropriate to a surface-brightness field, which
        inflates a point component's flux by exactly that factor (measured:
        coarsen x2 -> x4, coarsen x4 -> x16; refining and same-grid are exact).
        Degridding always uses the model's own grid and so is unaffected.

    Args:
        model_ds: An opened `.mds` dataset.
        time: Time to evaluate at, in the model's own time units.
        freq_out: Frequency to evaluate at, in Hz.
        nx: Output pixels along x. Defaults to the model's ``npix_x``.
        ny: Output pixels along y. Defaults to the model's ``npix_y``.
        cell_rad: Output pixel size in radians. Defaults to the model's.
        x0: Output phase-centre x offset. Defaults to the model's ``center_x``.
        y0: Output phase-centre y offset. Defaults to the model's ``center_y``.

    Returns:
        The rendered image, shape ``(nstokes, nx, ny)``, x-major. ``nstokes``
        is 1 for the ``genesis`` spec, which carries a single Stokes product;
        the axis is present so that adding a Stokes axis (#19) changes values
        rather than shapes.

    Raises:
        ValueError: If the `.mds` declares a spec this module does not know.
    """
    a = model_ds.attrs
    spec = str(a.get("spec", "genesis"))
    if spec not in _SUPPORTED_SPECS:
        raise ValueError(f"Unsupported .mds spec {spec!r}; this module understands {sorted(_SUPPORTED_SPECS)}")

    nxi, nyi = int(a["npix_x"]), int(a["npix_y"])
    cellxi, cellyi = float(a["cell_rad_x"]), float(a["cell_rad_y"])
    x0i, y0i = float(a["center_x"]), float(a["center_y"])

    nxo = nxi if nx is None else int(nx)
    nyo = nyi if ny is None else int(ny)
    cellxo = cellxi if cell_rad is None else float(cell_rad)
    cellyo = cellyi if cell_rad is None else float(cell_rad)
    x0o = x0i if x0 is None else float(x0)
    y0o = y0i if y0 is None else float(y0)

    image = eval_coeffs_to_slice(
        time,
        freq_out,
        model_ds.coefficients.values,
        model_ds.location_x.values,
        model_ds.location_y.values,
        a["parametrisation"],
        model_ds.params.values,
        a["texpr"],
        a["fexpr"],
        nxi,
        nyi,
        cellxi,
        cellyi,
        x0i,
        y0i,
        nxo,
        nyo,
        cellxo,
        cellyo,
        x0o,
        y0o,
    )
    # leading stokes axis: 1 for genesis, which stores a single product
    return image[None]


def apply_mueller(stokes_image: np.ndarray, mueller: np.ndarray) -> np.ndarray:
    """Attenuate a Stokes image with a Stokes-basis Mueller beam.

    ``apparent[i] = sum_j mueller[i, j] * intrinsic[j]``, pixel by pixel.

    The beam is whatever the caller supplies; this function neither builds one
    nor folds the wgridder's geometric ``1/n`` term into it. A caller that
    folds ``1/n`` into its beam (as pfb-imaging does) must correspondingly
    leave ``divide_by_n=False`` in :func:`degrid_stokes`.

    Args:
        stokes_image: Intrinsic model, shape ``(nstokes_in, nx, ny)``.
        mueller: Real Mueller block, shape ``(nstokes_out, nstokes_in, nx, ny)``
            on the same grid as ``stokes_image``.

    Returns:
        The apparent model, shape ``(nstokes_out, nx, ny)``.

    Raises:
        ValueError: If the Stokes axes or the image grids disagree.
    """
    if mueller.ndim != 4:
        raise ValueError(f"mueller must be 4-D (nso, nsi, nx, ny), got {mueller.shape}")
    if mueller.shape[1] != stokes_image.shape[0]:
        raise ValueError(
            f"mueller input axis {mueller.shape[1]} does not match the model's {stokes_image.shape[0]} Stokes planes"
        )
    if mueller.shape[2:] != stokes_image.shape[1:]:
        raise ValueError(f"mueller grid {mueller.shape[2:]} does not match the model grid {stokes_image.shape[1:]}")
    return np.einsum("ijxy,jxy->ixy", mueller, stokes_image)


def model_to_apparent_vis_for_region(
    model_ds: xr.Dataset,
    *,
    uvw: np.ndarray,
    freq: np.ndarray,
    corr_types: Sequence[str],
    time: float,
    freq_out: float,
    mueller: np.ndarray | None = None,
    stokes_out: str | Sequence[str] | None = None,
    mask: np.ndarray | None = None,
    region_mask: np.ndarray | None = None,
    epsilon: float = 1e-7,
    do_wgridding: bool = True,
    divide_by_n: bool = False,
    nthreads: int = 1,
) -> np.ndarray:
    """Render a model and degrid it to apparent correlated visibilities.

    Composes :func:`render_model_region`, :func:`apply_mueller`,
    :func:`degrid_stokes` and :func:`stokes_vis_to_corr` for one chunk of data.
    The primitives remain separately callable for consumers that need only part
    of the chain.

    The model is always rendered on its own grid, taken from the `.mds` attrs,
    so no resampling occurs (see :func:`render_model_region` on why resampling
    to a coarser grid would not conserve flux).

    Args:
        model_ds: An opened `.mds` dataset.
        uvw: Baseline coordinates in metres, shape ``(nrow, 3)``.
        freq: Channel frequencies in Hz, shape ``(nchan,)``.
        corr_types: Output correlations, e.g. ``("XX", "XY", "YX", "YY")``.
        time: Representative time for the chunk, conventionally the unweighted
            mean of its time axis.
        freq_out: Representative frequency in Hz for the chunk, conventionally
            the unweighted mean of its frequency axis. May lie between the
            bands the model was fitted at.
        mueller: Optional Stokes-basis Mueller block on the model grid, shape
            ``(nstokes_out, nstokes_in, nx, ny)``.
        stokes_out: Optional Stokes products of the planes being degridded (the
            rendered image after ``mueller``, if any, has been applied), in
            that axis's order, e.g. ``"IV"``. Passed as the ``stokes_in``
            argument to :func:`stokes_vis_to_corr`. When ``None`` (the
            default): if no Mueller was applied, the model's own ``stokes``
            attr is used; if a Mueller was applied, the planes are *assumed*
            to be an ``"IQUV"`` prefix of length ``nstokes_out`` -- this
            assumption is wrong whenever the Mueller's output axis is not such
            a prefix (e.g. a 2-plane (I, V) Mueller, mislabelled ``"IQ"`` by
            the default), so pass ``stokes_out`` explicitly in that case.
        mask: Optional ``(nrow, nchan)`` visibility mask; zero entries are not
            degridded and come back as exactly zero.
        region_mask: Optional ``(nx, ny)`` image mask applied to the rendered
            model, for degridding a sub-region into its own column.
        epsilon: Gridder accuracy.
        do_wgridding: Perform w-correction via improved w-stacking.
        divide_by_n: Divide by the geometric n-term; see :func:`apply_mueller`.
        nthreads: Threads for the gridder.

    Returns:
        Complex visibilities, shape ``(nrow, nchan, len(corr_types))``.

    Raises:
        ValueError: If ``region_mask``'s grid does not match the model grid,
            if ``stokes_out``'s length does not match the number of image
            planes being degridded, or if the composed primitives raise
            ValueError: non-square pixels from :func:`model_geometry`, an
            unsupported ``.mds`` spec from :func:`model_geometry` or
            :func:`render_model_region`, a Mueller shape mismatch from
            :func:`apply_mueller`, or an unsupported correlation from
            :func:`stokes_vis_to_corr`.
    """
    geom = model_geometry(model_ds)
    image = render_model_region(model_ds, time=time, freq_out=freq_out)
    stokes_in = geom["stokes"]

    if region_mask is not None:
        if region_mask.shape != image.shape[1:]:
            raise ValueError(f"region_mask grid {region_mask.shape} does not match the model grid {image.shape[1:]}")
        image = image * region_mask[None]

    if mueller is not None:
        image = apply_mueller(image, mueller)
        # a Mueller may map a single intrinsic Stokes product onto several
        # apparent ones, so the labels follow its output axis, not the model's
        stokes_in = "IQUV"[: image.shape[0]]

    if stokes_out is not None:
        stokes_out = tuple(stokes_out)
        if len(stokes_out) != image.shape[0]:
            raise ValueError(
                f"stokes_out has {len(stokes_out)} entries but the rendered image has {image.shape[0]} planes"
            )
        stokes_in = stokes_out

    stokes_vis = degrid_stokes(
        uvw,
        freq,
        image,
        cell_rad=geom["cell_rad"],
        x0=geom["x0"],
        y0=geom["y0"],
        flip_u=geom["flip_u"],
        flip_v=geom["flip_v"],
        flip_w=geom["flip_w"],
        epsilon=epsilon,
        do_wgridding=do_wgridding,
        divide_by_n=divide_by_n,
        nthreads=nthreads,
        mask=mask,
    )
    return stokes_vis_to_corr(stokes_vis, stokes_in, corr_types)
