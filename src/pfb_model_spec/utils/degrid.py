"""Render a component model to visibilities for a chunk of data.

Pure numpy: no dask, no Ray, no measurement-set handles. Data selection,
chunking, distribution and I/O belong to the calling application
(pfb-imaging's `degrid-msv4`, QuartiCal's predict); this module only turns a
model plus a chunk's `(uvw, freq)` into visibilities.

Axis convention: images are `(nstokes, ny, nx)`, the spec 0.1 / FITS order. ducc0's
`dirty2vis` is x-major, so :func:`degrid_stokes` transposes exactly once, right before the
gridder call; nothing else here transposes. Visibilities have no orientation, so a caller
of the fused wrapper never transposes at all.
"""

from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np
import xarray as xr
from ducc0.wgridder.experimental import dirty2vis

from pfb_model_spec.utils.modelspec import eval_coeffs_to_slice
from pfb_model_spec.utils.spec import upgrade

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
        stokes_image: Model image per Stokes plane, shape ``(nstokes, ny, nx)``
            (see the module docstring).
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
            dirty=np.ascontiguousarray(stokes_image[s].T),  # (ny, nx) -> ducc0's x-major (nx, ny)
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
        "stokes"}``; ``stokes`` is a list of Stokes labels.

    Raises:
        ValueError: If the `.mds` spec is unknown or newer (from `upgrade`), or if the
            pixels are not square, since ``degrid_stokes`` takes a single ``cell_rad``.
    """
    ds = upgrade(model_ds)
    a = ds.attrs
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
        "stokes": [str(s) for s in ds.stokes.values],
    }


def pixel_lm(
    geom: dict[str, Any],
    *,
    i0: int = 0,
    j0: int = 0,
    shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sky direction cosines of the pixels of a model grid or a crop of it.

    The convention is the one real measurement-set data follow through the
    wgridder, ``V = sum I(l, m) exp(+2πi(ul + vm + w(n-1))ν/c)``, with ``l``
    East and ``m`` North positive. Measured against ducc0 for every
    ``(flip_u, flip_v)``: ``l = -x0 ± (j - nx//2)·cell`` with ``+`` when
    ``flip_u``, and ``m = -y0 ± (i - ny//2)·cell`` with ``+`` when ``flip_v``.
    This is the coordinate a caller evaluates its beam on, so getting it wrong
    mirrors an asymmetric beam.

    Args:
        geom: :func:`model_geometry` output (``nx``, ``ny``, ``cell_rad``,
            ``x0``, ``y0``, ``flip_u``, ``flip_v`` are read).
        i0: Row of the crop's origin on the full grid (y).
        j0: Column of the crop's origin on the full grid (x).
        shape: ``(ny_crop, nx_crop)``; defaults to the rest of the grid.

    Returns:
        ``(l, m)`` in radians, shapes ``(nx_crop,)`` and ``(ny_crop,)``.
    """
    nx, ny, cell = geom["nx"], geom["ny"], geom["cell_rad"]
    nyc, nxc = shape if shape is not None else (ny - i0, nx - j0)
    sl = 1.0 if geom["flip_u"] else -1.0
    sm = 1.0 if geom["flip_v"] else -1.0
    j = j0 + np.arange(nxc)
    i = i0 + np.arange(nyc)
    ll = -geom["x0"] + sl * (j - nx // 2) * cell
    mm = -geom["y0"] + sm * (i - ny // 2) * cell
    return ll, mm


class RegionMask(NamedTuple):
    """A region's weights, cropped to the region's bounding box.

    Cropping is a pure optimisation, and a large one: a region is degridded by
    its own `dirty2vis` call, whose cost scales with the grid it is handed, not
    with the flux on it. Measured in pfb-imaging#331 on a 6720^2 model with three
    regions, the per-chunk gridder time fell from 3 x 5.29 s to 5.29 + 0.64 + 0.51 s. It
    also shrinks what the driver ships to each replica -- a full-grid float64
    mask is 361 MB at 6720^2, and a small region's is under 1 MB.

    Attributes:
        mask: `(nyc, nxc)` float64 weights, (Y, X) like the model.
        i0: Index along y of the mask's origin in the full model grid.
        j0: Index along x of the mask's origin in the full model grid.
    """

    mask: np.ndarray
    i0: int
    j0: int


def _even_span(lo: int, hi: int, n: int) -> tuple[int, int]:
    """Grow `[lo, hi)` to an even length without leaving `[0, n)`.

    ducc's wgridder asserts `nx_dirty must be even`, so a bounding box of odd
    extent cannot be handed to it. Growing outwards keeps the box a window on
    real model pixels, which padding with zeros would not.
    """
    if (hi - lo) % 2 == 0:
        return lo, hi
    if hi < n:
        return lo, hi + 1
    if lo > 0:
        return lo - 1, hi
    # the full axis is odd, so no even window exists; the uncropped grid would
    # fail the same assertion, and that is the error worth surfacing
    return lo, hi


def crop_bbox(mask: np.ndarray) -> RegionMask:
    """Crop a full-grid mask to the bounding box of its non-zero pixels.

    The box is grown by at most one pixel per axis so that both extents are
    even, which ducc requires (see `_even_span`).

    Args:
        mask: `(ny, nx)` weights, (Y, X) like the model.

    Returns:
        The cropped mask and its origin. An all-zero mask crops to a `(2, 2)`
        zero box at the origin -- `degrid_stokes` skips an empty plane
        outright, so the degenerate box is never gridded.
    """
    ny, nx = mask.shape
    ys = np.flatnonzero(mask.any(axis=1))
    xs = np.flatnonzero(mask.any(axis=0))
    if xs.size == 0 or ys.size == 0:
        return RegionMask(np.zeros((min(2, ny), min(2, nx)), dtype=mask.dtype), 0, 0)
    i0, i1 = _even_span(int(ys[0]), int(ys[-1]) + 1, ny)
    j0, j1 = _even_span(int(xs[0]), int(xs[-1]) + 1, nx)
    return RegionMask(np.ascontiguousarray(mask[i0:i1, j0:j1]), i0, j0)


def crop_phase_centre(
    *,
    x0: float,
    y0: float,
    cell_rad: float,
    flip_u: bool,
    flip_v: bool,
    nx: int,
    ny: int,
    rm: RegionMask,
) -> tuple[float, float]:
    """The `center_x`/`center_y` a cropped sub-grid must be degridded with.

    ducc places image pixel `i` at `(i - nx // 2) * cell` from the image
    centre, so cropping moves the centre by the shift in that origin --
    `j0 + nxc // 2 - nx // 2` pixels along x and `i0 + nyc // 2 - ny // 2`
    along y. The sign is the axis's flip convention: `flip_u`/`flip_v` negate
    the direction the offset is measured in. Getting either sign wrong is not
    a small error -- it puts the model in the wrong place on the sky, and the
    visibilities come back order-unity wrong (`test_crop_phase_centre_reproduces_the_full_grid_degrid`
    checks all four flip combinations).

    Args:
        x0: The model's own `center_x`, from `model_geometry`.
        y0: The model's own `center_y`.
        cell_rad: Pixel size in radians (square pixels).
        flip_u: The model's U-axis flip convention.
        flip_v: The model's V-axis flip convention.
        nx: Full model grid size along x.
        ny: Full model grid size along y.
        rm: The cropped mask whose grid the offsets are wanted for.

    Returns:
        `(center_x, center_y)` for `degrid_stokes`.
    """
    nyc, nxc = rm.mask.shape
    sx = -1.0 if flip_u else 1.0
    sy = -1.0 if flip_v else 1.0
    return (
        x0 + sx * (rm.j0 + nxc // 2 - nx // 2) * cell_rad,
        y0 + sy * (rm.i0 + nyc // 2 - ny // 2) * cell_rad,
    )


def _linear_weights(axis: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Left index, right weight and in-range flag for linear interpolation on `axis`."""
    k = np.clip(np.searchsorted(axis, x, side="right") - 1, 0, axis.size - 2)
    w = (x - axis[k]) / (axis[k + 1] - axis[k])
    inside = (x >= axis[0]) & (x <= axis[-1])
    return k, w, inside


def interp_mueller(
    mueller: np.ndarray,
    l_axis: np.ndarray,
    m_axis: np.ndarray,
    l: np.ndarray,  # noqa: E741 -- the (l, m) name of the interface
    m: np.ndarray,
) -> np.ndarray:
    """Bilinearly interpolate a coarse Mueller block onto a separable pixel grid.

    A primary beam is smooth on the scale of a model pixel, so callers evaluate
    it on a coarse grid and interpolate here, separably: along l for each coarse
    m row, then along m. The cost is O(ny·nx) per entry and no full-resolution
    coordinate mesh is built. Points outside the coarse axes get 0, matching
    meerkat-beams' off-cube value.

    Args:
        mueller: ``(..., nm, nl)`` on ``(m_axis, l_axis)``; real or complex.
        l_axis: ``(nl,)`` strictly ascending, radians.
        m_axis: ``(nm,)`` strictly ascending, radians.
        l: ``(nx,)`` target l, any order (from :func:`pixel_lm`).
        m: ``(ny,)`` target m, any order.

    Returns:
        ``(..., ny, nx)``, same dtype as ``mueller``.

    Raises:
        ValueError: If an axis is not strictly ascending, has fewer than two
            points, or does not match ``mueller``'s trailing shape.
    """
    for name, axis in (("l_axis", l_axis), ("m_axis", m_axis)):
        if axis.size < 2 or not np.all(np.diff(axis) > 0):
            raise ValueError(f"{name} must be strictly ascending with at least two points")
    if mueller.shape[-2:] != (m_axis.size, l_axis.size):
        raise ValueError(f"mueller grid {mueller.shape[-2:]} does not match (m, l) axes {(m_axis.size, l_axis.size)}")
    kl, wl, inl = _linear_weights(l_axis, l)
    km, wm, inm = _linear_weights(m_axis, m)
    # weights in the block's real dtype so float32/complex64 blocks are not promoted
    rdt = np.finfo(mueller.dtype).dtype
    wl, wm = wl.astype(rdt), wm.astype(rdt)
    along_l = mueller[..., kl] * (1 - wl) + mueller[..., kl + 1] * wl  # (..., nm, nx)
    out = along_l[..., km, :] * (1 - wm)[:, None] + along_l[..., km + 1, :] * wm[:, None]
    return out * (inm[:, None] & inl[None, :])


def model_sampling(model_ds: xr.Dataset) -> dict[str, Any]:
    """The optional sampling fields a `.mds` records (spec 0.1).

    Consumers derive defaults from these (for example a degridding chunk that
    spans one imaging band). Every field is optional in the spec, so each is
    ``None`` when absent -- including on any upgraded genesis `.mds`.

    Args:
        model_ds: An opened `.mds` dataset.

    Returns:
        ``{"freq_bounds", "time_bounds", "weights", "flux_scale"}``; arrays are
        ``(nfreq, 2)``, ``(ntime, 2)`` and ``(nstokes, ntime, nfreq)``.

    Raises:
        ValueError: If the `.mds` spec is unknown or newer (from `upgrade`).
    """
    ds = upgrade(model_ds)
    return {
        "freq_bounds": ds.freq_bounds.values if "freq_bounds" in ds.variables else None,
        "time_bounds": ds.time_bounds.values if "time_bounds" in ds.variables else None,
        "weights": ds.weights.values if "weights" in ds.variables else None,
        "flux_scale": ds.attrs.get("flux_scale"),
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
    kernel: str = "box",
) -> np.ndarray:
    """Render a component model to an image at one time and frequency.

    The model is a continuous function of time and frequency, so ``freq_out``
    may lie between the bands the model was fitted at -- that is what lets a
    consumer predict at finer spectral resolution than the imaging run used.

    The output grid defaults to the model's own. Supplying a different one
    remaps via ``eval_coeffs_to_slice``, which conserves flux whether the grid
    is refined, coarsened or shifted; flux outside the output grid is dropped.

    Args:
        model_ds: An opened `.mds` dataset.
        time: Time to evaluate at, in the model's own time units.
        freq_out: Frequency to evaluate at, in Hz.
        nx: Output pixels along x. Defaults to the model's ``npix_x``.
        ny: Output pixels along y. Defaults to the model's ``npix_y``.
        cell_rad: Output pixel size in radians. Defaults to the model's.
        x0: Output phase-centre x offset. Defaults to the model's ``center_x``.
        y0: Output phase-centre y offset. Defaults to the model's ``center_y``.
        kernel: Remap kernel for a different output grid: ``"box"``
            (piecewise-constant) or ``"tent"`` (minmod-limited piecewise-linear,
            smoother when refining extended emission).

    Returns:
        The rendered image, shape ``(nstokes, ny, nx)``, one plane per entry of
        the `.mds` ``stokes`` coord.

    Raises:
        ValueError: If the `.mds` spec is unknown or newer (from `upgrade`).
    """
    ds = upgrade(model_ds)
    a = ds.attrs

    nxi, nyi = int(a["npix_x"]), int(a["npix_y"])
    cellxi, cellyi = float(a["cell_rad_x"]), float(a["cell_rad_y"])
    x0i, y0i = float(a["center_x"]), float(a["center_y"])

    nxo = nxi if nx is None else int(nx)
    nyo = nyi if ny is None else int(ny)
    cellxo = cellxi if cell_rad is None else float(cell_rad)
    cellyo = cellyi if cell_rad is None else float(cell_rad)
    x0o = x0i if x0 is None else float(x0)
    y0o = y0i if y0 is None else float(y0)

    return eval_coeffs_to_slice(
        time,
        freq_out,
        ds.coefficients.values,
        ds.location_y.values,
        ds.location_x.values,
        a["parametrisation"],
        ds.params.values,
        a["texpr"],
        a["fexpr"],
        nxi=nxi,
        nyi=nyi,
        cellxi=cellxi,
        cellyi=cellyi,
        x0i=x0i,
        y0i=y0i,
        nxo=nxo,
        nyo=nyo,
        cellxo=cellxo,
        cellyo=cellyo,
        x0o=x0o,
        y0o=y0o,
        kernel=kernel,
    )


def apply_mueller(stokes_image: np.ndarray, mueller: np.ndarray) -> np.ndarray:
    """Attenuate a Stokes image with a Stokes-basis Mueller beam.

    ``apparent[i] = sum_j mueller[i, j] * intrinsic[j]``, pixel by pixel.

    The beam is whatever the caller supplies; this function neither builds one
    nor folds the wgridder's geometric ``1/n`` term into it. A caller that
    folds ``1/n`` into its beam (as pfb-imaging does) must correspondingly
    leave ``divide_by_n=False`` in :func:`degrid_stokes`.

    Args:
        stokes_image: Intrinsic model, shape ``(nstokes_in, ny, nx)``.
        mueller: Real or complex Mueller block, shape ``(nstokes_out, nstokes_in, ny, nx)``
            on the same grid as ``stokes_image``.

    Returns:
        The apparent model, shape ``(nstokes_out, ny, nx)``.

    Raises:
        ValueError: If the Stokes axes or the image grids disagree.
    """
    if mueller.ndim != 4:
        raise ValueError(f"mueller must be 4-D (nso, nsi, ny, nx), got {mueller.shape}")
    if mueller.shape[1] != stokes_image.shape[0]:
        raise ValueError(
            f"mueller input axis {mueller.shape[1]} does not match the model's {stokes_image.shape[0]} Stokes planes"
        )
    if mueller.shape[2:] != stokes_image.shape[1:]:
        raise ValueError(f"mueller grid {mueller.shape[2:]} does not match the model grid {stokes_image.shape[1:]}")
    return np.einsum("ijxy,jxy->ixy", mueller, stokes_image)


def _degrid_real_imag(
    uvw: np.ndarray, freq: np.ndarray, image: np.ndarray, *, sign: np.ndarray | None = None, **kw: Any
) -> np.ndarray:
    """Degrid a real or complex image; ducc takes only real ones, so a complex one goes as Re + i·sign·Im."""
    vis = degrid_stokes(uvw, freq, np.ascontiguousarray(image.real), **kw)
    if np.iscomplexobj(image):
        vis_im = degrid_stokes(uvw, freq, np.ascontiguousarray(image.imag), **kw)
        vis = vis + 1j * (vis_im if sign is None else sign[None, :, None] * vis_im)
    return vis


class BeamTerm(NamedTuple):
    """One primary beam and the visibility rows it applies to.

    The caller builds the beam (pfb-model-spec never depends on a beam package):
    a Stokes-basis Mueller block on a coarse, ascending (m, l) grid in the
    :func:`pixel_lm` convention, interpolated onto each region here. Several
    terms partition the rows between beams -- a mixed array's baseline groups --
    and ``conj_rows`` marks rows whose baseline ordering is reversed relative to
    the beam's, which take ``conj(mueller)`` (meerkat-beams' MeerKAT/MeerKAT+
    cross group). A complex block is predicted exactly, by degridding its real
    and imaginary parts separately. Pixels outside ``(l_axis, m_axis)`` see a
    zero beam, so the axes must cover every region the term is applied to.

    Attributes:
        mueller: ``(nso, nsi, nmb, nlb)``, real or complex, bare primary beam.
        l_axis: ``(nlb,)`` ascending, radians.
        m_axis: ``(nmb,)`` ascending, radians.
        rows: ``(nrow,)`` bool, the rows this term predicts; ``None`` means all.
        conj_rows: ``(nrow,)`` bool, a subset of ``rows`` taking the conjugate.
    """

    mueller: np.ndarray
    l_axis: np.ndarray
    m_axis: np.ndarray
    rows: np.ndarray | None = None
    conj_rows: np.ndarray | None = None


def _check_beam_terms(terms: Sequence[BeamTerm], nrow: int, nso: int, nsi: int) -> None:
    """Refuse beam terms that would mislabel planes or double-count rows."""
    covered = np.zeros(nrow, dtype=bool)
    for term in terms:
        if term.mueller.ndim != 4 or term.mueller.shape[:2] != (nso, nsi):
            raise ValueError(f"mueller must be (nso={nso}, nsi={nsi}, nm, nl), got {term.mueller.shape}")
        rows = np.ones(nrow, dtype=bool) if term.rows is None else np.asarray(term.rows, dtype=bool)
        if rows.shape != (nrow,):
            raise ValueError(f"rows must have shape ({nrow},), got {rows.shape}")
        if (covered & rows).any():
            raise ValueError("beam terms' rows must be disjoint")
        covered |= rows
        if term.conj_rows is not None:
            conj = np.asarray(term.conj_rows, dtype=bool)
            if conj.shape != (nrow,) or (conj & ~rows).any():
                raise ValueError("conj_rows must be a subset of the term's rows")


def _degrid_beam_term(
    uvw: np.ndarray,
    freq: np.ndarray,
    sub: np.ndarray,
    term: BeamTerm,
    l: np.ndarray,  # noqa: E741
    m: np.ndarray,
    mask: np.ndarray | None,
    kw: dict[str, Any],
) -> np.ndarray:
    """Attenuate one region by one beam term and degrid it onto the term's rows."""
    nrow = uvw.shape[0]
    nso, nsi = term.mueller.shape[:2]
    term_mask = mask
    if term.rows is not None:
        rows = np.asarray(term.rows, dtype=np.uint8)[:, None]
        term_mask = np.ascontiguousarray(np.broadcast_to(rows, (nrow, freq.size)) if mask is None else mask * rows)
    sign = None if term.conj_rows is None else np.where(term.conj_rows, -1.0, 1.0)
    vis = np.zeros((nso, nrow, freq.size), dtype=np.complex128)
    # one apparent plane at a time: neither the (nso, nsi, ny, nx) Mueller block
    # nor the (nso, ny, nx) complex apparent cube (2.9 GB at 6720^2 for a 4x4
    # beam) is ever held whole
    for o in range(nso):
        plane = sum(interp_mueller(term.mueller[o, i], term.l_axis, term.m_axis, l, m) * sub[i] for i in range(nsi))
        vis[o] = _degrid_real_imag(uvw, freq, plane[None], sign=sign, mask=term_mask, **kw)[0]
    return vis


def model_to_apparent_vis_for_regions(
    model_ds: xr.Dataset,
    *,
    uvw: np.ndarray,
    freq: np.ndarray,
    corr_types: Sequence[str],
    time: float,
    freq_out: float,
    region_masks: Sequence[RegionMask] | None = None,
    stokes_out: str | Sequence[str] | None = None,
    beam_terms: Sequence[BeamTerm] | None = None,
    mask: np.ndarray | None = None,
    epsilon: float = 1e-7,
    do_wgridding: bool = True,
    divide_by_n: bool = False,
    nthreads: int = 1,
) -> list[np.ndarray]:
    """Render a model once and degrid each region, optionally through beams.

    The fused per-chunk wrapper over :func:`render_model_region`,
    :func:`interp_mueller`, :func:`degrid_stokes` and :func:`stokes_vis_to_corr`,
    which stay separately callable for consumers that need only part of the
    chain. The model is rendered once per chunk,
    each region is degridded on its cropped bounding box with a shifted phase
    centre, and each :class:`BeamTerm` attenuates the region on the rows it
    owns. Rows covered by no term predict zero, so a caller that splits rows
    between beams must cover all of them. ``divide_by_n`` passes straight
    through; with a bare beam set it ``True`` and ducc folds in ``1/n``.

    Each term costs a full gridding pass per output plane (two for a complex
    block), whatever the share of rows it owns, so splitting rows between K
    beams costs about K times the FFT work of a single beam.

    Args:
        model_ds: An opened `.mds` dataset.
        uvw: ``(nrow, 3)`` metres; padded rows must be masked by ``mask``.
        freq: ``(nchan,)`` Hz.
        corr_types: Output correlations.
        time: Representative chunk time, unweighted mean, model time units.
        freq_out: Representative chunk frequency, unweighted mean, Hz.
        region_masks: Cropped region masks; ``None`` means one full-grid region.
        stokes_out: Stokes planes predicted. Without beams a subset of the
            model's; with beams the Mueller output axis. Defaults to the
            model's own planes.
        beam_terms: Beams and the rows they own; ``None`` applies no beam. An
            empty sequence is refused rather than predicting zeros.
        mask: ``(nrow, nchan)`` uint8 structural mask (padding), never flags.
        epsilon: Gridder accuracy.
        do_wgridding: w-correction via improved w-stacking.
        divide_by_n: Divide by the geometric n-term in ducc.
        nthreads: Gridder threads.

    Returns:
        One ``(nrow, nchan, len(corr_types))`` complex array per region.

    Raises:
        ValueError: On a Stokes plane the model lacks, inconsistent beam terms,
            or anything the composed primitives refuse.
    """
    model_ds = upgrade(model_ds)
    geom = model_geometry(model_ds)
    model_stokes = list(geom["stokes"])
    stokes_out = list(stokes_out) if stokes_out is not None else model_stokes
    if beam_terms is not None:
        if not beam_terms:
            raise ValueError("beam_terms is empty, which would predict zeros; pass None for no beam")
        _check_beam_terms(beam_terms, uvw.shape[0], len(stokes_out), len(model_stokes))

    image = render_model_region(model_ds, time=time, freq_out=freq_out)
    if beam_terms is None:
        missing = [s for s in stokes_out if s not in model_stokes]
        if missing:
            raise ValueError(f"stokes_out {stokes_out} names {missing}, which the model {model_stokes} lacks")
        planes = [model_stokes.index(s) for s in stokes_out]
        if planes != list(range(len(model_stokes))):
            # fancy indexing copies the whole grid, so only when not the identity
            image = image[planes]

    out = []
    # None is the whole grid: no crop, and no full-grid mask of ones to multiply by
    for rm in [None] if region_masks is None else region_masks:
        if rm is None:
            i0, j0, sub = 0, 0, image
            x0c, y0c = geom["x0"], geom["y0"]
        else:
            i0, j0 = rm.i0, rm.j0
            nyc, nxc = rm.mask.shape
            sub = image[:, i0 : i0 + nyc, j0 : j0 + nxc] * rm.mask[None]
            x0c, y0c = crop_phase_centre(
                x0=geom["x0"],
                y0=geom["y0"],
                cell_rad=geom["cell_rad"],
                flip_u=geom["flip_u"],
                flip_v=geom["flip_v"],
                nx=geom["nx"],
                ny=geom["ny"],
                rm=rm,
            )
        kw = dict(
            cell_rad=geom["cell_rad"],
            x0=x0c,
            y0=y0c,
            flip_u=geom["flip_u"],
            flip_v=geom["flip_v"],
            flip_w=geom["flip_w"],
            epsilon=epsilon,
            do_wgridding=do_wgridding,
            divide_by_n=divide_by_n,
            nthreads=nthreads,
        )
        if beam_terms is None:
            stokes_vis = degrid_stokes(uvw, freq, np.ascontiguousarray(sub), mask=mask, **kw)
        else:
            l, m = pixel_lm(geom, i0=i0, j0=j0, shape=sub.shape[1:])  # noqa: E741
            stokes_vis = np.zeros((len(stokes_out), uvw.shape[0], freq.size), dtype=np.complex128)
            for term in beam_terms:
                if term.rows is not None and not np.any(term.rows):
                    continue  # owns no rows here; skip its gridding passes
                stokes_vis += _degrid_beam_term(uvw, freq, sub, term, l, m, mask, kw)
        out.append(stokes_vis_to_corr(stokes_vis, stokes_out, corr_types))
    return out
