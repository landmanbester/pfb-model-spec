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

from typing import Sequence

import numpy as np
from ducc0.wgridder.experimental import dirty2vis

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
    out = np.zeros((nrow, nchan, len(corr_types)), dtype=stokes_vis.dtype)
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
