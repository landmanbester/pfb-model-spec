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
