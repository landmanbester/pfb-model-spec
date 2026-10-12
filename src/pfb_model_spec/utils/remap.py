"""Flux-conserving remapping of pixelated images between axis-aligned rectilinear grids.

Pixel values are flux per pixel (Jy/pixel), so remapping redistributes flux rather than
sampling a field. Along each axis pixel ``k`` of an ``n``-pixel grid with cell ``c`` and centre
offset ``x0`` covers ``[x_k - c/2, x_k + c/2]`` with ``x_k = (k - n//2)·c + x0``. A 2-D remap is
a 1-D remap along x (the last axis) followed by one along y.

Kernels:
    ``"box"``: brightness is uniform within each input pixel (first-order conservative).

Flux from input pixels outside the output footprint is dropped, never wrapped.
"""

import numpy as np
from scipy import sparse

KERNELS = ("box",)

# pixel edges this close to an integer (in output-pixel units) are snapped onto it, so that
# integer shifts and exact cell ratios produce exact weights instead of 1e-16 slivers
_SNAP = 1e-9


def _snap(p: np.ndarray) -> np.ndarray:
    nearest = np.rint(p)
    return np.where(np.abs(p - nearest) < _SNAP, nearest, p)


def _axis_weights(
    n_in: int, cell_in: float, x0_in: float, n_out: int, cell_out: float, x0_out: float
) -> tuple[sparse.csr_array, sparse.csr_array]:
    """1-D overlap weights ``(W, M)``, each a ``(n_out, n_in)`` sparse array.

    ``W[o, i]`` is the fraction of input pixel ``i`` lying in output pixel ``o``; ``M[o, i]`` is
    the first moment of that overlap, ``∫ q dq`` with ``q`` the position within pixel ``i`` in
    input-pixel units (``q`` in ``[-1/2, 1/2]``).
    """
    # input pixel edges in output-pixel units, where output pixel o spans [o, o + 1]
    origin = (-(n_out // 2) - 0.5) * cell_out + x0_out
    lo = _snap(((np.arange(n_in) - n_in // 2 - 0.5) * cell_in + x0_in - origin) / cell_out)
    hi = _snap(lo + cell_in / cell_out)
    width = hi - lo
    centre = (lo + hi) / 2

    # each input pixel overlaps at most nspan consecutive output pixels, starting at `first`
    first = np.clip(np.floor(lo).astype(np.int64), 0, n_out - 1)
    nspan = int(np.ceil(cell_in / cell_out)) + 1
    o = first[:, None] + np.arange(nspan)[None, :]
    a = np.maximum(lo[:, None], o)
    b = np.minimum(hi[:, None], o + 1)
    keep = (b > a) & (o < n_out)

    cols = np.broadcast_to(np.arange(n_in)[:, None], o.shape)[keep]
    rows = o[keep]
    qa = ((a - centre[:, None]) / width[:, None])[keep]
    qb = ((b - centre[:, None]) / width[:, None])[keep]
    w = sparse.csr_array((qb - qa, (rows, cols)), shape=(n_out, n_in))
    m = sparse.csr_array(((qb**2 - qa**2) / 2, (rows, cols)), shape=(n_out, n_in))
    return w, m


def _remap_axis(image: np.ndarray, axis: int, w: sparse.csr_array) -> np.ndarray:
    moved = np.moveaxis(image, axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    out = w @ flat
    return np.moveaxis(out.reshape((w.shape[0],) + moved.shape[1:]), 0, axis)


def _check_grid(name: str, n: int, cell: float) -> None:
    if int(n) != n or n < 1:
        raise ValueError(f"{name}: pixel count must be a positive integer, got {n}")
    if not cell > 0:
        raise ValueError(f"{name}: cell size must be positive, got {cell}")


def _check_geometry(
    nxi: int, nyi: int, cellxi: float, cellyi: float, nxo: int, nyo: int, cellxo: float, cellyo: float
) -> None:
    _check_grid("x input", nxi, cellxi)
    _check_grid("y input", nyi, cellyi)
    _check_grid("x output", nxo, cellxo)
    _check_grid("y output", nyo, cellyo)


def remap_image(
    image: np.ndarray,
    *,
    kernel: str = "box",
    nxi: int,
    nyi: int,
    cellxi: float,
    cellyi: float,
    x0i: float,
    y0i: float,
    nxo: int,
    nyo: int,
    cellxo: float,
    cellyo: float,
    x0o: float,
    y0o: float,
) -> np.ndarray:
    """Remap flux-per-pixel images onto another grid, conserving flux.

    Args:
        image: Shape ``(..., nyi, nxi)``; leading axes are remapped independently.
        kernel: One of ``KERNELS`` (see the module docstring).
        nxi, nyi, cellxi, cellyi, x0i, y0i: Input grid (pixels, cell sizes, centre offsets).
        nxo, nyo, cellxo, cellyo, x0o, y0o: Output grid.

    Returns:
        A float array of shape ``(..., nyo, nxo)``. Identical geometries return a copy.

    Raises:
        ValueError: On an unknown kernel, an invalid grid, or a mismatched ``image`` shape.
    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
    _check_geometry(nxi, nyi, cellxi, cellyi, nxo, nyo, cellxo, cellyo)
    image = np.asarray(image, dtype=float)
    if image.ndim < 2 or image.shape[-2:] != (nyi, nxi):
        raise ValueError(f"image must have trailing shape (nyi, nxi) = {(nyi, nxi)}, got {image.shape}")
    if (nxi, nyi, cellxi, cellyi, x0i, y0i) == (nxo, nyo, cellxo, cellyo, x0o, y0o):
        return image.copy()
    wx, _ = _axis_weights(nxi, cellxi, x0i, nxo, cellxo, x0o)
    wy, _ = _axis_weights(nyi, cellyi, y0i, nyo, cellyo, y0o)
    return _remap_axis(_remap_axis(image, -1, wx), -2, wy)
