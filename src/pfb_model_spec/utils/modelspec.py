from collections.abc import Callable

import numpy as np
import sympy as sm
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.lambdify import lambdify

from pfb_model_spec.utils.remap import remap_image
from pfb_model_spec.utils.spec import open_mds


def _scale(x: np.ndarray, sym: sm.Symbol, method: str) -> tuple[np.ndarray, sm.Expr]:
    """Map an axis onto the fit domain, returning ``(scaled values, sympy scaling expr)``.

    A length-1 axis is left unscaled: only the constant basis function can be fitted to
    it, so any scaling is irrelevant -- and ``x / x[0]`` would divide by zero at t=0.
    """
    if x.size == 1:
        return x.astype(float), sym
    if method == "poly":
        return x / x[0], sym / x[0]
    # Legendre: scale onto [-1, 1] for stability
    centre = (x.max() + x.min()) / 2
    half = (x - centre).max()
    return (x - centre) / half, (sym - centre) / half


def _fake_nodes(x: np.ndarray, sym: sm.Symbol, bounds: np.ndarray | None) -> tuple[np.ndarray, sm.Expr]:
    """Map an axis through the "fake nodes" cosine map, returning ``(mapped values, sympy map)``.

    ``S(x) = -cos(pi (x - lo) / (hi - lo))`` maps ``[lo, hi]`` onto ``[-1, 1]``. The domain is
    the outer edge of the samples' bins: ``bounds`` (``(n, 2)`` ``[lo, hi]`` per sample) when
    given, else half a sample spacing beyond the first and last samples. Equispaced bin
    centres then land exactly on the Chebyshev (first kind) nodes, so interpolating in ``S``
    neither shows Runge's phenomenon nor extrapolates anywhere inside the outer bins. Outside
    ``[lo, hi]`` the map folds back, mirroring the interpolant rather than diverging.

    A length-1 axis is left unscaled, as in :func:`_scale`.

    Raises:
        ValueError: If ``bounds`` has the wrong shape or an empty domain.
    """
    if x.size == 1:
        return x.astype(float), sym
    if bounds is None:
        xs = np.sort(x)
        lo = xs[0] - (xs[1] - xs[0]) / 2
        hi = xs[-1] + (xs[-1] - xs[-2]) / 2
    else:
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (x.size, 2):
            raise ValueError(f"bounds must have shape ({x.size}, 2), got {bounds.shape}")
        lo, hi = bounds.min(), bounds.max()
    if not hi > lo:
        raise ValueError(f"empty fake-nodes domain [{lo}, {hi}]")
    lo, hi = float(lo), float(hi)
    return -np.cos(np.pi * (x - lo) / (hi - lo)), -sm.cos(sm.pi * (sym - lo) / (hi - lo))


def fit_image_cube(
    time,
    freq,
    image,
    wgt=None,
    nbasist=None,
    nbasisf=None,
    method="poly",
    sigmasq=0,
    *,
    time_bounds=None,
    freq_bounds=None,
):
    """Fit the time and frequency axes of a multi-Stokes image cube.

    The basis is additive: functions of time (including the constant) plus functions of
    frequency (excluding the constant). Every Stokes plane is fitted separately against the
    same basis, and all planes share one set of component locations: the pixels that are
    non-zero in any plane.

    Args:
        time: Time axis, shape ``(ntime,)``.
        freq: Frequency axis, shape ``(nband,)``.
        image: Pixelated cube, shape ``(ntime, nband, nstokes, ny, nx)``.
        wgt: Optional weights, shape ``(ntime, nband, nstokes)``.
        nbasist: Number of time basis functions (default ``ntime``).
        nbasisf: Number of frequency basis functions, counting the shared constant
            (default ``nband``).
        method: ``"poly"`` (monomials), ``"Legendre"``, or ``"fakenodes"`` (Chebyshev
            polynomials of the cosine-mapped axis, see :func:`_fake_nodes`). Use
            ``"fakenodes"`` to evaluate between or beyond the band centres: the polynomial
            methods ring there (Runge's phenomenon) when ``nbasisf`` is close to ``nband``.
        sigmasq: Ridge term added to the Hessian; ignored when ``ntime == nband == 1``
            (nothing to fit: the data are the coefficients, whatever the weights).
        time_bounds: Optional ``(ntime, 2)`` ``[lo, hi]`` per time bin; ``"fakenodes"`` only.
        freq_bounds: Optional ``(nband, 2)`` ``[lo, hi]`` in Hz per band; ``"fakenodes"``
            only. Without bounds the domain extends half a sample spacing past the ends.

    Returns:
        ``(coeffs, y_index, x_index, expr, params, texpr, fexpr)``:
        - ``coeffs`` has shape ``(nstokes, npar, ncomps)``;
        - ``y_index``/``x_index`` are the component pixel locations, in array-axis order;
        - ``expr`` is the stringified sympy model in ``t``, ``f`` and ``params``;
        - ``texpr``/``fexpr`` map raw time/frequency onto the fit domain.

    Raises:
        ValueError: On inconsistent shapes or basis sizes.
        NotImplementedError: For an unknown ``method``.
    """
    from sympy.abc import f, t

    if image.ndim != 5:
        raise ValueError(f"image must be (ntime, nband, nstokes, ny, nx), got shape {image.shape}")
    ntime, nband, nstokes, ny, nx = image.shape
    if time.size != ntime or freq.size != nband:
        raise ValueError(f"time/freq sizes ({time.size}, {freq.size}) do not match image {image.shape}")
    if wgt is None:
        wgt = np.ones((ntime, nband, nstokes), dtype=float)
    elif wgt.shape != (ntime, nband, nstokes):
        raise ValueError(f"wgt must have shape {(ntime, nband, nstokes)}, got {wgt.shape}")
    nbasist = ntime if nbasist is None else nbasist
    nbasisf = nband if nbasisf is None else nbasisf
    if not 1 <= nbasist <= ntime or not 1 <= nbasisf <= nband:
        raise ValueError(f"need 1 <= nbasist <= {ntime} and 1 <= nbasisf <= {nband}")

    if method == "poly":

        def basis(i, w, sym):
            return w**i, sym**i

    elif method == "Legendre":

        def basis(i, w, sym):
            return np.polynomial.Legendre.basis(i)(w), sm.polys.orthopolys.legendre_poly(i, sym)

    elif method == "fakenodes":

        def basis(i, w, sym):
            return np.polynomial.Chebyshev.basis(i)(w), sm.chebyshevt_poly(i, sym)

    else:
        raise NotImplementedError(f"Method {method} not implemented")

    if method == "fakenodes":
        wt, tfunc = _fake_nodes(time, t, time_bounds)
        wf, ffunc = _fake_nodes(freq, f, freq_bounds)
    else:
        wt, tfunc = _scale(time, t, method)
        wf, ffunc = _scale(freq, f, method)
    tparams = sm.symbols(f"t(0:{nbasist})")
    fparams = sm.symbols(f"f(1:{nbasisf})") if nbasisf > 1 else ()

    expr = sm.Integer(0)
    xt = np.zeros((ntime, nbasist))
    for i, p in enumerate(tparams):
        xt[:, i], term = basis(i, wt, t)
        expr += term * p
    xf = np.zeros((nband, nbasisf - 1))
    for i, p in enumerate(fparams, start=1):
        xf[:, i - 1], term = basis(i, wf, f)
        expr += term * p
    # rows ordered (time, band), matching the C-order reshape of beta below
    xfit = np.hstack((np.repeat(xt, nband, axis=0), np.tile(xf, (ntime, 1))))

    mask = np.any(image, axis=(0, 1, 2))
    y_index, x_index = np.where(mask)
    beta = image[:, :, :, y_index, x_index].reshape(ntime * nband, nstokes, y_index.size)
    wgt = wgt.reshape(ntime * nband, nstokes)

    coeffs = np.zeros((nstokes, xfit.shape[1], y_index.size), dtype=np.result_type(beta.dtype, float))
    for s in range(nstokes):
        if ntime == 1 and nband == 1:
            # nothing to fit (the basis is the constant 1); a zero weight would make it singular
            coeffs[s] = beta[:, s]
            continue
        w = wgt[:, s : s + 1]
        hess = xfit.T.dot(w * xfit)
        if sigmasq:
            hess += sigmasq * np.eye(hess.shape[0])
        coeffs[s] = np.linalg.solve(hess, xfit.T.dot(w * beta[:, s]))

    params = [*tparams, *fparams]
    return coeffs, y_index, x_index, str(expr), [str(p) for p in params], str(tfunc), str(ffunc)


def _model_functions(expr: str, paramf: list[str], texpr: str, fexpr: str) -> tuple[Callable, Callable, Callable]:
    """Lambdify a stored parametrisation into ``(modelf, tfunc, ffunc)``."""
    params = sm.symbols(("t", "f"))
    params += sm.symbols(tuple(paramf))
    modelf = lambdify(params, parse_expr(expr))
    tfunc = lambdify(params[0], parse_expr(texpr))
    ffunc = lambdify(params[1], parse_expr(fexpr))
    return modelf, tfunc, ffunc


def eval_coeffs_to_cube(time, freq, nx, ny, coeffs, y_index, x_index, expr, paramf, texpr, fexpr):
    """Render ``(nstokes, npar, ncomps)`` coefficients to a ``(ntime, nfreq, nstokes, ny, nx)`` cube."""
    modelf, tfunc, ffunc = _model_functions(expr, paramf, texpr, fexpr)
    nstokes = coeffs.shape[0]
    image = np.zeros((time.size, freq.size, nstokes, ny, nx), dtype=float)
    for i, tval in enumerate(time):
        for j, fval in enumerate(freq):
            for s in range(nstokes):
                image[i, j, s, y_index, x_index] = modelf(tfunc(tval), ffunc(fval), *coeffs[s])
    return image


def eval_coeffs_to_slice(
    time,
    freq,
    coeffs,
    y_index,
    x_index,
    expr,
    paramf,
    texpr,
    fexpr,
    *,
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
    kernel="box",
):
    """Render coefficients at one (time, freq) onto an arbitrary output grid.

    The components are rendered on their own grid, then remapped onto the output grid with
    :func:`pfb_model_spec.utils.remap.remap_image`, which conserves flux (``kernel`` is
    ``"box"`` or ``"tent"``). Returns ``(nstokes, nyo, nxo)``. The geometry is keyword-only so
    that a call written against the old x-major positional signature fails loudly instead of
    swapping axes.
    """
    modelf, tfunc, ffunc = _model_functions(expr, paramf, texpr, fexpr)
    tval, fval = tfunc(time), ffunc(freq)
    image_in = np.zeros((coeffs.shape[0], nyi, nxi), dtype=float)
    for s in range(coeffs.shape[0]):
        image_in[s, y_index, x_index] = modelf(tval, fval, *coeffs[s])
    return remap_image(
        image_in,
        kernel=kernel,
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
    )


def model_from_mds(mds_name, freqs=None):
    """Render a `.mds` at any known spec at its own resolution: ``(ntime, nfreq, nstokes, ny, nx)``."""
    mds = open_mds(mds_name)
    if freqs is None:
        freqs = mds.freqs.values
    else:
        freqs = np.atleast_1d(freqs)
    return eval_coeffs_to_cube(
        mds.times.values,
        freqs,
        mds.npix_x,
        mds.npix_y,
        mds.coefficients.values,
        mds.location_y.values,
        mds.location_x.values,
        mds.parametrisation,
        mds.params.values,
        mds.texpr,
        mds.fexpr,
    )
