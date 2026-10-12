import warnings
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


HISTOPOLATION = ("box", "tent")


def _band_edges(freq: np.ndarray, bounds: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Per-band ``(lo, hi)`` edges: ``bounds`` when given, else midway between centres.

    Without bounds the outer edges lie half a spacing past the end centres.

    Raises:
        ValueError: If the bands are not increasing and non-overlapping, or ``bounds`` is
            misshapen.
    """
    if bounds is None:
        if freq.size == 1:
            return freq.astype(float), freq.astype(float)
        mid = (freq[1:] + freq[:-1]) / 2
        lo = np.concatenate(([freq[0] - (freq[1] - freq[0]) / 2], mid))
        hi = np.concatenate((mid, [freq[-1] + (freq[-1] - freq[-2]) / 2]))
    else:
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (freq.size, 2):
            raise ValueError(f"bounds must have shape ({freq.size}, 2), got {bounds.shape}")
        lo, hi = bounds[:, 0], bounds[:, 1]
    # contiguous bands built from floating-point channel edges may overlap by rounding
    tol = 1e-9 * (hi - lo).max(initial=0.0)
    if np.any(hi <= lo) or np.any(lo[1:] < hi[:-1] - tol):
        raise ValueError("bands must be increasing and non-overlapping, with hi > lo")
    return lo, hi


def _tent_slopes(means: np.ndarray, anchor: np.ndarray, half: np.ndarray, qlo: float, qhi: float) -> np.ndarray:
    """Limited slopes, in band value per half band width (``(nband, ...)``).

    Band ``b`` is ``means[b] + slope[b] * (f - anchor[b]) / half[b]``. Slopes come from the
    differences between neighbouring anchors: interior bands take their minmod, the end bands
    their one-sided difference, clipped so that the outer edge (``qlo``/``qhi`` half widths
    from the anchor) keeps the sign of the band value. Since a band's edge never lies beyond
    its neighbour's anchor, every edge value lies between zero and the neighbouring values, so
    non-negative bands give a non-negative spectrum, and a band at a local extremum is flat.
    """
    slopes = np.zeros_like(means)
    if means.shape[0] == 1:
        return slopes
    shape = (-1,) + (1,) * (means.ndim - 1)
    diff = np.diff(means, axis=0) / np.diff(anchor).reshape(shape)  # per Hz, between anchors
    fwd, bwd = diff[1:], diff[:-1]
    slopes[1:-1] = np.where(fwd * bwd > 0, np.sign(fwd) * np.minimum(np.abs(fwd), np.abs(bwd)), 0.0)
    slopes[0], slopes[-1] = diff[0], diff[-1]
    slopes *= half.reshape(shape)
    # outer edges: the low edge of the first band is m + s*qlo (qlo <= 0), the high edge of the
    # last m + s*qhi (qhi >= 0); a zero mean has no sign to keep, so its band is flat
    m0, m1 = means[0], means[-1]
    if qlo < 0:
        cap = m0 / -qlo
        slopes[0] = np.where(m0 > 0, np.minimum(slopes[0], cap), np.where(m0 < 0, np.maximum(slopes[0], cap), 0.0))
    if qhi > 0:
        cap = -m1 / qhi
        slopes[-1] = np.where(m1 > 0, np.maximum(slopes[-1], cap), np.where(m1 < 0, np.minimum(slopes[-1], cap), 0.0))
    return slopes


def _fit_histopolation(freq, image, wgt, freq_bounds, method):
    """Fit ``(nband, nstokes, ny, nx)`` band values piecewise over frequency (see ``fit_image_cube``)."""
    f = sm.Symbol("f")
    lo, hi = _band_edges(freq, freq_bounds)
    keep = np.any(wgt != 0, axis=1)
    if not keep.any():
        raise ValueError("every band has zero weight")
    lo, hi, anchor, image = lo[keep], hi[keep], np.asarray(freq, dtype=float)[keep], image[keep]
    if np.any(anchor < lo) or np.any(anchor > hi):
        raise ValueError("each band's freq must lie within its bounds")
    nb, nstokes = image.shape[:2]

    y_index, x_index = np.where(np.any(image, axis=(0, 1)))
    means = image[:, :, y_index, x_index]  # (nb, nstokes, ncomps)
    mparams = sm.symbols(f"m(0:{nb})")
    if nb == 1:
        # a single band carries no slope information: its value is the model
        return means.transpose(1, 0, 2), y_index, x_index, str(mparams[0]), [str(mparams[0])], "t", "f"

    # each piece is anchored at the band's own freq (its weighted frequency in pfb-imaging),
    # not the edge midpoint, so rendering at freq returns the band value exactly
    half = (hi - lo) / 2
    qlo, qhi = (lo - anchor) / half, (hi - anchor) / half
    terms = list(mparams)
    coeffs = [means]
    params = list(mparams)
    if method == "tent":
        sparams = sm.symbols(f"s(0:{nb})")
        for b, sp in enumerate(sparams):
            # q runs over [qlo, qhi] across the band and is held at its edge values outside it
            q = sm.Min(sm.Max((f - float(anchor[b])) / float(half[b]), float(qlo[b])), float(qhi[b]))
            terms[b] = terms[b] + sp * q
        coeffs.append(_tent_slopes(means, anchor, half, qlo[0], qhi[-1]))
        params += list(sparams)
    # each band owns its own range; a gap between bands is split at its midpoint
    breaks = (hi[:-1] + lo[1:]) / 2
    expr = sm.Piecewise(*[(term, f < float(brk)) for term, brk in zip(terms[:-1], breaks)], (terms[-1], True))
    coeffs = np.concatenate(coeffs, axis=0).transpose(1, 0, 2)
    return coeffs, y_index, x_index, str(expr), [str(p) for p in params], "t", "f"


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
        method: A basis -- ``"poly"`` (monomials), ``"Legendre"``, or ``"fakenodes"``
            (Chebyshev polynomials of the cosine-mapped axis, see :func:`_fake_nodes`) -- or
            a histopolation, ``"box"`` or ``"tent"``. The polynomial bases ring between band
            centres (Runge's phenomenon) when ``nbasisf`` is close to ``nband``; fake nodes
            does not, at the cost of an O(1/nband) bias near the outer band edges.
            Histopolation treats each band value as the band's weighted average, with
            ``freq`` its weighted frequency: ``"box"`` holds it across the band, ``"tent"``
            adds a limited slope through ``(freq, value)`` (see :func:`_tent_slopes`), so its
            average over the band under the weights that define ``freq`` is the band value.
            Both render every band value exactly at ``freq`` and keep non-negative bands
            non-negative. They need ``ntime == 1`` and each ``freq`` inside its band, ignore
            ``nbasist``/``sigmasq`` and ``nbasisf`` (warning if it differs from ``nband``),
            drop bands with zero weight in every Stokes plane, and hold the end values beyond
            the outer edges.
        sigmasq: Ridge term added to the Hessian; ignored when ``ntime == nband == 1``
            (nothing to fit: the data are the coefficients, whatever the weights).
        time_bounds: Optional ``(ntime, 2)`` ``[lo, hi]`` per time bin; ``"fakenodes"`` only.
        freq_bounds: Optional ``(nband, 2)`` ``[lo, hi]`` in Hz per band, for
            ``"fakenodes"``, ``"box"`` and ``"tent"``. Without bounds the band edges lie
            midway between centres and half a spacing past the ends.

    Returns:
        ``(coeffs, y_index, x_index, expr, params, texpr, fexpr)``:
        - ``coeffs`` has shape ``(nstokes, npar, ncomps)``;
        - ``y_index``/``x_index`` are the component pixel locations, in array-axis order;
        - ``expr`` is the stringified sympy model in ``t``, ``f`` and ``params``;
        - ``texpr``/``fexpr`` map raw time/frequency onto the fit domain.

    Raises:
        ValueError: On inconsistent shapes or basis sizes, or (histopolation) a ``freq``
            outside its band.
        NotImplementedError: For an unknown ``method``, or histopolation with ``ntime > 1``.
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
    if method in HISTOPOLATION and nbasisf is not None and nbasisf != nband:
        warnings.warn(
            f"method {method!r} has one piece per band: nbasisf={nbasisf} is ignored for {nband} bands",
            stacklevel=2,
        )
    nbasist = ntime if nbasist is None else nbasist
    nbasisf = nband if nbasisf is None else nbasisf
    if not 1 <= nbasist <= ntime or not 1 <= nbasisf <= nband:
        raise ValueError(f"need 1 <= nbasist <= {ntime} and 1 <= nbasisf <= {nband}")

    if method in HISTOPOLATION:
        if ntime != 1:
            raise NotImplementedError(f"method {method!r} needs ntime == 1, got {ntime}")
        return _fit_histopolation(freq, image[0], wgt[0], freq_bounds, method)

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
