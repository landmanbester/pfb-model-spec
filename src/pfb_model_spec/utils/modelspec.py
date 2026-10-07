import numpy as np
import sympy as sm
from scipy.interpolate import RegularGridInterpolator
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities.lambdify import lambdify

from pfb_model_spec.utils.spec import open_mds


def _scale(x, sym, method):
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


def fit_image_cube(time, freq, image, wgt=None, nbasist=None, nbasisf=None, method="poly", sigmasq=0):
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
        method: ``"poly"`` (monomials) or ``"Legendre"``.
        sigmasq: Ridge term added to the Hessian; ignored when ``ntime == nband == 1``
            (nothing to fit, the data are the coefficients).

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
    if ntime == 1 and nband == 1:
        sigmasq = 0

    if method == "poly":

        def basis(i, w, sym):
            return w**i, sym**i

    elif method == "Legendre":

        def basis(i, w, sym):
            return np.polynomial.Legendre.basis(i)(w), sm.polys.orthopolys.legendre_poly(i, sym)

    else:
        raise NotImplementedError(f"Method {method} not implemented")

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
        w = wgt[:, s : s + 1]
        hess = xfit.T.dot(w * xfit)
        if sigmasq:
            hess += sigmasq * np.eye(hess.shape[0])
        coeffs[s] = np.linalg.solve(hess, xfit.T.dot(w * beta[:, s]))

    params = [*tparams, *fparams]
    return coeffs, y_index, x_index, str(expr), [str(p) for p in params], str(tfunc), str(ffunc)


def _model_functions(expr, paramf, texpr, fexpr):
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
):
    """Render coefficients at one (time, freq) onto an arbitrary output grid.

    Returns ``(nstokes, nyo, nxo)``. The geometry is keyword-only so that a call written
    against the old x-major positional signature fails loudly instead of swapping axes.
    """
    modelf, tfunc, ffunc = _model_functions(expr, paramf, texpr, fexpr)
    tval, fval = tfunc(time), ffunc(freq)
    out = np.zeros((coeffs.shape[0], nyo, nxo), dtype=float)
    for s in range(coeffs.shape[0]):
        image_in = np.zeros((nxi, nyi), dtype=float)
        image_in[x_index, y_index] = modelf(tval, fval, *coeffs[s])
        # resampling works x-major internally; transpose once on the way out
        out[s] = _resample_xmajor(image_in, nxi, nyi, cellxi, cellyi, x0i, y0i, nxo, nyo, cellxo, cellyo, x0o, y0o).T
    return out


def _resample_xmajor(image_in, nxi, nyi, cellxi, cellyi, x0i, y0i, nxo, nyo, cellxo, cellyo, x0o, y0o):
    """Zero-pad and bilinearly resample an x-major ``(nxi, nyi)`` slice onto ``(nxo, nyo)``."""
    pix_area_in = cellxi * cellyi
    pix_area_out = cellxo * cellyo
    area_ratio = pix_area_out / pix_area_in

    xin = (-(nxi // 2) + np.arange(nxi)) * cellxi + x0i
    yin = (-(nyi // 2) + np.arange(nyi)) * cellyi + y0i
    xo = (-(nxo // 2) + np.arange(nxo)) * cellxo + x0o
    yo = (-(nyo // 2) + np.arange(nyo)) * cellyo + y0o

    # how many pixels to pad by to extrapolate with zeros
    xldiff = xin.min() - xo.min()
    if xldiff > 0.0:
        npadxl = int(np.ceil(xldiff / cellxi))
    else:
        npadxl = 0
    yldiff = yin.min() - yo.min()
    if yldiff > 0.0:
        npadyl = int(np.ceil(yldiff / cellyi))
    else:
        npadyl = 0

    xudiff = xo.max() - xin.max()
    if xudiff > 0.0:
        npadxu = int(np.ceil(xudiff / cellxi))
    else:
        npadxu = 0
    yudiff = yo.max() - yin.max()
    if yudiff > 0.0:
        npadyu = int(np.ceil(yudiff / cellyi))
    else:
        npadyu = 0

    do_pad = npadxl > 0
    do_pad |= npadxu > 0
    do_pad |= npadyl > 0
    do_pad |= npadyu > 0
    if do_pad:
        image_in = np.pad(image_in, ((npadxl, npadxu), (npadyl, npadyu)), mode="constant")

        xin = (-(nxi // 2 + npadxl) + np.arange(nxi + npadxl + npadxu)) * cellxi + x0i
        nxi = nxi + npadxl + npadxu
        yin = (-(nyi // 2 + npadyl) + np.arange(nyi + npadyl + npadyu)) * cellyi + y0i
        nyi = nyi + npadyl + npadyu

    do_interp = cellxi != cellxo
    do_interp |= cellyi != cellyo
    do_interp |= x0i != x0o
    do_interp |= y0i != y0o
    do_interp |= nxi != nxo
    do_interp |= nyi != nyo
    if do_interp:
        interpo = RegularGridInterpolator((xin, yin), image_in, bounds_error=True, method="linear")
        xx, yy = np.meshgrid(xo, yo, indexing="ij")
        return interpo((xx, yy)) * area_ratio
    else:
        return image_in


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
