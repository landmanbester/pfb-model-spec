"""Frozen writer for `genesis`-spec `.mds` fixtures.

The library no longer writes genesis, so these fixtures pin what a genesis `.mds` looked
like at pfb-model-spec 0.0.3. Do NOT update this module to track library changes -- its
whole value is that it does not move.

The model is written by hand rather than fitted: ``expr = t0 + f1 * fs`` with
``fs = (f - 1.1e9) / 1e8``, so the reference render below is exact and independent of
the library's fit/render code.
"""

import numpy as np
import xarray as xr

NX = 32  # non-square on purpose: a swapped axis must not pass silently
NY = 24
FREQS = np.array([1.0e9, 1.1e9, 1.2e9])
TIMES = np.array([0.0])
CELL_RAD = float(np.deg2rad(2.0 / 3600.0))
LOC_X = np.array([4, 27, 15])
LOC_Y = np.array([3, 20, 11])
# (par, comps): rows are the t0 and f1 coefficients
COEFFS = np.array([[1.0, -2.0, 0.5], [0.3, 0.1, -0.4]])
PARAMS = ["t0", "f1"]
EXPR = "f*f1 + t0"
TEXPR = "t"
FEXPR = "(f - 1100000000.0)/100000000.0"


def genesis_dataset(**attr_overrides) -> xr.Dataset:
    """A genesis `.mds` dataset; an override value of ``None`` deletes that attr."""
    attrs = {
        "pfb-imaging-version": "0.0.3",
        "spec": "genesis",
        "cell_rad_x": CELL_RAD,
        "cell_rad_y": CELL_RAD,
        "npix_x": NX,
        "npix_y": NY,
        "texpr": TEXPR,
        "fexpr": FEXPR,
        "center_x": 0.0,
        "center_y": 0.0,
        "flip_u": False,
        "flip_v": True,
        "flip_w": False,
        "ra": 0.1,
        "dec": -0.5,
        "stokes": "I",
        "parametrisation": EXPR,
    }
    for key, value in attr_overrides.items():
        if value is None:
            attrs.pop(key, None)
        else:
            attrs[key] = value
    return xr.Dataset(
        data_vars={"coefficients": (("par", "comps"), COEFFS)},
        coords={
            "location_x": (("x",), LOC_X),
            "location_y": (("y",), LOC_Y),
            "params": (("par",), PARAMS),
            "times": (("t",), TIMES),
            "freqs": (("f",), FREQS),
        },
        attrs=attrs,
    )


def genesis_cube(freqs: np.ndarray = FREQS) -> np.ndarray:
    """Exact reference render of ``genesis_dataset()``, x-major ``(nfreq, NX, NY)``."""
    cube = np.zeros((freqs.size, NX, NY))
    fs = (freqs - 1.1e9) / 1.0e8
    cube[:, LOC_X, LOC_Y] = COEFFS[0][None, :] + COEFFS[1][None, :] * fs[:, None]
    return cube
