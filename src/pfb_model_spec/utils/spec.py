"""`.mds` spec versions and the one-way upgrade path between them.

Policy (see `.claude/rules/component-model.md`): a spec is named after the ``major.minor``
of the pfb-model-spec release that introduced it, and any schema change requires a breaking
version bump. ``"genesis"`` is the legacy name of every pre-versioning ``.mds`` (a literal ``"0.0"`` is not
a known spec). Older specs
always upgrade to newer ones; there is no downgrade path, because a newer spec may hold
things an older one cannot express.

Every step in ``_UPGRADES`` is a pure ``Dataset -> Dataset`` function. Readers call
:func:`open_mds` (or :func:`upgrade` on an already-open dataset), so a `.mds` written under
any known spec keeps working; ``pfbspec convert`` persists the same upgrade to disk.
"""

import os
from collections.abc import Callable

import xarray as xr

SPEC_VERSION = "0.1"
LEGACY_SPEC = "genesis"

_STOKES = ("I", "Q", "U", "V")


def spec_of(ds: xr.Dataset) -> str:
    """Return the spec a `.mds` dataset declares; a missing attr means ``"genesis"``."""
    return str(ds.attrs.get("spec", LEGACY_SPEC))


def _genesis_to_0_1(ds: xr.Dataset) -> xr.Dataset:
    """genesis -> 0.1: add a length-1 Stokes axis and put the locations on ``comps``.

    Lossless: no values change. ``location_x`` indexes FITS ``NAXIS1`` in both specs; the
    ``(Y, X)`` change in 0.1 affects only the axis order of rendered cubes.
    """
    if ds.coefficients.ndim != 2:
        raise ValueError(
            f"This genesis .mds has multi-correlation coefficients (dims {ds.coefficients.dims}); "
            "it cannot be upgraded -- re-fit it with `pfbspec model2comps`"
        )
    if "stokes" not in ds.attrs:
        raise ValueError(
            "genesis .mds has no 'stokes' attr, so its Stokes product is unknown; "
            "it cannot be upgraded -- re-fit it with `pfbspec model2comps`"
        )
    attrs = dict(ds.attrs)
    stokes = str(attrs.pop("stokes"))
    if stokes not in _STOKES:
        raise ValueError(f"A genesis .mds carries a single Stokes product, got {stokes!r}")
    attrs["writer-version"] = attrs.pop("pfb-imaging-version", "unknown")
    attrs["spec"] = "0.1"
    # rebuilt from .values so no zarr encoding from the old layout leaks into a later write
    return xr.Dataset(
        data_vars={"coefficients": (("stokes", "par", "comps"), ds.coefficients.values[None])},
        coords={
            "stokes": (("stokes",), [stokes]),
            "location_x": (("comps",), ds.location_x.values),
            "location_y": (("comps",), ds.location_y.values),
            "params": (("par",), ds.params.values),
            "times": (("t",), ds.times.values),
            "freqs": (("f",), ds.freqs.values),
        },
        attrs=attrs,
    )


# from-spec -> (to-spec, step). A breaking release that leaves the schema alone still adds
# a no-op step here, so that every spec name is explicitly defined.
_UPGRADES: dict[str, tuple[str, Callable[[xr.Dataset], xr.Dataset]]] = {
    LEGACY_SPEC: ("0.1", _genesis_to_0_1),
}


def _as_version(spec: str) -> tuple[int, int] | None:
    parts = spec.split(".")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1])


def upgrade(ds: xr.Dataset) -> xr.Dataset:
    """Upgrade a `.mds` dataset to ``SPEC_VERSION``; a current dataset is returned as is.

    Args:
        ds: An opened `.mds` dataset at any known spec.

    Returns:
        The dataset at ``SPEC_VERSION``. The input is never mutated.

    Raises:
        ValueError: If the spec is newer than this package understands, unknown, or the
            dataset cannot be expressed in the newer spec.
    """
    spec = spec_of(ds)
    while spec != SPEC_VERSION:
        if spec not in _UPGRADES:
            version = _as_version(spec)
            if version is not None and version > _as_version(SPEC_VERSION):
                raise ValueError(
                    f"This .mds uses spec {spec!r}, which needs pfb-model-spec >= {spec}; "
                    f"this installation reads spec {SPEC_VERSION!r}"
                )
            known = sorted([*_UPGRADES, SPEC_VERSION])
            raise ValueError(f"Unknown .mds spec {spec!r}; known specs: {known}")
        spec, step = _UPGRADES[spec]
        ds = step(ds)
    return ds


def open_mds(path: str | os.PathLike) -> xr.Dataset:
    """Open a `.mds` zarr store (local path or remote URI) and upgrade it to ``SPEC_VERSION``."""
    return upgrade(xr.open_zarr(str(path), chunks=None))
