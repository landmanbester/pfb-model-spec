"""Upgrade a `.mds` component model to the current spec (``pfbspec convert``).

Conversion is one-way (older -> newer) and is exactly the in-memory upgrade every reader
already applies (`pfb_model_spec.utils.spec.upgrade`), persisted to a new store. In-place
conversion is not supported: the source is opened lazily, so writing over it would destroy
the data being read.
"""

import logging
import os

import xarray as xr
from upath import UPath

from pfb_model_spec.utils.spec import SPEC_VERSION, spec_of, upgrade

log = logging.getLogger("pfb_model_spec.convert")


def _canonical(path: str | os.PathLike) -> str:
    """Normalise a local path or remote URI so two spellings of one store compare equal."""
    up = UPath(path)
    if up.protocol in ("", "file", "local"):
        return os.path.realpath(up.path)
    return f"{up.protocol}://{up.path.rstrip('/')}"


def convert(input_mds: str | os.PathLike, output_mds: str | os.PathLike, overwrite: bool = False) -> None:
    """Upgrade ``input_mds`` to ``SPEC_VERSION`` and write the result to ``output_mds``.

    Args:
        input_mds: Source `.mds` (local path or remote URI) at any known spec.
        output_mds: Destination `.mds`; must neither be, contain, nor lie inside the source.
        overwrite: Replace ``output_mds`` if it exists.

    Raises:
        ValueError: If input and output are the same store or one contains the other, if the output exists and
            ``overwrite`` is False, or if the input's spec is unknown or newer than this
            package understands.
    """
    if not log.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    src, dst = _canonical(input_mds), _canonical(output_mds)
    if src == dst:
        raise ValueError(f"Input and output are the same store ({input_mds}); in-place conversion is not supported")
    if src.startswith(dst.rstrip("/") + "/") or dst.startswith(src.rstrip("/") + "/"):
        raise ValueError(f"One of {input_mds} and {output_mds} contains the other; writing would corrupt the input")
    if UPath(output_mds).exists() and not overwrite:
        raise ValueError(f"{output_mds} exists. Set overwrite=True to replace it.")

    raw = xr.open_zarr(str(input_mds), chunks=None)
    source = spec_of(raw)
    ds = upgrade(raw)  # raises before anything is written
    if source == SPEC_VERSION:
        log.info(f"{input_mds} is already at spec {SPEC_VERSION}; writing it through unchanged")
    else:
        log.info(f"Upgrading {input_mds} from spec {source} to {SPEC_VERSION}")
    ds.to_zarr(str(output_mds), mode="w")
    log.info(f"Wrote {output_mds}")
