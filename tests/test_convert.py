"""Tests for the `.mds` spec converter (core implementation)."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from pfb_model_spec.core.convert import convert
from pfb_model_spec.utils.modelspec import model_from_mds
from pfb_model_spec.utils.spec import SPEC_VERSION, upgrade
from tests._genesis import genesis_cube, genesis_dataset


def test_convert_upgrades_genesis_on_disk(tmp_path):
    src, dst = tmp_path / "g.mds", tmp_path / "new.mds"
    genesis_dataset().to_zarr(src)
    convert(str(src), str(dst))

    raw = xr.open_zarr(dst, chunks=None)  # read without upgrading: the store itself is 0.1
    assert raw.attrs["spec"] == SPEC_VERSION
    assert list(raw.stokes.values) == ["I"]
    assert_allclose(model_from_mds(str(dst))[0, :, 0], genesis_cube().transpose(0, 2, 1), atol=1e-12)
    # the source is untouched
    assert xr.open_zarr(src, chunks=None).attrs["spec"] == "genesis"


def test_convert_writes_a_current_spec_input_through(tmp_path):
    src, dst = tmp_path / "cur.mds", tmp_path / "copy.mds"
    upgrade(genesis_dataset()).to_zarr(src)
    convert(str(src), str(dst))
    xr.testing.assert_identical(xr.open_zarr(dst, chunks=None), xr.open_zarr(src, chunks=None))


def test_convert_refuses_to_overwrite_without_the_flag(tmp_path):
    src, dst = tmp_path / "g.mds", tmp_path / "new.mds"
    genesis_dataset().to_zarr(src)
    convert(str(src), str(dst))
    with pytest.raises(ValueError, match="exists"):
        convert(str(src), str(dst))
    convert(str(src), str(dst), overwrite=True)


@pytest.mark.parametrize("alias", ["{p}", "{p}/", "./{rel}", "file://{p}"])
def test_convert_refuses_in_place(tmp_path, monkeypatch, alias):
    """The same store named differently is still in-place, even with --overwrite."""
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "g.mds"
    genesis_dataset().to_zarr(src)
    with pytest.raises(ValueError, match="in-place"):
        convert(str(src), alias.format(p=src, rel="g.mds"), overwrite=True)


def test_convert_works_on_remote_uris():
    """fsspec's in-memory filesystem stands in for s3:// / gs:// here."""
    src, dst = "memory://convert-test/g.mds", "memory://convert-test/new.mds"
    genesis_dataset().to_zarr(src, mode="w")
    convert(src, dst)
    assert xr.open_zarr(dst, chunks=None).attrs["spec"] == SPEC_VERSION
    with pytest.raises(ValueError, match="exists"):
        convert(src, dst)
    with pytest.raises(ValueError, match="in-place"):
        convert(src, src + "/", overwrite=True)


def test_convert_rejects_a_newer_spec(tmp_path):
    src = tmp_path / "future.mds"
    genesis_dataset(spec="9.9").to_zarr(src)
    with pytest.raises(ValueError, match="needs pfb-model-spec"):
        convert(str(src), str(tmp_path / "out.mds"))
    assert not (tmp_path / "out.mds").exists()


def test_convert_accepts_path_objects(tmp_path):
    """The CLI hands core UPath objects, not strings."""
    from upath import UPath

    src = tmp_path / "g.mds"
    genesis_dataset().to_zarr(src)
    convert(UPath(src), UPath(tmp_path / "new.mds"))
    assert np.array_equal(xr.open_zarr(tmp_path / "new.mds", chunks=None).location_x.values, [4, 27, 15])
