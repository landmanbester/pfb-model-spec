"""Tests for the .mds spec registry and one-way upgrade path."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pfb_model_spec import __version__
from pfb_model_spec.utils.spec import LEGACY_SPEC, SPEC_VERSION, open_mds, spec_of, upgrade
from tests._genesis import COEFFS, LOC_X, LOC_Y, genesis_dataset


def _minor(version: str) -> tuple[int, int]:
    major, minor = version.split(".")[:2]
    return int(major), int(minor)


def test_spec_version_is_not_behind_the_package():
    """Bumping the package to a new minor needs a new spec step (even a no-op one)."""
    assert _minor(SPEC_VERSION) >= _minor(__version__)


def test_upgrade_genesis_gives_the_current_schema():
    up = upgrade(genesis_dataset())
    assert spec_of(up) == SPEC_VERSION == "0.1"
    assert up.coefficients.dims == ("stokes", "par", "comps")
    assert_array_equal(up.coefficients.values, COEFFS[None])
    assert list(up.stokes.values) == ["I"]
    assert up.location_x.dims == ("comps",) and up.location_y.dims == ("comps",)
    # stored pixel indices are untouched: the (Y, X) change is render-order only
    assert_array_equal(up.location_x.values, LOC_X)
    assert_array_equal(up.location_y.values, LOC_Y)
    assert up.attrs["writer-version"] == "0.0.3"
    assert "pfb-imaging-version" not in up.attrs
    assert "stokes" not in up.attrs


def test_upgrade_keeps_every_other_attr():
    g = genesis_dataset()
    up = upgrade(g)
    moved = {"spec", "stokes", "pfb-imaging-version"}
    for key, value in g.attrs.items():
        if key not in moved:
            assert up.attrs[key] == value, key


def test_upgrade_does_not_mutate_its_input():
    g = genesis_dataset()
    before = dict(g.attrs)
    upgrade(g)
    assert g.attrs == before
    assert g.coefficients.dims == ("par", "comps")


def test_missing_spec_attr_means_genesis():
    g = genesis_dataset(spec=None)
    assert spec_of(g) == LEGACY_SPEC
    assert spec_of(upgrade(g)) == SPEC_VERSION


def test_missing_version_attr_still_upgrades():
    up = upgrade(genesis_dataset(**{"pfb-imaging-version": None}))
    assert up.attrs["writer-version"] == "unknown"


def test_multi_stokes_genesis_is_rejected():
    with pytest.raises(ValueError, match="single Stokes"):
        upgrade(genesis_dataset(stokes="IQ"))


def test_unknown_spec_is_rejected():
    with pytest.raises(ValueError, match="Unknown .mds spec 'exodus'"):
        upgrade(genesis_dataset(spec="exodus"))


def test_newer_spec_asks_for_a_newer_package():
    with pytest.raises(ValueError, match=r"needs pfb-model-spec >= 9\.9"):
        upgrade(genesis_dataset(spec="9.9"))


def test_upgrade_of_current_spec_is_a_no_op():
    up = upgrade(genesis_dataset())
    assert upgrade(up) is up


def test_open_mds_upgrades_on_read(tmp_path):
    path = tmp_path / "g.mds"
    genesis_dataset().to_zarr(path)
    ds = open_mds(path)
    assert spec_of(ds) == SPEC_VERSION
    assert np.array_equal(ds.coefficients.values, COEFFS[None])


def test_genesis_without_a_stokes_attr_is_rejected_clearly():
    with pytest.raises(ValueError, match="no 'stokes' attr"):
        upgrade(genesis_dataset(stokes=None))


def test_genesis_with_multi_correlation_coefficients_is_rejected_clearly():
    import xarray as xr

    ds = genesis_dataset()
    ds = ds.assign(coefficients=(("corr", "par", "comps"), ds.coefficients.values[None]))
    assert isinstance(ds, xr.Dataset)
    with pytest.raises(ValueError, match="multi-correlation"):
        upgrade(ds)
