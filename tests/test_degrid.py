"""Synthetic, self-contained tests for the degrid kernel."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pfb_model_spec.utils.degrid import (
    apply_mueller,
    degrid_stokes,
    model_geometry,
    model_to_apparent_vis_for_region,
    render_model_region,
    stokes_vis_to_corr,
)
from pfb_model_spec.utils.io import build_mds_dataset
from pfb_model_spec.utils.modelspec import fit_image_cube


def test_stokes_vis_to_corr_linear():
    rng = np.random.default_rng(1)
    sv = rng.normal(size=(4, 5, 3)) + 1j * rng.normal(size=(4, 5, 3))
    i, q, u, v = sv
    out = stokes_vis_to_corr(sv, "IQUV", ("XX", "XY", "YX", "YY"))
    assert out.shape == (5, 3, 4)
    assert_allclose(out[..., 0], i + q)
    assert_allclose(out[..., 1], u + 1j * v)
    assert_allclose(out[..., 2], u - 1j * v)
    assert_allclose(out[..., 3], i - q)


def test_stokes_vis_to_corr_circular():
    rng = np.random.default_rng(2)
    sv = rng.normal(size=(4, 5, 3)) + 1j * rng.normal(size=(4, 5, 3))
    i, q, u, v = sv
    out = stokes_vis_to_corr(sv, "IQUV", ("RR", "RL", "LR", "LL"))
    assert_allclose(out[..., 0], i + v)
    assert_allclose(out[..., 1], q + 1j * u)
    assert_allclose(out[..., 2], q - 1j * u)
    assert_allclose(out[..., 3], i - v)


def test_stokes_vis_to_corr_stokes_i_only():
    """The v1 degrid path: an I-only model must give XX == YY == I, XY == YX == 0."""
    rng = np.random.default_rng(3)
    sv = rng.normal(size=(1, 5, 3)) + 0j
    out = stokes_vis_to_corr(sv, "I", ("XX", "XY", "YX", "YY"))
    assert_allclose(out[..., 0], sv[0])
    assert_allclose(out[..., 3], sv[0])
    assert np.abs(out[..., 1]).max() == 0.0
    assert np.abs(out[..., 2]).max() == 0.0


def test_stokes_vis_to_corr_two_correlations():
    rng = np.random.default_rng(4)
    sv = rng.normal(size=(2, 5, 3)) + 0j
    out = stokes_vis_to_corr(sv, "IQ", ("XX", "YY"))
    assert out.shape == (5, 3, 2)
    assert_allclose(out[..., 0], sv[0] + sv[1])
    assert_allclose(out[..., 1], sv[0] - sv[1])


def test_stokes_vis_to_corr_rejects_unknown_correlation():
    sv = np.zeros((1, 2, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="ZZ"):
        stokes_vis_to_corr(sv, "I", ("ZZ",))


def test_stokes_vis_to_corr_rejects_length_mismatch():
    sv = np.zeros((2, 2, 2), dtype=np.complex128)
    with pytest.raises(ValueError, match="stokes_in"):
        stokes_vis_to_corr(sv, "IQU", ("XX",))


# ducc0 wgridder conventions used throughout these tests; see
# https://github.com/mreineck/ducc/issues/34
_CONV = dict(flip_u=False, flip_v=True, flip_w=False, x0=0.0, y0=0.0)


def _uvw_freq(rng, nrow=2000, nchan=4):
    uvw = rng.normal(0.0, 3000.0, (nrow, 3))
    uvw[:, 2] *= 0.05
    return uvw, np.linspace(1.0e9, 1.1e9, nchan)


def test_degrid_stokes_centre_source_is_flat():
    """A point source at the phase centre gives V == flux on every baseline."""
    rng = np.random.default_rng(10)
    uvw, freq = _uvw_freq(rng)
    npix, cell = 128, np.deg2rad(2.0 / 3600.0)
    img = np.zeros((1, npix, npix))
    img[0, npix // 2, npix // 2] = 3.0
    vis = degrid_stokes(uvw, freq, img, cell_rad=cell, **_CONV)
    assert vis.shape == (1, uvw.shape[0], freq.size)
    assert_allclose(vis, 3.0, atol=1e-6)


def test_degrid_stokes_is_linear():
    rng = np.random.default_rng(11)
    uvw, freq = _uvw_freq(rng)
    npix, cell = 128, np.deg2rad(2.0 / 3600.0)
    a = np.zeros((1, npix, npix))
    a[0, 40, 70] = 1.5
    b = np.zeros((1, npix, npix))
    b[0, 90, 33] = -0.7
    kw = dict(cell_rad=cell, **_CONV)
    va = degrid_stokes(uvw, freq, a, **kw)
    vb = degrid_stokes(uvw, freq, b, **kw)
    vab = degrid_stokes(uvw, freq, a + b, **kw)
    assert_allclose(vab, va + vb, atol=1e-10)


def test_degrid_stokes_geometry_round_trip():
    """Degrid then grid back: the peak must land on the source pixel.

    A sign error in flip_u/flip_v mirrors the image, moving the peak to the
    mirrored pixel, so this pins the geometric conventions end to end.
    """
    from ducc0.wgridder.experimental import vis2dirty

    rng = np.random.default_rng(12)
    uvw, freq = _uvw_freq(rng)
    npix, cell = 128, np.deg2rad(2.0 / 3600.0)
    src = (37, 91)
    img = np.zeros((1, npix, npix))
    img[0, src[0], src[1]] = 1.0
    vis = degrid_stokes(uvw, freq, img, cell_rad=cell, **_CONV)
    back = vis2dirty(
        uvw=uvw,
        freq=freq,
        vis=np.ascontiguousarray(vis[0]),
        wgt=np.ones((uvw.shape[0], freq.size)),
        npix_x=npix,
        npix_y=npix,
        pixsize_x=cell,
        pixsize_y=cell,
        center_x=_CONV["x0"],
        center_y=_CONV["y0"],
        flip_u=_CONV["flip_u"],
        flip_v=_CONV["flip_v"],
        flip_w=_CONV["flip_w"],
        epsilon=1e-7,
        do_wgridding=True,
        divide_by_n=False,
        nthreads=1,
    )
    assert np.unravel_index(np.argmax(back), back.shape) == src


def test_degrid_stokes_masks_absent_rows():
    """xarray-ms pads absent (time, baseline) cells with NaN UVW."""
    rng = np.random.default_rng(13)
    uvw, freq = _uvw_freq(rng, nrow=500)
    uvw[10:20, :] = np.nan
    npix, cell = 64, np.deg2rad(2.0 / 3600.0)
    img = np.zeros((1, npix, npix))
    img[0, npix // 2, npix // 2] = 1.0
    mask = np.isfinite(uvw).all(axis=1)[:, None] & np.ones(freq.size, bool)[None, :]
    vis = degrid_stokes(uvw, freq, img, cell_rad=cell, mask=mask.astype(np.uint8), **_CONV)
    assert np.abs(vis[0][10:20]).max() == 0.0
    assert_allclose(vis[0][mask[:, 0]], 1.0, atol=1e-6)


def test_degrid_stokes_skips_empty_planes():
    """An all-zero Stokes plane costs no gridder call and returns zeros."""
    rng = np.random.default_rng(14)
    uvw, freq = _uvw_freq(rng, nrow=200)
    npix, cell = 64, np.deg2rad(2.0 / 3600.0)
    img = np.zeros((2, npix, npix))
    img[0, npix // 2, npix // 2] = 1.0
    vis = degrid_stokes(uvw, freq, img, cell_rad=cell, **_CONV)
    assert_allclose(vis[0], 1.0, atol=1e-6)
    assert np.abs(vis[1]).max() == 0.0


def _synthetic_mds(nx=64, ny=48, nband=4):
    """A two-component power-law model on a deliberately NON-square grid."""
    freq = np.linspace(1.0e9, 1.2e9, nband)
    time = np.array([0.0])
    cell = np.deg2rad(2.0 / 3600.0)
    cube = np.zeros((nband, nx, ny))
    cube[:, 20, 31] = (freq / freq[0]) ** -0.7
    cube[:, 44, 12] = 2.0 * (freq / freq[0]) ** -1.1
    coeffs, xi, yi, expr, params, texpr, fexpr = fit_image_cube(
        time, freq, cube[None], nbasisf=nband, method="Legendre", sigmasq=0
    )
    ds = build_mds_dataset(
        coeffs,
        xi,
        yi,
        expr,
        params,
        texpr,
        fexpr,
        time,
        freq,
        cell,
        nx,
        ny,
        0.0,
        0.0,
        False,
        True,
        False,
        (0.1, -0.5),
        "I",
        "test",
    )
    return ds, cube, time, freq, cell


def test_render_model_region_matches_the_fitted_cube():
    ds, cube, time, freq, _ = _synthetic_mds()
    for band in (0, freq.size - 1):
        got = render_model_region(ds, time=time[0], freq_out=freq[band])
        assert got.shape == (1, ds.attrs["npix_x"], ds.attrs["npix_y"])
        assert_allclose(got[0], cube[band], atol=1e-10)


def test_render_model_region_interpolates_between_fitted_bands():
    """The model is continuous in frequency -- this is what upsampling relies on."""
    ds, cube, time, freq, _ = _synthetic_mds()
    mid = 0.5 * (freq[0] + freq[1])
    got = render_model_region(ds, time=time[0], freq_out=mid)[0]
    lo, hi = cube[0, 20, 31], cube[1, 20, 31]
    assert min(lo, hi) < got[20, 31] < max(lo, hi)


def test_model_geometry_reads_the_mds_attrs():
    ds, _, _, _, cell = _synthetic_mds()
    geom = model_geometry(ds)
    assert geom["nx"] == 64 and geom["ny"] == 48
    assert geom["cell_rad"] == pytest.approx(cell)
    assert geom["x0"] == 0.0 and geom["y0"] == 0.0
    assert geom["flip_u"] is False and geom["flip_v"] is True
    assert geom["stokes"] == "I"


def test_model_geometry_rejects_non_square_pixels():
    """cell_rad_x != cell_rad_y has no single cell_rad to hand the gridder."""
    ds, _, _, _, cell = _synthetic_mds()
    ds.attrs["cell_rad_y"] = 2.0 * cell
    with pytest.raises(ValueError, match="square"):
        model_geometry(ds)


def test_render_model_region_rejects_unknown_spec():
    ds, _, time, freq, _ = _synthetic_mds()
    ds.attrs["spec"] = "some-future-spec"
    with pytest.raises(ValueError, match="some-future-spec"):
        render_model_region(ds, time=time[0], freq_out=freq[0])


def test_render_model_region_refine_conserves_flux():
    """Refining the grid (finer cell, more pixels) is exact -- see the module
    docstring's Note. This exercises the non-default-grid path, previously
    untested."""
    ds, time, freq, cell = _single_component_mds()
    nxi, nyi = ds.attrs["npix_x"], ds.attrs["npix_y"]
    got = render_model_region(
        ds,
        time=time[0],
        freq_out=freq[0],
        nx=2 * nxi,
        ny=2 * nyi,
        cell_rad=cell / 2,
    )
    assert got.sum() == pytest.approx(1.0, abs=1e-8)


def test_render_model_region_coarsen_inflates_flux_by_area_ratio():
    """Pins the documented `eval_coeffs_to_slice` limitation (module
    docstring's Note): coarsening is NOT flux-conserving, it inflates by
    `area_ratio` (here x4 for a x2 coarsening). This is a known limitation
    being pinned, not desired behaviour -- do not read this as an assertion
    that the inflation is correct."""
    ds, time, freq, cell = _single_component_mds()
    nxi, nyi = ds.attrs["npix_x"], ds.attrs["npix_y"]
    got = render_model_region(
        ds,
        time=time[0],
        freq_out=freq[0],
        nx=nxi // 2,
        ny=nyi // 2,
        cell_rad=cell * 2,
    )
    assert got.sum() == pytest.approx(4.0, rel=1e-6)


def _single_component_mds(nx=64, ny=48):
    """A single unit-flux, flat-spectrum component -- for pinning
    `eval_coeffs_to_slice`'s resampling scaling in isolation from any
    interaction between multiple components."""
    freq = np.linspace(1.0e9, 1.2e9, 2)
    time = np.array([0.0])
    cell = np.deg2rad(2.0 / 3600.0)
    cube = np.zeros((2, nx, ny))
    cube[:, nx // 2, ny // 2] = 1.0
    coeffs, xi, yi, expr, params, texpr, fexpr = fit_image_cube(
        time, freq, cube[None], nbasisf=2, method="Legendre", sigmasq=0
    )
    ds = build_mds_dataset(
        coeffs,
        xi,
        yi,
        expr,
        params,
        texpr,
        fexpr,
        time,
        freq,
        cell,
        nx,
        ny,
        0.0,
        0.0,
        False,
        True,
        False,
        (0.1, -0.5),
        "I",
        "test",
    )
    return ds, time, freq, cell


def _mueller(nso, nsi, nx, ny):
    m = np.zeros((nso, nsi, nx, ny))
    for i in range(min(nso, nsi)):
        m[i, i] = 1.0
    return m


def test_apply_mueller_identity_is_a_no_op():
    rng = np.random.default_rng(20)
    img = rng.normal(size=(4, 8, 6))
    out = apply_mueller(img, _mueller(4, 4, 8, 6))
    assert_allclose(out, img)


def test_apply_mueller_diagonal_scales_each_plane():
    rng = np.random.default_rng(21)
    img = rng.normal(size=(2, 8, 6))
    m = _mueller(2, 2, 8, 6)
    m[0, 0] *= 0.5
    m[1, 1] *= 3.0
    out = apply_mueller(img, m)
    assert_allclose(out[0], 0.5 * img[0])
    assert_allclose(out[1], 3.0 * img[1])


def test_apply_mueller_predicts_leakage_from_stokes_i():
    """The first Mueller column turns an I-only sky into apparent I, Q, U, V."""
    rng = np.random.default_rng(22)
    img = rng.normal(size=(1, 8, 6))
    m = rng.normal(size=(4, 1, 8, 6))
    out = apply_mueller(img, m)
    assert out.shape == (4, 8, 6)
    for i in range(4):
        assert_allclose(out[i], m[i, 0] * img[0])


def test_apply_mueller_rejects_shape_mismatch():
    img = np.zeros((2, 8, 6))
    with pytest.raises(ValueError, match="does not match"):
        apply_mueller(img, _mueller(4, 3, 8, 6))


def test_apply_mueller_rejects_grid_mismatch():
    img = np.zeros((1, 8, 6))
    with pytest.raises(ValueError, match="grid"):
        apply_mueller(img, _mueller(1, 1, 8, 7))


def test_fused_wrapper_equals_the_composition():
    ds, _, time, freq, cell = _synthetic_mds()
    rng = np.random.default_rng(30)
    uvw, chan = _uvw_freq(rng, nrow=400, nchan=3)
    corr = ("XX", "XY", "YX", "YY")

    fused = model_to_apparent_vis_for_region(ds, uvw=uvw, freq=chan, corr_types=corr, time=time[0], freq_out=freq[0])

    geom = model_geometry(ds)
    img = render_model_region(ds, time=time[0], freq_out=freq[0])
    sv = degrid_stokes(
        uvw,
        chan,
        img,
        cell_rad=geom["cell_rad"],
        x0=geom["x0"],
        y0=geom["y0"],
        flip_u=geom["flip_u"],
        flip_v=geom["flip_v"],
        flip_w=geom["flip_w"],
    )
    manual = stokes_vis_to_corr(sv, geom["stokes"], corr)

    assert fused.shape == (uvw.shape[0], chan.size, 4)
    assert_allclose(fused, manual)


def test_fused_wrapper_region_mask_selects_components():
    """Masking splits the model: the parts must sum back to the whole."""
    ds, _, time, freq, _ = _synthetic_mds()
    rng = np.random.default_rng(31)
    uvw, chan = _uvw_freq(rng, nrow=300, nchan=2)
    corr = ("XX", "YY")
    nx, ny = ds.attrs["npix_x"], ds.attrs["npix_y"]

    inside = np.zeros((nx, ny))
    inside[20, 31] = 1.0  # the first component only
    outside = 1.0 - inside

    kw = dict(uvw=uvw, freq=chan, corr_types=corr, time=time[0], freq_out=freq[0])
    whole = model_to_apparent_vis_for_region(ds, **kw)
    part_a = model_to_apparent_vis_for_region(ds, region_mask=inside, **kw)
    part_b = model_to_apparent_vis_for_region(ds, region_mask=outside, **kw)

    assert_allclose(part_a + part_b, whole, atol=1e-10)
    assert np.abs(part_a).max() > 0.0
    assert np.abs(part_b).max() > 0.0


def test_fused_wrapper_identity_mueller_matches_no_beam():
    ds, _, time, freq, _ = _synthetic_mds()
    rng = np.random.default_rng(32)
    uvw, chan = _uvw_freq(rng, nrow=200, nchan=2)
    corr = ("XX", "YY")
    nx, ny = ds.attrs["npix_x"], ds.attrs["npix_y"]
    kw = dict(uvw=uvw, freq=chan, corr_types=corr, time=time[0], freq_out=freq[0])
    no_beam = model_to_apparent_vis_for_region(ds, **kw)
    with_beam = model_to_apparent_vis_for_region(ds, mueller=_mueller(1, 1, nx, ny), **kw)
    assert_allclose(with_beam, no_beam, atol=1e-10)


def test_stokes_out_relabels_a_non_prefix_mueller():
    """A 2-plane (I, V) Mueller output is mislabelled "IQ" by the default
    IQUV-prefix assumption; `stokes_out` lets the caller name it correctly.
    Circular correlations make the I/V distinction visible (Q does not enter
    RR/LL)."""
    ds, _, time, freq, _ = _synthetic_mds()
    rng = np.random.default_rng(33)
    uvw, chan = _uvw_freq(rng, nrow=200, nchan=2)
    corr = ("RR", "LL")
    nx, ny = ds.attrs["npix_x"], ds.attrs["npix_y"]

    # a genuine (I, V) Mueller: plane 0 keeps I, plane 1 leaks some I into V
    mueller = np.zeros((2, 1, nx, ny))
    mueller[0, 0] = 1.0
    mueller[1, 0] = 0.3

    kw = dict(uvw=uvw, freq=chan, corr_types=corr, time=time[0], freq_out=freq[0], mueller=mueller)
    labelled = model_to_apparent_vis_for_region(ds, stokes_out="IV", **kw)
    default = model_to_apparent_vis_for_region(ds, **kw)

    geom = model_geometry(ds)
    image = apply_mueller(render_model_region(ds, time=time[0], freq_out=freq[0]), mueller)
    stokes_vis = degrid_stokes(
        uvw,
        chan,
        image,
        cell_rad=geom["cell_rad"],
        x0=geom["x0"],
        y0=geom["y0"],
        flip_u=geom["flip_u"],
        flip_v=geom["flip_v"],
        flip_w=geom["flip_w"],
    )
    manual = stokes_vis_to_corr(stokes_vis, "IV", corr)

    assert_allclose(labelled, manual)
    assert not np.allclose(labelled, default)


def test_stokes_out_rejects_length_mismatch():
    ds, _, time, freq, _ = _synthetic_mds()
    rng = np.random.default_rng(34)
    uvw, chan = _uvw_freq(rng, nrow=50, nchan=2)
    with pytest.raises(ValueError, match="stokes_out"):
        model_to_apparent_vis_for_region(
            ds,
            uvw=uvw,
            freq=chan,
            corr_types=("XX", "YY"),
            time=time[0],
            freq_out=freq[0],
            stokes_out="IQU",
        )
