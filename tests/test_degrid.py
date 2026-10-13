"""Synthetic, self-contained tests for the degrid kernel."""

import numpy as np
import pytest
from ducc0.wgridder.experimental import dirty2vis
from numpy.testing import assert_allclose

from pfb_model_spec.utils.degrid import (
    BeamTerm,
    apply_mueller,
    crop_bbox,
    degrid_stokes,
    interp_mueller,
    model_geometry,
    model_to_apparent_vis_for_regions,
    pixel_lm,
    render_model_region,
    stokes_vis_to_corr,
)
from pfb_model_spec.utils.io import build_mds_dataset
from pfb_model_spec.utils.modelspec import fit_image_cube
from tests._genesis import FREQS, genesis_cube, genesis_dataset


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
    assert np.unravel_index(np.argmax(back), back.shape) == (src[1], src[0])


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
    cube = np.zeros((nband, 1, ny, nx))
    cube[:, 0, 31, 20] = (freq / freq[0]) ** -0.7
    cube[:, 0, 12, 44] = 2.0 * (freq / freq[0]) ** -1.1
    coeffs, yi, xi, expr, params, texpr, fexpr = fit_image_cube(
        time, freq, cube[None], nbasisf=nband, method="Legendre", sigmasq=0
    )
    ds = build_mds_dataset(
        coeffs,
        yi,
        xi,
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
        ["I"],
        "test",
    )
    return ds, cube[:, 0], time, freq, cell


def test_render_model_region_matches_the_fitted_cube():
    ds, cube, time, freq, _ = _synthetic_mds()
    for band in (0, freq.size - 1):
        got = render_model_region(ds, time=time[0], freq_out=freq[band])
        assert got.shape == (1, ds.attrs["npix_y"], ds.attrs["npix_x"])
        assert_allclose(got[0], cube[band], atol=1e-10)


def test_render_model_region_interpolates_between_fitted_bands():
    """The model is continuous in frequency -- this is what upsampling relies on."""
    ds, cube, time, freq, _ = _synthetic_mds()
    mid = 0.5 * (freq[0] + freq[1])
    got = render_model_region(ds, time=time[0], freq_out=mid)[0]
    lo, hi = cube[0, 31, 20], cube[1, 31, 20]
    assert min(lo, hi) < got[31, 20] < max(lo, hi)


def test_model_geometry_reads_the_mds_attrs():
    ds, _, _, _, cell = _synthetic_mds()
    geom = model_geometry(ds)
    assert geom["nx"] == 64 and geom["ny"] == 48
    assert geom["cell_rad"] == pytest.approx(cell)
    assert geom["x0"] == 0.0 and geom["y0"] == 0.0
    assert geom["flip_u"] is False and geom["flip_v"] is True
    assert geom["stokes"] == ["I"]


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
    """Refining the grid (finer cell, more pixels) conserves flux."""
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


@pytest.mark.parametrize("kernel", ["box", "tent"])
def test_render_model_region_coarsen_conserves_flux(kernel):
    """#23: coarsening used to inflate flux by `area_ratio` (x4 for this x2 coarsening)."""
    ds, time, freq, cell = _single_component_mds()
    nxi, nyi = ds.attrs["npix_x"], ds.attrs["npix_y"]
    got = render_model_region(
        ds,
        time=time[0],
        freq_out=freq[0],
        nx=nxi // 2,
        ny=nyi // 2,
        cell_rad=cell * 2,
        kernel=kernel,
    )
    assert got.sum() == pytest.approx(1.0, abs=1e-8)


def _single_component_mds(nx=64, ny=48):
    """A single unit-flux, flat-spectrum component -- for pinning
    `eval_coeffs_to_slice`'s resampling scaling in isolation from any
    interaction between multiple components."""
    freq = np.linspace(1.0e9, 1.2e9, 2)
    time = np.array([0.0])
    cell = np.deg2rad(2.0 / 3600.0)
    cube = np.zeros((2, 1, ny, nx))
    cube[:, 0, ny // 2, nx // 2] = 1.0
    coeffs, yi, xi, expr, params, texpr, fexpr = fit_image_cube(
        time, freq, cube[None], nbasisf=2, method="Legendre", sigmasq=0
    )
    ds = build_mds_dataset(
        coeffs,
        yi,
        xi,
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
        ["I"],
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


def _composed(ds, uvw, chan, corr, time, freq_out, *, mueller=None, region_mask=None, stokes=None, epsilon=1e-7):
    """The fused wrapper by hand: render, mask, attenuate, degrid (complex as Re + i·Im), convert."""
    geom = model_geometry(ds)
    image = render_model_region(ds, time=time, freq_out=freq_out)
    if region_mask is not None:
        image = image * region_mask[None]
    if mueller is not None:
        image = apply_mueller(image, mueller)
    kw = dict(
        cell_rad=geom["cell_rad"],
        x0=geom["x0"],
        y0=geom["y0"],
        flip_u=geom["flip_u"],
        flip_v=geom["flip_v"],
        flip_w=geom["flip_w"],
        epsilon=epsilon,
    )
    sv = degrid_stokes(uvw, chan, np.ascontiguousarray(image.real), **kw)
    if np.iscomplexobj(image):
        sv = sv + 1j * degrid_stokes(uvw, chan, np.ascontiguousarray(image.imag), **kw)
    return stokes_vis_to_corr(sv, geom["stokes"] if stokes is None else stokes, corr)


def test_fused_wrapper_equals_the_composition():
    ds, _, time, freq, cell = _synthetic_mds()
    rng = np.random.default_rng(30)
    uvw, chan = _uvw_freq(rng, nrow=400, nchan=3)
    corr = ("XX", "XY", "YX", "YY")
    (fused,) = model_to_apparent_vis_for_regions(
        ds, uvw=uvw, freq=chan, corr_types=corr, time=time[0], freq_out=freq[0]
    )
    assert fused.shape == (uvw.shape[0], chan.size, 4)
    assert_allclose(fused, _composed(ds, uvw, chan, corr, time[0], freq[0]))


def test_fused_wrapper_region_mask_selects_components():
    """Masking splits the model: the parts must sum back to the whole."""
    ds, _, time, freq, _ = _synthetic_mds()
    rng = np.random.default_rng(31)
    uvw, chan = _uvw_freq(rng, nrow=300, nchan=2)
    corr = ("XX", "YY")
    nx, ny = ds.attrs["npix_x"], ds.attrs["npix_y"]

    inside = np.zeros((ny, nx))
    inside[31, 20] = 1.0  # the first component only
    outside = 1.0 - inside

    kw = dict(uvw=uvw, freq=chan, corr_types=corr, time=time[0], freq_out=freq[0])
    (whole,) = model_to_apparent_vis_for_regions(ds, **kw)
    part_a, part_b = model_to_apparent_vis_for_regions(ds, region_masks=[crop_bbox(inside), crop_bbox(outside)], **kw)

    assert_allclose(part_a + part_b, whole, atol=1e-10)
    assert np.abs(part_a).max() > 0.0
    assert np.abs(part_b).max() > 0.0


def test_stokes_out_labels_the_mueller_output_axis():
    """With a beam, `stokes_out` names the Mueller's output planes, here a non-prefix (I, V).

    Circular correlations make the I/V distinction visible (Q does not enter
    RR/LL). There is no default to guess from: leaving `stokes_out` as the
    model's single plane is refused for a 2-plane output.
    """
    ds, _, time, freq, _ = _synthetic_mds()
    geom = model_geometry(ds)
    rng = np.random.default_rng(33)
    uvw, chan = _uvw_freq(rng, nrow=200, nchan=2)
    corr = ("RR", "LL")

    # a genuine (I, V) Mueller: plane 0 keeps I, plane 1 leaks some I into V
    la, ma = _flat_axes(geom)
    block = np.zeros((2, 1, ma.size, la.size))
    block[0, 0] = 1.0
    block[1, 0] = 0.3

    kw = dict(uvw=uvw, freq=chan, corr_types=corr, time=time[0], freq_out=freq[0])
    (labelled,) = model_to_apparent_vis_for_regions(ds, stokes_out="IV", beam_terms=[BeamTerm(block, la, ma)], **kw)
    mueller = np.broadcast_to(np.array([[1.0], [0.3]])[:, :, None, None], (2, 1, geom["ny"], geom["nx"]))
    manual = _composed(ds, uvw, chan, corr, time[0], freq[0], mueller=mueller, stokes="IV")
    assert_allclose(labelled, manual)

    with pytest.raises(ValueError, match="mueller"):
        model_to_apparent_vis_for_regions(ds, beam_terms=[BeamTerm(block, la, ma)], **kw)


def test_stokes_out_rejects_length_mismatch():
    ds, _, time, freq, _ = _synthetic_mds()
    rng = np.random.default_rng(34)
    uvw, chan = _uvw_freq(rng, nrow=50, nchan=2)
    with pytest.raises(ValueError, match="stokes_out"):
        model_to_apparent_vis_for_regions(
            ds,
            uvw=uvw,
            freq=chan,
            corr_types=("XX", "YY"),
            time=time[0],
            freq_out=freq[0],
            stokes_out="IQU",
        )


def test_genesis_model_degrids_like_its_x_major_render():
    """Pins orientation across the upgrade: a genesis .mds (x-major, non-square) must predict
    exactly what degridding its own x-major render with ducc0 directly predicts."""
    ds = genesis_dataset()
    rng = np.random.default_rng(40)
    uvw, chan = _uvw_freq(rng, nrow=300, nchan=2)
    (got,) = model_to_apparent_vis_for_regions(ds, uvw=uvw, freq=chan, corr_types=("XX",), time=0.0, freq_out=FREQS[2])
    want = dirty2vis(
        uvw=uvw,
        freq=chan,
        dirty=genesis_cube(FREQS[2:])[0],
        pixsize_x=ds.attrs["cell_rad_x"],
        pixsize_y=ds.attrs["cell_rad_y"],
        center_x=0.0,
        center_y=0.0,
        flip_u=False,
        flip_v=True,
        flip_w=False,
        epsilon=1e-7,
        do_wgridding=True,
        divide_by_n=False,
        nthreads=1,
    )
    assert_allclose(got[..., 0], want, atol=1e-8)


def test_model_geometry_rejects_a_newer_spec():
    with pytest.raises(ValueError, match="needs pfb-model-spec"):
        model_geometry(genesis_dataset(spec="9.9"))


def test_model_sampling_reads_present_and_absent_fields():
    from pfb_model_spec.utils.degrid import model_sampling

    freq = np.array([1.0e9, 1.1e9])
    fb = np.array([[0.95e9, 1.05e9], [1.05e9, 1.15e9]])
    ds = build_mds_dataset(
        np.ones((1, 1, 1)), np.array([1]), np.array([2]), "t0", ["t0"], "t", "f",
        np.array([0.0]), freq, 1e-5, 8, 8, 0.0, 0.0, False, False, False,
        (0.0, 0.0), ["I"], "t", freq_bounds=fb, flux_scale="apparent",
    )  # fmt: skip
    s = model_sampling(ds)
    np.testing.assert_allclose(s["freq_bounds"], fb)
    assert s["time_bounds"] is None and s["weights"] is None
    assert s["flux_scale"] == "apparent"

    g = model_sampling(genesis_dataset())
    assert all(v is None for v in g.values())


@pytest.mark.parametrize("flip_u", [False, True])
@pytest.mark.parametrize("flip_v", [False, True])
def test_pixel_lm_matches_the_visibility_phase(flip_u, flip_v):
    """A unit pixel degrids to exp(+2πi(ul + vm)ν/c) at the (l, m) pixel_lm reports.

    +2πi is the convention real MS data follow through this pipeline (pfb-imaging's
    image-and-beam-orientation wiki §2). A wrong sign on either axis mirrors any
    asymmetric beam, invisibly to Stokes-I tests, so every flip pair is checked and
    the grid is non-square.
    """
    from pfb_model_spec.utils.degrid import pixel_lm

    c = 299792458.0
    rng = np.random.default_rng(5)
    uvw = rng.normal(0.0, 300.0, (64, 3))
    uvw[:, 2] = 0.0
    freq = np.array([1.0e9])
    ny, nx, cell = 48, 64, 1e-4
    geom = dict(nx=nx, ny=ny, cell_rad=cell, x0=3e-3, y0=-2e-3, flip_u=flip_u, flip_v=flip_v)
    iy, ix = 10, 40
    image = np.zeros((1, ny, nx))
    image[0, iy, ix] = 1.0
    vis = degrid_stokes(
        uvw,
        freq,
        image,
        cell_rad=cell,
        x0=geom["x0"],
        y0=geom["y0"],
        flip_u=flip_u,
        flip_v=flip_v,
        flip_w=False,
        epsilon=1e-10,
        do_wgridding=False,
    )[0, :, 0]
    lax, max_ = pixel_lm(geom)
    assert lax.shape == (nx,) and max_.shape == (ny,)
    expected = np.exp(2j * np.pi * (uvw[:, 0] * lax[ix] + uvw[:, 1] * max_[iy]) * freq[0] / c)
    assert_allclose(vis, expected, atol=1e-7)

    # a crop reports the same coordinates for the same pixels
    lc, mc = pixel_lm(geom, i0=4, j0=30, shape=(12, 16))
    assert_allclose(lc, lax[30:46])
    assert_allclose(mc, max_[4:16])


@pytest.mark.parametrize("flip_u", [False, True])
@pytest.mark.parametrize("flip_v", [False, True])
def test_crop_phase_centre_reproduces_the_full_grid_degrid(flip_u, flip_v):
    """Degridding a cropped sub-grid must equal degridding the full one.

    This is the guard on `crop_phase_centre`'s arithmetic, and it has to run
    over every flip combination: each axis's offset sign is set by its flip
    convention, and a sign error moves the model on the sky rather than
    changing it slightly, so it would sail past any loose tolerance. The
    check is that a wrong sign is order-unity wrong, not that the right one
    is merely close.
    """
    from ducc0.wgridder import dirty2vis

    from pfb_model_spec.utils.degrid import crop_bbox, crop_phase_centre

    rng = np.random.default_rng(42)
    nx, ny, cell = 64, 48, 1e-5
    x0, y0 = -3e-4, 2e-4

    uvw = rng.normal(0.0, 3000.0, (500, 3))
    uvw[:, 2] *= 0.05
    freq = np.linspace(1.0e9, 1.1e9, 4)

    # flux confined to a box away from the grid centre, so a mis-set centre
    # cannot be masked by symmetry. (Y, X) like the .mds; ducc is x-major, so
    # it is handed the transpose
    image = np.zeros((ny, nx))
    image[30:42, 10:26] = rng.normal(size=(12, 16))

    kw = dict(
        uvw=uvw,
        freq=freq,
        pixsize_x=cell,
        pixsize_y=cell,
        flip_u=flip_u,
        flip_v=flip_v,
        flip_w=False,
        epsilon=1e-9,
        do_wgridding=True,
        divide_by_n=False,
        nthreads=1,
    )
    ref = dirty2vis(dirty=np.ascontiguousarray(image.T), center_x=x0, center_y=y0, **kw)

    rm = crop_bbox(np.where(image != 0.0, 1.0, 0.0))
    assert (rm.mask.shape, rm.i0, rm.j0) == ((12, 16), 30, 10)
    nyc, nxc = rm.mask.shape
    sub = np.ascontiguousarray(image[rm.i0 : rm.i0 + nyc, rm.j0 : rm.j0 + nxc].T)

    geom = dict(x0=x0, y0=y0, cell_rad=cell, flip_u=flip_u, flip_v=flip_v, nx=nx, ny=ny)
    xc, yc = crop_phase_centre(rm=rm, **geom)
    got = dirty2vis(dirty=sub, center_x=xc, center_y=yc, **kw)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6 * np.abs(ref).max())

    # every other sign choice is order-unity wrong
    for sx, sy in ((1, -1), (-1, 1), (-1, -1)):
        bad = dirty2vis(
            dirty=sub,
            center_x=x0 + sx * (xc - x0),
            center_y=y0 + sy * (yc - y0),
            **kw,
        )
        assert np.abs(bad - ref).max() > 0.1 * np.abs(ref).max()


def test_crop_bbox_of_an_empty_mask_is_degenerate():
    """An empty region must not crop to a zero-sized grid ducc would reject."""
    from pfb_model_spec.utils.degrid import crop_bbox

    rm = crop_bbox(np.zeros((16, 8)))
    assert rm.mask.shape == (2, 2)
    assert (rm.i0, rm.j0) == (0, 0)
    assert not rm.mask.any()


def test_crop_bbox_grows_odd_extents_to_even():
    from pfb_model_spec.utils.degrid import crop_bbox

    mask = np.zeros((10, 12))
    mask[3:6, 4:9] = 1.0  # 3 x 5, both odd
    rm = crop_bbox(mask)
    assert rm.mask.shape == (4, 6)
    assert (rm.i0, rm.j0) == (3, 4)


def test_interp_mueller_is_exact_on_a_bilinear_field():
    from pfb_model_spec.utils.degrid import interp_mueller

    la = np.linspace(-0.02, 0.02, 9)
    ma = np.linspace(-0.01, 0.015, 7)
    field = 2.0 + 3.0 * la[None, :] - 5.0 * ma[:, None] + 7.0 * la[None, :] * ma[:, None]
    block = np.stack([field, 1j * field])[:, None]  # (2, 1, nm, nl), complex
    l = np.array([0.0103, -0.0199, 0.004])  # noqa: E741 -- arbitrary order
    m = np.array([0.0, 0.0149, -0.0099, 0.007])
    out = interp_mueller(block, la, ma, l, m)
    expected = 2.0 + 3.0 * l[None, :] - 5.0 * m[:, None] + 7.0 * l[None, :] * m[:, None]
    assert out.shape == (2, 1, 4, 3) and np.iscomplexobj(out)
    assert_allclose(out[0, 0], expected, rtol=1e-12)
    assert_allclose(out[1, 0], 1j * expected, rtol=1e-12)


def test_interp_mueller_keeps_the_blocks_precision():
    from pfb_model_spec.utils.degrid import interp_mueller

    la = np.linspace(-0.02, 0.02, 9)
    ma = np.linspace(-0.01, 0.015, 7)
    rng = np.random.default_rng(0)
    block = rng.standard_normal((2, 7, 9)) + 1j * rng.standard_normal((2, 7, 9))
    l = np.array([0.0103, -0.0199, 0.004])  # noqa: E741 -- arbitrary order
    m = np.array([0.0, 0.0149, -0.0099, 0.007])
    ref = interp_mueller(block, la, ma, l, m)
    out32 = interp_mueller(block.real.astype(np.float32), la, ma, l, m)
    outc64 = interp_mueller(block.astype(np.complex64), la, ma, l, m)
    assert out32.dtype == np.float32 and outc64.dtype == np.complex64
    assert_allclose(out32, interp_mueller(block.real, la, ma, l, m), rtol=1e-6, atol=1e-6)
    assert_allclose(outc64, ref, rtol=1e-6, atol=1e-6)


def test_interp_mueller_is_zero_outside_its_axes():
    from pfb_model_spec.utils.degrid import interp_mueller

    la = np.linspace(0.0, 1.0, 5)
    block = np.ones((1, 1, 5, 5))
    out = interp_mueller(block, la, la, np.array([-0.1, 0.5, 1.1]), np.array([0.5]))
    assert_allclose(out[0, 0, 0], [0.0, 1.0, 0.0])


def test_interp_mueller_rejects_descending_axes():
    from pfb_model_spec.utils.degrid import interp_mueller

    with pytest.raises(ValueError, match="ascending"):
        interp_mueller(np.ones((1, 1, 3, 3)), np.array([2.0, 1.0, 0.0]), np.arange(3.0), np.zeros(1), np.zeros(1))


def _flat_axes(geom, n=5):
    l, m = pixel_lm(geom)  # noqa: E741
    pad = geom["cell_rad"]
    return np.linspace(l.min() - pad, l.max() + pad, n), np.linspace(m.min() - pad, m.max() + pad, n)


def _sloped_block(geom, nso=1, nsi=1, scale=1.0):
    """A Mueller field with an E-W and N-S gradient, so a mirrored axis shows."""
    la, ma = _flat_axes(geom)
    field = 1.0 + 40.0 * la[None, :] - 25.0 * ma[:, None]
    block = np.zeros((nso, nsi, ma.size, la.size), dtype=np.result_type(scale, float))
    for k in range(min(nso, nsi)):
        block[k, k] = scale * field
    return block, la, ma


def test_regions_identity_beam_term_matches_no_beam():
    ds, _, time, freq, cell = _synthetic_mds()
    geom = model_geometry(ds)
    rng = np.random.default_rng(32)
    uvw, chan = _uvw_freq(rng, nrow=300, nchan=3)
    la, ma = _flat_axes(geom)
    nsi = len(geom["stokes"])
    ident = np.zeros((nsi, nsi, ma.size, la.size))
    for k in range(nsi):
        ident[k, k] = 1.0
    kw = dict(uvw=uvw, freq=chan, corr_types=("XX", "YY"), time=time[0], freq_out=freq[0], divide_by_n=True)
    (beam,) = model_to_apparent_vis_for_regions(ds, beam_terms=[BeamTerm(ident, la, ma)], **kw)
    (bare,) = model_to_apparent_vis_for_regions(ds, **kw)
    assert_allclose(beam, bare, rtol=0, atol=1e-12 * np.abs(bare).max())


def test_regions_beam_term_crop_matches_full_grid():
    """A cropped region sees the beam at the same sky positions as the full grid."""
    ds, _, time, freq, cell = _synthetic_mds()
    geom = model_geometry(ds)
    nsi = len(geom["stokes"])
    rng = np.random.default_rng(33)
    uvw, chan = _uvw_freq(rng, nrow=300, nchan=2)
    block, la, ma = _sloped_block(geom, nso=nsi, nsi=nsi)
    full_mask = np.zeros((geom["ny"], geom["nx"]))
    full_mask[:, : geom["nx"] // 2] = 1.0
    kw = dict(uvw=uvw, freq=chan, corr_types=("XX", "YY"), time=time[0], freq_out=freq[0], epsilon=1e-10)
    (got,) = model_to_apparent_vis_for_regions(
        ds, region_masks=[crop_bbox(full_mask)], beam_terms=[BeamTerm(block, la, ma)], **kw
    )
    full_block = interp_mueller(block, la, ma, *pixel_lm(geom))
    ref = _composed(
        ds, uvw, chan, kw["corr_types"], time[0], freq[0], mueller=full_block, region_mask=full_mask, epsilon=1e-10
    )
    assert_allclose(got, ref, rtol=0, atol=1e-7 * np.abs(ref).max())


def test_regions_conj_rows_predict_the_reversed_baseline():
    """At -uvw, conj_rows must give exactly conj of the forward prediction (Stokes I, XX)."""
    ds, _, time, freq, cell = _synthetic_mds()
    geom = model_geometry(ds)
    nsi = len(geom["stokes"])
    rng = np.random.default_rng(34)
    uvw, chan = _uvw_freq(rng, nrow=200, nchan=2)
    block, la, ma = _sloped_block(geom, nso=1, nsi=nsi, scale=1.0 + 0.3j)
    kw = dict(freq=chan, corr_types=("XX",), time=time[0], freq_out=freq[0], stokes_out="I", epsilon=1e-10)
    every = np.ones(uvw.shape[0], dtype=bool)
    (fwd,) = model_to_apparent_vis_for_regions(ds, uvw=uvw, beam_terms=[BeamTerm(block, la, ma)], **kw)
    (rev,) = model_to_apparent_vis_for_regions(
        ds, uvw=-uvw, beam_terms=[BeamTerm(block, la, ma, rows=every, conj_rows=every)], **kw
    )
    assert np.abs(fwd.imag).max() > 1e-3 * np.abs(fwd).max()  # the imaginary beam part matters
    assert_allclose(rev, np.conj(fwd), rtol=0, atol=1e-7 * np.abs(fwd).max())


def test_regions_terms_over_a_row_split_add_up():
    ds, _, time, freq, cell = _synthetic_mds()
    geom = model_geometry(ds)
    nsi = len(geom["stokes"])
    rng = np.random.default_rng(35)
    uvw, chan = _uvw_freq(rng, nrow=200, nchan=2)
    a, la, ma = _sloped_block(geom, nso=nsi, nsi=nsi)
    b = 0.5 * a
    even = np.arange(uvw.shape[0]) % 2 == 0
    kw = dict(uvw=uvw, freq=chan, corr_types=("XX", "YY"), time=time[0], freq_out=freq[0])
    (both,) = model_to_apparent_vis_for_regions(
        ds, beam_terms=[BeamTerm(a, la, ma, rows=even), BeamTerm(b, la, ma, rows=~even)], **kw
    )
    (only_a,) = model_to_apparent_vis_for_regions(ds, beam_terms=[BeamTerm(a, la, ma, rows=even)], **kw)
    (only_b,) = model_to_apparent_vis_for_regions(ds, beam_terms=[BeamTerm(b, la, ma, rows=~even)], **kw)
    assert_allclose(both, only_a + only_b, rtol=0, atol=1e-12 * np.abs(both).max())
    assert not only_a[~even].any() and not only_b[even].any()


@pytest.mark.parametrize(
    "rows, conj_rows, match",
    [
        ("overlap", None, "disjoint"),
        ("even", "odd", "subset"),
    ],
)
def test_regions_refuse_inconsistent_beam_terms(rows, conj_rows, match):
    ds, _, time, freq, cell = _synthetic_mds()
    geom = model_geometry(ds)
    nsi = len(geom["stokes"])
    rng = np.random.default_rng(36)
    uvw, chan = _uvw_freq(rng, nrow=10, nchan=1)
    a, la, ma = _sloped_block(geom, nso=nsi, nsi=nsi)
    even = np.arange(10) % 2 == 0
    if rows == "overlap":
        terms = [BeamTerm(a, la, ma, rows=even), BeamTerm(a, la, ma, rows=np.ones(10, bool))]
    else:
        terms = [BeamTerm(a, la, ma, rows=even, conj_rows=~even)]
    with pytest.raises(ValueError, match=match):
        model_to_apparent_vis_for_regions(
            ds, uvw=uvw, freq=chan, corr_types=("XX",), time=time[0], freq_out=freq[0], beam_terms=terms
        )


def test_regions_refuse_a_mueller_of_the_wrong_shape():
    ds, _, time, freq, cell = _synthetic_mds()
    geom = model_geometry(ds)
    rng = np.random.default_rng(37)
    uvw, chan = _uvw_freq(rng, nrow=10, nchan=1)
    block, la, ma = _sloped_block(geom, nso=2, nsi=len(geom["stokes"]) + 1)
    with pytest.raises(ValueError, match="mueller"):
        model_to_apparent_vis_for_regions(
            ds,
            uvw=uvw,
            freq=chan,
            corr_types=("XX",),
            time=time[0],
            freq_out=freq[0],
            stokes_out="IQ",
            beam_terms=[BeamTerm(block, la, ma)],
        )


def test_regions_that_partition_the_grid_add_up_to_the_whole():
    """Two cropped regions must sum to the uncropped prediction, beam or not."""
    ds, _, time, freq, cell = _synthetic_mds()
    geom = model_geometry(ds)
    nsi = len(geom["stokes"])
    rng = np.random.default_rng(38)
    uvw, chan = _uvw_freq(rng, nrow=300, nchan=2)
    left = np.zeros((geom["ny"], geom["nx"]))
    left[:, : geom["nx"] // 2] = 1.0
    regions = [crop_bbox(left), crop_bbox(1.0 - left)]
    block, la, ma = _sloped_block(geom, nso=nsi, nsi=nsi)
    kw = dict(uvw=uvw, freq=chan, corr_types=("XX", "YY"), time=time[0], freq_out=freq[0], epsilon=1e-10)
    for terms in (None, [BeamTerm(block, la, ma)]):
        parts = model_to_apparent_vis_for_regions(ds, region_masks=regions, beam_terms=terms, **kw)
        (whole,) = model_to_apparent_vis_for_regions(ds, beam_terms=terms, **kw)
        assert len(parts) == 2 and all(np.abs(p).max() > 1e-3 * np.abs(whole).max() for p in parts)
        assert_allclose(parts[0] + parts[1], whole, rtol=0, atol=1e-8 * np.abs(whole).max())


def test_regions_conj_rows_give_the_hermitian_of_the_reversed_baseline():
    """With Q/U/V in the apparent sky, a reversed baseline must predict V_qp = V_pq^H.

    That is XX and YY conjugated, and XY swapped with conj(YX). Conjugating the
    Stokes-basis Mueller gives this only because XY = U + iV and YX = U - iV, so
    it is checked here on all four correlations, not on XX alone.
    """
    ds, _, time, freq, cell = _synthetic_mds()
    geom = model_geometry(ds)
    rng = np.random.default_rng(39)
    uvw, chan = _uvw_freq(rng, nrow=200, nchan=2)
    block, la, ma = _sloped_block(geom, nso=4, nsi=1, scale=1.0 + 0.3j)
    block[1, 0] = 0.2 * block[0, 0]
    block[2, 0] = (0.1 - 0.05j) * block[0, 0]
    block[3, 0] = 0.07j * block[0, 0]
    every = np.ones(uvw.shape[0], dtype=bool)
    kw = dict(freq=chan, corr_types=("XX", "XY", "YX", "YY"), time=time[0], freq_out=freq[0], stokes_out="IQUV")
    (fwd,) = model_to_apparent_vis_for_regions(ds, uvw=uvw, beam_terms=[BeamTerm(block, la, ma)], **kw)
    (rev,) = model_to_apparent_vis_for_regions(
        ds, uvw=-uvw, beam_terms=[BeamTerm(block, la, ma, rows=every, conj_rows=every)], **kw
    )
    assert np.abs(fwd[..., 1] - fwd[..., 2]).max() > 1e-3 * np.abs(fwd).max()  # XY and YX differ
    assert_allclose(rev, np.conj(fwd[..., [0, 2, 1, 3]]), rtol=0, atol=1e-7 * np.abs(fwd).max())


def test_regions_beam_rows_respect_the_padding_mask():
    """A term's rows and the structural mask combine: zero wherever either is zero."""
    ds, _, time, freq, cell = _synthetic_mds()
    geom = model_geometry(ds)
    nsi = len(geom["stokes"])
    rng = np.random.default_rng(40)
    uvw, chan = _uvw_freq(rng, nrow=200, nchan=3)
    block, la, ma = _sloped_block(geom, nso=nsi, nsi=nsi)
    even = np.arange(uvw.shape[0]) % 2 == 0
    mask = (rng.random((uvw.shape[0], chan.size)) > 0.3).astype(np.uint8)
    kw = dict(uvw=uvw, freq=chan, corr_types=("XX",), time=time[0], freq_out=freq[0], epsilon=1e-10)
    (got,) = model_to_apparent_vis_for_regions(ds, beam_terms=[BeamTerm(block, la, ma, rows=even)], mask=mask, **kw)
    (ref,) = model_to_apparent_vis_for_regions(ds, beam_terms=[BeamTerm(block, la, ma)], **kw)
    keep = (mask.astype(bool) & even[:, None])[..., None]
    assert not got[~keep[..., 0]].any()  # exactly zero, not merely small
    assert_allclose(got, np.where(keep, ref, 0.0), rtol=0, atol=1e-8 * np.abs(ref).max())


def test_regions_an_empty_region_predicts_zeros():
    ds, _, time, freq, cell = _synthetic_mds()
    geom = model_geometry(ds)
    rng = np.random.default_rng(41)
    uvw, chan = _uvw_freq(rng, nrow=50, nchan=2)
    empty = crop_bbox(np.zeros((geom["ny"], geom["nx"])))
    (got,) = model_to_apparent_vis_for_regions(
        ds, uvw=uvw, freq=chan, corr_types=("XX",), time=time[0], freq_out=freq[0], region_masks=[empty]
    )
    assert got.shape == (50, 2, 1) and not got.any()


def test_regions_refuse_an_empty_list_of_beam_terms():
    ds, _, time, freq, cell = _synthetic_mds()
    rng = np.random.default_rng(42)
    uvw, chan = _uvw_freq(rng, nrow=10, nchan=1)
    with pytest.raises(ValueError, match="empty"):
        model_to_apparent_vis_for_regions(
            ds, uvw=uvw, freq=chan, corr_types=("XX",), time=time[0], freq_out=freq[0], beam_terms=[]
        )


def test_regions_complex_beam_matches_the_composition():
    """A complex block is degridded as Re + i·Im, one plane at a time, without changing the answer."""
    ds, _, time, freq, cell = _synthetic_mds()
    geom = model_geometry(ds)
    rng = np.random.default_rng(43)
    uvw, chan = _uvw_freq(rng, nrow=200, nchan=2)
    block, la, ma = _sloped_block(geom, nso=4, nsi=1, scale=1.0 + 0.3j)
    block[3, 0] = 0.07j * block[0, 0]
    corr = ("XX", "XY", "YX", "YY")
    (got,) = model_to_apparent_vis_for_regions(
        ds,
        uvw=uvw,
        freq=chan,
        corr_types=corr,
        time=time[0],
        freq_out=freq[0],
        stokes_out="IQUV",
        beam_terms=[BeamTerm(block, la, ma)],
        epsilon=1e-10,
    )
    full = interp_mueller(block, la, ma, *pixel_lm(geom))
    ref = _composed(ds, uvw, chan, corr, time[0], freq[0], mueller=full, stokes="IQUV", epsilon=1e-10)
    assert_allclose(got, ref, rtol=0, atol=1e-8 * np.abs(ref).max())
