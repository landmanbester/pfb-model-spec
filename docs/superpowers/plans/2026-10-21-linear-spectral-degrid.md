# Linear-in-Frequency Degrid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `model_to_apparent_vis_for_regions` predict a model, and optionally a beam, that varies linearly in frequency across a degrid chunk, instead of holding both flat at `freq_out`.

**Architecture:**
- A public `frequency_kinks` lists every frequency where the stored parametrisation switches piece or clamps. Consumers cut chunks there, and the wrapper refuses a chunk that crosses one.
- `render_model_slope` returns dS/dν at `freq_out` by forward difference against the already-rendered value image, through the unchanged `render_model_region`.
- The fused wrapper gains `spectral="linear"`. It degrids the value image A₀ and the slope image A₁ through the same regions and beam terms, and combines them per channel as `V = V₀ + (ν − ν₀)·V₁`.
- A beam term may carry a Mueller slope `dmueller`, giving A₁ = M₀S₁ + M₁S₀; the (ν−ν₀)² cross term is dropped.
- `utils/modelspec.py` and the `.mds` schema are untouched.

**Tech Stack:** numpy, sympy (parsing the stored parametrisation only), ducc0, xarray.

**Spec:** landmanbester/pfb-model-spec#39, as amended by:
- the review comment https://github.com/landmanbester/pfb-model-spec/issues/39#issuecomment-6309952078;
- the pfb-imaging review on #39 (kinks, expansion point, beam consistency, memory), accepted in our reply there.

Companion plan: `~/software/meerkat-beams/docs/superpowers/plans/2026-10-21-freq-extrapolation.md` (meerkat-beams#41). Its Task 4, measurement B, decides whether pfb-imaging supplies a `dmueller`, and which one.

## Global Constraints

- **Branch:** `issue39`, created from `main`. Set `CONTAINER_IMAGE` in `src/pfb_model_spec/_container_image.py` to the `:issue39` tag (architecture.md §6).
- **`"flat"` is the default and stays byte-identical.** Every existing test in `tests/test_degrid.py` must pass unmodified.
- **Kinks are the contract.** A kink is any `Piecewise` break, plus any `Min`/`Max` switch point solved for `f`. In linear mode, every channel in `freq` and both `freq_out` and `freq_out + step` must lie between the same two consecutive kinks; otherwise the wrapper raises `ValueError`. The wrapper never splits a chunk and never clips Δν. A chunk wholly in a held region (beyond the outer kink, or in a gap) has a zero slope and is exact.
- **Intervals are half-open, `[k_i, k_{i+1})`, matching `Piecewise`'s `f < brk`:** a frequency exactly on a break belongs to the upper piece. Every interval label is `np.searchsorted(kinks, x, side="right")`: in `split_at_kinks`, in the wrapper's check and in `render_model_slope`. A clamp kink within 1 Hz of a break is dropped in favour of the break's exact value, so an on-break channel cannot land on the wrong side of a ULP-shifted clamp.
- **Expansion point:** ν₀ = `freq_out` may be any frequency in the chunk's piece; the chunk mean is recommended. Inside a tent piece the result must not depend on ν₀. For the polynomial bases, the centre minimises the second-order error.
- **Slope by forward difference, not `sm.diff`:** `(R(ν₀ + h) − R(ν₀)) / h`, where `R(ν₀)` is the value image the wrapper already holds and the default `h = 1e-6 * freq_out` Hz. Difference in place, so linear mode keeps exactly one extra full image resident. Only the model's own grid is supported, because a `"tent"` remap is nonlinear and does not commute with differencing.
- **No schema change, so no spec bump.** `BeamTerm.dmueller` is appended **last**, with default `None`, so positional construction is unchanged.
- **Do not restyle `utils/modelspec.py`** (python-standards.md §6). Nothing in this plan edits it.
- **Lint:** `uv run ruff format . && uv run ruff check . --fix`. **Test:** `uv run --extra full pytest -v`.
- **Commit with the venv active:** `source /home/bester/software/pfb-model-spec/.venv/bin/activate && git add <files> && git commit -m "..."`. Conventional Commits.
- Type hints on every signature; Google-style docstrings; the `SPECTRAL = ("flat", "linear")` tuple follows `remap.KERNELS`.

## Cross-repo interface (what the consumers must do)

| Who | Must |
|---|---|
| pfb-imaging `degrid` | On the driver, before Ray Serve starts: cut each node's frequency axis with `split_at_kinks(freq, frequency_kinks(ds))`, then subdivide by `--channels-per-chunk`. Fixed-width chunks drift off the imager's `linspace` band edges; with 4096 channels and 15 bands the first straddle is chunk 8. |
| pfb-imaging `degrid` | Pass any `freq_out` in the chunk's piece. The chunk's unweighted mean (D39) is recommended. |
| pfb-imaging, flat beam (`dmueller=None`) | Evaluate `mueller` at the chunk's piece anchor, taken from `frequency_pieces(ds)` (the band's weighted frequency, D28), not at the chunk mean. The weighted band average of the prediction is then `B(ν_anchor)·m_b`, the gauge `deconv` solved in. At the chunk mean it is off by about `2 ln B · (ν_anchor − ν_mean)/ν`. |
| pfb-imaging + meerkat-beams, sloped beam | `mueller` = M(`freq_out`), and `dmueller` = the slope of any linear fit to M(ν) through that point: chord, least squares or λ/D `(l·∂ₗM + m·∂ₘM)/ν₀`. Units per Hz, in the `mueller`'s own `(nso, nsi, nmb, nlb)` layout and axes. M₀, S₀ and Δν must share one ν₀, because mixing expansion points is a first-order error. meerkat-beams#41 Task 4 (measurement B) picks the fit; λ/D needs no extra beam evaluations. |
| meerkat-beams | Evaluate M at band-edge channels (856–869 / 1698–1712 MHz) via `freq_extrapolation="scale"` (meerkat-beams#41). |

## Review Focus

1. **A one-channel chunk** (`nchan == 1`) in linear mode. It must equal the per-channel prediction, not divide by a zero width.
2. **`freq_out` outside the chunk's channel span but inside the same piece.** It must still be exact for the tent, because the kink check spans ν₀ too.
3. **Padded rows** (`mask` zeroed, NaN `uvw`). Both passes must honour the mask, and masked visibilities stay exactly zero.
4. **A genesis or upgraded `.mds` without `freq_bounds`** (polynomial basis). Linear mode works, and `frequency_kinks` is empty.
5. **A Stokes subset** (`stokes_out` smaller than the model's planes) in linear mode. The slope image must be subset exactly like the value image.

---

## File Structure

| File | Change |
|---|---|
| Modify: `src/pfb_model_spec/utils/degrid.py` | `frequency_kinks`, `split_at_kinks`, `render_model_slope`, `SPECTRAL`, `BeamTerm.dmueller`, linear path in `model_to_apparent_vis_for_regions` and `_degrid_beam_term`, `dmueller` checks in `_check_beam_terms` |
| Modify: `tests/test_degrid.py` | `_hist_mds` and `_per_channel` helpers plus the tests below (imports merged into the top block, E402) |
| Modify: `.claude/rules/component-model.md`, `CLAUDE.md` | Degrid API entries (Task 3) |

---

### Task 1: Kinks, chunk splitting and `render_model_slope`

**Files:**
- Modify: `src/pfb_model_spec/utils/degrid.py` (after `render_model_region`)
- Test: `tests/test_degrid.py`

**Interfaces:**
- Produces: `frequency_kinks(model_ds: xr.Dataset) -> np.ndarray`.
  - Returns the sorted, deduplicated kink frequencies in Hz: kinks less than 1 Hz apart are merged, because `sm.solve` round-off makes a clamp land a few ULP off its break.
  - Returns an empty array for a parametrisation with no `Piecewise`, `Min` or `Max`.
  - Raises `ValueError` if there are kinks and `fexpr != "f"`.
  - Raises `NotImplementedError` for a `Min`/`Max` whose `f`-dependent core is not linear in `f`.
- Produces: `split_at_kinks(freq: np.ndarray, kinks: np.ndarray) -> list[slice]`. Returns contiguous slices of the ascending `freq` that each lie in one half-open interval `[k_i, k_{i+1})`, in order. Their union covers `freq`.
- Produces: `FrequencyPiece(NamedTuple)` with fields `lo: float`, `hi: float` and `anchor: float`, and `frequency_pieces(model_ds: xr.Dataset) -> list[FrequencyPiece]`.
  - One entry per surviving `Piecewise` piece, in order. `[lo, hi)` is the piece's break range, with `-inf`/`inf` at the ends.
  - A non-piecewise model with exactly one kept band returns one piece `(-inf, inf, that band's freq)`. Otherwise (the polynomial bases) it returns `[]`.
  - Anchor source:
    1. For a piece whose term has a clamp core (`tent`), take the root of the core, `(f − anchor)/half = 0`, which is the fit's own truth.
    2. Otherwise (`box`), take the unique `freqs` value in `[lo, hi)` among the kept bands. Kept bands are those whose stored `weights` are non-zero in any Stokes plane, or all bands when `weights` is absent.
    3. Raise `ValueError` ("ambiguous anchor") if the result is not unique. This happens for a box `.mds` with a dropped band and no `weights`, as `model2comps` writes them.
- Produces: `render_model_slope(model_ds: xr.Dataset, *, time: float, freq_out: float, step: float | None = None, value: np.ndarray | None = None) -> np.ndarray`.
  - Returns `(nstokes, ny, nx)` per Hz on the model's grid.
  - `step=None` means `1e-6 * freq_out`. `value` is `render_model_region(..., freq_out=freq_out)` when the caller already holds it, and is rendered otherwise.
  - Raises `ValueError` if a kink lies in `(freq_out, freq_out + step]`, that is, if `freq_out` and `freq_out + step` get different `side="right"` labels. A kink exactly at `freq_out` is allowed, because the difference is taken on the upper side.

- [ ] **Step 1: Create the branch and set the image tag**

`git checkout -b issue39`, then set the tag in `_container_image.py` to `issue39`.

- [ ] **Step 2: Write the failing tests**

```python
def _hist_mds(method="tent", nband=4, nx=64, ny=48, drop=(), stokes=("I",)):
    """The _synthetic_mds sources, fitted by histopolation over contiguous bands.

    The anchors sit off the band midpoints, like pfb-imaging's weighted freq_out. Bands in
    `drop` get zero weight. With stokes=("I", "Q"), Q = -0.3 * I.
    """
    edges = np.linspace(1.0e9, 1.2e9, nband + 1)
    bounds = np.stack([edges[:-1], edges[1:]], axis=1)
    freq = bounds.mean(axis=1) + 0.1 * np.diff(edges)
    # cube as in _synthetic_mds, evaluated at freq; wgt zero on `drop`
    # fit_image_cube(..., wgt=wgt, method=method, freq_bounds=bounds); build_mds_dataset(..., freq_bounds=bounds)
    return ds, time, freq, bounds


def test_frequency_kinks_of_contiguous_bands_are_the_band_edges():
    ds, _, _, bounds = _hist_mds()
    assert_allclose(frequency_kinks(ds), np.r_[bounds[:, 0], bounds[-1, 1]])
    assert frequency_kinks(_synthetic_mds()[0]).size == 0


def test_frequency_kinks_see_the_clamps_around_a_dropped_band():
    ds, _, _, bounds = _hist_mds(drop=(2,))
    gap_break = 0.5 * (bounds[1, 1] + bounds[3, 0])           # the Piecewise break; equals bounds[2] midpoint here
    expected = np.r_[bounds[:2, 0], bounds[1, 1], gap_break, bounds[3, 0], bounds[3, 1]]
    assert_allclose(frequency_kinks(ds), np.unique(expected))  # 1000, 1050, 1100, 1125, 1150, 1200 MHz


def test_split_at_kinks_cuts_between_kinks():
    freq = np.arange(10.0)
    assert split_at_kinks(freq, np.array([2.5, 6.5])) == [slice(0, 3), slice(3, 7), slice(7, 10)]
    assert split_at_kinks(freq, np.array([-1.0, 20.0])) == [slice(0, 10)]
    assert split_at_kinks(freq, np.array([3.0])) == [slice(0, 3), slice(3, 10)]   # on a kink -> upper


@pytest.mark.parametrize("method", ["tent", "box"])
def test_frequency_pieces_anchor_each_surviving_band(method):
    ds, _, freq, bounds = _hist_mds(method=method)
    pieces = frequency_pieces(ds)
    assert [p.anchor for p in pieces] == pytest.approx(list(freq))
    assert pieces[0].lo == -np.inf and pieces[-1].hi == np.inf
    assert [p.lo for p in pieces[1:]] == pytest.approx(list(bounds[1:, 0]))


@pytest.mark.parametrize("method", ["tent", "box"])
def test_frequency_pieces_skip_a_dropped_band(method):
    ds, _, freq, bounds = _hist_mds(method=method, drop=(2,))
    pieces = frequency_pieces(ds)
    assert [p.anchor for p in pieces] == pytest.approx([freq[0], freq[1], freq[3]])
    assert pieces[2].lo == pytest.approx(0.5 * (bounds[1, 1] + bounds[3, 0]))


def test_frequency_pieces_refuse_an_ambiguous_box_anchor():
    ds, *_ = _hist_mds(method="box", drop=(2,))
    with pytest.raises(ValueError, match="ambiguous"):
        frequency_pieces(ds.drop_vars("weights"))


def test_frequency_pieces_of_a_polynomial_basis_are_empty():
    assert frequency_pieces(_synthetic_mds()[0]) == []


def test_render_model_slope_of_a_tent_is_its_piece_slope():
    ds, time, freq, bounds = _hist_mds()
    for b in range(freq.size):
        nu = freq[b] + 0.2 * (bounds[b, 1] - freq[b])   # inside the piece, off the anchor
        got = render_model_slope(ds, time=time[0], freq_out=nu)
        h = 1.0e3
        fd = (render_model_region(ds, time=time[0], freq_out=freq[b] + h)
              - render_model_region(ds, time=time[0], freq_out=freq[b] - h)) / (2 * h)
        assert_allclose(got, fd, rtol=1e-6, atol=1e-18)   # one slope per piece


def test_render_model_slope_reuses_a_supplied_value():
    ds, time, freq, _ = _hist_mds()
    v = render_model_region(ds, time=time[0], freq_out=freq[1])
    assert_array_equal(render_model_slope(ds, time=time[0], freq_out=freq[1], value=v),
                       render_model_slope(ds, time=time[0], freq_out=freq[1]))


def test_render_model_slope_of_a_box_is_zero():
    ds, time, freq, _ = _hist_mds(method="box")
    assert not render_model_slope(ds, time=time[0], freq_out=freq[1]).any()


def test_render_model_slope_matches_the_symbolic_derivative():
    """For a polynomial basis the forward difference agrees with sm.diff."""
    ds, _, time, freq, _ = _synthetic_mds()
    # oracle: lambdify(sm.diff(parse_expr(parametrisation), f)) * d(fexpr)/dnu, at the component pixels
    assert_allclose(got[0, yi, xi], oracle, rtol=1e-5)


def test_render_model_slope_refuses_a_kink_within_the_step():
    ds, time, _, bounds = _hist_mds()
    with pytest.raises(ValueError, match="kink"):
        render_model_slope(ds, time=time[0], freq_out=bounds[1, 0] - 1.0e2, step=1.0e3)
```

- [ ] **Step 3: Run them to confirm they fail**

Run: `uv run --extra full pytest tests/test_degrid.py -k "kinks or render_model_slope" -v`
Expected: FAIL with `ImportError` (the names are not defined yet).

- [ ] **Step 4: Implement the three functions in `degrid.py`**

- `frequency_kinks`: `e = parse_expr(attrs["parametrisation"])`, then collect two sets of points.
  - If `e` is a `sm.Piecewise`, take `float(cond.rhs)` for each `(expr, cond)` with `cond is not sm.true`.
  - For each node in `e.atoms(sm.Min, sm.Max)`, find its core `p`: descend through its single `f`-dependent argument until that argument is no longer a `Min`/`Max`. Then, for each constant argument `c` of the node, add the solutions of `p = c` in `f`.
  - This was prototyped against `_fit_histopolation` output, with and without a dropped interior band.
- `split_at_kinks`: `np.searchsorted(kinks, freq, side="right")` labels each channel with its interval, and each run of equal labels becomes one slice.
- `frequency_pieces`: the piece ranges come from the `Piecewise` breaks only. The anchor sources are as in the Interfaces block. `_hist_mds` must pass `weights` to `build_mds_dataset`, zero on `drop`, as `model_to_ds` does.
- `render_model_slope`: check the kinks, then compute `rp = render_model_region(..., freq_out=freq_out + step)`, then `rp -= value; rp /= step` in place. The docstring states the model's-own-grid restriction and the one extra resident image.

- [ ] **Step 5: Run the tests**

Run: `uv run --extra full pytest tests/test_degrid.py -v`
Expected: all pass, including every pre-existing test.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check . --fix
source .venv/bin/activate && git add src/pfb_model_spec tests/test_degrid.py && git commit -m "feat(degrid): expose frequency kinks and render the model's slope"
```

---

### Task 2: `spectral="linear"` in the fused wrapper (no beam)

**Files:**
- Modify: `src/pfb_model_spec/utils/degrid.py` (`model_to_apparent_vis_for_regions`)
- Test: `tests/test_degrid.py`

**Interfaces:**
- Consumes: `frequency_kinks` and `render_model_slope` (Task 1).
- Produces: `SPECTRAL = ("flat", "linear")`. Also produces `model_to_apparent_vis_for_regions(..., spectral: str = "flat")`, a keyword after `nthreads`. The return shape is unchanged.
- Produces (internal, Task 3 reuses it): `dnu = freq - freq_out`, `(nchan,)`, which is `None` in flat mode. The per-region combination is `stokes_vis = V0 + dnu[None, None, :] * V1`.

- [ ] **Step 1: Write the failing tests**

```python
def _per_channel(ds, uvw, chan, corr, time, beam_at=None, **kw):
    """Exact reference: one flat prediction per channel at that channel's frequency.

    beam_at(nu) -> list[BeamTerm], or None for no beam.
    """
    # concatenate model_to_apparent_vis_for_regions(..., freq=chan[c:c+1], freq_out=chan[c],
    #   beam_terms=None if beam_at is None else beam_at(chan[c]), **kw)[0] along axis 1


def test_linear_matches_per_channel_inside_a_tent_piece():
    ds, time, freq, bounds = _hist_mds()
    rng = np.random.default_rng(40)
    uvw, _ = _uvw_freq(rng, nrow=300)
    b = 1
    chan = np.linspace(bounds[b, 0], bounds[b, 1], 9)[1:-1]   # strictly inside band b
    corr = ("XX", "YY")
    kw = dict(uvw=uvw, freq=chan, corr_types=corr, time=time[0], epsilon=1e-10)
    (got,) = model_to_apparent_vis_for_regions(ds, freq_out=chan.mean(), spectral="linear", **kw)
    ref = _per_channel(ds, uvw, chan, corr, time[0], epsilon=1e-10)
    assert_allclose(got, ref, rtol=0, atol=1e-8 * np.abs(ref).max())


def test_linear_does_not_depend_on_the_expansion_point_inside_a_piece():
    # same chunk with freq_out = chan.mean(), freq[b], and bounds[b, 0] + 0.05 * width
    # (the last is outside the channel span but inside the piece: Review Focus 2) -> equal to 1e-8 relative


def test_linear_equals_flat_for_a_box_model():
    ds, time, freq, bounds = _hist_mds(method="box")
    # chunk inside band 1: linear == flat exactly (assert_array_equal); the slope image is empty


def test_linear_error_is_second_order_for_a_polynomial_basis():
    ds, _, time, freq, _ = _synthetic_mds()
    nu0 = 0.5 * (freq[1] + freq[2])
    # chunks of width W = 40 MHz and W/2, 5 channels each, centred on nu0, freq_out = nu0
    # err(W) = max|prediction - per_channel|, for linear and for flat
    assert 3.0 < err_lin(W) / err_lin(W / 2) < 5.0
    assert 1.5 < err_flat(W) / err_flat(W / 2) < 2.5
    assert err_lin(W) < 0.1 * err_flat(W)


def test_linear_refuses_a_chunk_that_straddles_a_kink():
    ds, time, freq, bounds = _hist_mds()
    chan = np.linspace(freq[1], freq[2], 4)
    with pytest.raises(ValueError, match="kink"):
        model_to_apparent_vis_for_regions(ds, ..., freq=chan, freq_out=freq[1], spectral="linear")


def test_linear_beyond_the_outer_kink_matches_per_channel():
    ds, time, freq, bounds = _hist_mds()
    w = bounds[-1, 1] - bounds[-1, 0]
    chan = np.linspace(bounds[-1, 1] + 0.1 * w, bounds[-1, 1] + 0.5 * w, 5)   # wholly held flat
    # linear (freq_out = chan.mean()) == _per_channel to 1e-8 relative


def test_linear_refuses_a_chunk_that_straddles_the_outer_kink():
    # chan from bounds[-1, 1] - 0.2 * w to bounds[-1, 1] + 0.2 * w -> ValueError, match="kink"


def test_linear_around_a_dropped_band():
    ds, time, freq, bounds = _hist_mds(drop=(2,))
    # chunk inside (bounds[1, 1], gap_break): held flat -> matches _per_channel to 1e-8 relative
    # chunk from bounds[1, 1] - 0.2 * w to bounds[1, 1] + 0.2 * w (crosses band 1's clamp) -> ValueError, match="kink"


def test_linear_channel_on_a_break_belongs_to_the_upper_piece():
    ds, time, freq, bounds = _hist_mds()
    brk = bounds[2, 0]
    up = np.r_[brk, np.linspace(brk, bounds[2, 1], 6)[1:-1]]
    # linear over `up` with freq_out = up.mean() == _per_channel to 1e-8 relative (the jump is ~0.5%)
    down = np.r_[np.linspace(bounds[1, 0], brk, 6)[1:-1], brk]
    # linear over `down` with freq_out = down[:-1].mean() -> ValueError, match="kink"


def test_linear_rejects_an_unknown_spectral_mode():
    with pytest.raises(ValueError, match="spectral"):
        model_to_apparent_vis_for_regions(ds, ..., spectral="cubic")
```

Also add the Review Focus tests, in the same style:
- `test_linear_single_channel_chunk_matches_per_channel` (Focus 1): `chan = freq[1:2] + 1e6`.
- `test_linear_masked_rows_are_zero` (Focus 3): rows 0–9 with NaN `uvw` and `mask[:10] = 0`. Masked output is `== 0`, and the rest equals the unmasked run.
- `test_linear_on_a_genesis_model` (Focus 4): `genesis_dataset()`, a 5-channel, 40 MHz chunk centred on `FREQS[1]`. It raises nothing (no kinks), and `err_lin < 0.1 * err_flat` against `_per_channel`.
- `test_linear_stokes_subset` (Focus 5): `_hist_mds(stokes=("I", "Q"))`. `stokes_out=["Q"]` linear equals the Q plane of the full linear prediction.
- `test_linear_regions_add_up_to_the_whole`: mirror `test_regions_that_partition_the_grid_add_up_to_the_whole` with `spectral="linear"`.

- [ ] **Step 2: Run them to confirm they fail**

Run: `uv run --extra full pytest tests/test_degrid.py -k linear -v`
Expected: FAIL with `TypeError: unexpected keyword argument 'spectral'`.

- [ ] **Step 3: Implement the linear path in `model_to_apparent_vis_for_regions`**

- Validate `spectral in SPECTRAL`.
- In linear mode:
  - compute `step = 1e-6 * freq_out`, and raise `ValueError` (naming the kink and pointing at `split_at_kinks`) unless `freq`, `freq_out` and `freq_out + step` all fall in one interval of `frequency_kinks`;
  - render `slope = render_model_slope(..., value=image)` before the `planes` subset, then subset it the same way;
  - set `dnu = freq - freq_out`.
- Per region, crop and mask `slope` exactly like `sub`. Degrid both with the same `kw` and `mask`, and combine with `dnu`.
- Docstring:
  - document `spectral`, the kink contract and the expansion-point freedom;
  - say that linear mode doubles every gridding pass and keeps one extra full image resident;
  - say that without a beam slope, an empty slope image (box) costs no extra pass, because `degrid_stokes` skips empty planes.

- [ ] **Step 4: Run the full suite**

Run: `uv run --extra full pytest -v`
Expected: all pass, with every pre-existing `test_degrid.py` test unchanged.

- [ ] **Step 5: Lint and commit**

`git commit -m "feat(degrid): predict a model linear in frequency across a chunk"`

---

### Task 3: Beam slope `BeamTerm.dmueller`, and docs

**Files:**
- Modify: `src/pfb_model_spec/utils/degrid.py` (`BeamTerm`, `_check_beam_terms`, `_degrid_beam_term`, wrapper beam branch)
- Modify: `.claude/rules/component-model.md` ("The degrid API"), `CLAUDE.md` ("Current status", degrid bullet)
- Test: `tests/test_degrid.py`

**Interfaces:**
- Consumes: `spectral`, `dnu` and the slope image from Task 2.
- Produces: `BeamTerm(mueller, l_axis, m_axis, rows=None, conj_rows=None, dmueller=None)`. `dmueller` has the same shape and axes as `mueller`, in units per Hz, real or complex.
- Produces: `_degrid_beam_term(uvw, freq, sub, term, l, m, mask, kw, sub1=None, dnu=None)`. When `sub1` is given it returns `V0 + dnu·V1` for that term. Otherwise it behaves as today.

- [ ] **Step 1: Write the failing tests**

```python
def test_linear_frozen_beam_matches_linear_without_a_beam():
    # identity mueller, dmueller=None, divide_by_n=True on both; _hist_mds chunk inside band 1
    # linear with beam == linear without beam to 1e-12 relative (mirrors test_regions_identity_beam_term_matches_no_beam)


def test_linear_beam_slope_matches_per_channel():
    """A flat (box) model under a beam linear in frequency is linear overall, so exact."""
    ds, time, freq, bounds = _hist_mds(method="box")
    geom = model_geometry(ds)
    block, la, ma = _sloped_block(geom)
    kappa = 2.0e-9                                   # fractional beam change per Hz
    chan = np.linspace(bounds[1, 0], bounds[1, 1], 9)[1:-1]
    nu0 = chan.mean()
    beam_at = lambda nu: [BeamTerm(block * (1 + kappa * (nu - nu0)), la, ma)]
    term = BeamTerm(block, la, ma, dmueller=kappa * block)
    # linear with [term], freq_out=nu0 == _per_channel(..., beam_at=beam_at), 1e-8 relative


def test_linear_box_with_a_beam_slope_is_not_flat():
    # same setup: the linear prediction differs from flat (the M1·S0 pass runs), max|diff| > 1e-3 * max|flat|


def test_linear_complex_beam_slope_with_conj_rows():
    # block scaled by (1 + 0.3j), dmueller likewise complex; conj_rows = rows[::2]
    # == _per_channel with beam_at(nu) carrying the same conj_rows, 1e-8 relative


def test_dmueller_needs_linear_mode():
    with pytest.raises(ValueError, match="dmueller"):
        model_to_apparent_vis_for_regions(ds, ..., beam_terms=[BeamTerm(block, la, ma, dmueller=block)])  # spectral="flat"


def test_dmueller_must_match_the_mueller_shape():
    with pytest.raises(ValueError, match="dmueller"):
        model_to_apparent_vis_for_regions(ds, ..., beam_terms=[BeamTerm(block, la, ma, dmueller=block[..., :-1])],
                                          spectral="linear")
```

- [ ] **Step 2: Run them to confirm they fail**

Run: `uv run --extra full pytest tests/test_degrid.py -k "beam_slope or frozen_beam or dmueller" -v`
Expected: FAIL with `TypeError` on the `dmueller` keyword.

- [ ] **Step 3: Implement**

- Add `dmueller: np.ndarray | None = None` as the last `BeamTerm` field. Its docstring must say all of the following:
  - It is the slope per Hz of any linear fit to M(ν) (chord, least squares or λ/D).
  - That fit must pass through `mueller` at the wrapper's `freq_out`. M₀, S₀ and Δν must share one ν₀; mixing expansion points is a first-order error.
  - With `dmueller=None`, the caller should evaluate `mueller` at the band's weighted anchor, not at the chunk mean.
- `_check_beam_terms`: when `dmueller` is not `None`, require `dmueller.shape == mueller.shape`. The wrapper raises if any term has a `dmueller` while `spectral == "flat"`.
- `_degrid_beam_term`, per output plane `o`, interpolating each Mueller entry once and reusing it:

  ```
  plane0 = Σ_i M0[o,i]·sub[i]
  plane1 = Σ_i (M0[o,i]·sub1[i] + M1[o,i]·sub[i])     # M1 term only when dmueller is not None
  vis[o] = V(plane0) + dnu[None, :] · V(plane1)        # V = _degrid_real_imag with this term's mask and sign
  ```

  `conj_rows` applies to `dmueller` exactly as to `mueller`, through the same `sign`.
- Extend the wrapper's cost note: with a `dmueller`, the slope pass runs even for a box model, because M₁S₀ ≠ 0.
- `component-model.md` → "The degrid API":
  - add `frequency_kinks`, `split_at_kinks`, `frequency_pieces`, `render_model_slope`, `spectral` and `BeamTerm.dmueller`, plus the half-open convention;
  - add the kink contract and the expansion-point and beam-consistency rules;
  - mirror the cross-repo table in two or three lines.
- `CLAUDE.md` degrid bullet: one clause, "optionally linear in frequency across a chunk (#39)".

- [ ] **Step 4: Run the full suite**

Run: `uv run --extra full pytest -v`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

`git commit -m "feat(degrid): carry a beam slope through the linear prediction"`
