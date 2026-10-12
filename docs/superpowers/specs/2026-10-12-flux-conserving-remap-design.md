# Design: flux-conserving spatial remapping

**Date:** 2026-10-12
**Status:** Approved (design); pending spec review
**Issues:** #23 (`area_ratio` flux inflation); related: #29 (frequency parametrisation),
ratt-ru/pfb-imaging#309 (`--transfer-model-from`)
**Release:** rides along with pfb-model-spec **0.1.0** (no `.mds` schema change)

## Background

`eval_coeffs_to_slice` renders a component model on its own grid, then moves it onto an
arbitrary output grid by bilinear interpolation (`RegularGridInterpolator`) followed by a
multiplication by `area_ratio = pix_area_out / pix_area_in`. That is correct for a sampled
surface-brightness field but not for what a `.mds` stores, **flux per pixel (Jy/pixel)**.
Refining conserves flux by coincidence; coarsening ×2 inflates it ×4, ×4 inflates it ×16 (#23).

This is no longer just a hypothetical problem. pfb-imaging's `--transfer-model-from` (#309) already calls
`render_model_region(..., nx=, ny=, cell_rad=, x0=, y0=)` with the *target* imaging grid
(`pfb_imaging/core/imager.py`), so a transfer to a coarser cell currently seeds an inflated
model.

## Decisions taken in discussion

- **Conservative area-weighted regridding (option 2 of #23)**, not depositing components onto
  the nearest output pixel (option 1). Option 1 turns extended emission into a CLEAN-like set of
  spikes.
- **No NUFFT/sinc resampling.** It conserves total flux and is Fourier-exact inside the output
  band, but it rings: off-grid compact emission becomes a sampled sinc with negative sidelobes.
  It is also periodic, so emission outside the output footprint wraps in. Both are bad for
  seeding a positivity-constrained deconvolver. It may come back later as an optional mode; it
  is not part of this design.
- **The kernel is a user choice:** `"box"` (piecewise-constant) or `"linear"` (piecewise-linear
  with a minmod slope limiter).
- **The image-space path is always available.** A coefficient-space shortcut is offered only
  for parametrisations that are linear in their coefficients. Planned power-law models and the
  #29 parametrisation work may not be, so nothing may depend on the shortcut.

## Goals

1. Flux-conserving, positivity-preserving remapping of `(…, ny, nx)` images between any two
   axis-aligned rectilinear grids (any cell sizes, sizes and centre offsets, non-square allowed),
   in both directions.
2. A selectable kernel, `"box"` or `"linear"`, defaulting to `"box"`.
3. `eval_coeffs_to_slice` and `render_model_region` use it, and the `area_ratio` scaling goes away.
4. A helper that remaps the **coefficients** of a linear parametrisation directly (box only),
   for writing a `.mds` on a new grid without rendering every (time, freq).

## Non-goals

- A NUFFT/sinc mode (see above).
- Rotation, reprojection, flips or differing tangent points. Both grids share the model's
  orientation and phase-centre convention, as the current resampler assumes.
- Any `.mds` schema change, spec bump or CLI command. Coefficient remapping is a library
  helper; wiring it into pfb-imaging's #309 is a follow-up there.
- #29 itself, or changes to the time/frequency evaluation.

## Pixel semantics

Along each axis, pixel `k` of an `n`-pixel grid with cell `c` and centre offset `x0` is centred at
`x_k = (k - n//2)·c + x0`, the convention `_resample_xmajor` already uses. It covers the interval
`[x_k - c/2, x_k + c/2]`. A pixel value is the **flux in that pixel**, so remapping redistributes
flux; it does not sample a field.

All weights are separable: a 2-D remap is a 1-D remap along x (`NAXIS1`, last axis) followed by a
1-D remap along y. For `"box"` the order is irrelevant. For `"linear"` the limiter makes the
result depend slightly on the order, so the order is fixed (x then y) and documented.

## Kernels (1-D)

Let `u_i` be input fluxes. Then `b_i = u_i / c_in` is the mean brightness in input pixel `i`, and
`O(o, i)` is the overlap of output pixel `o` with input pixel `i`.

**`"box"`**: brightness is uniform within each input pixel.

```
W[o, i] = |O(o, i)| / c_in            out = W @ u
```

**`"linear"`**: brightness is linear within each input pixel, with a limited slope:

```
b(x) = b_i + s_i (x - x_i)       for x in pixel i
s_i  = minmod((b_{i+1} - b_i)/c_in, (b_i - b_{i-1})/c_in),   b_{-1} = b_n = 0
M[o, i] = ∫_{O(o, i)} (x - x_i) dx
out = W @ u + M @ s
```

`minmod(a, b)` is `0` when `a` and `b` differ in sign and otherwise whichever of `a` and `b` is
smaller in magnitude. Consequences, each of which is tested:

- **Conservation.** The linear term integrates to zero over each input pixel, so each input pixel's
  flux is fully redistributed, exactly as with box.
- **Positivity.** For `u ≥ 0`, the limiter guarantees `b(x) ≥ b_i / 2 ≥ 0` inside every pixel, so
  every overlap integral is non-negative.
- **Same grid gives the identity.** Overlaps are whole pixels, `M = 0`, so `out = u` for both kernels.
- **An isolated component behaves exactly like box.** Its neighbour differences have opposite
  signs, so `s = 0` and point sources are never smeared. Smooth extended emission refines smoothly.
- **Exactness.** It is exact for brightness that varies linearly in the interior.
- **Nonlinearity.** The limiter makes `"linear"` nonlinear in `u`, so `"linear"` is image-space only.

`W` and `M` are `scipy.sparse` matrices of shape `(n_out, n_in)`, each with
`O(n_in + n_out)` non-zeros. Dense matrices would be prohibitive at 10k-pixel grids.

**Footprint.** Flux from input pixels outside the output footprint is dropped, never wrapped.
An input pixel that straddles the edge contributes only its overlapping part. So `sum(out)`
equals `sum(in)` exactly when the input footprint lies inside the output footprint, and
otherwise equals the flux inside the output footprint.

## API

### New module `utils/remap.py` (numpy + scipy.sparse; no sympy)

- `remap_image(image, *, kernel="box", nxi, nyi, cellxi, cellyi, x0i, y0i, nxo, nyo, cellxo,
  cellyo, x0o, y0o) -> np.ndarray`. Takes `image` of shape `(..., nyi, nxi)` and returns
  `(..., nyo, nxo)`; leading axes (e.g. Stokes) are remapped independently. If the two
  geometries are identical it returns a copy, so the same-grid path stays bit-exact.
- `remap_coefficients(coeffs, y_index, x_index, expr, params, *, nxi, …, y0o)
  -> (coeffs_out, y_index_out, x_index_out)`. Remaps `(nstokes, npar, ncomps)` coefficients
  with the box kernel, treating each `(stokes, par)` row as a sparse image. It builds sparse
  `Wy[:, y_index] · diag(c) · Wx[:, x_index]ᵀ` products and never renders a dense cube. The output
  components are the union of output pixels that are non-zero in any `(stokes, par)` plane, in
  the same `np.where` order `fit_image_cube` uses.
  - It first checks that `expr` is **linear and homogeneous** in `params` (all second derivatives
    w.r.t. params vanish, and `expr` is `0` when every param is `0`; checked with sympy). Otherwise
    it raises `ValueError`, naming `remap_image` as the alternative. This is what keeps power-law
    or other nonlinear parametrisations correct: they fail loudly instead of being remapped wrongly.
  - Box only, because the linear kernel is not linear in the data. The signature has no
    `kernel` argument.
  - It returns only coefficients and locations; the caller writes the new `.mds` via
    `build_mds_dataset` with the target geometry.
  - It lives in `remap.py` but imports sympy lazily for the linearity check, so `remap_image`
    stays numpy/scipy-only.
- Invalid arguments raise `ValueError`: an unknown `kernel`, a non-positive cell or size, or an
  `image` whose trailing shape does not match `(nyi, nxi)`.

### Changes to existing functions

- `eval_coeffs_to_slice(..., *, <geometry>, kernel="box")` adds a new keyword-only `kernel`. It
  renders `(nstokes, nyi, nxi)` directly in `(Y, X)` order (no x-major intermediate or transpose)
  and calls `remap_image`. `_resample_xmajor`, `area_ratio` and the `RegularGridInterpolator`
  import are removed.
- `render_model_region(..., kernel="box")` passes `kernel` through. Its "not flux-conserving"
  Note is replaced by a description of the kernels.
- `model_to_ds` and `model2comps` render on their own grid, take the identity path and are
  unchanged.

### Contract impact

- Every existing call keeps working: the new arguments are additive, keyword-only and defaulted.
- **Numerics change for every off-grid call.** That is the fix, and it is a behavioural contract
  change (`component-model.md` → "Ownership"). The only off-grid consumer is pfb-imaging's
  `--transfer-model-from`, which gets correct flux without any code change. Its tests should be
  re-run against this branch before release.
- There is no `.mds` change, so no `SPEC_VERSION` step is needed. It ships in the unreleased 0.1.0.

## Testing (`tests/test_remap.py`, plus edits)

Synthetic only, following the existing style:

- **Identity:** for both kernels, a same-grid remap equals the input exactly.
- **Conservation:** for both kernels, with refine ×2 and ×4, coarsen ×2 and ×4, non-integer ratios
  (e.g. ×1.7) and sub-pixel `x0`/`y0` offsets, `sum(out) == sum(in)` when the input is contained
  in the output footprint. A single unit component reproduces the #23 table with `sum = 1` everywhere.
- **Box exactness:** coarsening by an integer factor equals block sums; an integer-pixel shift
  moves values exactly. This keeps `test_slice_matches_cube_and_is_shift_invariant` passing.
- **Positivity:** `out ≥ 0` for random non-negative input with both kernels.
- **Linear kernel:**
  - an isolated component gives the same result as box;
  - a linear ramp, refined ×2, is reproduced exactly in the interior;
  - for a smooth Gaussian refined ×4, the error is smaller than box's.
- **Footprint:** emission outside the output grid is dropped and does not reappear on the opposite
  edge (no wrap-around).
- **Validation:** an unknown `kernel` and mismatched shapes raise `ValueError`.
- **`remap_coefficients`:**
  - rendering its output on the target grid equals `remap_image(kernel="box")` of the rendering
    on the source grid, at several (time, freq) values (this is what linearity means);
  - a non-linear `expr` (e.g. `t0*exp(f1*f)`) raises `ValueError`.
- **Edits to existing tests:**
  - `test_render_model_region_coarsen_inflates_flux_by_area_ratio` (`tests/test_degrid.py`) is
    replaced by a test that coarsening conserves flux;
  - the refine test there stays as it is.

## Documentation

- `.claude/rules/component-model.md`:
  - replace "Known limitation: resampling is not flux-conserving" with the kernel description;
  - add `utils/remap.py` to the library API section;
  - note that the coefficient shortcut is limited to linear parametrisations.
- `CLAUDE.md` project-structure line for `utils/`: add `remap.py`.
- Close #23 from the PR.
