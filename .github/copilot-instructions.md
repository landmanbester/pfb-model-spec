# Copilot instructions for pfb-model-spec

This is a **[hip-cargo](https://github.com/landmanbester/hip-cargo) package**: a lightweight Typer
CLI (`pfbspec`) whose Stimela cab definitions and container image tag are managed by automation.
Several things that look like bugs are intentional and automated — please **do not flag them in
reviews**.

## Do NOT flag these (they are automated / by design)

- **The container image tag** in `src/pfb_model_spec/_container_image.py` (`CONTAINER_IMAGE`) and the
  `image:` field in `src/pfb_model_spec/cabs/*.yml`. The tag is **deliberately not always `latest`**:
  - feature branches use the **branch name** as the tag (e.g. `:my-feature`);
  - `main` is reset to `:latest` automatically by the `update-cabs.yml` workflow on merge;
  - releases use the **semantic version** via `tbump`.
  A non-`latest` tag on a branch, or a cab whose `image:` matches the branch tag, is **correct** —
  do not suggest changing it to `latest` or "fixing" the mismatch.
- **`src/pfb_model_spec/cabs/*.yml`** are **generated artefacts**. Do not suggest hand-editing them,
  and do not flag drift between branches — they are regenerated from `cli/*.py` by a pre-commit hook,
  by `update-cabs.yml` on `main`, and by `tbump` on release.
- **`src/pfb_model_spec/utils/modelspec.py`** is the **canonical** spec library — pfb-imaging imports
  it from here, so it is no longer a vendored copy *of* pfb-imaging (`tests/_synth.py` likewise
  originated upstream). Do not suggest renaming, restyling, refactoring, or "modernizing" them: the
  public signatures, return tuples, and `.mds` schema are a **cross-repo contract** with pfb-imaging,
  so cosmetic churn risks silent behavioural drift. (`src/pfb_model_spec/utils/io.py` and
  `utils/fits.py` are newer, non-vendored code — normal review rules apply there.)
- **`[skip checks]`** in commit messages (used only by the `update-cabs` bot) — not a mistake.
- The **two-tier dependency split**: top-level `dependencies` stays tiny (`hip-cargo`); the
  scientific stack lives in `[project.optional-dependencies].full`. The top-level `__init__.py`
  intentionally imports nothing heavy. This is required for the lightweight install — do not suggest
  consolidating them.

## DO flag / care about

- Behavioural changes to `utils/modelspec.py` public signatures, return tuples, or the `.mds` schema
  (these break the cross-repo contract with pfb-imaging).
- `cli/` modules importing heavy deps at module scope, or importing from `core/` outside the function
  body (must stay lazy).
- Non-Conventional-Commit messages (types: `feat fix refactor perf docs deps chore ci style test
  build`).
- Typer options using `None` as a positional `typer.Option()` default (raises `AttributeError`).
- The Dockerfile failing to `COPY … LICENSE ./` while `pyproject.toml` sets
  `license-files = ["LICENSE"]` (the image build fails under newer `uv_build`).

## Context

- Lightweight install (`pip install pfb-model-spec`) → `hip-cargo` + `typer` only. Full install
  (`[full]`) adds the runtime stack. CI tests must run with the `full` extra.
- Tests are synthetic (no measurement set). `pytest` needs `uv run --extra full`.
- More detail lives in `CLAUDE.md` and `.claude/rules/`.
