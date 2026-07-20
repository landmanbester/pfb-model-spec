# Architecture & hip-cargo Mechanics

Read this when editing `src/pfb_model_spec/**`.

This is a hip-cargo package. The patterns below are **load-bearing** — they make the round-trip
between CLI source, generated cabs, and container-fallback execution work. If you want to
deviate, check [hip-cargo's docs](https://github.com/landmanbester/hip-cargo) first.

## 1. Package layout & directory roles

| Directory | What lives there | What does NOT |
|---|---|---|
| `cli/` | Thin Typer wrappers with `@stimela_cab` (+ optional `@stimela_output`). One file per command. Imports from `core/` are **lazy** (inside the function body). | Heavy imports at module top. Business logic. NumPy/domain libs. |
| `core/` | The real implementation — same function name as the CLI wrapper, **no** Typer/hip-cargo decorators. Free to import anything. | Typer. `@stimela_cab`. `typer.Exit(...)`. UI concerns. |
| `utils/` | The component-model spec library, outside the command-oriented layout: `modelspec.py` (fit + render, numpy/sympy/scipy/xarray) and `io.py` (`model_to_ds` — fit, write `.mds`, re-render; numpy/xarray/zarr). Self-contained library code, **not** CLI commands. See `component-model.md`. | Typer. `@stimela_cab`. Any dependency on `pfb_imaging`. |
| `cabs/` | Generated `<command>.yml` files, committed to source control, loaded by Stimela. | Anything hand-written. Drift from `cli/*.py`. |

**`__init__.py`** — must stay light (only `__version__`). It must **not** import `utils.modelspec`
(or `utils.io`, which itself imports `modelspec`), or `import pfb_model_spec` would pull the
scientific stack and break the lightweight install. This invariant is guarded by
`tests/test_lightweight_import.py`.

## 2. Adding a new command

1. `src/pfb_model_spec/cli/<name>.py` — a `@stimela_cab`-decorated Typer function; lazily import
   the core implementation inside the body.
2. `src/pfb_model_spec/core/<name>.py` — the real implementation, same function name, no decorators.
3. Register it in `src/pfb_model_spec/cli/__init__.py` (mirror the `onboard` registration).
4. Add a round-trip case to `tests/test_roundtrip.py`.
5. Commit — the pre-commit hook regenerates `cabs/<name>.yml`.

**Never** create files under `cabs/` by hand.

## 3. Lightweight vs full install

| Mode | Command | Pulls | Use when |
|---|---|---|---|
| Lightweight | `pip install pfb-model-spec` | `hip-cargo` + `typer` | cab consumers (Stimela), dispatching into containers |
| Full | `pip install pfb-model-spec[full]` | + everything in `[project.optional-dependencies].full` | local dev, native execution |

When you add a heavy dep: add it to the **`full`** extra (not top-level `dependencies`), and import
it **only from `core/`** or `utils/` — never from `cli/` at module scope.

## 4. Container fallback & backends

Every generated `cli/*.py` wrapper has this shape (emitted by `hip-cargo generate-function`):

```python
def my_command(...):
    if backend in ("native", "auto"):
        try:
            from hip_cargo.utils.runner import preflight_remote_must_exist
            preflight_remote_must_exist(my_command, dict(...))
            from pfb_model_spec.core.my_command import my_command as core
            core(...); return
        except ImportError:
            if backend == "native":
                raise
    # heavy deps missing OR a container backend was chosen → run in container
    from hip_cargo.utils.config import get_container_image
    from hip_cargo.utils.runner import run_in_container
    run_in_container(my_command, dict(...), image=get_container_image("pfb-model-spec"), backend=backend, ...)
```

Every command auto-grows two params via `generate-function`, both marked `StimelaMeta(skip=True)`
so they appear in the Python CLI but **not** the cab YAML:
- `--backend` (`auto`|`native`|`apptainer`|`singularity`|`docker`|`podman`)
- `--always-pull-images` (bool)

Path-typed params (`File`/`Directory`/`MS`/`URI`) accept local paths and remote URIs
(`s3://`, `gs://`, `az://`); credentials are forwarded into the container by `run_in_container`.

## 5. Cab generation is automatic

`src/pfb_model_spec/cabs/*.yml` are **generated artefacts — never hand-edit them.** Three paths
keep them in sync with `cli/*.py`:

1. **Pre-commit hook** (`generate-cabs`): regenerates cabs on every commit. If it modifies a file
   the commit "fails" — re-stage and re-commit. **It uses `language: system`, so it runs `hip-cargo`
   from the *active* venv** — see `testing-and-ci.md` → "Committing" for why this matters.
2. **`update-cabs.yml`** workflow: on merge to `main`, resets the image tag to `latest`,
   regenerates cabs, and pushes a `[skip checks]` commit.
3. **`tbump`**: on release, sets the image tag to the version and regenerates cabs.

## 6. Image-tag lifecycle (`_container_image.py`)

`CONTAINER_IMAGE` in `src/pfb_model_spec/_container_image.py` is the single source of truth.
**It is deliberately *not* always `:latest`** — its value depends on context, and a non-`latest`
value on a branch is expected, not a bug:

- **Feature branch:** set the tag to the branch name (e.g. `:my-feature`); `publish-container.yml`
  builds/pushes that tag on each PR push.
- **`main`:** `update-cabs.yml` resets it to `:latest`.
- **Release:** `tbump <version>` sets it to the semantic version.
