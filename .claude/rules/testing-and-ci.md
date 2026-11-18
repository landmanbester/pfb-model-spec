# Testing, Committing & CI/CD

Read this when editing `tests/**` or `.github/workflows/**`, when committing, or when cutting a
release.

## 1. Testing
- Tests are **synthetic and self-contained** — no measurement set, no `daskms`/`africanus`. Build
  inputs from hard-coded arrays (see `tests/test_modelspec.py` + `tests/_synth.py`).
- Run with the `full` extra, which carries the scientific stack:
  ```bash
  uv run --extra full pytest -v
  ```
  Plain `uv run pytest` may omit the extra and the modelspec test will fail on `import numpy`.
- `tests/test_roundtrip.py` (hip-cargo's template) discovers every `cli/<cmd>.py` and checks both
  directions: the committed cab is what `generate-cabs` emits, and the committed CLI is what
  `generate-function` emits from that cab, byte for byte. New commands need no change here. If the
  CLI direction fails, regenerate the module from its cab — never hand-fix `cli/*.py`.
- `tests/test_precommit.py` guards that the ruff `rev:` in `.pre-commit-config.yaml` equals the
  ruff in `uv.lock`; bump the hook when ruff moves in the lock.
- `tests/test_spec.py` guards that `SPEC_VERSION` is not behind the package `major.minor`.
- `tests/_genesis.py` is a frozen fixture (the genesis writer) and must never be updated to follow
  the library.
- `tests/test_lightweight_import.py` guards that `import pfb_model_spec` does **not** import
  `modelspec` (keeps the lightweight install intact).

## 2. Committing (read before you commit)
- **Run `git commit` with the project venv active**, inline, in the same shell:
  ```bash
  source /path/to/pfb-model-spec/.venv/bin/activate && git add <files> && git commit -m "..."
  ```
  Why: the `generate-cabs` pre-commit hook uses `language: system`, so it runs whatever `hip-cargo`
  is on `PATH`. If the wrong venv is active (e.g. a sibling project's), `get_container_image` can't
  resolve and the hook **silently strips the `image:` line** from `cabs/model2comps.yml`, failing the
  commit. `uv run …` is not enough for commits because the git hook runs in the ambient shell.
- **Commit messages must be Conventional Commits.** A `commit-msg` hook (`conventional-pre-commit`)
  enforces it. Allowed types: `feat fix refactor perf docs deps chore ci style test build`.
  After cloning, run `pre-commit install` once — `default_install_hook_types` wires up both the
  `pre-commit` and `commit-msg` stages.
- Don't prefix human commits with `[skip checks]` (reserved for the `update-cabs` bot) and avoid the
  `chore(release):` prefix for ordinary work (it's skipped from the changelog).

## 3. GitHub workflows
- **`ci.yml`** — `quality` job (ruff, lint-only), a `test` matrix and a `lightweight` job. The
  full suite runs on the oldest and newest Python the `full` extra supports (`Tests (3.11)`,
  `Tests (3.14)`) plus `Tests (3.14, arm64)`; `UV_PYTHON` comes from the matrix (otherwise uv
  follows `.python-version`). It installs with `uv sync --extra full --group test` and runs
  `uv run --extra full pytest` (the `full` extra is required, or modelspec tests fail to import).
  **Python 3.10 is lightweight-only**: `Lightweight install (3.10)` installs without extras and runs
  `pfbspec --help`, which imports every `cli/` module; its Python must equal the `requires-python`
  floor. These job names are the required checks on `main` — renaming a job means updating the
  ruleset. Honors a `[skip checks]` commit tag.
- **`publish.yml`** — on `v*` tags: lint + `uv build` + PyPI trusted publishing (environment `pypi`).
  Does **not** run tests (they run in `ci.yml`).
- **`publish-container.yml`** — builds/pushes the image to GHCR on `v*` tags and pushes to `main`
  only, never on PRs. The image is multi-arch: amd64 and arm64 build on native runners and push by
  digest, and a `merge` job tags one manifest (failing if an architecture is missing). A per-ref
  `concurrency` group cancels superseded builds, so `:latest` comes from the newest `main` commit.
  Feature-branch images are built on demand by the maintainer
  (`gh workflow run publish-container.yml --ref <branch>`, tag = branch name with `/` → `-`); agents
  never trigger it or change the image tag (see `architecture.md` §6). The image build runs `uv pip install ".[full]"`; the Dockerfile must `COPY … LICENSE ./` because
  `pyproject.toml` sets `license-files = ["LICENSE"]` (newer `uv_build` errors if the license glob
  matches nothing in the build context).
- **`update-cabs.yml`** — on push to `main`, resets the image tag to `latest`, regenerates cabs, and
  pushes a `[skip checks]` commit. Requires the `APP_CLIENT_ID`/`APP_PRIVATE_KEY` GitHub App secrets.

## 4. Dependencies (Dependabot)
`.github/dependabot.yml` uses the **`uv` ecosystem** (keeps `pyproject.toml` + `uv.lock` in sync;
the `pip` ecosystem would not update `uv.lock`). Bumps are grouped by update-type
(minor+patch batched, major individual). Review requests come from `CODEOWNERS`, not the deprecated
Dependabot `reviewers` key.

## 5. Releases
```bash
uv run tbump <version>      # e.g. uv run tbump 0.0.1
```
`tbump` (via `uv run`, so hooks resolve the image) bumps `pyproject.toml` + `__init__.py`,
regenerates the changelog (`git-cliff` → `CHANGELOG.md` per `cliff.toml`), rewrites the image tag to
the version, regenerates cabs, commits, tags `v<version>`, and pushes; the tag triggers
`publish.yml` and `publish-container.yml`.

`tbump` to a new minor version fails CI until a spec step (possibly a no-op) is added to
`utils/spec.py`; schema changes ship only in a new minor release.

**Release ordering:** after merging a PR to `main`, let `update-cabs.yml` finish (it pushes a
`[skip checks]` commit), then `git checkout main && git pull` and confirm a clean tree **before**
running `tbump` — otherwise tbump's push is rejected as non-fast-forward. Verify the PyPI trusted
publisher matches workflow `publish.yml` + environment `pypi`, and that the `update-cabs` GitHub App
secrets exist.
