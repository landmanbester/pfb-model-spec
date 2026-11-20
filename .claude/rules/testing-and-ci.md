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
  CLI direction fails, regenerate the module from its cab — never hand-fix `cli/*.py`. It also checks
  two contracts: every CLI parameter is a parameter of `<cmd>_core` (the wrapper calls it with
  `**params`), and every cab command is registered in `cli/__init__.py`. If one fails, fix the
  source, not the test. A generated core import whose module or function no longer exists fails;
  the signature check skips only a hand-written wrapper or a core function it cannot follow with
  `ast` (re-exported, aliased or defined conditionally).
- `tests/test_precommit.py` guards that the ruff `rev:` in `.pre-commit-config.yaml` equals the
  ruff in `uv.lock`; bump the hook when ruff moves in the lock.
- Both template files set `pytestmark = pytest.mark.unit`. Mark new test files too: an unmarked file
  is silently deselected by `pytest -m unit`.
- `tests/test_spec.py` guards that `SPEC_VERSION` is not behind the package `major.minor`.
- `tests/_genesis.py` is a frozen fixture (the genesis writer) and must never be updated to follow
  the library.
- `tests/test_lightweight_import.py` guards that `import pfb_model_spec` does **not** import
  `modelspec` (keeps the lightweight install intact).

## 2. Committing (read before you commit)
- **Keep the project `.venv` synced** (`uv sync --extra full --group test`) before committing. The
  `generate-cabs` hook (and tbump's cab step) run `uv run --no-sync hip-cargo …`, so they use the
  hip-cargo locked in `uv.lock` from the project `.venv`, whichever venv is active. `--no-sync`
  never installs anything, so a missing or stale `.venv` fails the hook rather than being fixed by
  it. (A bare `hip-cargo` from `PATH` could come from a sibling project's venv, which cannot
  resolve the image and silently strips the `image:` line from a cab.)
- `default_stages: [pre-commit]` makes each hook run once per commit; the commit-msg stage runs only
  the conventional-commit hook.
- **Commit messages must be Conventional Commits.** A `commit-msg` hook (`conventional-pre-commit`)
  enforces it. Allowed types: `feat fix refactor perf docs deps chore ci style test build`.
  After cloning, run `pre-commit install` once — `default_install_hook_types` wires up both the
  `pre-commit` and `commit-msg` stages.
- Don't prefix human commits with `[skip checks]` (reserved for the `update-cabs` bot) and avoid the
  `chore(release):` prefix for ordinary work (it's skipped from the changelog).

## 3. GitHub workflows
- **`ci.yml`** — runs on **pull requests only** (into `main` or a long-lived `dev-*` integration
  branch) plus `workflow_dispatch`; there is no `push` trigger, since a merge is the merge of a PR
  that already passed. A PR into any other branch (e.g. a stacked PR) runs only on demand:
  `gh workflow run ci.yml --ref <branch>`. Do not add a `push` trigger back. A PR is tested against
  its base as of the run, so the ruleset requires branches to be up to date before merging.
  Jobs: `quality` (ruff, lint-only), a `test` matrix and a `lightweight` job. The
  full suite runs on the oldest and newest Python the `full` extra supports (`Tests (3.11)`,
  `Tests (3.14)`) plus `Tests (3.14, arm64)`; `UV_PYTHON` comes from the matrix (otherwise uv
  follows `.python-version`). It installs with `uv sync --extra full --group test` and runs
  `uv run --extra full pytest` (the `full` extra is required, or modelspec tests fail to import).
  **Python 3.10 is lightweight-only**: `Lightweight install (3.10)` installs without extras and runs
  `pfbspec --help`, which imports every `cli/` module; its Python must equal the `requires-python`
  floor. These job names are the required checks on `main` — renaming a job means updating the
  ruleset. Until hip-cargo 0.5.0 is on PyPI, `Lightweight install (3.10)` does not check the
  hip-cargo floor: `uv pip install .` honours the git source in `[tool.uv.sources]`.
- **`publish.yml`** — on `v*` tags: lint + a `test` job + `uv build` + PyPI trusted publishing
  (environment `pypi`); `publish` needs both `quality` and `test`. The tbump release commit never
  goes through a PR, so this is the only CI run that sees it. The `test` job's Python (`UV_PYTHON`
  and `setup-python`, `3.14`) must equal the newest Python in the `ci.yml` matrix.
- **`publish-container.yml`** — builds/pushes the image to GHCR on `v*` tags and pushes to `main`
  only, never on PRs. The image is multi-arch: amd64 and arm64 build on native runners and push by
  digest, and a `merge` job tags one manifest (failing if an architecture is missing). A shared
  `meta` job computes tags, labels and index annotations once: labels live in each image's config,
  so the per-arch builds set them, and sharing them keeps the `version` label equal to the published
  tag. A per-ref `concurrency` group cancels superseded builds, so `:latest` comes from the newest
  `main` commit.
  Feature-branch images are built on demand by the maintainer
  (`gh workflow run publish-container.yml --ref <branch>`, tag = branch name with `/` → `-`); agents
  never trigger it or change the image tag (see `architecture.md` §6). The image installs from the
  lock (`uv export --locked --extra full` → `uv pip install -r`, then the project with `--no-deps`),
  so every architecture runs the versions CI tested and dependencies move only when `uv.lock` does.
  `--locked` fails the build if `uv.lock` is stale — run `uv lock` after editing dependencies. The
  dependency layer copies only `pyproject.toml`/`uv.lock`, so it stays cached until the lock
  changes. The Dockerfile must also `COPY … LICENSE ./`, because `pyproject.toml` sets
  `license-files = ["LICENSE"]` (newer `uv_build` errors if the license glob matches nothing in the
  build context).
- **`update-cabs.yml`** — on push to `main`, resets the image tag to `latest`, regenerates cabs, and
  pushes a commit tagged `[skip checks]`, which stops this workflow re-triggering on its own push
  (`ci.yml` no longer reads the tag). Requires the `APP_CLIENT_ID`/`APP_PRIVATE_KEY` GitHub App secrets.

## 4. Dependencies (Dependabot)
`.github/dependabot.yml` uses the **`uv` ecosystem** (keeps `pyproject.toml` + `uv.lock` in sync;
the `pip` ecosystem would not update `uv.lock`). Bumps are grouped by update-type
(minor+patch batched, major individual). Review requests come from `CODEOWNERS`, not the deprecated
Dependabot `reviewers` key.

A monthly **`docker`** entry bumps the Dockerfile's base images. The image's uv is a named
`FROM ghcr.io/astral-sh/uv:<tag> AS uv` stage because Dependabot only reads `FROM` lines (a
`COPY --from=<image>` is never updated). Keep that uv no older than the one that writes `uv.lock`:
`uv export --locked` re-checks the lock with it, and an old uv can reject a valid lock. Python
minor/major bumps are ignored (the image's Python minor is chosen by hand). When a uv bump moves
past the `uv_build` cap in `[build-system]` (`<0.14.0`), raise the cap.

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
