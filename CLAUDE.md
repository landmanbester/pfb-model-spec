# CLAUDE.md — Project Context

**pfb-model-spec** owns the *component model* used by
[pfb-imaging](https://github.com/ratt-ru/pfb-imaging): a compact `.mds` ("model dataset")
representation of a sky model as Legendre/polynomial coefficients over time and frequency,
together with the routines that fit a pixelated image cube into those coefficients and render
them back to images.

It is a **[hip-cargo](https://github.com/landmanbester/hip-cargo) package**: a lightweight
Typer CLI (`pfbspec`) whose commands are decorated so that Stimela cab definitions are generated
automatically, with container-fallback execution. The project prioritizes **simplicity and
minimalism**; when in doubt, consult [The Twelve Factor App](https://12factor.net/).

*Detailed architecture, domain logic, Python standards, and CI/CD rules are modularized into
`.claude/rules/` for progressive disclosure — read the relevant file when working in that area.*

## Current status (phase 1)

- **Implemented:** the portable spec library `pfb_model_spec/modelspec.py` (fit + render),
  vendored **byte-for-byte** from `pfb_imaging/utils/modelspec.py` so pfb-imaging can later
  import it as a drop-in replacement. Synthetic, measurement-set-free tests. Release/CI tooling
  (changelog, conventional commits, `uv`-ecosystem Dependabot, CODEOWNERS) at parity with the
  sibling repos.
- **Scaffold:** the `onboard` command (safe to delete once project setup is complete).
- **Deferred (not yet built):** the `model2comps` CLI converter, the pfb-imaging `.dds` reading
  path, and FITS I/O. See `.claude/rules/component-model.md`.

The full design + plan live in `docs/superpowers/specs/` and `docs/superpowers/plans/`.

## Mandatory dev workflow

```bash
uv run ruff format . && uv run ruff check . --fix      # after every code change
uv run --extra full pytest -v                           # run tests (the `full` extra is REQUIRED)
```

- **Commits must run with the project venv active** so the `generate-cabs` pre-commit hook can
  resolve the container image — see `.claude/rules/testing-and-ci.md` → "Committing".
- **Commit messages must be [Conventional Commits](https://www.conventionalcommits.org/)**
  (enforced by a `commit-msg` hook).

## Detailed rules (read on demand)

| Read this | When |
|---|---|
| `.claude/rules/architecture.md` | editing `src/pfb_model_spec/**` — package layout, cli/core split, container fallback, cab generation, image-tag lifecycle |
| `.claude/rules/component-model.md` | working on the spec library, the `.mds` format, or the fit/render routines |
| `.claude/rules/python-standards.md` | writing any `**/*.py` — type hints, Typer syntax, imports |
| `.claude/rules/testing-and-ci.md` | editing `tests/**` or `.github/workflows/**`, committing, or cutting a release |

## Project structure

```
src/pfb_model_spec/
├── __init__.py            # stays light: only __version__ (must NOT import modelspec)
├── _container_image.py    # CONTAINER_IMAGE — single source of truth for the image tag
├── modelspec.py           # the component-model spec library (fit + render)
├── cli/                   # lightweight Typer wrappers — generate-cabs parses these
├── core/                  # heavy implementations mirroring cli/ commands (one per command)
└── cabs/                  # AUTO-GENERATED Stimela YAMLs — never hand-edit
tests/                     # synthetic, MS-free tests (+ _synth.py helpers)
```
