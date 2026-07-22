# Python Standards & CLI Implementation

Read this when editing or creating `**/*.py`.

## 1. Modern Python
- Python 3.10+; use modern syntax (`X | Y`, `list[int]`, etc.).
- Type hints on every function signature. Use `from typing import Any` for generics.
- Only import from `typing_extensions` if genuinely required.

## 2. Typer Option/Argument syntax (CRITICAL)
**Never** pass `None` as the positional default to `typer.Option()` — it raises `AttributeError`.

- **Required:** `Annotated[T, typer.Option(..., help="...")]` (no `= default`).
- **Optional w/ default:** `Annotated[T, typer.Option(help="...")] = default`.
- **Optional None:** `Annotated[T | None, typer.Option(help="...")] = None`.

## 3. Imports
- **`cli/` modules stay lightweight:** import `core/` (and any heavy dep) **lazily**, inside the
  function body. This keeps `pfb --help` and cab generation fast and the lightweight install viable.
- **`core/` and `utils/` (`modelspec.py`, `io.py`)** may import heavy deps at module top.
- **`src/pfb_model_spec/__init__.py` must not import `utils.modelspec`** (or any heavy dep) — the
  lightweight-import guard test enforces this.
- Comma-separated list options use `ListInt`/`ListFloat`/`ListStr` from `hip_cargo` with their
  `parse_list_*` parsers. Path types `File`/`Directory`/`MS`/`URI` use `parser=parse_upath`.

## 4. Architectural style
- Prefer pure functions and explicit behaviour; use classes when state/polymorphism genuinely helps.
- Keep it DRY but don't over-engineer. Prefer the standard library; add a dep via `uv add` only
  when it meaningfully reduces complexity (and put runtime deps in the `full` extra — see
  `architecture.md`).
- Be explicit about errors and let exceptions propagate unless there's a reason to catch them. Use
  `typer.Exit(code=1)` for CLI-level errors.

## 5. Documentation
- Google-style docstrings; document Args/Returns/Raises. Keep them concise.
- Comment only where intent isn't obvious; prefer short inline comments.

## 6. Canonical spec library
`utils/modelspec.py` is the **canonical** spec library — pfb-imaging now imports it from here rather
than the other way around, so it is no longer a vendored copy *of* pfb-imaging (`tests/_synth.py`
likewise originated upstream). Do **not** restyle or "improve" them: the public signatures, return
tuples, and `.mds` schema are a cross-repo contract, so cosmetic churn risks silent behavioural
drift. See `component-model.md` → "Ownership".
