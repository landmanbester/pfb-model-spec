"""The pre-commit ruff hook must be the ruff that uv locked.

CI runs `uv run ruff format --check`, which uses the ruff in uv.lock, and
hip-cargo's generate-function formats generated CLIs with it too. The
pre-commit hook runs its own ruff, pinned by `rev:` in .pre-commit-config.yaml.
If the two versions differ, a formatter change between them makes the hook and
CI disagree. When ruff moves in uv.lock, bump the hook's `rev` to `v<version>`.
"""

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RUFF_PRE_COMMIT = "https://github.com/astral-sh/ruff-pre-commit"


def test_precommit_ruff_matches_locked_ruff() -> None:
    match = re.search(r'^name = "ruff"\nversion = "([^"]+)"', (REPO_ROOT / "uv.lock").read_text(), re.MULTILINE)
    assert match, "ruff is not in uv.lock"
    repos = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text())["repos"]
    rev = next(repo["rev"] for repo in repos if repo["repo"] == RUFF_PRE_COMMIT)
    assert rev == f"v{match.group(1)}", (
        f"ruff-pre-commit is at {rev} but uv.lock has ruff {match.group(1)}; "
        f"set `rev: v{match.group(1)}` in .pre-commit-config.yaml"
    )
