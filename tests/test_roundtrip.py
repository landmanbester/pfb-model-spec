"""Round-trip tests: cli/*.py <-> cabs/*.yml.

The committed CLI modules and the committed cabs are two renderings of one
definition. These tests assert they still agree, in both directions:

* every committed cab is what ``generate-cabs`` emits from the committed CLI
  module -- a failure means the cabs are stale, so regenerate them;
* every committed CLI module is what ``generate-function`` emits from the
  committed cab, byte for byte.

If the second one fails, do NOT hand-fix cli/*.py. Either edit the cab and
regenerate the module, or file a hip-cargo bug if the regeneration is wrong.

Commands are discovered, not listed: adding one under src/pfb_model_spec/cli/
needs no change here.
"""

import ast
from pathlib import Path

import pytest
import yaml
from hip_cargo.core.generate_cabs import generate_cabs
from hip_cargo.core.generate_function import generate_function

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CLI_DIR = REPO_ROOT / "src" / "pfb_model_spec" / "cli"
CABS_DIR = REPO_ROOT / "src" / "pfb_model_spec" / "cabs"
CONTAINER_IMAGE_PY = REPO_ROOT / "src" / "pfb_model_spec" / "_container_image.py"


def _cli_modules() -> list[Path]:
    """Every CLI module, excluding __init__ and private helpers."""
    return sorted(path for path in CLI_DIR.glob("*.py") if not path.stem.startswith("_"))


def _has_stimela_cab(path: Path) -> bool:
    """True if a function in the module is decorated with @stimela_cab.

    Parsed rather than searched as text, so a command that is deliberately left
    uncabbed and says so in its docstring is not mistaken for a cab command.
    """
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
                if name == "stimela_cab":
                    return True
    return False


def _commands() -> list[str]:
    """Commands with a committed cab to compare against."""
    return [path.stem for path in _cli_modules() if (CABS_DIR / f"{path.stem}.yml").is_file()]


def _container_image() -> str | None:
    """CONTAINER_IMAGE, read from the source file rather than imported.

    generate-cabs resolves this from installed distribution metadata. Reading
    the constant instead keeps this suite runnable on a bare checkout, and
    checks the committed cabs against the real source of truth rather than
    against themselves.
    """
    if not CONTAINER_IMAGE_PY.is_file():
        return None
    for node in ast.parse(CONTAINER_IMAGE_PY.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CONTAINER_IMAGE" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    return None


COMMANDS = _commands()
CONTAINER_IMAGE = _container_image()


def test_there_is_at_least_one_command() -> None:
    """Guard against the discovery above silently matching nothing."""
    assert COMMANDS, f"no CLI modules with cabs found under {CLI_DIR}"


def test_every_generated_cab_has_a_cli_module() -> None:
    """A generated cab whose CLI module was deleted would otherwise go unchecked.

    Only generated cabs count: ``cabs/<cmd>.yml`` defining ``cabs.<cmd>`` with a
    ``command``. A hand-written aggregate such as ``cabs/cabs.yml`` (stimela
    ``_include`` entries) is not one.
    """
    orphans = []
    for cab_file in sorted(CABS_DIR.glob("*.yml")):
        cab = (yaml.safe_load(cab_file.read_text()) or {}).get("cabs", {}).get(cab_file.stem)
        if isinstance(cab, dict) and "command" in cab and not (CLI_DIR / f"{cab_file.stem}.py").is_file():
            orphans.append(cab_file.name)
    assert not orphans, f"committed cabs with no cli/ module: {orphans}; delete them or restore the module"


def test_every_decorated_command_has_a_committed_cab() -> None:
    """A @stimela_cab command with no cab means generate-cabs was not run."""
    missing = [
        path.stem for path in _cli_modules() if _has_stimela_cab(path) and not (CABS_DIR / f"{path.stem}.yml").is_file()
    ]
    assert not missing, f"no committed cab for {missing}; run generate-cabs"


@pytest.mark.parametrize("command", COMMANDS)
def test_the_committed_cab_is_what_generate_cabs_emits(command: str, tmp_path: Path) -> None:
    """Forward direction: cli/<command>.py -> cabs/<command>.yml."""
    generate_cabs([CLI_DIR / f"{command}.py"], output_dir=tmp_path, image=CONTAINER_IMAGE)

    committed = yaml.safe_load((CABS_DIR / f"{command}.yml").read_text())
    regenerated = yaml.safe_load((tmp_path / f"{command}.yml").read_text())

    assert regenerated == committed, f"cabs/{command}.yml is stale; regenerate it"


@pytest.mark.parametrize("command", COMMANDS)
def test_the_committed_cli_is_what_generate_function_emits(command: str, tmp_path: Path) -> None:
    """Reverse direction: cabs/<command>.yml -> cli/<command>.py, byte for byte."""
    regenerated_file = tmp_path / f"{command}.py"
    generate_function(CABS_DIR / f"{command}.yml", output_file=regenerated_file, config_file=PYPROJECT)

    committed_lines = (CLI_DIR / f"{command}.py").read_text().splitlines()
    regenerated_lines = regenerated_file.read_text().splitlines()

    assert len(regenerated_lines) == len(committed_lines), (
        f"{command}: committed {len(committed_lines)} lines, regenerated {len(regenerated_lines)}"
    )
    for number, (committed_line, regenerated_line) in enumerate(zip(committed_lines, regenerated_lines), 1):
        assert committed_line == regenerated_line, (
            f"{command} line {number} differs:\n  committed:   {committed_line}\n  regenerated: {regenerated_line}"
        )
