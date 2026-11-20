"""Round-trip tests: cli/*.py <-> cabs/*.yml.

The committed CLI modules and the committed cabs are two renderings of one
definition. These tests assert they still agree, in both directions:

* every committed cab is what ``generate-cabs`` emits from the committed CLI
  module -- a failure means the cabs are stale, so regenerate them;
* every committed CLI module is what ``generate-function`` emits from the
  committed cab, byte for byte.

If the second one fails, do NOT hand-fix cli/*.py. Either edit the cab and
regenerate the module, or file a hip-cargo bug if the regeneration is wrong.

It also checks the two contracts the round trip cannot see: each wrapper's
``<name>_core(**params)`` call binds to the core function's signature, and each
cab command is registered on the CLI app.

Commands are discovered, not listed: adding one under src/pfb_model_spec/cli/
needs no change here.
"""

import ast
from pathlib import Path

import pytest
import yaml
from hip_cargo.core.generate_cabs import generate_cabs
from hip_cargo.core.generate_function import generate_function

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CLI_DIR = REPO_ROOT / "src" / "pfb_model_spec" / "cli"
CABS_DIR = REPO_ROOT / "src" / "pfb_model_spec" / "cabs"
CONTAINER_IMAGE_PY = REPO_ROOT / "src" / "pfb_model_spec" / "_container_image.py"


def _cli_modules() -> list[Path]:
    """Every CLI module, excluding __init__ and private helpers."""
    return sorted(path for path in CLI_DIR.glob("*.py") if not path.stem.startswith("_"))


def _cab_function(path: Path) -> ast.FunctionDef | None:
    """The function in the module decorated with @stimela_cab, if any.

    Parsed rather than searched as text, so a command that is deliberately left
    uncabbed and says so in its docstring is not mistaken for a cab command.
    """
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", None)
                if name == "stimela_cab":
                    return node
    return None


def _has_stimela_cab(path: Path) -> bool:
    return _cab_function(path) is not None


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


def _core_binding(command: str) -> tuple[set[str], ast.FunctionDef] | None:
    """The keys the wrapper passes as ``**params`` and the core function it calls.

    None when the wrapper is not in the generated shape (a hand-written command),
    or when the core function lives outside this package or is re-exported rather
    than defined. Fails when the generated import points at a module or function
    that does not exist: a stale import must not turn the check into a skip.
    """
    wrapper = _cab_function(CLI_DIR / f"{command}.py")
    if wrapper is None:
        return None
    nodes = list(ast.walk(wrapper))
    call = next(
        (
            node
            for node in nodes
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and not node.args
            and len(node.keywords) == 1
            and node.keywords[0].arg is None
            and isinstance(node.keywords[0].value, ast.Name)
        ),
        None,
    )
    if call is None:
        return None
    params_var = call.keywords[0].value.id
    keys = next(
        (
            {kw.arg for kw in node.value.keywords} if isinstance(node.value, ast.Call) else set()
            for node in nodes
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == params_var for t in node.targets)
        ),
        None,
    )
    source = next(
        (
            (node.module, alias.name)
            for node in nodes
            if isinstance(node, ast.ImportFrom) and node.module
            for alias in node.names
            if (alias.asname or alias.name) == call.func.id
        ),
        None,
    )
    if keys is None or source is None:
        return None
    module, name = source
    if module.split(".")[0] != CLI_DIR.parent.name:
        return None
    core_file = REPO_ROOT / "src" / f"{module.replace('.', '/')}.py"
    if not core_file.is_file():
        core_file = core_file.with_suffix("") / "__init__.py"
    if not core_file.is_file():
        pytest.fail(f"{command} imports {name} from {module}, which does not exist")
    bindings = _module_bindings(ast.parse(core_file.read_text()), name)
    if not bindings:
        pytest.fail(f"{command} imports {name} from {module}, which does not define it")
    core = bindings[0]
    # A re-export, an alias or a conditional definition cannot be followed with ast.
    if len(bindings) > 1 or not isinstance(core, ast.FunctionDef):
        return None
    return keys, core


def _module_bindings(tree: ast.Module, name: str) -> list[ast.stmt]:
    """Module-scope statements that bind ``name``, including inside ``if``/``try``/``with`` blocks."""
    found = []
    pending = list(tree.body)
    while pending:
        node = pending.pop(0)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                found.append(node)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any((alias.asname or alias.name.split(".")[0]) == name for alias in node.names):
                found.append(node)
            continue
        # A bare annotation (`run: Callable`) binds nothing at run time.
        targets = [node.target] if isinstance(node, ast.AnnAssign) and node.value else getattr(node, "targets", [])
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            found.append(node)
        for field in ("body", "orelse", "finalbody", "handlers"):
            pending.extend(getattr(node, field, []))
    return found


@pytest.mark.parametrize("command", COMMANDS)
def test_the_core_function_accepts_what_the_wrapper_passes(command: str) -> None:
    """The wrapper calls ``<name>_core(**params)``, so a renamed parameter fails at run time, not import."""
    binding = _core_binding(command)
    if binding is None:
        pytest.skip(f"{command} does not call its core function with **params")
    keys, core = binding
    args = core.args
    assert not args.posonlyargs, f"{core.name}: positional-only parameters cannot be passed by name"
    named = args.args + args.kwonlyargs
    defaults = [None] * (len(args.args) - len(args.defaults)) + list(args.defaults) + list(args.kw_defaults)
    required = {arg.arg for arg, default in zip(named, defaults) if default is None}
    unknown = set() if args.kwarg else keys - {arg.arg for arg in named}
    assert not unknown, f"{core.name}() does not accept {sorted(unknown)}, which the CLI passes"
    assert required <= keys, f"{core.name}() requires {sorted(required - keys)}, which the CLI does not pass"


def test_every_cab_command_is_registered_on_the_cli() -> None:
    """A command with a cab but no ``app.command(...)`` passes the round trip yet is invisible on the CLI."""
    import typer

    from pfb_model_spec.cli import app

    def registered(typer_app: typer.Typer) -> set[str]:
        modules = {command.callback.__module__ for command in typer_app.registered_commands if command.callback}
        for group in typer_app.registered_groups:
            if group.typer_instance is not None:
                modules |= registered(group.typer_instance)
        return modules

    modules = registered(app)
    package = CLI_DIR.parent.name
    missing = [
        path.stem for path in _cli_modules() if _has_stimela_cab(path) and f"{package}.cli.{path.stem}" not in modules
    ]
    assert not missing, f"cab commands not registered in cli/__init__.py: {missing}"
