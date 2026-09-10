"""Guard: importing the top-level package must NOT pull in the heavy spec
library, so the lightweight install (hip-cargo + typer) and cab generation
stay fast and dependency-free."""

import sys


def test_top_level_import_does_not_load_any_submodule():
    # Drop any prior import so the assertion reflects a fresh top-level import.
    for mod in list(sys.modules):
        if mod == "pfb_model_spec" or mod.startswith("pfb_model_spec."):
            del sys.modules[mod]

    import pfb_model_spec  # noqa: F401

    # Naming modules one at a time (e.g. just "utils.modelspec") is what let a
    # leak drift in unnoticed; assert none at all leaked instead.
    leaked = [m for m in sys.modules if m.startswith("pfb_model_spec.")]
    assert not leaked, f"top-level import pulled in submodules: {leaked}"
