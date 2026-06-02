"""Guard: importing the top-level package must NOT pull in the heavy spec
library, so the lightweight install (hip-cargo + typer) and cab generation
stay fast and dependency-free."""

import sys


def test_top_level_import_does_not_load_modelspec():
    # Drop any prior import so the assertion reflects a fresh top-level import.
    for mod in list(sys.modules):
        if mod == "pfb_model_spec" or mod.startswith("pfb_model_spec."):
            del sys.modules[mod]

    import pfb_model_spec  # noqa: F401

    assert "pfb_model_spec.modelspec" not in sys.modules
