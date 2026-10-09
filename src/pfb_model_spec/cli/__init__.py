"""CLI for pfb-model-spec."""

import typer

app = typer.Typer(
    name="pfbspec",
    help="Model specification for pfb-imaging",
    no_args_is_help=True,
)


@app.callback()
def callback() -> None:
    """Model specification for pfb-imaging"""
    pass


# Register subcommands below. Imports go here (bottom) to avoid circular imports.
from pfb_model_spec.cli.model2comps import model2comps  # noqa: E402

app.command(name="model2comps")(model2comps)

from pfb_model_spec.cli.convert import convert  # noqa: E402

app.command(name="convert")(convert)

__all__ = ["app"]
