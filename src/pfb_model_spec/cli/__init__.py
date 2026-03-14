"""CLI for pfb-model-spec."""

import typer

app = typer.Typer(no_args_is_help=True)

# Import command at bottom to avoid circular imports.
from pfb_model_spec.cli.onboard import onboard  # noqa: E402

app.command()(onboard)

__all__ = ["app"]
