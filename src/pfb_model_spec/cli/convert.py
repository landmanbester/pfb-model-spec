from typing import Annotated, Literal

import typer
from hip_cargo import Directory, StimelaMeta, parse_upath, stimela_cab, stimela_output


@stimela_cab(
    name="convert",
    info="Upgrade a component model (.mds) to the current spec.",
)
@stimela_output(
    dtype="Directory",
    name="output-mds",
    info="Path to write the upgraded component model (.mds) to.",
    required=True,
    must_exist=False,
    metadata={"rich_help_panel": "Output"},
)
def convert(
    input_mds: Annotated[
        Directory,
        typer.Option(
            ...,
            parser=parse_upath,
            help="Component model (.mds) to upgrade. Its spec is detected from the dataset attrs.",
            rich_help_panel="Input",
        ),
        StimelaMeta(
            must_exist=True,
        ),
    ],
    output_mds: Annotated[
        Directory,
        typer.Option(
            ...,
            parser=parse_upath,
            help="Path to write the upgraded component model (.mds) to.",
            rich_help_panel="Output",
        ),
        StimelaMeta(
            must_exist=False,
        ),
    ],
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Allow overwrite of an existing output.",
            rich_help_panel="Control",
        ),
    ] = False,
    backend: Annotated[
        Literal["auto", "native", "apptainer", "singularity", "docker", "podman"],
        typer.Option(
            help="Execution backend.",
        ),
        StimelaMeta(
            skip=True,
        ),
    ] = "auto",
    always_pull_images: Annotated[
        bool,
        typer.Option(
            help="Always pull container images, even if cached locally.",
        ),
        StimelaMeta(
            skip=True,
        ),
    ] = False,
):
    """
    Upgrade a component model (.mds) to the current spec.
    """
    params = dict(
        input_mds=input_mds,
        overwrite=overwrite,
        output_mds=output_mds,
    )

    if backend == "native" or backend == "auto":
        # Only a missing install falls back to the container, so only the
        # imports and the pre-flight are guarded; the core call is not.
        try:
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            from pfb_model_spec.core.convert import convert as convert_core  # noqa: E402

            # Pre-flight must_exist for remote URIs before dispatching.
            preflight_remote_must_exist(convert, params)
        except ImportError as exc:
            # A failed import from this package is a bug, not a missing install.
            if backend == "native" or (exc.name or "").split(".")[0] == "pfb_model_spec":
                raise
            typer.echo(f"Cannot run natively ({exc!r}); falling back to a container.", err=True)
        else:
            convert_core(**params)
            return

    # Resolve container image from installed package metadata
    from hip_cargo.utils.config import get_container_image  # noqa: E402
    from hip_cargo.utils.runner import run_in_container  # noqa: E402

    image = get_container_image("pfb-model-spec")
    if image is None:
        raise RuntimeError("No Container URL in pfb-model-spec metadata.")

    run_in_container(
        convert,
        params,
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
