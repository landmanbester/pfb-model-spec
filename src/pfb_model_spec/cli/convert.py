from pathlib import Path
from typing import Annotated, Literal, NewType

import typer
from hip_cargo import StimelaMeta, parse_upath, stimela_cab, stimela_output

Directory = NewType("Directory", Path)


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
    if backend == "native" or backend == "auto":
        try:
            # Pre-flight must_exist for remote URIs before dispatching.
            from hip_cargo.utils.runner import preflight_remote_must_exist  # noqa: E402

            preflight_remote_must_exist(
                convert,
                dict(
                    input_mds=input_mds,
                    overwrite=overwrite,
                    output_mds=output_mds,
                ),
            )

            # Lazy import the core implementation
            from pfb_model_spec.core.convert import convert as convert_core  # noqa: E402

            # Call the core function with all parameters
            convert_core(
                input_mds,
                overwrite=overwrite,
                output_mds=output_mds,
            )
            return
        except ImportError:
            if backend == "native":
                raise

    # Resolve container image from installed package metadata
    from hip_cargo.utils.config import get_container_image  # noqa: E402
    from hip_cargo.utils.runner import run_in_container  # noqa: E402

    image = get_container_image("pfb-model-spec")
    if image is None:
        raise RuntimeError("No Container URL in pfb-model-spec metadata.")

    run_in_container(
        convert,
        dict(
            input_mds=input_mds,
            overwrite=overwrite,
            output_mds=output_mds,
        ),
        image=image,
        backend=backend,
        always_pull_images=always_pull_images,
    )
