import click

from storage_optimizer.optimizer import RasterOptimizer
from storage_optimizer.visualizer import Visualizer


@click.group()
def main():
    """Storage Optimizer: Enterprise Geospatial Raster Compression Engine."""


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option(
    "--auto/--no-auto",
    default=True,
    help="Automatically determine or benchmark best codec profile.",
)
@click.option(
    "--max-error",
    type=float,
    default=0.0,
    help="Max error bound for LERC_ZSTD (e.g. 0.0001 for 70%+ savings).",
)
@click.option(
    "--aux/--no-aux", default=True, help="Generate .aux.xml statistics sidecar."
)
@click.option(
    "--stac/--no-stac", default=True, help="Generate .stac.json metadata sidecar."
)
def optimize(
    input_path: str,
    output_path: str,
    auto: bool,
    max_error: float,
    aux: bool,
    stac: bool,
):
    """Compresses a GeoTIFF raster using automated decision heuristics."""
    click.echo(f"Processing: {input_path} -> {output_path}")

    optimizer = RasterOptimizer(generate_aux=aux, generate_stac=stac)
    result = optimizer.optimize(
        input_path=input_path,
        output_path=output_path,
        auto_tune=auto,
        max_z_error=max_error,
    )

    if not result.success:
        click.secho(
            f"\nOptimization Failed: {result.error_message or 'Parity verification failed.'}",
            fg="red",
        )
        return

    Visualizer.render_summary(result)


if __name__ == "__main__":
    main()
