import click

from storage_optimizer.models import OptimizationResult


class Visualizer:
    @staticmethod
    def render_summary(result: OptimizationResult) -> None:
        """Renders the terminal summary report matching your console formatting."""
        m = result.metadata
        p = result.profile_used
        v = result.verification

        click.echo("\n======================================================")
        click.echo("         STORAGE OPTIMIZER RUN SUMMARY               ")
        click.echo("======================================================")
        click.echo(f" File Name         : {m.filepath if m else result.source_path}")
        if m:
            click.echo(
                f" Raster Dimensions : {m.width} x {m.height} px ({m.count} bands)"
            )
            click.echo(
                f" Data Type         : {p.target_dtype or m.dtype if p else m.dtype}"
            )
        if p:
            click.echo(
                f" Selected Codec    : {p.codec.value.upper()} (Predictor: {p.predictor})"
            )
            if p.max_z_error > 0.0:
                click.echo(f" Max Error Bound   : {p.max_z_error}")
        click.echo("------------------------------------------------------")
        if v:
            click.echo(
                f" Original Size     : {v.source_size_bytes / (1024 * 1024):.2f} MB"
            )
            click.echo(
                f" Compressed Size   : {v.compressed_size_bytes / (1024 * 1024):.2f} MB"
            )
            click.echo(f" Space Savings     : {v.space_savings_percent:.2f}%")
            if v.max_allowed_error > 0.0:
                click.echo(
                    f" Verification      : {'PASSED' if v.is_valid else 'FAILED'} (Max Diff: {v.max_observed_error:.6f})"
                )
            else:
                click.echo(
                    f" Verification      : {'PASSED (100% Bit-Exact)' if v.is_valid else 'FAILED'}"
                )
        click.echo("======================================================\n")
