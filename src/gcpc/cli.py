"""CLI entry point – orchestrates the full georeferencing pipeline."""

import os
from pathlib import Path

import click
import numpy as np
from scipy.spatial.transform import Rotation

from . import averaging, io, plotting, report, registration

DEFAULT_NUM_THREADS = max(os.cpu_count() or 1, 1)


def _fmt_stat(value: float) -> str:
    """Format CLI statistics and render unavailable values as n/a."""
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


def _format_transform_matrix(matrix: np.ndarray, decimals: int = 12) -> str:
    """Return a readable multi-line transform matrix string."""
    rows = [
        "      " + "  ".join(f"{value:16.{decimals}f}" for value in row)
        for row in np.asarray(matrix, dtype=np.float64)
    ]
    return "\n".join(rows)


def _print_pair_transform(result: registration.PairResult, center: np.ndarray, accepted: bool) -> None:
    """Print one per-GCPC transform in centred and global summaries."""
    T_global = averaging.global_transform_from_centered(result.T_centered, center)
    euler_centered = Rotation.from_matrix(result.T_centered[:3, :3]).as_euler("xyz", degrees=True)
    euler_global = Rotation.from_matrix(T_global[:3, :3]).as_euler("xyz", degrees=True)
    status = "accepted" if accepted else "rejected"

    click.echo(
        f"\n    [{status}] {result.name}"
        f"  fitness={result.fitness:.3f}  rmse={result.inlier_rmse:.4f} m"
    )
    click.echo(
        "      centred translation (m): "
        f"({result.T_centered[0, 3]:.9f}, {result.T_centered[1, 3]:.9f}, {result.T_centered[2, 3]:.9f})"
    )
    click.echo(
        "      centred Euler xyz (deg): "
        f"({euler_centered[0]:.9f}, {euler_centered[1]:.9f}, {euler_centered[2]:.9f})"
    )
    click.echo(
        "      global translation (m): "
        f"({T_global[0, 3]:.9f}, {T_global[1, 3]:.9f}, {T_global[2, 3]:.9f})"
    )
    click.echo(
        "      global Euler xyz (deg): "
        f"({euler_global[0]:.9f}, {euler_global[1]:.9f}, {euler_global[2]:.9f})"
    )
    click.echo("      centred transform:")
    click.echo(_format_transform_matrix(result.T_centered))


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--pc", "pc_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to the point cloud (.las / .laz).",
)
@click.option(
    "--gcpc-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing georeferenced GCPC .las files.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for the report and optional georeferenced point cloud.",
)
@click.option(
    "--voxel-size", default=0.10, show_default=True, type=float,
    help="Voxel downsampling resolution before GICP (m).",
)
@click.option(
    "--max-distance", default=2.0, show_default=True, type=float,
    help="Maximum GICP correspondence distance (m).",
)
@click.option(
    "--inlier-max-distance",
    default=0.20,
    show_default=True,
    type=float,
    help="Maximum distance used to count inliers and compute RMSE/weights (m).",
)
@click.option(
    "--registration-method",
    default="gicp",
    show_default=True,
    type=click.Choice(["icp", "plane_icp", "gicp", "vgicp"], case_sensitive=False),
    help="small_gicp registration backend to use.",
)
@click.option(
    "--num-threads",
    default=DEFAULT_NUM_THREADS,
    show_default=True,
    type=int,
    help="Number of threads for small_gicp preprocessing and registration.",
)
@click.option(
    "--min-fitness", default=0.50, show_default=True, type=float,
    help="Minimum GICP fitness score to accept a GCPC pair [0–1].",
)
@click.option(
    "--save-georef-las", is_flag=True, default=False,
    help="Write pc_georeferenced.las to the output directory (loads full point cloud into RAM).",
)
@click.option(
    "--save-georef-las-opt", is_flag=True, default=False,
    help=(
        "Write pc_georeferenced.las using memory-efficient chunked streaming "
        "(two passes over the file; preferred for large point clouds). "
        "Takes precedence over --save-georef-las when both are set."
    ),
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    default=False,
    help="Print all individual GCPC transforms after registration.",
)
def main(
    pc_path: Path,
    gcpc_dir: Path,
    output_dir: Path,
    voxel_size: float,
    max_distance: float,
    inlier_max_distance: float,
    registration_method: str,
    num_threads: int,
    min_fitness: float,
    save_georef_las: bool,
    save_georef_las_opt: bool,
    verbose: bool,
) -> None:
    """Georeference a point cloud using GNSS-enabled GCPCs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    gcpc_files = sorted(gcpc_dir.glob("*.las")) + sorted(gcpc_dir.glob("*.laz"))
    if not gcpc_files:
        raise click.BadParameter(f"No .las / .laz files found in {gcpc_dir}")

    method = registration_method.upper()
    if num_threads < 1:
        raise click.BadParameter("--num-threads must be >= 1")
    if inlier_max_distance <= 0:
        raise click.BadParameter("--inlier-max-distance must be > 0")

    # Preprocess the point cloud once and reuse its search structures across all GCPCs.
    click.echo(f"Loading point cloud: {pc_path}")
    center, pc_point_count = io.get_las_center_and_count(pc_path)
    pc_source = registration.preprocess_pc_las(
        pc_path,
        voxel_size,
        center,
        num_threads=num_threads,
        build_voxel_map=(method == "VGICP"),
    )
    click.echo(f"  registration backend: {method}")
    click.echo(f"  {pc_point_count:,} pts  ->  {pc_source.num_points:,} pts after downsampling")
    click.echo(f"  center used for centering: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    click.echo(f"  registration max distance: {max_distance:.2f} m | inlier cutoff: {inlier_max_distance:.2f} m")

    accepted: list[registration.PairResult] = []
    accepted_paths: list[Path] = []
    rejected: list[registration.PairResult] = []
    processed_pairs: list[tuple[registration.PairResult, bool]] = []

    with click.progressbar(gcpc_files, label="Registering GCPCs") as bar:
        for gcpc_file in bar:
            r = registration.register_pair(
                gcpc_file,
                pc_source,
                voxel_size,
                max_distance,
                inlier_max_distance,
                center,
                registration_method=method,
                num_threads=num_threads,
            )
            if r.fitness >= min_fitness:
                accepted.append(r)
                accepted_paths.append(gcpc_file)
                processed_pairs.append((r, True))
            else:
                rejected.append(r)
                processed_pairs.append((r, False))

    click.echo(
        f"\n  {len(accepted)} accepted  |  {len(rejected)} rejected "
        f"(fitness < {min_fitness})"
    )
    for r in rejected:
        click.echo(f"    [rejected] {r.name}  fitness={r.fitness:.3f}  rmse={r.inlier_rmse:.4f} m")

    if verbose:
        click.echo("\n  Individual GCPC transforms (point cloud -> GCPC):")
        for result, is_accepted in processed_pairs:
            _print_pair_transform(result, center, is_accepted)

    if not accepted:
        raise click.ClickException(
            "No GCPC pairs accepted."
        )
    if len(accepted) == 1:
        click.echo("  single accepted GCPC -> using its transform directly (no averaging)")

    avg = averaging.average_transforms(accepted, center)
    local_residuals = averaging.compute_pair_local_residuals(
        accepted,
        avg,
        center,
    )

    heatmap_path = output_dir / "gcpc_coverage_heatmap.png"
    plotting.save_gcpc_coverage_heatmap(
        pc_points_centered=pc_source.point_cloud.points(),
        pc_center=center,
        pc_bounds=io.get_las_bounds(pc_path),
        gcpc_paths=accepted_paths,
        output_path=heatmap_path,
    )

    report_path = output_dir / "report.md"
    report_path.write_text(
        report.build_report(
            accepted,
            avg,
            n_rejected=len(rejected),
            coverage_plot_name=heatmap_path.name,
            local_residuals=local_residuals,
        ),
        encoding="utf-8",
    )
    click.echo(f"\nReport  ->  {report_path}")
    click.echo(f"Coverage heatmap  ->  {heatmap_path}")

    if save_georef_las or save_georef_las_opt:
        out_las = output_dir / "pc_georeferenced.las"
        chunked = save_georef_las_opt  # opt flag takes precedence
        mode_label = "chunked streaming" if chunked else "in-memory"
        click.echo(f"Writing georeferenced point cloud ({mode_label})  ->  {out_las}")
        io.save_transformed_las(pc_path, avg.T_avg, out_las, chunked=chunked)

    click.echo(
        f"\n3-D translation sigma (absolute) : {_fmt_stat(avg.sigma_t_abs_3d)} m\n"
        f"Translation sigma (absolute)     : ({_fmt_stat(avg.sigma_t_abs[0])}, {_fmt_stat(avg.sigma_t_abs[1])}, {_fmt_stat(avg.sigma_t_abs[2])}) m\n"
        f"3-D translation residual RMSE    : {_fmt_stat(avg.t_rmse)} m\n"
        f"Rotation sigma (posterior)       : {_fmt_stat(avg.sigma_r)} deg"
    )
