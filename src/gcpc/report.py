"""Markdown accuracy report generator."""

from datetime import date

import numpy as np
from scipy.spatial.transform import Rotation

from .averaging import AveragingResult, PairLocalResidual
from .registration import PairResult


def build_report(
    pair_results: list[PairResult],
    avg: AveragingResult,
    n_rejected: int = 0,
    coverage_plot_name: str | None = None,
    local_residuals: list[PairLocalResidual] | None = None,
) -> str:
    """Return a Markdown string summarising the georeferencing result."""
    R = Rotation.from_matrix(avg.T_avg[:3, :3])
    euler = R.as_euler("xyz", degrees=True)
    quat = R.as_quat()          # xyzw, scalar-last
    t = avg.T_avg[:3, 3]

    matrix_lines = [
        "  " + "  ".join(f"{v:12.6f}" for v in row)
        for row in avg.T_avg
    ]

    cov_t_lines = [
        "  " + "  ".join(f"{v:12.8f}" for v in row)
        for row in avg.cov_t
    ]
    cov_t_abs_lines = [
        "  " + "  ".join(f"{v:12.8f}" for v in row)
        for row in avg.cov_t_abs
    ]
    deg2_per_rad2 = (180.0 / np.pi) ** 2
    cov_r_lines = [
        "  " + "  ".join(f"{v * deg2_per_rad2:12.8f}" for v in row)
        for row in avg.cov_r
    ]

    if local_residuals is None:
        residual_rows = [
            f"| {r.name:>4} | {r.fitness:.3f} | {r.inlier_rmse:.4f} "
            f"| {avg.t_residuals[i][0]:+.4f} | {avg.t_residuals[i][1]:+.4f} "
            f"| {avg.t_residuals[i][2]:+.4f} | {avg.r_residuals[i]:.4f} |"
            for i, r in enumerate(pair_results)
        ]
        residual_heading = "## Per-GCPC Residuals"
        residual_header = "| GCPC | Fitness | RMSE (m) | dTX (m) | dTY (m) | dTZ (m) | dR (deg) |"
        residual_divider = "|-----:|--------:|---------:|--------:|--------:|--------:|-------:|"
    else:
        residual_rows = [
            f"| {pair.name:>4} | {pair.fitness:.3f} | {pair.inlier_rmse:.4f} "
            f"| {local.translation[0]:+.4f} | {local.translation[1]:+.4f} | {local.translation[2]:+.4f} "
            f"| {local.rotation_vec_deg[0]:+.4f} | {local.rotation_vec_deg[1]:+.4f} | {local.rotation_vec_deg[2]:+.4f} |"
            for pair, local in zip(pair_results, local_residuals)
        ]
        residual_heading = "## Per-GCPC Relative Transform Residuals"
        residual_header = "| GCPC | Fitness | RMSE (m) | dTX (m) | dTY (m) | dTZ (m) | dRX (deg) | dRY (deg) | dRZ (deg) |"
        residual_divider = "|-----:|--------:|---------:|--------:|--------:|--------:|----------:|----------:|----------:|"

    sections = [
        "# Point Cloud Georeferencing Accuracy Report",
        "",
        f"**Date:** {date.today().isoformat()}  ",
        f"**GCPCs accepted:** {len(pair_results)}  ",
        f"**GCPCs rejected:** {n_rejected}",
        "",
        "---",
        "",
        "## Final Transformation  (Point cloud -> Global)",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
        f"| T_X (m) | {t[0]:.4f} |",
        f"| T_Y (m) | {t[1]:.4f} |",
        f"| T_Z (m) | {t[2]:.4f} |",
        f"| 𝜙 (deg) | {euler[0]:.6f} |",
        f"| 𝜃 (deg) | {euler[1]:.6f} |",
        f"| 𝜓 (deg) | {euler[2]:.6f} |",
        f"| Quaternion_xyzw | `{quat[0]:.6f}  {quat[1]:.6f}  {quat[2]:.6f}  {quat[3]:.6f}` |",
        "",
        "**4 x 4 Transformation Matrix:**",
        "",
        "```",
        *matrix_lines,
        "```",
        "",
        "---",
        "",
        residual_heading,
        "",
        residual_header,
        residual_divider,
        *residual_rows,
        "",
        "---",
        "",
        "## Georeferencing Uncertainty",
        "",
        "### Error Propagation from GCPC Matching",
        "",
        "| Component | Value |",
        "|-----------|-------|",
        f"| σ_X (m) | {avg.sigma_t_abs[0]:.4f} |",
        f"| σ_Y (m) | {avg.sigma_t_abs[1]:.4f} |",
        f"| σ_Z (m) | {avg.sigma_t_abs[2]:.4f} |",
        f"| σ_3D(m) | {avg.sigma_t_abs_3d:.4f} |",
        "",
        "### Posterior Consistency From GCPC Single Transformation Result Scatter",
        "",
        "| Component | Value |",
        "|-----------|-------|",
        f"| σ_X (m) | {avg.sigma_t[0]:.4f} |",
        f"| σ_Y (m) | {avg.sigma_t[1]:.4f} |",
        f"| σ_Z (m) | {avg.sigma_t[2]:.4f} |",
        f"| σ_3D(m) | {avg.sigma_t_3d:.4f} |",
        f"| σ_𝜙 (deg) | {avg.sigma_r_vec[0]:.4f} |",
        f"| σ_𝜃 (deg) | {avg.sigma_r_vec[1]:.4f} |",
        f"| σ_𝜓 (deg) | {avg.sigma_r_vec[2]:.4f} |",
        f"| σ_rot (deg) | {avg.sigma_r:.4f} |",
        "",
        "**Translation covariance matrix from posterior transform scatter (m^2):**",
        "",
        "```",
        *cov_t_lines,
        "```",
        "",
        "**Rotation covariance matrix (local tangent frame, deg^2):**",
        "",
        "```",
        *cov_r_lines,
        "```",
    ]

    if coverage_plot_name is not None:
        sections.extend(
            [
                "",
                "---",
                "",
                "## GCPC Coverage Heatmap",
                "",
                f"![GCPC coverage heatmap]({coverage_plot_name})",
            ]
        )

    return "\n".join(sections)
