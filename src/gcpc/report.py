"""Markdown accuracy report generator."""

from datetime import date

import numpy as np
from scipy.spatial.transform import Rotation

from .averaging import AveragingResult, PairLocalResidual
from .registration import PairResult


FINAL_TRANSLATION_DECIMALS = 9
FINAL_ROTATION_DECIMALS = 9
FINAL_MATRIX_DECIMALS = 12
FINAL_QUATERNION_DECIMALS = 12
RESIDUAL_DECIMALS = 6


def _fmt_scalar(value: float, digits: int = 4) -> str:
    """Format finite floats consistently and render missing values as n/a."""
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_matrix_rows(matrix: np.ndarray, digits: int = 8) -> list[str]:
    """Format a matrix for fenced-code rendering in the Markdown report."""
    rows: list[str] = []
    for row in matrix:
        formatted = [
            f"{value:12.{digits}f}" if np.isfinite(value) else f"{'n/a':>12}"
            for value in row
        ]
        rows.append("  " + "  ".join(formatted))
    return rows


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
        "  " + "  ".join(f"{v:16.{FINAL_MATRIX_DECIMALS}f}" for v in row)
        for row in avg.T_avg
    ]

    cov_t_lines = _fmt_matrix_rows(avg.cov_t)
    deg2_per_rad2 = (180.0 / np.pi) ** 2
    cov_r_lines = _fmt_matrix_rows(avg.cov_r * deg2_per_rad2)
    posterior_note = None
    if len(pair_results) == 1:
        posterior_note = (
            "Posterior consistency from transform scatter is unavailable with a single "
            "accepted GCPC."
        )

    if local_residuals is None:
        residual_rows = [
            f"| {r.name:>4} | {r.fitness:.3f} | {r.inlier_rmse:.4f} "
            f"| {avg.t_residuals[i][0]:+.{RESIDUAL_DECIMALS}f} | {avg.t_residuals[i][1]:+.{RESIDUAL_DECIMALS}f} "
            f"| {avg.t_residuals[i][2]:+.{RESIDUAL_DECIMALS}f} | {avg.r_residuals[i]:.{RESIDUAL_DECIMALS}f} |"
            for i, r in enumerate(pair_results)
        ]
        residual_heading = "## Per-GCPC Residuals"
        residual_header = "| GCPC | Fitness | RMSE (m) | dTX (m) | dTY (m) | dTZ (m) | dR (deg) |"
        residual_divider = "|-----:|--------:|---------:|--------:|--------:|--------:|-------:|"
    else:
        residual_rows = [
            f"| {pair.name:>4} | {pair.fitness:.3f} | {pair.inlier_rmse:.4f} "
            f"| {local.translation[0]:+.{RESIDUAL_DECIMALS}f} | {local.translation[1]:+.{RESIDUAL_DECIMALS}f} | {local.translation[2]:+.{RESIDUAL_DECIMALS}f} "
            f"| {local.rotation_vec_deg[0]:+.{RESIDUAL_DECIMALS}f} | {local.rotation_vec_deg[1]:+.{RESIDUAL_DECIMALS}f} | {local.rotation_vec_deg[2]:+.{RESIDUAL_DECIMALS}f} |"
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
        f"| T_X (m) | {t[0]:.{FINAL_TRANSLATION_DECIMALS}f} |",
        f"| T_Y (m) | {t[1]:.{FINAL_TRANSLATION_DECIMALS}f} |",
        f"| T_Z (m) | {t[2]:.{FINAL_TRANSLATION_DECIMALS}f} |",
        f"| 𝜙 (deg) | {euler[0]:.{FINAL_ROTATION_DECIMALS}f} |",
        f"| 𝜃 (deg) | {euler[1]:.{FINAL_ROTATION_DECIMALS}f} |",
        f"| 𝜓 (deg) | {euler[2]:.{FINAL_ROTATION_DECIMALS}f} |",
        f"| Quaternion_xyzw | `{quat[0]:.{FINAL_QUATERNION_DECIMALS}f}  {quat[1]:.{FINAL_QUATERNION_DECIMALS}f}  {quat[2]:.{FINAL_QUATERNION_DECIMALS}f}  {quat[3]:.{FINAL_QUATERNION_DECIMALS}f}` |",
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
        f"| σ_X (m) | {_fmt_scalar(avg.sigma_t_abs[0])} |",
        f"| σ_Y (m) | {_fmt_scalar(avg.sigma_t_abs[1])} |",
        f"| σ_Z (m) | {_fmt_scalar(avg.sigma_t_abs[2])} |",
        f"| σ_3D(m) | {_fmt_scalar(avg.sigma_t_abs_3d)} |",
        "",
        "### Posterior Consistency From GCPC Single Transformation Result Scatter",
        "",
        *([posterior_note, ""] if posterior_note is not None else []),
        "| Component | Value |",
        "|-----------|-------|",
        f"| σ_X (m) | {_fmt_scalar(avg.sigma_t[0])} |",
        f"| σ_Y (m) | {_fmt_scalar(avg.sigma_t[1])} |",
        f"| σ_Z (m) | {_fmt_scalar(avg.sigma_t[2])} |",
        f"| σ_3D(m) | {_fmt_scalar(avg.sigma_t_3d)} |",
        f"| σ_𝜙 (deg) | {_fmt_scalar(avg.sigma_r_vec[0])} |",
        f"| σ_𝜃 (deg) | {_fmt_scalar(avg.sigma_r_vec[1])} |",
        f"| σ_𝜓 (deg) | {_fmt_scalar(avg.sigma_r_vec[2])} |",
        f"| σ_rot (deg) | {_fmt_scalar(avg.sigma_r)} |",
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
