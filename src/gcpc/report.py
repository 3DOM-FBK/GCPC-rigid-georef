"""Markdown accuracy report generator."""

from datetime import date

import numpy as np
from scipy.spatial.transform import Rotation

from .averaging import AveragingResult, ConsensusRejection, PairLocalResidual
from .registration import FitMetrics, PairResult

FINAL_TRANSLATION_DECIMALS = 9
FINAL_ROTATION_DECIMALS = 9
FINAL_MATRIX_DECIMALS = 12
FINAL_QUATERNION_DECIMALS = 12
RESIDUAL_DECIMALS = 6


def _fmt_scalar(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_matrix_rows(matrix: np.ndarray, digits: int = 8) -> list[str]:
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
    final_fit_metrics: list[FitMetrics] | None = None,
    consensus_rejections: list[ConsensusRejection] | None = None,
    consensus_ambiguous: bool = False,
    rejected_pair_results: list[PairResult] | None = None,
    excluded_final_fit_metrics: list[FitMetrics] | None = None,
    inlier_max_distance: float = 0.20,
    max_anchor_residual: float = 0.20,
    max_rotation_residual: float = 1.0,
) -> str:
    """Return a concise, survey-oriented Markdown georeferencing report."""
    rotation = Rotation.from_matrix(avg.T_avg[:3, :3])
    euler = rotation.as_euler("xyz", degrees=True)
    quat = rotation.as_quat()
    translation = avg.T_avg[:3, 3]
    consensus_rejections = consensus_rejections or []
    rejected_pair_results = rejected_pair_results or []
    final_fit_metrics = final_fit_metrics or []
    excluded_final_fit_metrics = excluded_final_fit_metrics or []
    total_evaluated = len(pair_results) + len(consensus_rejections) + len(rejected_pair_results)

    matrix_lines = [
        "  " + "  ".join(f"{value:16.{FINAL_MATRIX_DECIMALS}f}" for value in row)
        for row in avg.T_avg
    ]
    cov_t_lines = _fmt_matrix_rows(avg.cov_t)
    cov_r_lines = _fmt_matrix_rows(avg.cov_r * (180.0 / np.pi) ** 2)

    if local_residuals is None:
        position_norms = np.linalg.norm(avg.t_residuals, axis=1)
        rotation_norms = avg.r_residuals
        detail_rows = [
            f"| {pair.name:>4} | {avg.t_residuals[i][0]:+.{RESIDUAL_DECIMALS}f} | "
            f"{avg.t_residuals[i][1]:+.{RESIDUAL_DECIMALS}f} | "
            f"{avg.t_residuals[i][2]:+.{RESIDUAL_DECIMALS}f} | n/a | n/a | n/a |"
            for i, pair in enumerate(pair_results)
        ]
    else:
        position_norms = np.array([np.linalg.norm(item.translation) for item in local_residuals])
        rotation_norms = np.array([np.linalg.norm(item.rotation_vec_deg) for item in local_residuals])
        detail_rows = [
            f"| {pair.name:>4} | {item.translation[0]:+.{RESIDUAL_DECIMALS}f} | "
            f"{item.translation[1]:+.{RESIDUAL_DECIMALS}f} | "
            f"{item.translation[2]:+.{RESIDUAL_DECIMALS}f} | "
            f"{item.rotation_vec_deg[0]:+.{RESIDUAL_DECIMALS}f} | "
            f"{item.rotation_vec_deg[1]:+.{RESIDUAL_DECIMALS}f} | "
            f"{item.rotation_vec_deg[2]:+.{RESIDUAL_DECIMALS}f} |"
            for pair, item in zip(pair_results, local_residuals)
        ]

    sections = [
        "# Point Cloud Georeferencing Report",
        "",
        "## Georeferencing Result",
        "",
        f"**Date:** {date.today().isoformat()}  ",
        f"**GCPCs evaluated:** {total_evaluated}  ",
        f"**GCPCs used:** {len(pair_results)}  ",
        f"**GCPCs excluded:** {n_rejected + len(consensus_rejections)}",
        "",
        "| Transformation parameter | Value |",
        "|--------------------------|------:|",
        f"| X translation (m) | {translation[0]:.{FINAL_TRANSLATION_DECIMALS}f} |",
        f"| Y translation (m) | {translation[1]:.{FINAL_TRANSLATION_DECIMALS}f} |",
        f"| Z translation (m) | {translation[2]:.{FINAL_TRANSLATION_DECIMALS}f} |",
        f"| X rotation (deg) | {euler[0]:.{FINAL_ROTATION_DECIMALS}f} |",
        f"| Y rotation (deg) | {euler[1]:.{FINAL_ROTATION_DECIMALS}f} |",
        f"| Z rotation (deg) | {euler[2]:.{FINAL_ROTATION_DECIMALS}f} |",
        f"| Quaternion (xyzw) | `{quat[0]:.{FINAL_QUATERNION_DECIMALS}f}  {quat[1]:.{FINAL_QUATERNION_DECIMALS}f}  {quat[2]:.{FINAL_QUATERNION_DECIMALS}f}  {quat[3]:.{FINAL_QUATERNION_DECIMALS}f}` |",
        "",
        "<details>",
        "<summary>Transformation matrix</summary>",
        "",
        "```",
        *matrix_lines,
        "```",
        "",
        "</details>",
        "",
        "---",
        "",
        "## Accuracy Indicators",
        "",
        "Internal metrics. Absolute accuracy requires independent data.",
        "",
        "### Estimated Georeferencing Precision",
        "",
        "1σ estimate from agreement between the accepted GCPC solutions.",
        "",
        "| Quantity | X | Y | Z | 3D / combined |",
        "|----------|--:|--:|--:|--------------:|",
        f"| Position (m) | {_fmt_scalar(avg.sigma_t[0])} | {_fmt_scalar(avg.sigma_t[1])} | "
        f"{_fmt_scalar(avg.sigma_t[2])} | {_fmt_scalar(avg.sigma_t_3d)} |",
        f"| Rotation (deg) | {_fmt_scalar(avg.sigma_r_vec[0])} | {_fmt_scalar(avg.sigma_r_vec[1])} | "
        f"{_fmt_scalar(avg.sigma_r_vec[2])} | {_fmt_scalar(avg.sigma_r)} |",
        "",
        "Position values apply at the TLS centering origin; rotation uncertainty increases "
        "position uncertainty with distance from that origin.",
        "",
        "### Final Alignment With Reference Clouds",
        "",
        f"Distance tolerance: **{inlier_max_distance:.3f} m**",
        "",
        "| GCPC | Status | Points within tolerance (%) | RMSE within tolerance (m) |",
        "|-----:|--------|----------------------------:|--------------------------:|",
        *[
            f"| {metric.name:>4} | Used | {100.0 * metric.fitness:.1f} | {_fmt_scalar(metric.inlier_rmse)} |"
            for metric in final_fit_metrics
        ],
        *[
            f"| {metric.name:>4} | Excluded—check only | {100.0 * metric.fitness:.1f} | {_fmt_scalar(metric.inlier_rmse)} |"
            for metric in excluded_final_fit_metrics
        ],
        "",
        "---",
        "",
        "## Internal Precision and Consistency",
        "",
        "### Between-GCPC Consistency",
        "",
        "Disagreement among GCPC-derived solutions.",
        "",
        "| Indicator | Value |",
        "|-----------|------:|",
        f"| Weighted anchor RMS | {_fmt_scalar(avg.anchor_weighted_rms)} m |",
        f"| Maximum anchor disagreement | {_fmt_scalar(avg.anchor_max_residual)} m |",
        f"| 3D position consistency (1σ) | {_fmt_scalar(avg.sigma_t_3d)} m |",
        f"| Rotation consistency (1σ) | {_fmt_scalar(avg.sigma_r)} deg |",
        "",
        "| GCPC | Position disagreement (m) | Rotation disagreement (deg) |",
        "|-----:|--------------------------:|----------------------------:|",
        *[
            f"| {pair.name:>4} | {position_norms[i]:.4f} | {rotation_norms[i]:.4f} |"
            for i, pair in enumerate(pair_results)
        ],
        "",
        "### Local-Matching Precision Proxy",
        "",
        f"**Heuristic 3D precision proxy:** {_fmt_scalar(avg.sigma_t_abs_3d)} m",
        "",
        "Assumes independent, unbiased position errors represented by local matching RMSE. Represents precision expected from individual ICP fit quality.",
    ]

    if consensus_rejections or rejected_pair_results:
        sections.extend(["", "---", "", "## Excluded GCPCs", ""])
        if consensus_rejections:
            sections.extend(
                [
                    "Excluded GCPCs have zero weight in the final transformation.",
                    "",
                    f"Consensus limits: **{max_anchor_residual:.3f} m** position and "
                    f"**{max_rotation_residual:.3f} deg** rotation.",
                    "",
                    "| GCPC | Reason | Position disagreement (m) | Rotation disagreement (deg) |",
                    "|-----:|--------|--------------------------:|----------------------------:|",
                    *[
                        f"| {item.result.name:>4} | {item.reason} | {item.anchor_residual:.4f} | "
                        f"{item.rotation_residual:.4f} |"
                        for item in consensus_rejections
                    ],
                ]
            )
        if rejected_pair_results:
            sections.extend(
                [
                    "",
                    "### Failed Local Registration Limits",
                    "",
                    "| GCPC | Points within tolerance (%) | Local RMSE (m) |",
                    "|-----:|----------------------------:|---------------:|",
                    *[
                        f"| {item.name:>4} | {100.0 * item.fitness:.1f} | {_fmt_scalar(item.inlier_rmse)} |"
                        for item in rejected_pair_results
                    ],
                ]
            )
        if consensus_ambiguous:
            sections.extend(
                [
                    "",
                    "**Consensus warning:** Multiple equal-size compatible GCPC groups were found; "
                    "review the exclusions manually.",
                ]
            )

    sections.extend(
        [
            "",
            "---",
            "",
            "## Processing Details",
            "",
            "<details>",
            "<summary>Registration, network geometry, and detailed statistics</summary>",
            "",
            f"**Fusion method:** {avg.fusion_method}  ",
            f"**Anchor geometry ratio:** {_fmt_scalar(avg.anchor_geometry_ratio, 6)}",
            "",
            "| GCPC | Local fitness | Local RMSE (m) |",
            "|-----:|--------------:|---------------:|",
            *[
                f"| {item.name:>4} | {item.fitness:.3f} | {_fmt_scalar(item.inlier_rmse)} |"
                for item in pair_results
            ],
            "",
            "| GCPC | dX (m) | dY (m) | dZ (m) | dRX (deg) | dRY (deg) | dRZ (deg) |",
            "|-----:|-------:|-------:|-------:|----------:|----------:|----------:|",
            *detail_rows,
            "",
            "**Position covariance from between-GCPC scatter (m²):**",
            "",
            "```",
            *cov_t_lines,
            "```",
            "",
            "**Rotation covariance from between-GCPC scatter (deg²):**",
            "",
            "```",
            *cov_r_lines,
            "```",
            "",
            "</details>",
        ]
    )

    if coverage_plot_name is not None:
        sections.extend(
            [
                "",
                "---",
                "",
                "## Reference-Cloud Coverage",
                "",
                f"![GCPC coverage heatmap]({coverage_plot_name})",
            ]
        )

    return "\n".join(sections)
