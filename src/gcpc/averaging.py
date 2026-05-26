"""Weighted averaging of N rigid transforms from pairwise GICP registration.

Key design note
---------------
All averaging and residual analysis is performed in the **centred coordinate
frame** (where the point-cloud center == origin).  Working in global UTM coordinates
directly would amplify sub-degree rotation differences into km-scale
translation noise via the (I – R) @ centre term (|centre| ≈ 5 M m for
typical UTM zones).  The final point-cloud -> global transform is composed from the
mean centred transform at the very end.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .registration import PairResult


@dataclass
class AveragingResult:
    T_avg: np.ndarray         # (4, 4) weighted mean point cloud -> global  (global frame)
    t_residuals: np.ndarray   # (N, 3) per-pair centred translation residuals  (m)
    r_residuals: np.ndarray   # (N,)   per-pair rotation residual magnitudes    (°)
    weights: np.ndarray       # (N,)   normalised RMSE-based weights            (sum = 1)
    sigma_t_abs: np.ndarray   # (3,)   translation 1σ from absolute RMSE propagation (m)
    sigma_t_abs_3d: float     # 3-D translation 1σ from absolute RMSE propagation (m)
    sigma_t: np.ndarray       # (3,)   translation 1σ from LS covariance        (m)
    sigma_t_3d: float         # 3-D translation 1σ from posterior consistency covariance (m)
    sigma_r_vec: np.ndarray   # (3,)   rotation-vector 1σ from LS covariance    (°)
    sigma_r: float            # combined rotation 1σ                            (°)
    t_rmse: float             # overall 3-D translation residual RMSE           (m)
    cov_t_abs: np.ndarray     # (3, 3) translation covariance from absolute RMSE propagation (m²)
    cov_t: np.ndarray         # (3, 3) centred translation covariance           (m²)
    cov_r: np.ndarray         # (3, 3) local rotation-vector covariance         (rad²)


@dataclass
class PairLocalResidual:
    name: str
    translation: np.ndarray       # (3,) anchor-point residual in final world frame (m)
    rotation_vec_deg: np.ndarray  # (3,) relative rotation vector final <- single (deg)


def _weighted_quaternion_seed(rotations: list[Rotation], weights: np.ndarray) -> Rotation:
    """Return a robust initial rotation estimate from weighted quaternion averaging."""
    quats = np.array([rot.as_quat() for rot in rotations])
    for i in range(1, len(quats)):
        if quats[0] @ quats[i] < 0:
            quats[i] *= -1

    gram = np.einsum("i,ij,ik->jk", weights, quats, quats)
    q_avg = np.linalg.eigh(gram)[1][:, -1]
    return Rotation.from_quat(q_avg)


def _solve_rotation_mean(
    rotations: list[Rotation],
    weights: np.ndarray,
    max_iterations: int = 20,
    tolerance: float = 1e-12,
) -> tuple[Rotation, np.ndarray]:
    """Solve the weighted least-squares mean rotation in the tangent space."""
    R_avg = _weighted_quaternion_seed(rotations, weights)

    for _ in range(max_iterations):
        residuals = np.array([(R_avg.inv() * rot).as_rotvec() for rot in rotations])
        delta = np.average(residuals, axis=0, weights=weights)
        if np.linalg.norm(delta) < tolerance:
            break
        R_avg = R_avg * Rotation.from_rotvec(delta)

    residuals = np.array([(R_avg.inv() * rot).as_rotvec() for rot in rotations])
    return R_avg, residuals


def _covariance_from_weighted_ls(residuals: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the posterior covariance of the weighted mean from transform scatter.

    Each GCPC pair contributes one 3-vector observation with relative weight
    ``w_i = 1 / rmse_i²``.  For the repeated-observation model

        l_i = x + v_i

    the covariance of the least-squares estimate is

        Cov(x̂) = Σ̂0 / Σw_i

    where Σ̂0 is the posterior covariance of unit weight estimated from the
    weighted residuals.  This yields a 3×3 covariance matrix for either the
    centred translation or the local rotation-vector parameters.
    """
    dof = residuals.shape[0] - 1
    if dof <= 0:
        raise ValueError("At least two accepted GCPCs are required for least-squares covariance")

    unit_weight_cov = residuals.T @ (weights[:, None] * residuals) / dof
    return unit_weight_cov / weights.sum()


def _translation_covariance_from_absolute_sigmas(sigmas: np.ndarray) -> np.ndarray:
    """Return translation covariance using pairwise RMSE values as absolute sigmas.

    Each GCPC pair contributes an isotropic translation observation covariance
    ``Sigma_i = sigma_i^2 I`` in the centred frame, with ``sigma_i`` taken
    directly from the pairwise inlier RMSE. The covariance of the weighted mean is

        Cov_abs(x̂) = (sum_i Sigma_i^-1)^-1 = I / sum_i(1 / sigma_i^2)
    """
    weights = 1.0 / np.asarray(sigmas, dtype=np.float64) ** 2
    return np.eye(3, dtype=np.float64) / weights.sum()


def _solve_translation_mean(
    anchors_pc: np.ndarray,
    anchors_global: np.ndarray,
    rotation_matrix: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the weighted translation LS after fixing the mean rotation.

    Each pair contributes one translation observation implied at its own anchor:

        t_i = anchor_global_i - R_avg @ anchor_pc_i

    The final translation is the weighted LS mean of these observations.
    """
    translation_observations = anchors_global - anchors_pc @ rotation_matrix.T
    translation_mean = np.average(translation_observations, axis=0, weights=weights)
    residuals = translation_observations - translation_mean
    return translation_mean, residuals


def centered_transform_from_global(T_global: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Recover the centred-frame point-cloud->global transform from the global-frame one."""
    T_centered = np.eye(4)
    T_centered[:3, :3] = T_global[:3, :3]
    T_centered[:3, 3] = T_global[:3, 3] - center + T_global[:3, :3] @ center
    return T_centered


def global_transform_from_centered(T_centered: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Compose the global-frame point-cloud->global transform from the centred-frame one."""
    T_global = np.eye(4)
    T_global[:3, :3] = T_centered[:3, :3]
    T_global[:3, 3] = center - T_centered[:3, :3] @ center + T_centered[:3, 3]
    return T_global


def single_transform(result: PairResult, center: np.ndarray) -> AveragingResult:
    """Return a direct point-cloud->global solution from a single accepted GCPC."""
    rmse = float(result.inlier_rmse)
    sigma_t_abs = np.full(3, rmse, dtype=np.float64)
    nan_vec = np.full(3, np.nan, dtype=np.float64)
    nan_mat = np.full((3, 3), np.nan, dtype=np.float64)

    return AveragingResult(
        T_avg=global_transform_from_centered(result.T_centered, center),
        t_residuals=np.zeros((1, 3), dtype=np.float64),
        r_residuals=np.zeros(1, dtype=np.float64),
        weights=np.ones(1, dtype=np.float64),
        sigma_t_abs=sigma_t_abs,
        sigma_t_abs_3d=float(np.sqrt(3.0) * rmse),
        sigma_t=nan_vec.copy(),
        sigma_t_3d=float("nan"),
        sigma_r_vec=nan_vec.copy(),
        sigma_r=float("nan"),
        t_rmse=0.0,
        cov_t_abs=np.eye(3, dtype=np.float64) * rmse**2,
        cov_t=nan_mat.copy(),
        cov_r=nan_mat.copy(),
    )


def compute_pair_local_residuals(
    results: list[PairResult],
    avg: AveragingResult,
    center: np.ndarray,
) -> list[PairLocalResidual]:
    """Compute per-GCPC local residuals at each GCPC anchor.

    For each accepted GCPC, translation is evaluated at the GCPC anchor implied
    by that single-registration result, not at the centred origin. This avoids
    reporting lever-arm artefacts from small rotation differences.

    Rotation residuals remain the relative final <- single rotation vector.
    """
    T_avg_c = centered_transform_from_global(avg.T_avg, center)
    R_avg = Rotation.from_matrix(T_avg_c[:3, :3])
    diagnostics: list[PairLocalResidual] = []

    for result in results:
        anchor_residual = result.anchor_pc @ T_avg_c[:3, :3].T + T_avg_c[:3, 3] - result.anchor_global
        rotation_vec_deg = np.degrees(
            (R_avg * Rotation.from_matrix(result.T_centered[:3, :3]).inv()).as_rotvec()
        )

        diagnostics.append(
            PairLocalResidual(
                name=result.name,
                translation=anchor_residual.astype(np.float64),
                rotation_vec_deg=rotation_vec_deg.astype(np.float64),
            )
        )

    return diagnostics


def average_transforms(results: list[PairResult], center: np.ndarray) -> AveragingResult:
    """Weighted LS solution with rotation averaging and anchor-coupled translation.

    When a single accepted GCPC is available, return its transform directly.

    Rotation is solved first by weighted least squares on SO(3) using the
    individual pair rotations. Translation is then solved in a coupled way at
    the GCPC anchors using that final rotation:

        t_i = anchor_global_i - R_avg @ anchor_pc_i

    so small rotation differences are converted into translation observations at
    the actual GCPC locations rather than at the arbitrary centred origin.
    """
    if len(results) == 1:
        return single_transform(results[0], center)

    weights = np.array([1.0 / r.inlier_rmse**2 for r in results], dtype=np.float64)
    weights_norm = weights / weights.sum()

    anchors_pc = np.array([r.anchor_pc for r in results], dtype=np.float64)
    anchors_global = np.array([r.anchor_global for r in results], dtype=np.float64)
    rotations = [Rotation.from_matrix(r.T_centered[:3, :3]) for r in results]

    R_avg, r_residual_vecs = _solve_rotation_mean(rotations, weights)
    R_avg_matrix = R_avg.as_matrix()
    t_avg_c, t_residuals = _solve_translation_mean(
        anchors_pc,
        anchors_global,
        R_avg_matrix,
        weights,
    )

    T_avg_c = np.eye(4)
    T_avg_c[:3, :3] = R_avg_matrix
    T_avg_c[:3, 3] = t_avg_c
    T_avg = global_transform_from_centered(T_avg_c, center)

    r_residuals = np.degrees(np.linalg.norm(r_residual_vecs, axis=1))

    cov_t_abs = _translation_covariance_from_absolute_sigmas(
        np.array([r.inlier_rmse for r in results], dtype=np.float64)
    )
    sigma_t_abs = np.sqrt(np.diag(cov_t_abs))
    sigma_t_abs_3d = float(np.sqrt(np.trace(cov_t_abs)))

    cov_t = _covariance_from_weighted_ls(t_residuals, weights)
    cov_r = _covariance_from_weighted_ls(r_residual_vecs, weights)

    sigma_t = np.sqrt(np.diag(cov_t))
    sigma_t_3d = float(np.sqrt(np.trace(cov_t)))
    sigma_r_vec = np.degrees(np.sqrt(np.diag(cov_r)))
    sigma_r = float(np.degrees(np.sqrt(np.trace(cov_r))))

    t_rmse = float(np.sqrt(np.average(np.sum(t_residuals**2, axis=1), weights=weights)))

    return AveragingResult(
        T_avg=T_avg,
        t_residuals=t_residuals,
        r_residuals=r_residuals,
        weights=weights_norm,
        sigma_t_abs=sigma_t_abs,
        sigma_t_abs_3d=sigma_t_abs_3d,
        sigma_t=sigma_t,
        sigma_t_3d=sigma_t_3d,
        sigma_r_vec=sigma_r_vec,
        sigma_r=sigma_r,
        t_rmse=t_rmse,
        cov_t_abs=cov_t_abs,
        cov_t=cov_t,
        cov_r=cov_r,
    )
