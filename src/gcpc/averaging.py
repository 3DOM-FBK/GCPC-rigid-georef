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
    translation: np.ndarray       # (3,) relative translation vector in final world frame (m)
    rotation_vec_deg: np.ndarray  # (3,) relative rotation vector in final world frame (deg)


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


def centered_transform_from_global(T_global: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Recover the centred-frame point-cloud->global transform from the global-frame one."""
    T_centered = np.eye(4)
    T_centered[:3, :3] = T_global[:3, :3]
    T_centered[:3, 3] = T_global[:3, 3] - center + T_global[:3, :3] @ center
    return T_centered


def compute_pair_local_residuals(
    results: list[PairResult],
    avg: AveragingResult,
    center: np.ndarray,
) -> list[PairLocalResidual]:
    """Compute per-GCPC transform residuals in the final averaged world frame.

    For each accepted GCPC, this returns the residual rigid transform from the
    single-registration world frame to the final averaged world frame:

        T_residual_i = T_final_world <- T_single_world

    Residuals are reported as one translation vector and one rotation vector.
    """
    T_avg_c = centered_transform_from_global(avg.T_avg, center)
    diagnostics: list[PairLocalResidual] = []

    for result in results:
        T_single_c = np.linalg.inv(result.T_centered)  # pc_centered -> global_centered from one GCPC
        T_single_to_final = T_avg_c @ np.linalg.inv(T_single_c)
        rotation_vec_deg = np.degrees(
            Rotation.from_matrix(T_single_to_final[:3, :3]).as_rotvec()
        )

        diagnostics.append(
            PairLocalResidual(
                name=result.name,
                translation=T_single_to_final[:3, 3].astype(np.float64),
                rotation_vec_deg=rotation_vec_deg.astype(np.float64),
            )
        )

    return diagnostics


def average_transforms(results: list[PairResult], center: np.ndarray) -> AveragingResult:
    """Weighted average of rigid transforms in the centred frame.

    Weight per pair:  w_i = 1 / inlier_rmse_i²
    GICP RMSE is treated as the relative observation standard deviation, so the
    transform mean is solved with inverse-variance weighting.

    Each GICP result T_centered_i maps GCPC_centred -> pc_centred.
    Inverting gives T_pc_c_i: pc_centred -> GCPC_centred (≡ global_centred).
    These centred transforms are averaged; the global-frame point-cloud->global
    is constructed once from the mean.

    The final georeferencing uncertainty is derived from least-squares error
    propagation in the centred frame:

    - translation covariance from the repeated-observation WLS mean
    - rotation covariance from the local tangent-space WLS mean on SO(3)
    """
    weights = np.array([1.0 / r.inlier_rmse**2 for r in results], dtype=np.float64)
    weights_norm = weights / weights.sum()

    # Work with inv(T_centered): pc_centred -> global_centred
    T_pc_c = [np.linalg.inv(r.T_centered) for r in results]
    rotations = [Rotation.from_matrix(T[:3, :3]) for T in T_pc_c]
    translations = np.array([T[:3, 3] for T in T_pc_c])  # small, centred-frame (m)

    R_avg, r_residual_vecs = _solve_rotation_mean(rotations, weights)
    t_avg_c = np.average(translations, axis=0, weights=weights)

    # Compose global-frame point-cloud -> global transform:
    #   p_global = R_avg @ (p_pc – centre) + t_avg_c + centre
    T_avg = np.eye(4)
    T_avg[:3, :3] = R_avg.as_matrix()
    T_avg[:3, 3] = center - R_avg.as_matrix() @ center + t_avg_c

    # --- residuals in centred frame ---
    t_residuals = translations - t_avg_c
    r_residuals = np.degrees(np.linalg.norm(r_residual_vecs, axis=1))

    # --- uncertainty from absolute RMSE propagation ---
    cov_t_abs = _translation_covariance_from_absolute_sigmas(
        np.array([r.inlier_rmse for r in results], dtype=np.float64)
    )
    sigma_t_abs = np.sqrt(np.diag(cov_t_abs))
    sigma_t_abs_3d = float(np.sqrt(np.trace(cov_t_abs)))

    # --- posterior consistency uncertainty from transform scatter ---
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
