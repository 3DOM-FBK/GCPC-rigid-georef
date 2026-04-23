"""Pairwise GCPC -> point-cloud registration using small_gicp."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import small_gicp

from . import io

SUPPORTED_REGISTRATION_METHODS = ("ICP", "PLANE_ICP", "GICP", "VGICP")
PC_CHUNK_SIZE = 2_000_000


@dataclass
class PreparedPointCloud:
    """Preprocessed point cloud plus reusable search structures."""

    point_cloud: small_gicp.PointCloud
    kdtree: small_gicp.KdTree
    num_points: int
    voxel_map: small_gicp.GaussianVoxelMap | None = None


@dataclass
class PairResult:
    """Outcome of a single GCPC-point-cloud registration."""

    name: str                 # GCPC stem, e.g. "3"
    T_centered: np.ndarray    # (4, 4) result: maps GCPC_centered -> pc_centered
    fitness: float            # inlier fraction  [0, 1]
    inlier_rmse: float        # inlier RMSE  (m)


def _as_xyz_array(point_cloud: small_gicp.PointCloud) -> np.ndarray:
    """Return PointCloud coordinates as an (N, 3) float64 array."""
    return np.asarray(point_cloud.points(), dtype=np.float64)[:, :3]


def _build_voxel_map(
    point_cloud: small_gicp.PointCloud,
    voxel_size: float,
) -> small_gicp.GaussianVoxelMap:
    """Build the target voxel map used by small_gicp VGICP."""
    voxel_map = small_gicp.GaussianVoxelMap(voxel_size)
    voxel_map.insert(point_cloud)
    return voxel_map


def preprocess(
    points: np.ndarray,
    voxel_size: float,
    center: np.ndarray | None = None,
    num_threads: int = 1,
    build_voxel_map: bool = False,
) -> PreparedPointCloud:
    """Downsample, optionally center, and precompute small_gicp search structures."""
    pts = np.asarray(points, dtype=np.float64)
    if center is not None:
        pts = pts - center

    point_cloud, kdtree = small_gicp.preprocess_points(
        pts,
        downsampling_resolution=voxel_size,
        num_neighbors=20,
        num_threads=num_threads,
    )

    voxel_map = _build_voxel_map(point_cloud, voxel_size) if build_voxel_map else None

    return PreparedPointCloud(
        point_cloud=point_cloud,
        kdtree=kdtree,
        num_points=len(point_cloud),
        voxel_map=voxel_map,
    )


def preprocess_pc_las(
    pc_path: Path,
    voxel_size: float,
    center: np.ndarray,
    num_threads: int = 1,
    build_voxel_map: bool = False,
    chunk_size: int = PC_CHUNK_SIZE,
) -> PreparedPointCloud:
    """Preprocess a large point-cloud LAS by chunked voxel sampling before full preprocessing.

    The point cloud is therefore read in chunks, each chunk is centred and voxel-sampled,
    and the concatenated reduced cloud is then passed to preprocess_points.
    """
    sampled_chunks: list[np.ndarray] = []

    for chunk_xyz in io.iter_xyz_chunks(pc_path, chunk_size):
        centered_chunk = chunk_xyz - center
        sampled_chunk = small_gicp.voxelgrid_sampling(
            centered_chunk,
            downsampling_resolution=voxel_size,
            num_threads=num_threads,
        )
        sampled_chunks.append(_as_xyz_array(sampled_chunk))

    if not sampled_chunks:
        raise ValueError(f"No points were read from {pc_path}")

    sampled_points = np.vstack(sampled_chunks)
    return preprocess(
        sampled_points,
        voxel_size,
        center=None,
        num_threads=num_threads,
        build_voxel_map=build_voxel_map,
    )


def _compute_registration_metrics(
    transform: np.ndarray,
    source: PreparedPointCloud,
    target: PreparedPointCloud,
    inlier_max_distance: float,
    num_threads: int,
) -> tuple[float, float]:
    """Compute inlier fraction and RMSE from the target KD-tree."""
    source_xyz = _as_xyz_array(source.point_cloud)
    transformed = source_xyz @ transform[:3, :3].T + transform[:3, 3]

    indices, sq_dists = target.kdtree.batch_nearest_neighbor_search(
        transformed,
        num_threads=num_threads,
    )
    indices = np.asarray(indices, dtype=np.int64)
    sq_dists = np.asarray(sq_dists, dtype=np.float64)
    inlier_mask = (indices >= 0) & (sq_dists <= inlier_max_distance**2)
    inlier_count = int(np.count_nonzero(inlier_mask))

    if source.num_points == 0 or inlier_count == 0:
        return 0.0, np.inf

    fitness = inlier_count / source.num_points
    inlier_rmse = float(np.sqrt(np.mean(sq_dists[inlier_mask])))
    return fitness, max(inlier_rmse, 1e-9)


def register_pair(
    gcpc_path: Path,
    pc_target: PreparedPointCloud,
    voxel_size: float = 0.10,
    max_distance: float = 2.0,
    inlier_max_distance: float = 0.20,
    center: np.ndarray | None = None,
    registration_method: str = "GICP",
    num_threads: int = 1,
) -> PairResult:
    """Register one GCPC against the preprocessed point-cloud target.

    Both clouds must be in the same optionally centered frame. The returned
    transform maps GCPC_centered -> pc_centered.
    """
    method = registration_method.upper()
    if method not in SUPPORTED_REGISTRATION_METHODS:
        raise ValueError(f"Unsupported registration method: {registration_method}")

    gcpc_xyz = io.load_xyz(gcpc_path)
    gcpc_source = preprocess(gcpc_xyz, voxel_size, center, num_threads)

    align_kwargs = {
        "init_T_target_source": np.eye(4),
        "max_correspondence_distance": max_distance,
        "num_threads": num_threads,
        "max_iterations": 100,
        "verbose": False,
    }

    if method == "VGICP":
        if pc_target.voxel_map is None:
            raise ValueError("VGICP requires a precomputed target voxel map")
        result = small_gicp.align(
            pc_target.voxel_map,
            gcpc_source.point_cloud,
            **align_kwargs,
        )
    else:
        result = small_gicp.align(
            pc_target.point_cloud,
            gcpc_source.point_cloud,
            target_tree=pc_target.kdtree,
            registration_type=method,
            **align_kwargs,
        )

    fitness, inlier_rmse = _compute_registration_metrics(
        np.asarray(result.T_target_source, dtype=np.float64),
        gcpc_source,
        pc_target,
        inlier_max_distance,
        num_threads,
    )

    return PairResult(
        name=gcpc_path.stem,
        T_centered=np.asarray(result.T_target_source, dtype=np.float64),
        fitness=float(fitness),
        inlier_rmse=float(inlier_rmse),
    )
