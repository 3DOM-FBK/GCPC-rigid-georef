"""Pairwise point-cloud -> GCPC registration using small_gicp."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import small_gicp
from scipy.spatial import cKDTree

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
    T_centered: np.ndarray    # (4, 4) result: maps pc_centered -> GCPC_centered
    anchor_pc: np.ndarray     # (3,) anchor point in pc_centered coordinates
    anchor_global: np.ndarray # (3,) corresponding anchor point in GCPC/global_centered coordinates
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


def _transform_xyz(xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a rigid transform to an (N, 3) XYZ array."""
    return xyz @ transform[:3, :3].T + transform[:3, 3]


def _crop_source_xyz_to_target_distance(
    source_xyz: np.ndarray,
    target_xyz_in_source_frame: np.ndarray,
    max_distance: float,
    num_threads: int,
) -> np.ndarray:
    """Return source points within max_distance of the target points."""
    bounds_min = target_xyz_in_source_frame.min(axis=0) - max_distance
    bounds_max = target_xyz_in_source_frame.max(axis=0) + max_distance
    bbox_mask = np.all((source_xyz >= bounds_min) & (source_xyz <= bounds_max), axis=1)
    bbox_source_xyz = source_xyz[bbox_mask]
    if len(bbox_source_xyz) == 0:
        return bbox_source_xyz

    target_tree = cKDTree(target_xyz_in_source_frame)
    distances, _ = target_tree.query(bbox_source_xyz, workers=num_threads)
    return bbox_source_xyz[distances <= max_distance]


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
    reference: str = "source",
) -> tuple[float, float]:
    """Compute inlier fraction and RMSE using either source or target coverage.

    When registration is performed as source -> target, a small target and a much
    larger source should be scored against the target coverage rather than the
    full source size.
    """
    if reference == "source":
        query_xyz = _as_xyz_array(source.point_cloud)
        query_xyz = query_xyz @ transform[:3, :3].T + transform[:3, 3]
        search_cloud = target
        reference_count = source.num_points
    elif reference == "target":
        inverse_transform = np.linalg.inv(transform)
        query_xyz = _as_xyz_array(target.point_cloud)
        query_xyz = query_xyz @ inverse_transform[:3, :3].T + inverse_transform[:3, 3]
        search_cloud = source
        reference_count = target.num_points
    else:
        raise ValueError(f"Unsupported metric reference: {reference}")

    indices, sq_dists = search_cloud.kdtree.batch_nearest_neighbor_search(
        query_xyz,
        num_threads=num_threads,
    )
    indices = np.asarray(indices, dtype=np.int64)
    sq_dists = np.asarray(sq_dists, dtype=np.float64)
    inlier_mask = (indices >= 0) & (sq_dists <= inlier_max_distance**2)
    inlier_count = int(np.count_nonzero(inlier_mask))

    if reference_count == 0 or inlier_count == 0:
        return 0.0, np.inf

    fitness = inlier_count / reference_count
    inlier_rmse = float(np.sqrt(np.mean(sq_dists[inlier_mask])))
    return fitness, max(inlier_rmse, 1e-9)


def register_pair(
    gcpc_path: Path,
    pc_reference: PreparedPointCloud,
    voxel_size: float = 0.10,
    max_distance: float = 2.0,
    inlier_max_distance: float = 0.20,
    center: np.ndarray | None = None,
    registration_method: str = "GICP",
    num_threads: int = 1,
) -> PairResult:
    """Register the preprocessed point cloud against one GCPC target.

    Both clouds must be in the same optionally centered frame. The returned
    transform maps pc_centered -> GCPC_centered.
    """
    method = registration_method.upper()
    if method not in SUPPORTED_REGISTRATION_METHODS:
        raise ValueError(f"Unsupported registration method: {registration_method}")

    gcpc_xyz = io.load_xyz(gcpc_path)
    if center is not None:
        gcpc_xyz = gcpc_xyz - center

    gcpc_target = preprocess(
        gcpc_xyz,
        voxel_size,
        center=None,
        num_threads=num_threads,
        build_voxel_map=(method == "VGICP"),
    )

    align_kwargs = {
        "init_T_target_source": np.eye(4),
        "max_correspondence_distance": max_distance,
        "num_threads": num_threads,
        "max_iterations": 100,
        "verbose": False,
    }

    if method == "VGICP":
        if pc_reference.voxel_map is None:
            raise ValueError("VGICP requires a precomputed TLS voxel map")
        coarse_result = small_gicp.align(
            pc_reference.voxel_map,
            gcpc_target.point_cloud,
            **align_kwargs,
        )
    else:
        coarse_result = small_gicp.align(
            pc_reference.point_cloud,
            gcpc_target.point_cloud,
            target_tree=pc_reference.kdtree,
            registration_type=method,
            **align_kwargs,
        )
    coarse_transform = np.asarray(coarse_result.T_target_source, dtype=np.float64)

    cropped_source_xyz = _crop_source_xyz_to_target_distance(
        _as_xyz_array(pc_reference.point_cloud),
        _transform_xyz(_as_xyz_array(gcpc_target.point_cloud), coarse_transform),
        max_distance,
        num_threads,
    )
    if len(cropped_source_xyz) == 0:
        return PairResult(
            name=gcpc_path.stem,
            T_centered=np.eye(4, dtype=np.float64),
            anchor_pc=np.zeros(3, dtype=np.float64),
            anchor_global=np.zeros(3, dtype=np.float64),
            fitness=0.0,
            inlier_rmse=np.inf,
        )

    local_pc_source = preprocess(
        cropped_source_xyz,
        voxel_size,
        center=None,
        num_threads=num_threads,
    )

    fine_align_kwargs = dict(align_kwargs)
    fine_align_kwargs["init_T_target_source"] = np.linalg.inv(coarse_transform)

    if method == "VGICP":
        if gcpc_target.voxel_map is None:
            raise ValueError("VGICP requires a precomputed target voxel map")
        result = small_gicp.align(
            gcpc_target.voxel_map,
            local_pc_source.point_cloud,
            **fine_align_kwargs,
        )
    else:
        result = small_gicp.align(
            gcpc_target.point_cloud,
            local_pc_source.point_cloud,
            target_tree=gcpc_target.kdtree,
            registration_type=method,
            **fine_align_kwargs,
        )

    fitness, inlier_rmse = _compute_registration_metrics(
        np.asarray(result.T_target_source, dtype=np.float64),
        local_pc_source,
        gcpc_target,
        inlier_max_distance,
        num_threads,
        reference="target",
    )

    T_centered = np.asarray(result.T_target_source, dtype=np.float64)
    gcpc_anchor = _as_xyz_array(gcpc_target.point_cloud).mean(axis=0).astype(np.float64)
    anchor_pc = (gcpc_anchor - T_centered[:3, 3]) @ T_centered[:3, :3]

    return PairResult(
        name=gcpc_path.stem,
        T_centered=T_centered,
        anchor_pc=anchor_pc.astype(np.float64),
        anchor_global=gcpc_anchor,
        fitness=float(fitness),
        inlier_rmse=float(inlier_rmse),
    )
