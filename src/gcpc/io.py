"""LAS file I/O helpers."""

from pathlib import Path
from typing import Iterator

import laspy
import numpy as np


def get_las_center_and_count(path: Path) -> tuple[np.ndarray, int]:
    """Return the bounding-box center and point count from the LAS header."""
    with laspy.open(path) as reader:
        header = reader.header
        center = 0.5 * (np.asarray(header.mins) + np.asarray(header.maxs))
        return center.astype(np.float64), int(header.point_count)


def get_las_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return LAS bounding-box minima and maxima from the header."""
    with laspy.open(path) as reader:
        header = reader.header
        return (
            np.asarray(header.mins, dtype=np.float64),
            np.asarray(header.maxs, dtype=np.float64),
        )


def iter_xyz_chunks(path: Path, chunk_size: int) -> Iterator[np.ndarray]:
    """Yield LAS coordinates in fixed-size chunks as float64 XYZ arrays."""
    with laspy.open(path) as reader:
        for points in reader.chunk_iterator(chunk_size):
            yield np.column_stack((points.x, points.y, points.z)).astype(np.float64, copy=False)


def load_xyz(path: Path) -> np.ndarray:
    """Load a LAS/LAZ file and return float64 XYZ coordinates."""
    las = laspy.read(path)
    return np.column_stack((las.x, las.y, las.z)).astype(np.float64, copy=False)


def _transform_chunk_xyz(chunk_xyz: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply rotation R and translation t in-place-friendly fashion.

    Returns a new (N, 3) float64 array without retaining the input.
    Peak extra allocation is one chunk-sized array.
    """
    xyz_t = chunk_xyz @ R.T   # new array, same size as chunk
    xyz_t += t                 # in-place, no extra allocation
    return xyz_t


def save_transformed_las(
    src_path: Path,
    transform: np.ndarray,
    dst_path: Path,
    chunked: bool = False,
    chunk_size: int = 500_000,
) -> None:
    """Apply a 4×4 rigid transform to a LAS file and write the result.

    All non-coordinate point attributes (intensity, classification, …) are
    preserved unchanged.  The header offsets are updated so that the new global
    coordinates fit within the existing scale precision.

    Args:
        src_path:   Source LAS/LAZ file.
        transform:  4×4 rigid-body transformation matrix.
        dst_path:   Destination LAS file.
        chunked:    When *True*, stream the file in chunks of ``chunk_size``
                    points so that only O(chunk_size) coordinates are resident
                    at any time (two passes over the file are required).
                    When *False* (default), load the full point cloud into RAM
                    in a single pass — faster but memory-intensive.
        chunk_size: Number of points per chunk (only used when chunked=True).
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    R = transform[:3, :3]
    t = transform[:3, 3]

    if not chunked:
        # ------------------------------------------------------------------
        # Single-pass in-memory path (original behaviour).
        # ------------------------------------------------------------------
        src = laspy.read(src_path)
        xyz = np.column_stack((src.x, src.y, src.z)).astype(np.float64)
        xyz_t = xyz @ R.T + t
        src.header.offsets = xyz_t.min(axis=0)
        src.x = xyz_t[:, 0]
        src.y = xyz_t[:, 1]
        src.z = xyz_t[:, 2]
        src.write(dst_path)
        return

    # ------------------------------------------------------------------
    # Two-pass chunked path (memory-efficient).
    # Pass 1 – find the global minimum of the transformed coordinates.
    # laspy needs the header offset *before* it can open the writer.
    # ------------------------------------------------------------------
    global_min = np.full(3, np.inf, dtype=np.float64)
    with laspy.open(src_path) as reader:
        src_header = reader.header
        for chunk in reader.chunk_iterator(chunk_size):
            xyz = np.column_stack((chunk.x, chunk.y, chunk.z)).astype(np.float64, copy=False)
            xyz_t = _transform_chunk_xyz(xyz, R, t)
            del xyz
            np.minimum(global_min, xyz_t.min(axis=0), out=global_min)

    # Pass 2 – stream-transform and write.
    out_header = laspy.LasHeader(
        point_format=src_header.point_format,
        version=src_header.version,
    )
    out_header.offsets = global_min
    out_header.scales = src_header.scales

    with laspy.open(src_path) as reader, open(dst_path, "wb") as f, laspy.LasWriter(f, header=out_header) as writer:
        for chunk in reader.chunk_iterator(chunk_size):
            xyz = np.column_stack((chunk.x, chunk.y, chunk.z)).astype(np.float64, copy=False)
            xyz_t = _transform_chunk_xyz(xyz, R, t)
            del xyz
            chunk.x = xyz_t[:, 0]
            chunk.y = xyz_t[:, 1]
            chunk.z = xyz_t[:, 2]
            writer.write_points(chunk)
