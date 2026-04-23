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


def save_transformed_las(src_path: Path, transform: np.ndarray, dst_path: Path) -> None:
    """Apply a 4×4 rigid transform to a LAS file and write the result.

    All non-coordinate point attributes (intensity, classification, …) are
    preserved unchanged.  The header offsets are updated so that the new global
    coordinates fit within the existing scale precision.
    """
    src = laspy.read(src_path)
    xyz = np.column_stack((src.x, src.y, src.z)).astype(np.float64)
    xyz_t = xyz @ transform[:3, :3].T + transform[:3, 3]

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    src.header.offsets = xyz_t.min(axis=0)
    src.x = xyz_t[:, 0]
    src.y = xyz_t[:, 1]
    src.z = xyz_t[:, 2]
    src.write(dst_path)
