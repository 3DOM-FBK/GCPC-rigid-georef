"""Plotting helpers for report figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull

from . import io

MAX_HULL_POINTS = 250_000
GRID_LONG_EDGE = 240


def _sample_pc_xy_hull_points(pc_points_centered: np.ndarray, pc_center: np.ndarray) -> np.ndarray:
    """Return a simplified set of point-cloud XY points for convex hull computation."""
    pc_xy_centered = np.asarray(pc_points_centered, dtype=np.float64)[:, :2]
    if len(pc_xy_centered) > MAX_HULL_POINTS:
        step = max(len(pc_xy_centered) // MAX_HULL_POINTS, 1)
        pc_xy_centered = pc_xy_centered[::step]
    return pc_xy_centered + pc_center[:2]


def _fallback_outline(min_xy: np.ndarray, max_xy: np.ndarray) -> np.ndarray:
    """Return a rectangular fallback outline if convex hull computation fails."""
    xmin, ymin = min_xy
    xmax, ymax = max_xy
    return np.array(
        [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
            [xmin, ymin],
        ],
        dtype=np.float64,
    )


def _simplify_xy_points(xy: np.ndarray, max_points: int = MAX_HULL_POINTS) -> np.ndarray:
    """Subsample XY points for stable and cheap convex-hull computation."""
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) > max_points:
        step = max(len(xy) // max_points, 1)
        xy = xy[::step]
    return xy


def _convex_hull_outline(xy: np.ndarray) -> np.ndarray:
    """Return a closed convex-hull polygon, with a rectangle fallback."""
    xy = _simplify_xy_points(xy)
    if len(xy) == 0:
        raise ValueError("Cannot compute a hull from an empty point set")

    try:
        hull = ConvexHull(xy)
        hull_xy = xy[hull.vertices]
        return np.vstack([hull_xy, hull_xy[0]])
    except Exception:
        return _fallback_outline(xy.min(axis=0), xy.max(axis=0))


def _polygon_row_col_bounds(
    polygon_xy: np.ndarray,
    min_xy: np.ndarray,
    max_xy: np.ndarray,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
) -> tuple[int, int, int, int]:
    """Return the clipped heatmap row/column bounds overlapped by a polygon."""
    poly_min = polygon_xy.min(axis=0)
    poly_max = polygon_xy.max(axis=0)

    x0 = int(np.clip(np.floor((poly_min[0] - min_xy[0]) / dx), 0, nx - 1))
    x1 = int(np.clip(np.floor((poly_max[0] - min_xy[0]) / dx), 0, nx - 1))
    y0 = int(np.clip(np.floor((poly_min[1] - min_xy[1]) / dy), 0, ny - 1))
    y1 = int(np.clip(np.floor((poly_max[1] - min_xy[1]) / dy), 0, ny - 1))

    row_start = int(np.clip((ny - 1) - y1, 0, ny - 1))
    row_end = int(np.clip((ny - 1) - y0, 0, ny - 1))
    return row_start, row_end, x0, x1


def _accumulate_polygon_coverage(
    coverage: np.ndarray,
    polygon_xy: np.ndarray,
    min_xy: np.ndarray,
    max_xy: np.ndarray,
    dx: float,
    dy: float,
) -> None:
    """Add one polygon's occupancy to the heatmap grid."""
    ny, nx = coverage.shape
    row_start, row_end, col_start, col_end = _polygon_row_col_bounds(
        polygon_xy,
        min_xy,
        max_xy,
        dx,
        dy,
        nx,
        ny,
    )

    rows = np.arange(row_start, row_end + 1)
    cols = np.arange(col_start, col_end + 1)
    x_centers = min_xy[0] + (cols + 0.5) * dx
    y_centers = max_xy[1] - (rows + 0.5) * dy

    xx, yy = np.meshgrid(x_centers, y_centers)
    query_xy = np.column_stack((xx.ravel(), yy.ravel()))
    polygon_path = MplPath(polygon_xy, closed=True)
    inside = polygon_path.contains_points(query_xy, radius=1e-9).reshape(len(rows), len(cols))
    coverage[row_start : row_end + 1, col_start : col_end + 1] += inside.astype(np.float64)


def _world_to_heatmap(
    xy: np.ndarray,
    min_xy: np.ndarray,
    dx: float,
    dy: float,
    ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map world XY coordinates into seaborn heatmap cell coordinates."""
    x_plot = (xy[:, 0] - min_xy[0]) / dx
    y_plot = (ny - 1) - (xy[:, 1] - min_xy[1]) / dy
    return x_plot, y_plot


def save_gcpc_coverage_heatmap(
    pc_points_centered: np.ndarray,
    pc_center: np.ndarray,
    pc_bounds: tuple[np.ndarray, np.ndarray],
    gcpc_paths: list[Path],
    output_path: Path,
) -> None:
    """Save a heatmap of accepted GCPC convex hull overlap over the point-cloud XY hull."""
    pc_outline_points = _sample_pc_xy_hull_points(pc_points_centered, pc_center)
    min_xyz, max_xyz = pc_bounds
    min_xy = min_xyz[:2]
    max_xy = max_xyz[:2]

    hull_xy = _convex_hull_outline(pc_outline_points)

    width = max(max_xy[0] - min_xy[0], 1e-6)
    height = max(max_xy[1] - min_xy[1], 1e-6)
    if width >= height:
        nx = GRID_LONG_EDGE
        ny = max(int(np.ceil(height / width * GRID_LONG_EDGE)), 1)
    else:
        ny = GRID_LONG_EDGE
        nx = max(int(np.ceil(width / height * GRID_LONG_EDGE)), 1)

    dx = width / nx
    dy = height / ny
    coverage = np.zeros((ny, nx), dtype=np.float64)

    for gcpc_path in gcpc_paths:
        gcpc_xy = io.load_xyz(gcpc_path)[:, :2]
        gcpc_hull = _convex_hull_outline(gcpc_xy)
        _accumulate_polygon_coverage(coverage, gcpc_hull, min_xy, max_xy, dx, dy)

    mask = coverage <= 0
    aspect = width / height if height > 0 else 1.0
    fig_width = 10.0
    fig_height = max(fig_width / max(aspect, 1e-6), 4.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        coverage,
        mask=mask,
        cmap="YlOrRd",
        alpha=0.78,
        cbar_kws={"label": "Accepted GCPC overlap count"},
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )

    x_plot, y_plot = _world_to_heatmap(hull_xy, min_xy, dx, dy, ny)
    ax.plot(x_plot, y_plot, color="black", linewidth=1.6)

    xticks = np.linspace(0, nx - 1, 6)
    yticks = np.linspace(0, ny - 1, 6)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{min_xy[0] + tick * dx:.0f}" for tick in xticks], rotation=0)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{max_xy[1] - tick * dy:.0f}" for tick in yticks], rotation=0)

    ax.set_title("Accepted GCPC convex-hull coverage over point-cloud XY extent")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_facecolor("white")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)