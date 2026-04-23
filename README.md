# GCPC Rigid Georeferencing

Georeference a survey-grade point cloud to a global coordinate system using a set of
small GNSS-enabled **Ground Control Point Clouds (GCPCs)**.

## Concept

Each GCPC is a small area captured with a GNSS-enabled scanner (global coordinates).
The point cloud covers the full site but is in a local frame that is only roughly pre-aligned to
global.  The tool:

1. Registers each GCPC -> point cloud with **small_gicp** and inverts the result to
   obtain *N* independent point-cloud -> global transform estimates.
2. Combines them into a single rigid transform via **weighted quaternion averaging**
   with **least-squares weights proportional to `1 / GICP_RMSE^2`**.
3. Writes an **accuracy report** with the final transform, per-GCPC residuals, and
   georeferencing uncertainty propagated from the weighted least-squares covariance.
4. Generates a **coverage heatmap** showing the overlap count of accepted GCPC extents
   over a simplified point-cloud XY hull.

## Installation

```bash
# requires uv  (https://docs.astral.sh/uv/)
uv sync
```

## Usage

```bash
uv run gcpc-rigid-georef \
   --pc         data/TLS.las \
  --gcpc-dir   data/GCPCs \
   --output-dir output/ \
   --registration-method gicp
```

Add `--save-georef-las` to also write the transformed point cloud:

```bash
uv run gcpc-rigid-georef \
   --pc         data/TLS.las \
  --gcpc-dir   data/GCPCs \
  --output-dir output/ \
  --save-georef-las
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--pc` | *required* | Path to the point cloud `.las` file |
| `--gcpc-dir` | *required* | Directory containing georeferenced GCPC `.las` files |
| `--output-dir` | *required* | Output directory |
| `--voxel-size` | `0.10` | Voxel downsampling before registration (m) |
| `--max-distance` | `2.0` | Max registration correspondence distance (m) |
| `--inlier-max-distance` | `0.20` | Max distance used to count inliers and compute RMSE/weights (m) |
| `--registration-method` | `gicp` | `small_gicp` backend: `icp`, `plane_icp`, `gicp`, `vgicp` |
| `--num-threads` | CPU count | Threads for `small_gicp` preprocessing and registration |
| `--min-fitness` | `0.50` | Min GICP fitness to accept a GCPC pair (0–1) |
| `--save-georef-las` | off | Write `pc_georeferenced.las` to the output directory |

## Outputs

| File | Description |
|------|-------------|
| `output/report.md` | Final transform, per-GCPC residuals, uncertainty (1 sigma) |
| `output/gcpc_coverage_heatmap.png` | Seaborn heatmap of accepted GCPC XY extent overlap over the point-cloud outline |
| `output/pc_georeferenced.las` | *(optional)* Transformed point cloud in global coordinates |

## Algorithm summary

| Step | Method |
|------|--------|
| Pairwise registration | `small_gicp` (`icp`, `plane_icp`, `gicp`, `vgicp`) — source = GCPC, target = point cloud |
| Transform inversion | T_pc->global = T_GCPC->pc^-1 |
| Rotation averaging | Gram-matrix eigenvector (Markley et al., 2007) |
| Translation averaging | Weighted arithmetic mean |
| Weighting | Inverse-variance style, `1 / GICP_RMSE^2` |
| Uncertainty | Weighted least-squares covariance propagation in the centred frame |
| Coverage visualization | Seaborn heatmap of accepted GCPC XY extents over a simplified point-cloud convex hull |
