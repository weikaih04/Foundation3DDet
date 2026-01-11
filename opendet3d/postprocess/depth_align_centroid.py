"""Depth-based 3D box alignment using centroid matching (V3).

Improvements over SAM3D V2:
1. Asymmetric trimming: remove front 10% + back 30% (instead of symmetric 30%)
2. Use projected 3D box for tighter 2D region
3. Align 3D box center to depth point cloud centroid
4. Support both uniform scaling and translation-based alignment
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any

import numpy as np
import torch
from vis4d.data.const import AxisMode
from vis4d.op.box.box3d import boxes3d_to_corners


@dataclass(frozen=True)
class CentroidAlignParams:
    """Parameters for centroid-based alignment."""
    trim_front_ratio: float = 0.10  # Remove closest 10% points
    trim_back_ratio: float = 0.30   # Remove farthest 30% points
    min_points: int = 10
    scale_min: float = 0.5
    scale_max: float = 1.5
    max_depth: float = 65.0
    # Alignment mode: 'scale' (uniform scaling) or 'translate' (translate only)
    mode: str = 'scale'


def _box3d_to_corners_opencv(box3d: np.ndarray) -> np.ndarray:
    """Convert single box3d to 8 corners using vis4d (OPENCV axis mode)."""
    box_t = torch.from_numpy(box3d).float().unsqueeze(0)
    corners = boxes3d_to_corners(box_t, AxisMode.OPENCV)
    return corners[0].numpy()


def correct_3d_box_centroid_v3(
    pred_box3d: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    box3d_to_corners_func: Callable = None,
    params: CentroidAlignParams = CentroidAlignParams(),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Align 3D box using centroid matching with asymmetric trimming.

    Algorithm:
    1. Project 3D box to 2D to get tight bbox
    2. Extract depth points in projected 2D region
    3. Trim front 10% and back 30% of points by depth
    4. Compute 3D centroid of remaining points
    5. Align 3D box to match centroid:
       - 'scale' mode: uniform scaling (preserves 2D projection)
       - 'translate' mode: translate center (changes 2D projection)

    Args:
        pred_box3d: [10] predicted 3D box [x, y, z, w, l, h, qw, qx, qy, qz]
        depth_map: [H, W] depth map
        intrinsics: [3, 3] camera intrinsic matrix
        box3d_to_corners_func: Function to convert box3d to 8 corners [8, 3]
        params: Alignment parameters

    Returns:
        corrected_box3d: [10] aligned 3D box
        info: Dictionary with alignment information
    """
    if box3d_to_corners_func is None:
        box3d_to_corners_func = _box3d_to_corners_opencv

    h, w = depth_map.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # 1. Get 3D box corners and project to 2D
    corners_3d = box3d_to_corners_func(pred_box3d)  # [8, 3]

    # Filter corners with positive depth
    valid_corners = corners_3d[corners_3d[:, 2] > 0]
    if len(valid_corners) < 4:
        return pred_box3d, {'success': False, 'reason': 'corners_behind_camera'}

    # Project to 2D
    u_2d = fx * valid_corners[:, 0] / valid_corners[:, 2] + cx
    v_2d = fy * valid_corners[:, 1] / valid_corners[:, 2] + cy

    # Get tight 2D bbox from projected corners
    x1 = int(max(0, np.floor(u_2d.min())))
    x2 = int(min(w, np.ceil(u_2d.max())))
    y1 = int(max(0, np.floor(v_2d.min())))
    y2 = int(min(h, np.ceil(v_2d.max())))

    if x2 <= x1 or y2 <= y1:
        return pred_box3d, {'success': False, 'reason': 'invalid_projected_box2d'}

    # 2. Extract depth region and back-project to 3D
    v_coords, u_coords = np.meshgrid(
        np.arange(y1, y2), np.arange(x1, x2), indexing='ij'
    )
    depth_region = depth_map[y1:y2, x1:x2]

    depth_values = depth_region.flatten()
    u_flat = u_coords.flatten()
    v_flat = v_coords.flatten()

    # Filter valid depths
    valid_mask = (depth_values > 0) & (depth_values <= params.max_depth)
    if valid_mask.sum() < params.min_points:
        return pred_box3d, {'success': False, 'reason': 'insufficient_depth_points'}

    depth_valid = depth_values[valid_mask]
    u_valid = u_flat[valid_mask]
    v_valid = v_flat[valid_mask]

    # 3. Asymmetric trimming: front 10% + back 30%
    sorted_indices = np.argsort(depth_valid)
    n_points = len(depth_valid)
    trim_front = int(n_points * params.trim_front_ratio)
    trim_back = int(n_points * params.trim_back_ratio)

    if n_points - trim_front - trim_back < params.min_points:
        return pred_box3d, {'success': False, 'reason': 'insufficient_points_after_trim'}

    keep_indices = sorted_indices[trim_front:n_points - trim_back]
    depth_trimmed = depth_valid[keep_indices]
    u_trimmed = u_valid[keep_indices]
    v_trimmed = v_valid[keep_indices]

    # 4. Back-project trimmed points to 3D
    x_3d = (u_trimmed - cx) * depth_trimmed / fx
    y_3d = (v_trimmed - cy) * depth_trimmed / fy
    z_3d = depth_trimmed

    # 5. Compute centroid of trimmed points
    centroid_target = np.array([x_3d.mean(), y_3d.mean(), z_3d.mean()])

    # 6. Align based on mode
    pred_center = pred_box3d[:3]
    pred_depth = pred_box3d[2]

    if pred_depth <= 0:
        return pred_box3d, {'success': False, 'reason': 'invalid_pred_depth'}

    if params.mode == 'translate':
        # Translate mode: directly move center to target centroid
        corrected_box3d = pred_box3d.copy()
        corrected_box3d[:3] = centroid_target
        scale = 1.0
    else:
        # Scale mode: uniform scaling (preserves 2D projection approximately)
        scale = centroid_target[2] / pred_depth
        if scale < params.scale_min or scale > params.scale_max:
            return pred_box3d, {
                'success': False, 'reason': 'scale_out_of_range',
                'scale': float(scale), 'scale_min': params.scale_min, 'scale_max': params.scale_max,
            }
        corrected_box3d = pred_box3d.copy()
        corrected_box3d[:6] *= scale  # Scale position + dimensions

    info = {
        'success': True, 'method': f'centroid_v3_{params.mode}', 'scale': float(scale),
        'original_center': pred_center.tolist(), 'target_centroid': centroid_target.tolist(),
        'projected_box2d': [x1, y1, x2, y2],
        'num_points_before_trim': int(n_points), 'num_points_after_trim': int(len(keep_indices)),
        'trim_front_ratio': params.trim_front_ratio, 'trim_back_ratio': params.trim_back_ratio,
    }
    return corrected_box3d, info


def align_boxes3d_centroid_v3(
    boxes3d_raw: np.ndarray,
    boxes2d_xyxy: np.ndarray,  # Not used, kept for API compatibility
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    *,
    params: CentroidAlignParams = CentroidAlignParams(),
) -> np.ndarray:
    """Apply centroid-based V3 alignment per instance.

    Args:
        boxes3d_raw: [N, 10] raw 3D boxes
        boxes2d_xyxy: [N, 4] 2D boxes (not used, for API compatibility)
        depth_map: [H, W] depth map
        intrinsics: [3, 3] camera intrinsics
        params: Alignment parameters

    Returns:
        boxes3d_aligned: [N, 10] aligned 3D boxes
    """
    out = boxes3d_raw.copy()
    if out.size == 0:
        return out

    for i in range(out.shape[0]):
        aligned, info = correct_3d_box_centroid_v3(
            pred_box3d=out[i],
            depth_map=depth_map,
            intrinsics=intrinsics,
            params=params,
        )
        out[i] = aligned.astype(out.dtype, copy=False)

    return out

