"""Depth-based 3D box post-processing (SAM-3D style one-shot scaling).

This wraps the existing implementation in:
  single_frame_3d_gen/ground_select_and_3d_bbox_predict/step25_align/align_simple.py

We load by file path to avoid importing step25_align as a package (its __init__
triggers non-relative imports when used outside that directory).
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable

import numpy as np
import torch
from vis4d.data.const import AxisMode
from vis4d.op.box.box3d import boxes3d_to_corners


def _load_align_module() -> ModuleType:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    mod_path = os.path.join(
        repo_root,
        "single_frame_3d_gen",
        "ground_select_and_3d_bbox_predict",
        "step25_align",
        "align_simple.py",
    )
    if not os.path.isfile(mod_path):
        raise FileNotFoundError(f"SAM3D align file not found: {mod_path}")

    spec = importlib.util.spec_from_file_location("_align_simple_sam3d", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for: {mod_path}")
    module = importlib.util.module_from_spec(spec)
    assert isinstance(module, ModuleType)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


_align_module = _load_align_module()
_correct_3d_box_sam3d_style = getattr(_align_module, "correct_3d_box_sam3d_style")
_correct_3d_box_sam3d_v2 = getattr(_align_module, "correct_3d_box_sam3d_v2")
_correct_3d_box_sam3d_v3 = getattr(_align_module, "correct_3d_box_sam3d_v3")


def _box3d_to_corners_opencv(box3d: np.ndarray) -> np.ndarray:
    """Convert one box3d [10] -> corners [8,3] using AxisMode.OPENCV."""
    t = torch.from_numpy(box3d[None, :].astype(np.float32, copy=False))
    corners = boxes3d_to_corners(t, AxisMode.OPENCV)[0]
    return corners.detach().cpu().numpy()


@dataclass(frozen=True)
class Sam3DParams:
    """Parameters for SAM3D-style depth alignment."""
    # V1 (original) params
    filter_percentile: float = 90.0
    min_points: int = 10
    # V2/V3 params
    use_v2: bool = False  # Use V2 (uniform depth scaling)
    use_v3: bool = True   # Use V3 (height scaling + translation) - default
    trim_ratio: float = 0.3  # Trim front/back 30% of points
    scale_min: float = 0.5
    scale_max: float = 1.5
    # Selective application params
    min_depth_for_alignment: float = 0.0  # Only apply SAM3D to boxes with depth >= this value (0 = apply to all)


def align_boxes3d_sam3d_from_depth(
    boxes3d_raw: np.ndarray,
    boxes2d_xyxy: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    *,
    params: Sam3DParams = Sam3DParams(),
) -> np.ndarray:
    """Apply SAM3D-style alignment per instance. Returns aligned boxes3d array.

    V3 (default): Projected 3D->2D box, trim front/back, height-based scale + translation.
    V2: Projected 3D->2D box, uniform depth scaling, scale clipped to [scale_min, scale_max].
    V1: Uses predicted 2D box, height-based scaling + translation.

    Notes:
      - V3 uses height ratio for scale (more stable than depth ratio).
      - V2 preserves 2D projection approximately (uniform scaling along ray).
      - V1 does NOT preserve 2D projection (scales + translates).
      - If min_depth_for_alignment > 0, only boxes with depth >= min_depth_for_alignment are aligned.
    """
    out = boxes3d_raw.copy()
    if out.size == 0:
        return out

    for i in range(out.shape[0]):
        # Check if this box should be aligned based on depth threshold
        box_depth = out[i, 2]  # Z coordinate (depth in camera frame)
        if box_depth < params.min_depth_for_alignment:
            # Skip alignment for this box (keep original)
            continue
        if params.use_v3:
            # V3: Hybrid - projected box + height scaling + translation
            aligned, info = _correct_3d_box_sam3d_v3(
                pred_box3d=out[i],
                depth_map=depth_map,
                intrinsics=intrinsics,
                box3d_to_corners_func=_box3d_to_corners_opencv,
                trim_ratio=params.trim_ratio,
                min_points=params.min_points,
            )
        elif params.use_v2:
            # V2: Use projected 3D box, uniform depth scaling
            aligned, info = _correct_3d_box_sam3d_v2(
                pred_box3d=out[i],
                depth_map=depth_map,
                intrinsics=intrinsics,
                box3d_to_corners_func=_box3d_to_corners_opencv,
                trim_ratio=params.trim_ratio,
                min_points=params.min_points,
                scale_min=params.scale_min,
                scale_max=params.scale_max,
            )
        else:
            # V1: Original SAM3D style with predicted 2D box
            box2d = boxes2d_xyxy[i].astype(np.float32, copy=False)
            aligned, info = _correct_3d_box_sam3d_style(
                pred_box3d=out[i],
                gt_box2d=box2d,
                depth_map=depth_map,
                intrinsics=intrinsics,
                box3d_to_corners_func=_box3d_to_corners_opencv,
                filter_percentile=params.filter_percentile,
                min_points=params.min_points,
            )
        out[i] = aligned.astype(out.dtype, copy=False)

    return out


