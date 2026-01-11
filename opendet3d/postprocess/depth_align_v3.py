"""Depth-based 3D box post-processing (V3 scale-only) for 3D-MOOD.

This module provides:
- a fast pseudo-mask from (2D box + depth_map): seed + depth gate + center CC
- a wrapper that runs the existing V3 alignment implementation from step25_align

Design goal: keep this lightweight so later you can swap in real instance masks
without changing the alignment code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    cv2 = None


import importlib.util
import os
from types import ModuleType


def _load_align_2d_constraint_scaling() -> Any:
    """Load V3 align function without importing step25_align as a package.

    Rationale: `step25_align/__init__.py` imports `alignment_manager`, which uses
    non-package imports (e.g. `from align_simple import ...`) and breaks when
    imported from outside that directory. Loading by file path avoids executing
    that package __init__ entirely.
    """

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    mod_path = os.path.join(
        repo_root,
        "single_frame_3d_gen",
        "ground_select_and_3d_bbox_predict",
        "step25_align",
        "align_2d_constraint_scaling.py",
    )
    if not os.path.isfile(mod_path):
        raise FileNotFoundError(f"V3 align file not found: {mod_path}")

    spec = importlib.util.spec_from_file_location("_align_2d_constraint_scaling_v3", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for: {mod_path}")
    module = importlib.util.module_from_spec(spec)
    assert isinstance(module, ModuleType)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    fn = getattr(module, "align_3d_box_2d_constraint_scaling", None)
    if fn is None:
        raise AttributeError("align_3d_box_2d_constraint_scaling not found in align_2d_constraint_scaling.py")
    return fn


# Load once at import time (fast) and avoid step25_align package side-effects.
align_3d_box_2d_constraint_scaling = _load_align_2d_constraint_scaling()


@dataclass(frozen=True)
class PseudoMaskParams:
    seed_frac: float = 0.4  # center window size as fraction of bbox size
    depth_gate_alpha: float = 0.15  # keep pixels with depth <= z0*(1+alpha)
    min_seed_points: int = 30
    min_mask_points: int = 50
    cc_connectivity: int = 8  # 4 or 8


def _clip_box_xyxy(box: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.astype(float).tolist()
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(0, min(x2, w)))
    y2 = int(max(0, min(y2, h)))
    return x1, y1, x2, y2


def pseudo_mask_seed_depth_gate(
    box2d_xyxy: np.ndarray,
    depth_map: np.ndarray,
    params: PseudoMaskParams = PseudoMaskParams(),
) -> np.ndarray | None:
    """Create a fast pseudo instance mask inside box2d using depth gating.

    Returns:
        mask (H, W) bool, or None if insufficient depth.
    """
    if cv2 is None:
        raise RuntimeError("cv2 is required for connected-components pseudo mask.")

    H, W = depth_map.shape[:2]
    x1, y1, x2, y2 = _clip_box_xyxy(box2d_xyxy, W, H)
    if x2 <= x1 or y2 <= y1:
        return None

    # Seed ROI: centered smaller box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    sx = int(bw * params.seed_frac)
    sy = int(bh * params.seed_frac)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    sx1 = max(x1, cx - sx // 2)
    sx2 = min(x2, cx + sx // 2)
    sy1 = max(y1, cy - sy // 2)
    sy2 = min(y2, cy + sy // 2)

    seed_depth = depth_map[sy1:sy2, sx1:sx2]
    seed_valid = seed_depth[np.isfinite(seed_depth) & (seed_depth > 0)]
    if seed_valid.size < params.min_seed_points:
        return None

    z0 = float(np.median(seed_valid))
    if not np.isfinite(z0) or z0 <= 0:
        return None

    region = depth_map[y1:y2, x1:x2]
    valid = np.isfinite(region) & (region > 0)
    gate = valid & (region <= z0 * (1.0 + params.depth_gate_alpha))

    if int(gate.sum()) < params.min_mask_points:
        return None

    # Keep only the connected component containing the bbox center (in region coords)
    gate_u8 = gate.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(gate_u8, connectivity=params.cc_connectivity)
    if num_labels <= 1:
        return None

    rcx = int(np.clip(cx - x1, 0, (x2 - x1) - 1))
    rcy = int(np.clip(cy - y1, 0, (y2 - y1) - 1))
    center_label = int(labels[rcy, rcx])
    if center_label == 0:
        # If center pixel isn't in foreground, pick the largest component.
        counts = np.bincount(labels.reshape(-1))
        counts[0] = 0
        center_label = int(np.argmax(counts))
        if center_label == 0:
            return None

    cc = labels == center_label
    if int(cc.sum()) < params.min_mask_points:
        return None

    mask = np.zeros((H, W), dtype=bool)
    mask[y1:y2, x1:x2] = cc
    return mask


def _to_whl_from_wlh(box3d: np.ndarray) -> np.ndarray:
    """Convert box3d dims from (w,l,h) to (w,h,l).

    NOTE: 3D-MOOD code comments indicate internal dims are wlh. The V3 alignment
    utilities expect whl.
    """
    out = box3d.copy()
    out[4], out[5] = out[5], out[4]
    return out


def _to_wlh_from_whl(box3d: np.ndarray) -> np.ndarray:
    """Inverse of _to_whl_from_wlh."""
    out = box3d.copy()
    out[4], out[5] = out[5], out[4]
    return out


def align_boxes3d_v3_from_depth(
    boxes3d_raw: np.ndarray,
    boxes2d_xyxy: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    *,
    pseudo_mask_params: PseudoMaskParams = PseudoMaskParams(),
    v3_kwargs: dict[str, Any] | None = None,
) -> np.ndarray:
    """Apply V3 scale-only alignment per instance. Returns aligned boxes3d array.

    If pseudo mask generation fails for a box, that instance is left unchanged.
    """
    v3_kwargs = v3_kwargs or {}
    out = boxes3d_raw.copy()

    if boxes3d_raw.size == 0:
        return out

    for i in range(out.shape[0]):
        box2d = boxes2d_xyxy[i]
        mask = pseudo_mask_seed_depth_gate(box2d, depth_map, params=pseudo_mask_params)
        if mask is None:
            continue

        box3d_whl = _to_whl_from_wlh(out[i])
        aligned_whl, info = align_3d_box_2d_constraint_scaling(
            pred_box3d=box3d_whl,
            mask=mask,
            depth_map=depth_map,
            intrinsics=intrinsics,
            gt_box2d=box2d.astype(np.float32),
            verbose=False,
            class_label=None,
            **v3_kwargs,
        )
        if aligned_whl is None:
            continue

        out[i] = _to_wlh_from_whl(aligned_whl.astype(out.dtype))

    return out


