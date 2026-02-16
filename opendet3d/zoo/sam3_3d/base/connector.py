"""SAM3_3D data connector and collator configuration.

This module provides:
1. DataConnector key mappings for train/test
2. SAM3_3DCollator: converts per-image DataLoader output to SAM3_3DBatchedInputs
3. Point prompt sampling (from mask or box region)
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import List, Literal, Optional

import numpy as np
import torch
from torch import Tensor

from opendet3d.utils.profiler import profile_start, profile_stop

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import DataConnector, data_key, pred_key

from opendet3d.model.detect3d.sam3_3d import SAM3_3DBatchedInputs


# ============================================================================
# Point Sampling Utilities
# ============================================================================

def sample_points_from_mask(
    mask: np.ndarray,
    n_points: int,
    mode: Literal["centered", "random_mask", "random_box"],
    box: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Sample points from a binary mask.

    Args:
        mask: Binary mask (H, W), 1=foreground, 0=background
        n_points: Number of points to sample
        mode: Sampling mode
            - "centered": sample from mask center (farthest from edges)
            - "random_mask": uniform sample from mask interior
            - "random_box": uniform sample from box, label from mask
        box: Box in xyxy format (required for random_box mode)

    Returns:
        Points array (n_points, 3) with (x, y, label)
    """
    if mode == "centered":
        return _center_positive_sample(mask, n_points)
    elif mode == "random_mask":
        return _uniform_positive_sample(mask, n_points)
    elif mode == "random_box":
        assert box is not None, "'random_box' mode requires a provided box."
        return _uniform_sample_from_box(mask, box, n_points)
    else:
        raise ValueError(f"Unknown point sampling mode {mode}.")


def _uniform_positive_sample(mask: np.ndarray, n_points: int) -> np.ndarray:
    """Sample positive points uniformly from mask interior."""
    mask_points = np.stack(np.nonzero(mask), axis=0).transpose(1, 0)
    if len(mask_points) == 0:
        # Empty mask, return center of image as fallback
        h, w = mask.shape
        return np.array([[w // 2, h // 2, 1]] * n_points)

    selected_idxs = np.random.randint(low=0, high=len(mask_points), size=n_points)
    selected_points = mask_points[selected_idxs]
    selected_points = selected_points[:, ::-1]  # (y, x) -> (x, y)
    labels = np.ones((len(selected_points), 1))
    return np.concatenate([selected_points, labels], axis=1)


def _center_positive_sample(mask: np.ndarray, n_points: int) -> np.ndarray:
    """Sample points farthest from mask edges (using distance transform)."""
    try:
        import cv2
    except ImportError:
        # Fallback to uniform sampling if cv2 not available
        return _uniform_positive_sample(mask, n_points)

    if np.max(mask) == 0:
        h, w = mask.shape
        return np.array([[w // 2, h // 2, 1]] * n_points)

    padded_mask = np.pad(mask.astype(np.uint8), 1)
    points = []

    for _ in range(n_points):
        if np.max(padded_mask) == 0:
            break
        dist = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 0)
        point = np.unravel_index(dist.argmax(), dist.shape)
        padded_mask[point[0], point[1]] = 0
        points.append(point[::-1])  # (y, x) -> (x, y)

    if len(points) == 0:
        h, w = mask.shape
        return np.array([[w // 2, h // 2, 1]] * n_points)

    points = np.stack(points, axis=0)
    points = points - 1  # Subtract padding offset
    labels = np.ones((len(points), 1))
    return np.concatenate([points, labels], axis=1)


def _uniform_sample_from_box(
    mask: np.ndarray,
    box: np.ndarray,
    n_points: int,
) -> np.ndarray:
    """Sample points uniformly from box, determine labels from mask."""
    int_box = np.ceil(box).astype(int)
    x1, y1, x2, y2 = int_box

    # Ensure valid box
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)

    x = np.random.randint(low=x1, high=x2, size=n_points)
    y = np.random.randint(low=y1, high=y2, size=n_points)

    # Clip to mask boundaries
    h, w = mask.shape
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)

    labels = mask[y, x]
    return np.stack([x, y, labels], axis=1)


def sample_points_without_mask(
    box: np.ndarray,
    n_positive: int,
    n_negative: int,
    H: int,
    W: int,
) -> np.ndarray:
    """Sample points when no mask is available.

    Uses box region as pseudo-mask:
    - Positive points: uniformly from inside box
    - Negative points: uniformly from outside box

    Args:
        box: Box in xyxy format (x1, y1, x2, y2)
        n_positive: Number of positive points to sample
        n_negative: Number of negative points to sample
        H: Image height
        W: Image width

    Returns:
        Points array (n_positive + n_negative, 3) with (x, y, label)
    """
    x1, y1, x2, y2 = map(int, box)

    # Ensure valid box
    x1 = max(0, min(x1, W - 1))
    x2 = max(x1 + 1, min(x2, W))
    y1 = max(0, min(y1, H - 1))
    y2 = max(y1 + 1, min(y2, H))

    points_list = []

    # Positive points: inside box
    if n_positive > 0:
        pos_x = np.random.randint(x1, x2, size=n_positive)
        pos_y = np.random.randint(y1, y2, size=n_positive)
        pos_labels = np.ones(n_positive)
        pos_points = np.stack([pos_x, pos_y, pos_labels], axis=1)
        points_list.append(pos_points)

    # Negative points: outside box
    if n_negative > 0:
        neg_points = []
        max_attempts = n_negative * 100

        for _ in range(max_attempts):
            if len(neg_points) >= n_negative:
                break
            x = np.random.randint(0, W)
            y = np.random.randint(0, H)
            # Check if outside box
            if not (x1 <= x < x2 and y1 <= y < y2):
                neg_points.append([x, y, 0])

        if len(neg_points) < n_negative:
            # Fallback: sample from image corners if box is too large
            corners = [(0, 0), (W-1, 0), (0, H-1), (W-1, H-1)]
            while len(neg_points) < n_negative:
                cx, cy = corners[len(neg_points) % 4]
                neg_points.append([cx, cy, 0])

        neg_points = np.array(neg_points[:n_negative])
        points_list.append(neg_points)

    if points_list:
        return np.concatenate(points_list, axis=0)
    else:
        return np.zeros((0, 3))


def noise_box(
    box: np.ndarray,
    im_size: tuple,
    box_noise_std: float = 0.1,
    box_noise_max: Optional[float] = 20.0,
    min_box_area: float = 100.0,
) -> np.ndarray:
    """Add noise to a box for data augmentation.

    Args:
        box: Box in xyxy format (x1, y1, x2, y2)
        im_size: Image size (H, W)
        box_noise_std: Noise std relative to box size
        box_noise_max: Max noise in pixels
        min_box_area: Min area after noising

    Returns:
        Noised box in xyxy format
    """
    if box_noise_std <= 0.0:
        return box

    noise = box_noise_std * np.random.randn(4)
    w, h = box[2] - box[0], box[3] - box[1]
    scale_factor = np.array([w, h, w, h])
    noise = noise * scale_factor

    if box_noise_max is not None:
        noise = np.clip(noise, -box_noise_max, box_noise_max)

    noised_box = box + noise

    # Clamp to image bounds
    H, W = im_size
    noised_box = np.maximum(noised_box, 0)
    noised_box = np.minimum(noised_box, [W, H, W, H])

    # Check min area
    new_w = noised_box[2] - noised_box[0]
    new_h = noised_box[3] - noised_box[1]
    if new_w * new_h <= min_box_area:
        return box

    return noised_box


# ============================================================================
# SAM3_3D Collator
# ============================================================================

class SAM3_3DCollator:
    """Collator that converts per-image data to SAM3_3DBatchedInputs.

    Design (SAM3 original - per-category queries):
    - DataLoader produces per-image samples
    - Collator groups GT boxes by category
    - Each category creates ONE query with multi-instance targets
    - This aligns with SAM3's multi-instance detection design

    Per-prompt batch strategy:
    - N_prompts = sum of unique categories across batch (NOT sum of boxes!)
    - img_ids[i] indicates which image prompt i belongs to
    - Each prompt can have multiple GT boxes (multi-instance targets)

    Coordinate format:
    - Input boxes2d: pixel xyxy (from dataset)
    - geo_boxes: normalized cxcywh [0,1] (for SAM3)
    - geo_points: normalized xy [0,1] (for SAM3)
    - gt_boxes2d: normalized xyxy [0,1] (for loss)
    - gt_boxes2d shape: (N_prompts, max_gts, 4) for multi-instance
    - num_gts: (N_prompts,) number of GT boxes per query (can be > 1)

    Text/Visual Query:
    - text_query_prob controls the ratio of text vs visual queries
    - text_query_prob=1.0: all text queries (SAM3 default for training)
    - text_query_prob=0.7: 70% text, 30% visual (recommended by SAM3)
    - Visual queries use one randomly selected target box as geo_box
    """

    def __init__(
        self,
        max_prompts_per_image: int = 50,
        use_text_prompts: bool = True,
        default_text: str = "visual",
        # Point prompt options
        use_point_prompts: bool = False,
        num_positive_points: int | tuple[int, int] = 1,
        num_negative_points: int | tuple[int, int] = 0,
        point_sample_mode: Literal["centered", "random_mask", "random_box"] = "random_mask",
        # Box prompt options
        use_box_prompts: bool = True,
        box_noise_std: float = 0.0,
        box_noise_max: float | None = 20.0,
        # Text/Visual query ratio (SAM3 original design)
        text_query_prob: float = 0.7,  # 70% text, 30% visual (SAM3 recommended)
        keep_text_for_visual: bool = False,  # If True, visual queries keep category text
        # Geometry prompt options (NEW: text + geometry training)
        use_geometry_prompts: bool = False,  # If True, create geometry queries per category
        geometric_query_str: str = "geometric",  # Text for geometry queries (SAM3 convention)
    ):
        """Initialize collator.

        Args:
            max_prompts_per_image: Max number of prompts (categories) per image
            use_text_prompts: Whether to include text with geometric prompts
            default_text: Default text when class name not available
            use_point_prompts: Whether to sample point prompts (for ablation)
            num_positive_points: Number of positive points to sample
                Can be int or (min, max) tuple for random range
            num_negative_points: Number of negative points to sample
                Can be int or (min, max) tuple for random range
            point_sample_mode: How to sample points when mask is available
                - "centered": sample from mask center (farthest from edges)
                - "random_mask": uniform sample from mask interior
                - "random_box": uniform sample from box, label from mask
            use_box_prompts: Whether to use box prompts
            box_noise_std: Noise std for box jittering (0 = no noise)
            box_noise_max: Max noise in pixels
            text_query_prob: Probability of text-only queries (SAM3 recommended: 0.7)
                1.0 = all text queries (pure text training)
                0.7 = 70% text, 30% visual (SAM3 mixed training)
                0.0 = all visual queries (DetAny3D style)
            keep_text_for_visual: If True, visual queries keep category text
                If False (default), visual queries use "visual" as text
            use_geometry_prompts: If True, create geometry queries per category
                This implements text + geometry training (SAM3 style):
                - Each category gets 1 TEXT query (one-to-many targets)
                - Each category gets 1 GEOMETRY query (one-to-one target)
            geometric_query_str: Text for geometry queries (default "geometric")
        """
        self.max_prompts_per_image = max_prompts_per_image
        self.use_text_prompts = use_text_prompts
        self.default_text = default_text

        # Point prompt options
        self.use_point_prompts = use_point_prompts
        self.num_positive_points = num_positive_points
        self.num_negative_points = num_negative_points
        self.point_sample_mode = point_sample_mode

        # Box prompt options
        self.use_box_prompts = use_box_prompts
        self.box_noise_std = box_noise_std
        self.box_noise_max = box_noise_max

        # Text/Visual query ratio
        self.text_query_prob = text_query_prob
        self.keep_text_for_visual = keep_text_for_visual

        # Geometry prompt options (SAM3 style text + geometry training)
        self.use_geometry_prompts = use_geometry_prompts
        self.geometric_query_str = geometric_query_str

    def _sample_num_points(self, num_spec: int | tuple[int, int]) -> int:
        """Sample number of points from spec."""
        if isinstance(num_spec, int):
            return num_spec
        else:
            low, high = num_spec
            return np.random.randint(low, high + 1)

    def _sample_points_for_box(
        self,
        box_xyxy: np.ndarray,
        mask: Optional[np.ndarray],
        H: int,
        W: int,
    ) -> np.ndarray:
        """Sample points for a single box.

        Args:
            box_xyxy: Box in pixel xyxy format
            mask: Optional binary mask (H, W)
            H, W: Image dimensions

        Returns:
            Points array (N, 3) with (x, y, label) in pixel coords
        """
        n_pos = self._sample_num_points(self.num_positive_points)
        n_neg = self._sample_num_points(self.num_negative_points)

        if mask is not None:
            # Sample from actual mask
            points = sample_points_from_mask(
                mask, n_pos + n_neg, self.point_sample_mode, box_xyxy
            )
        else:
            # Use box as pseudo-mask
            points = sample_points_without_mask(box_xyxy, n_pos, n_neg, H, W)

        return points

    def __call__(self, batch: List[dict]) -> SAM3_3DBatchedInputs:
        """Collate batch of per-image samples to SAM3_3DBatchedInputs.

        Args:
            batch: List of dicts, each containing:
                - images: (3, H, W)
                - boxes2d: (N_i, 4) pixel xyxy
                - boxes2d_classes: (N_i,) class indices
                - boxes2d_names: List[str] class names (optional)
                - boxes3d: (N_i, 7+) 3D box params
                - intrinsics: (3, 3)
                - masks2d: (N_i, H, W) binary masks (optional)

        Returns:
            SAM3_3DBatchedInputs with per-prompt batch
        """
        profile_start("  collator_total")

        # Filter out images with no GT boxes to avoid empty prompts
        # This reduces the probability of empty batches during training
        original_batch_size = len(batch)
        batch = [
            item for item in batch
            if item.get("boxes2d") is not None and len(item["boxes2d"]) > 0
        ]

        # if len(batch) < original_batch_size:
        #     import torch.distributed as dist
        #     rank = dist.get_rank() if dist.is_initialized() else 0
        #     filtered_count = original_batch_size - len(batch)
        #     print(
        #         f"[SAM3_3DCollator] Filtered {filtered_count}/{original_batch_size} "
        #         f"empty images on rank {rank}"
        #     )

        B = len(batch)

        # Handle completely empty batch (all images filtered out)
        if B == 0:
            # import torch.distributed as dist
            # rank = dist.get_rank() if dist.is_initialized() else 0
            # print(
            #     f"[SAM3_3DCollator] WARNING: Entire batch empty after filtering "
            #     f"({original_batch_size} images all had 0 GT boxes) on rank {rank}"
            # )
            # Return minimal empty batch - model will handle this gracefully
            return SAM3_3DBatchedInputs(
                images=torch.zeros(0, 3, 1, 1),  # (0, 3, H, W)
                intrinsics=torch.zeros(0, 3, 3),  # (0, 3, 3)
                img_ids=torch.zeros(0, dtype=torch.long),
                text_ids=torch.zeros(0, dtype=torch.long),
                unique_texts=[self.default_text],
                sample_names=None,
                dataset_name=None,
                original_hw=None,
                original_images=None,
                original_intrinsics=None,
                padding=None,
            )

        device = batch[0]["images"].device if batch[0]["images"].is_cuda else "cpu"

        # Collect image-level data
        profile_start("  collator_image_stack")
        # Images might be (3, H, W) or (1, 3, H, W) depending on data pipeline
        images_list = []
        for b in batch:
            img = b["images"]
            # Handle case where img might have extra batch dim
            if img.dim() == 4 and img.shape[0] == 1:
                img = img.squeeze(0)  # (1, 3, H, W) -> (3, H, W)
            images_list.append(img)
        images = torch.stack(images_list)  # (B, 3, H, W)
        intrinsics = torch.stack([b["intrinsics"] for b in batch])  # (B, 3, 3)
        H, W = images.shape[-2:]  # Use -2: and -1 for H, W to be safe
        profile_stop("  collator_image_stack")

        # Collect metadata for evaluation/visualization
        sample_names = []
        dataset_name_list = []
        original_hw_list = []
        original_images_list = []
        original_intrinsics_list = []
        padding_list = []
        for b_idx, b in enumerate(batch):
            # sample_names - image identifier for evaluation
            if "sample_names" in b:
                sample_names.append(b["sample_names"])
            elif "image_id" in b:
                sample_names.append(b["image_id"])
            else:
                sample_names.append(None)

            # dataset_name - for evaluator to route to correct dataset
            if "dataset_name" in b:
                dataset_name_list.append(b["dataset_name"])
            else:
                dataset_name_list.append(None)

            # original_hw - for coordinate scaling back
            if "original_hw" in b:
                original_hw_list.append(b["original_hw"])
            else:
                original_hw_list.append(None)

            # original_images - unresized images for visualization
            if "original_images" in b:
                original_images_list.append(b["original_images"])
            else:
                original_images_list.append(None)

            # original_intrinsics - intrinsics before resize
            if "original_intrinsics" in b:
                original_intrinsics_list.append(b["original_intrinsics"])
            else:
                original_intrinsics_list.append(None)

            # padding - CenterPad offsets [pad_left, pad_right, pad_top, pad_bottom]
            if "padding" in b:
                padding_list.append(b["padding"])
            else:
                padding_list.append(None)

        # Collect depth maps for geometry backend supervision
        depth_maps_list = []
        for b in batch:
            # depth_maps - K.depth_maps key from dataset
            if "depth_maps" in b and b["depth_maps"] is not None:
                depth_maps_list.append(b["depth_maps"])
            else:
                depth_maps_list.append(None)

        # Stack depth maps if available (all images must have depth)
        depth_gt = None
        if depth_maps_list and all(d is not None for d in depth_maps_list):
            try:
                depth_gt = torch.stack(depth_maps_list, dim=0)  # (B, H, W) or (B, 1, H, W)
                if depth_gt.dim() == 3:
                    depth_gt = depth_gt.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
            except (RuntimeError, TypeError):
                depth_gt = None

        # Convert to proper format (None if all are None)
        sample_names = sample_names if any(s is not None for s in sample_names) else None
        dataset_name = dataset_name_list if any(d is not None for d in dataset_name_list) else None
        original_hw = original_hw_list if any(h is not None for h in original_hw_list) else None
        padding = padding_list if any(p is not None for p in padding_list) else None
        original_images = None
        if any(img is not None for img in original_images_list):
            # Convert numpy arrays to tensors, then try stacking.
            # Different-sized images (e.g. cross-dataset) cannot be stacked;
            # in that case keep as list for the visualizer.
            imgs = []
            for img in original_images_list:
                if img is None:
                    continue
                if not isinstance(img, torch.Tensor):
                    img = torch.as_tensor(img)
                imgs.append(img)
            if len(imgs) == 1:
                original_images = imgs[0].unsqueeze(0) if imgs[0].dim() == 3 else imgs[0]
            elif len(imgs) > 1:
                try:
                    original_images = torch.stack(imgs)
                except RuntimeError:
                    # Different shapes across batch - keep first only
                    original_images = imgs[0].unsqueeze(0) if imgs[0].dim() == 3 else imgs[0]
        original_intrinsics = None
        if any(intr is not None for intr in original_intrinsics_list):
            intrs = []
            for intr in original_intrinsics_list:
                if intr is None:
                    continue
                if not isinstance(intr, torch.Tensor):
                    intr = torch.as_tensor(intr)
                intrs.append(intr)
            try:
                original_intrinsics = torch.stack(intrs)
            except (RuntimeError, TypeError):
                original_intrinsics = None

        # Build per-prompt data (SAM3 original: per-category queries)
        # If use_geometry_prompts=True: Each category creates TWO queries
        #   - TEXT query (one-to-many targets)
        #   - GEOMETRY query (one-to-one target)
        # If use_geometry_prompts=False: Original behavior (text or visual per category)
        img_ids_list = []
        text_ids_list = []
        geo_boxes_list = []  # normalized cxcywh (for visual/geometry queries)
        geo_points_list = []  # normalized xy with labels
        is_visual_query_list = []  # Track which queries have visual prompts
        query_types_list = []  # Track query types: 0=TEXT, 2=GEOMETRY

        # Multi-instance targets: list of lists
        # gt_boxes2d_per_query[i] = list of normalized xyxy boxes for query i
        gt_boxes2d_per_query = []
        gt_boxes3d_per_query = []
        gt_category_ids_list = []

        # Build unique text list
        unique_texts = []
        text_to_id = {}

        profile_start("  collator_category_group")
        for img_idx, sample in enumerate(batch):
            boxes2d = sample.get("boxes2d")  # (N_i, 4) pixel xyxy
            boxes3d = sample.get("boxes3d")  # (N_i, 7+)
            class_ids = sample.get("boxes2d_classes")  # (N_i,)
            class_names = sample.get("boxes2d_names", None)  # List[str] or None
            masks2d = sample.get("masks2d", None)  # (N_i, H, W) or None

            if boxes2d is None or len(boxes2d) == 0:
                continue

            # ========== SAM3 Original: Group boxes by category ==========
            # This is the key difference from the old per-box design
            cat_to_box_indices = defaultdict(list)
            for box_idx in range(len(boxes2d)):
                if class_ids is not None:
                    cat_id = class_ids[box_idx]
                    if isinstance(cat_id, torch.Tensor):
                        cat_id = cat_id.item()
                else:
                    cat_id = 0  # Default category if no class info
                cat_to_box_indices[cat_id].append(box_idx)

            # Limit number of categories (queries) per image
            categories = list(cat_to_box_indices.keys())
            if len(categories) > self.max_prompts_per_image:
                categories = categories[:self.max_prompts_per_image]

            # ========== Create queries per category ==========
            # If use_geometry_prompts=True: Create TWO queries per category
            #   - TEXT query (one-to-many targets)
            #   - GEOMETRY query (one-to-one target)
            # If use_geometry_prompts=False: Original text/visual random selection
            for cat_id in categories:
                box_indices = cat_to_box_indices[cat_id]

                # Get category name for text
                if self.use_text_prompts and class_names is not None:
                    cat_name = class_names[cat_id] if cat_id < len(class_names) else self.default_text
                else:
                    cat_name = self.default_text

                # Helper function to normalize box to xyxy [0,1]
                def normalize_box_xyxy(box_xyxy_raw):
                    if isinstance(box_xyxy_raw, torch.Tensor):
                        gt_box_norm = box_xyxy_raw.clone().float()
                    else:
                        gt_box_norm = torch.tensor(box_xyxy_raw, dtype=torch.float32)
                    gt_box_norm[0::2] /= W
                    gt_box_norm[1::2] /= H
                    return gt_box_norm.to(device)

                # Helper function to convert xyxy to cxcywh
                def xyxy_to_cxcywh(box_norm_xyxy):
                    cx = (box_norm_xyxy[0] + box_norm_xyxy[2]) / 2
                    cy = (box_norm_xyxy[1] + box_norm_xyxy[3]) / 2
                    w_box = box_norm_xyxy[2] - box_norm_xyxy[0]
                    h_box = box_norm_xyxy[3] - box_norm_xyxy[1]
                    return torch.tensor([cx, cy, w_box, h_box], device=device)

                if self.use_geometry_prompts:
                    # ========== NEW: Text + Geometry Training ==========
                    # Create TWO queries per category

                    # ----- Query 1: TEXT query (one-to-many) -----
                    img_ids_list.append(img_idx)
                    gt_category_ids_list.append(cat_id)
                    query_types_list.append(0)  # TEXT
                    is_visual_query_list.append(False)

                    # Text = category name
                    if cat_name not in text_to_id:
                        text_to_id[cat_name] = len(unique_texts)
                        unique_texts.append(cat_name)
                    text_ids_list.append(text_to_id[cat_name])

                    # No geometry prompt for text query
                    geo_boxes_list.append(None)

                    # Targets: ALL boxes of this category (one-to-many)
                    query_gt_boxes2d = []
                    query_gt_boxes3d = []
                    for box_idx in box_indices:
                        query_gt_boxes2d.append(normalize_box_xyxy(boxes2d[box_idx]))
                        if boxes3d is not None and box_idx < len(boxes3d):
                            query_gt_boxes3d.append(boxes3d[box_idx].to(device))
                    gt_boxes2d_per_query.append(query_gt_boxes2d)
                    gt_boxes3d_per_query.append(query_gt_boxes3d if query_gt_boxes3d else None)

                    # ----- Query 2: GEOMETRY query (one-to-one) -----
                    img_ids_list.append(img_idx)
                    gt_category_ids_list.append(cat_id)
                    query_types_list.append(2)  # GEOMETRY
                    is_visual_query_list.append(True)

                    # Text = "geometric" (SAM3 convention)
                    if self.geometric_query_str not in text_to_id:
                        text_to_id[self.geometric_query_str] = len(unique_texts)
                        unique_texts.append(self.geometric_query_str)
                    text_ids_list.append(text_to_id[self.geometric_query_str])

                    # Randomly select ONE box as geometry prompt
                    selected_idx = random.choice(box_indices)
                    box_xyxy = boxes2d[selected_idx]
                    box_xyxy_np = box_xyxy.cpu().numpy() if isinstance(box_xyxy, torch.Tensor) else box_xyxy

                    # Optionally add noise (using SAM3's noise_box logic)
                    if self.box_noise_std > 0:
                        box_xyxy_np = noise_box(
                            box_xyxy_np,
                            im_size=(H, W),
                            box_noise_std=self.box_noise_std,
                            box_noise_max=self.box_noise_max,
                        )

                    # Convert to normalized cxcywh for geometry encoder
                    box_norm_xyxy = torch.tensor([
                        box_xyxy_np[0] / W,
                        box_xyxy_np[1] / H,
                        box_xyxy_np[2] / W,
                        box_xyxy_np[3] / H,
                    ], dtype=torch.float32, device=device)
                    geo_boxes_list.append(xyxy_to_cxcywh(box_norm_xyxy))

                    # Target: ONLY the selected box (one-to-one)
                    query_gt_boxes2d = [normalize_box_xyxy(boxes2d[selected_idx])]
                    query_gt_boxes3d = []
                    if boxes3d is not None and selected_idx < len(boxes3d):
                        query_gt_boxes3d.append(boxes3d[selected_idx].to(device))
                    gt_boxes2d_per_query.append(query_gt_boxes2d)
                    gt_boxes3d_per_query.append(query_gt_boxes3d if query_gt_boxes3d else None)

                else:
                    # ========== Original: Text/Visual random selection ==========
                    img_ids_list.append(img_idx)
                    gt_category_ids_list.append(cat_id)

                    # Decide query type: text-only or visual
                    is_text_query = random.random() < self.text_query_prob
                    is_visual_query = not is_text_query

                    # Track query type (0=TEXT for both text and visual in original mode)
                    query_types_list.append(0 if is_text_query else 1)  # 1=VISUAL
                    is_visual_query_list.append(is_visual_query)

                    # Determine text for this query
                    if is_visual_query and not self.keep_text_for_visual:
                        text = "visual"
                    else:
                        text = cat_name

                    if text not in text_to_id:
                        text_to_id[text] = len(unique_texts)
                        unique_texts.append(text)
                    text_ids_list.append(text_to_id[text])

                    # Visual query: pick one target as geo_box
                    if is_visual_query and self.use_box_prompts:
                        selected_idx = random.choice(box_indices)
                        box_xyxy = boxes2d[selected_idx]
                        box_xyxy_np = box_xyxy.cpu().numpy() if isinstance(box_xyxy, torch.Tensor) else box_xyxy

                        if self.box_noise_std > 0:
                            box_xyxy_np = noise_box(
                                box_xyxy_np,
                                im_size=(H, W),
                                box_noise_std=self.box_noise_std,
                                box_noise_max=self.box_noise_max,
                            )

                        box_norm_xyxy = torch.tensor([
                            box_xyxy_np[0] / W,
                            box_xyxy_np[1] / H,
                            box_xyxy_np[2] / W,
                            box_xyxy_np[3] / H,
                        ], dtype=torch.float32, device=device)
                        geo_boxes_list.append(xyxy_to_cxcywh(box_norm_xyxy))
                    else:
                        geo_boxes_list.append(None)

                    # Multi-instance targets: ALL boxes of this category
                    query_gt_boxes2d = []
                    query_gt_boxes3d = []
                    for box_idx in box_indices:
                        query_gt_boxes2d.append(normalize_box_xyxy(boxes2d[box_idx]))
                        if boxes3d is not None and box_idx < len(boxes3d):
                            query_gt_boxes3d.append(boxes3d[box_idx].to(device))
                    gt_boxes2d_per_query.append(query_gt_boxes2d)
                    gt_boxes3d_per_query.append(query_gt_boxes3d if query_gt_boxes3d else None)

        profile_stop("  collator_category_group")

        N_prompts = len(img_ids_list)

        if N_prompts == 0:
            import torch.distributed as dist
            rank = dist.get_rank() if dist.is_initialized() else 0
            print(
                f"[SAM3_3DCollator] WARNING: Unexpected N_prompts=0 "
                f"(B={B} images passed filter) on rank {rank}"
            )
            return SAM3_3DBatchedInputs(
                images=images,
                intrinsics=intrinsics,
                img_ids=torch.zeros(0, dtype=torch.long, device=device),
                text_ids=torch.zeros(0, dtype=torch.long, device=device),
                unique_texts=[self.default_text],
                sample_names=sample_names,
                dataset_name=dataset_name,
                original_hw=original_hw,
                original_images=original_images,
                original_intrinsics=original_intrinsics,
                padding=padding,
            )

        # Stack tensors
        profile_start("  collator_tensor_stack")
        img_ids = torch.tensor(img_ids_list, dtype=torch.long, device=device)
        text_ids = torch.tensor(text_ids_list, dtype=torch.long, device=device)

        # ========== Box prompts for visual queries ==========
        # geo_boxes: (N_prompts, 1, 4) - None for text-only queries
        geo_boxes = None
        geo_boxes_mask = None
        geo_box_labels = None

        # Check if any visual queries exist
        has_visual = any(g is not None for g in geo_boxes_list)
        if has_visual:
            # Stack geo_boxes, use zeros for text-only queries
            stacked_geo_boxes = []
            for g in geo_boxes_list:
                if g is not None:
                    stacked_geo_boxes.append(g)
                else:
                    stacked_geo_boxes.append(torch.zeros(4, device=device))
            geo_boxes = torch.stack(stacked_geo_boxes).unsqueeze(1)  # (N, 1, 4)

            # Mask: True = padding (i.e., text-only queries have no valid box)
            geo_boxes_mask = torch.tensor(
                [[g is None] for g in geo_boxes_list],
                dtype=torch.bool, device=device
            )  # (N, 1)

            # Labels: 1 for positive (valid) boxes
            geo_box_labels = torch.tensor(
                [[1 if g is not None else 0] for g in geo_boxes_list],
                dtype=torch.long, device=device
            )  # (N, 1)

        # ========== Multi-instance GT boxes: pad to (N_prompts, max_gt, 4) ==========
        # Find max number of targets per query
        max_gt = max(len(q) for q in gt_boxes2d_per_query) if gt_boxes2d_per_query else 1
        num_gts_list = []

        gt_boxes2d_padded = []
        for query_boxes in gt_boxes2d_per_query:
            n_gt = len(query_boxes)
            num_gts_list.append(n_gt)

            if n_gt < max_gt:
                # Pad with zeros
                padded = query_boxes + [torch.zeros(4, device=device)] * (max_gt - n_gt)
            else:
                padded = query_boxes
            gt_boxes2d_padded.append(torch.stack(padded))

        gt_boxes2d = torch.stack(gt_boxes2d_padded)  # (N_prompts, max_gt, 4)
        num_gts = torch.tensor(num_gts_list, dtype=torch.long, device=device)  # (N_prompts,)

        # 3D boxes (if available)
        gt_boxes3d = None
        if any(q is not None for q in gt_boxes3d_per_query):
            # Get 3D box dimension from first valid entry
            box3d_dim = None
            for q in gt_boxes3d_per_query:
                if q is not None and len(q) > 0:
                    box3d_dim = q[0].shape[-1]
                    break

            if box3d_dim is not None:
                gt_boxes3d_padded = []
                for query_boxes in gt_boxes3d_per_query:
                    if query_boxes is None or len(query_boxes) == 0:
                        # No 3D boxes for this query
                        padded = [torch.zeros(box3d_dim, device=device)] * max_gt
                    else:
                        n_gt = len(query_boxes)
                        if n_gt < max_gt:
                            padded = query_boxes + [torch.zeros(box3d_dim, device=device)] * (max_gt - n_gt)
                        else:
                            padded = query_boxes
                    gt_boxes3d_padded.append(torch.stack(padded))
                gt_boxes3d = torch.stack(gt_boxes3d_padded)  # (N_prompts, max_gt, box3d_dim)

        gt_category_ids = torch.tensor(gt_category_ids_list, dtype=torch.long, device=device)

        # Query types: 0=TEXT, 1=VISUAL, 2=GEOMETRY
        query_types = torch.tensor(query_types_list, dtype=torch.long, device=device)
        profile_stop("  collator_tensor_stack")
        profile_stop("  collator_total")

        return SAM3_3DBatchedInputs(
            images=images,
            intrinsics=intrinsics,
            img_ids=img_ids,
            text_ids=text_ids,
            unique_texts=unique_texts,
            geo_boxes=geo_boxes,
            geo_boxes_mask=geo_boxes_mask,
            geo_box_labels=geo_box_labels,
            geo_points=None,  # Point prompts not implemented in per-category design yet
            geo_points_mask=None,
            geo_point_labels=None,
            gt_boxes2d=gt_boxes2d,
            gt_boxes3d=gt_boxes3d,
            num_gts=num_gts,
            gt_category_ids=gt_category_ids,
            query_types=query_types,
            # Metadata for evaluation/visualization
            sample_names=sample_names,
            dataset_name=dataset_name,
            original_hw=original_hw,
            original_images=original_images,
            original_intrinsics=original_intrinsics,
            padding=padding,
            # Depth ground truth for geometry backend supervision
            depth_gt=depth_gt,
            depth_mask=None,  # Not yet implemented
        )


# ============================================================================
# SAM3_3D Specific Connectors
# ============================================================================

# Training connector for SAM3_3D
# Note: SAM3 uses geometric prompts (boxes/points) instead of text
CONN_SAM3_3D_TRAIN = {
    "images": K.images,
    "input_hw": K.input_hw,
    # Geometric prompts (boxes as prompts)
    "prompt_boxes": K.boxes2d,  # Use GT boxes as prompts during training
    "prompt_box_labels": K.boxes2d_classes,
    # Targets
    "boxes2d": K.boxes2d,
    "boxes2d_classes": K.boxes2d_classes,
    "boxes3d": K.boxes3d,
    # Camera
    "intrinsics": K.intrinsics,
    # Depth for geometry backend
    "depth_gt": K.depth_maps,
}

# Test connector for SAM3_3D
CONN_SAM3_3D_TEST = {
    "images": K.images,
    "input_hw": K.input_hw,
    "original_hw": K.original_hw,
    # Geometric prompts (from external detector or user input)
    "prompt_boxes": K.boxes2d,  # External 2D detections as prompts
    # Camera
    "intrinsics": K.intrinsics,
    "padding": "padding",
}

# Loss connector for SAM3_3D
CONN_SAM3_3D_LOSS = {
    # Model outputs
    "pred_logits": pred_key("pred_logits"),
    "pred_boxes_2d": pred_key("pred_boxes_2d"),
    "pred_boxes_3d": pred_key("pred_boxes_3d"),
    "aux_outputs": pred_key("aux_outputs"),
    "geom_losses": pred_key("geom_losses"),
    # Matching indices (computed by model)
    "indices": pred_key("indices"),
    # Targets
    "targets": {
        "boxes": data_key(K.boxes2d),
        "boxes_xyxy": data_key(K.boxes2d),  # Will be converted
        "boxes_3d": data_key(K.boxes3d),
        "num_boxes": data_key("num_boxes"),
        "image_size": data_key(K.input_hw),  # (H, W) for pixel coordinate conversion
    },
    # Camera
    "intrinsics": data_key(K.intrinsics),
    # Image size for pixel coordinate conversion (following GDino3D)
    "image_size": data_key(K.input_hw),
}

# Evaluation connector
CONN_SAM3_3D_EVAL = {
    "coco_image_id": data_key(K.sample_names),
    "pred_boxes": pred_key("boxes"),
    "pred_scores": pred_key("scores"),
    "pred_classes": pred_key("class_ids"),
    "pred_boxes3d": pred_key("boxes3d"),
}

# Visualization connector
CONN_SAM3_3D_VIS = {
    "images": data_key(K.original_images),
    "image_names": data_key(K.sample_names),
    "intrinsics": data_key("original_intrinsics"),
    "boxes3d": pred_key("boxes3d"),
    "class_ids": pred_key("class_ids"),
    "scores": pred_key("scores"),
}


class SAM3_3DPassthroughConnector:
    """Data connector that passes SAM3_3DBatchedInputs directly to model.

    Since SAM3_3DCollator already produces SAM3_3DBatchedInputs with all needed
    data, we just pass it through as the 'batch' parameter to model.forward().

    This bypasses the key_mapping approach used by vis4d's DataConnector,
    which expects raw DataLoader output format.
    """

    def __call__(self, data: SAM3_3DBatchedInputs) -> dict:
        """Pass batch directly to model.

        Args:
            data: SAM3_3DBatchedInputs from collator

        Returns:
            Dict with 'batch' key pointing to the input data
        """
        return {"batch": data}


class SAM3_3DLossConnector:
    """Loss connector that passes model output and batch directly to loss.

    Similar to SAM3_3DPassthroughConnector, this bypasses vis4d's key_mapping
    since SAM3_3DLoss expects structured objects (SAM3_3DOut, SAM3_3DBatchedInputs).

    This connector is used with LossModule to enable proper wandb logging of
    individual loss components (loss_cls, loss_bbox, loss_giou, etc.).
    """

    def __call__(self, predictions, batch: SAM3_3DBatchedInputs) -> dict:
        """Map model output and batch to loss function inputs.

        Args:
            predictions: SAM3_3DOut from model.forward()
            batch: SAM3_3DBatchedInputs from collator

        Returns:
            Dict with 'out' and 'batch' keys for SAM3_3DLoss.forward()
        """
        return {
            "out": predictions,
            "batch": batch,
        }


class SAM3_3DVisConnector:
    """Vis connector that extracts from SAM3_3DBatchedInputs for visualization.

    vis4d's CallbackConnector uses dict access (data[key]) which doesn't
    work with SAM3_3DBatchedInputs dataclass. This connector does the
    extraction manually.

    Args:
        score_threshold: Only visualize boxes with score >= this value.
            Separate from model's score_threshold so evaluation AP is unaffected.
    """

    def __init__(self, score_threshold: float = 0.0):
        self.score_threshold = score_threshold

    def __call__(self, prediction, data: SAM3_3DBatchedInputs) -> dict:
        """Extract visualization data from dataclass + prediction.

        Args:
            prediction: Det3DOut NamedTuple from model.
            data: SAM3_3DBatchedInputs from collator.

        Returns:
            Dict with keys expected by BoundingBox3DVisualizer.
        """
        # When the collator filters out images with no GT boxes (empty batch),
        # original_images is None. Return empty tensor so the visualizer's
        # for-loop iterates 0 times instead of crashing.
        images = data.original_images
        if images is None:
            images = torch.zeros(0, 3, 1, 1)

        boxes3d = prediction.boxes3d
        class_ids = prediction.class_ids
        scores = prediction.scores

        # Filter by score threshold per image for cleaner visualization
        if self.score_threshold > 0.0 and scores is not None:
            filtered_boxes3d = []
            filtered_class_ids = []
            filtered_scores = []
            for i in range(len(scores)):
                mask = scores[i] >= self.score_threshold
                filtered_scores.append(scores[i][mask])
                filtered_class_ids.append(class_ids[i][mask])
                filtered_boxes3d.append(boxes3d[i][mask])
            boxes3d = filtered_boxes3d
            class_ids = filtered_class_ids
            scores = filtered_scores

        return {
            "images": images,
            "image_names": data.sample_names,
            "intrinsics": data.original_intrinsics,
            "boxes3d": boxes3d,
            "class_ids": class_ids,
            "scores": scores,
        }


class SAM3_3DEvalConnector:
    """Eval connector that extracts from SAM3_3DBatchedInputs for evaluator.

    Same issue as SAM3_3DVisConnector: CallbackConnector doesn't work with
    dataclass. This connector manually extracts fields.
    """

    def __call__(self, prediction, data: SAM3_3DBatchedInputs) -> dict:
        """Extract evaluation data from dataclass + prediction.

        Args:
            prediction: Det3DOut NamedTuple from model.
            data: SAM3_3DBatchedInputs from collator.

        Returns:
            Dict with keys expected by Omni3DEvaluator.
        """
        return {
            "coco_image_id": data.sample_names,
            "dataset_names": data.dataset_name,
            "pred_boxes": prediction.boxes,
            "pred_scores": prediction.scores,
            "pred_classes": prediction.class_ids,
            "pred_boxes3d": prediction.boxes3d,
        }


def get_sam3_3d_data_connector_cfg() -> tuple[ConfigDict, ConfigDict]:
    """Get SAM3_3D data connector configuration.

    Returns:
        Tuple of (train_connector, test_connector).

    Note:
        Uses SAM3_3DPassthroughConnector which passes the collated batch
        directly to model.forward(batch=...), rather than mapping individual
        keys like standard vis4d DataConnector.
    """
    train_data_connector = class_config(SAM3_3DPassthroughConnector)
    test_data_connector = class_config(SAM3_3DPassthroughConnector)

    return train_data_connector, test_data_connector


def get_sam3_3d_collator_cfg(
    max_prompts_per_image: int = 50,
    use_text_prompts: bool = True,
    # Point prompt options (for ablation)
    use_point_prompts: bool = False,
    num_positive_points: int | tuple[int, int] = 1,
    num_negative_points: int | tuple[int, int] = 0,
    point_sample_mode: Literal["centered", "random_mask", "random_box"] = "random_mask",
    # Box prompt options
    use_box_prompts: bool = True,
    box_noise_std: float = 0.0,
    box_noise_max: float | None = 20.0,
    # Text/Visual query ratio (SAM3 original design)
    text_query_prob: float = 0.7,
    keep_text_for_visual: bool = False,
) -> ConfigDict:
    """Get SAM3_3D collator configuration.

    The collator converts per-image DataLoader output to SAM3_3DBatchedInputs.
    Following SAM3 original design: per-category queries with multi-instance targets.

    Args:
        max_prompts_per_image: Max prompts (categories) per image
        use_text_prompts: Whether to include text with geometric prompts
        use_point_prompts: Whether to sample point prompts (for ablation)
        num_positive_points: Number of positive points to sample
            Can be int or (min, max) tuple for random range
        num_negative_points: Number of negative points to sample
            Can be int or (min, max) tuple for random range
        point_sample_mode: How to sample points when mask is available
            - "centered": sample from mask center (farthest from edges)
            - "random_mask": uniform sample from mask interior
            - "random_box": uniform sample from box, label from mask
        use_box_prompts: Whether to use box prompts
        box_noise_std: Noise std for box jittering (0 = no noise)
        box_noise_max: Max noise in pixels
        text_query_prob: Probability of text-only queries (SAM3 recommended: 0.7)
            1.0 = all text queries (pure text training)
            0.7 = 70% text, 30% visual (SAM3 mixed training)
            0.0 = all visual queries (DetAny3D style)
        keep_text_for_visual: If True, visual queries keep category text
            If False (default), visual queries use "visual" as text

    Returns:
        Collator configuration
    """
    return class_config(
        SAM3_3DCollator,
        max_prompts_per_image=max_prompts_per_image,
        use_text_prompts=use_text_prompts,
        use_point_prompts=use_point_prompts,
        num_positive_points=num_positive_points,
        num_negative_points=num_negative_points,
        point_sample_mode=point_sample_mode,
        use_box_prompts=use_box_prompts,
        box_noise_std=box_noise_std,
        box_noise_max=box_noise_max,
        text_query_prob=text_query_prob,
        keep_text_for_visual=keep_text_for_visual,
    )
