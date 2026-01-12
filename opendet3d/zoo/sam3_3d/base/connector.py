"""SAM3_3D data connector and collator configuration.

This module provides:
1. DataConnector key mappings for train/test
2. SAM3_3DCollator: converts per-image DataLoader output to SAM3_3DBatchedInputs
3. Point prompt sampling (from mask or box region)
"""

from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
import torch
from torch import Tensor

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

    Design:
    - DataLoader produces per-image samples
    - Collator expands to per-prompt batch (one prompt per GT box)
    - Each prompt uses one GT box as geometric prompt
    - Optionally samples point prompts from mask or box region

    Per-prompt batch strategy:
    - N_prompts = sum of all GT boxes across batch
    - img_ids[i] indicates which image prompt i belongs to
    - Each prompt gets one box from the corresponding image

    Coordinate format:
    - Input boxes2d: pixel xyxy (from dataset)
    - geo_boxes: normalized cxcywh [0,1] (for SAM3)
    - geo_points: normalized xy [0,1] (for SAM3)
    - gt_boxes2d: normalized xyxy [0,1] (for loss)

    Point Sampling Strategy:
    - If mask available: sample from mask (centered/random_mask mode)
    - If no mask: use box as pseudo-mask (positive inside, negative outside)
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
    ):
        """Initialize collator.

        Args:
            max_prompts_per_image: Max number of prompts per image
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
        B = len(batch)
        device = batch[0]["images"].device if batch[0]["images"].is_cuda else "cpu"

        # Collect image-level data
        images = torch.stack([b["images"] for b in batch])  # (B, 3, H, W)
        intrinsics = torch.stack([b["intrinsics"] for b in batch])  # (B, 3, 3)
        H, W = images.shape[2:]

        # Build per-prompt data
        img_ids_list = []
        text_ids_list = []
        geo_boxes_list = []  # normalized cxcywh
        geo_points_list = []  # normalized xy with labels
        gt_boxes2d_list = []  # normalized xyxy
        gt_boxes3d_list = []
        gt_category_ids_list = []

        # Build unique text list
        unique_texts = []
        text_to_id = {}

        for img_idx, sample in enumerate(batch):
            boxes2d = sample.get("boxes2d")  # (N_i, 4) pixel xyxy
            boxes3d = sample.get("boxes3d")  # (N_i, 7+)
            class_ids = sample.get("boxes2d_classes")  # (N_i,)
            class_names = sample.get("boxes2d_names", None)  # List[str] or None
            masks2d = sample.get("masks2d", None)  # (N_i, H, W) or None

            # Check if sample has query_types from SAM3DPromptSampler
            query_types = sample.get("query_types", None)  # List[str] or None
            text_prompts_from_sampler = sample.get("text_prompts", None)  # List[str] or None
            geo_boxes_from_sampler = sample.get("geo_boxes", None)  # (N_i, K, 4) or None
            geo_points_from_sampler = sample.get("geo_points", None)  # (N_i, P, 2) or None

            if boxes2d is None or len(boxes2d) == 0:
                continue

            n_boxes = min(len(boxes2d), self.max_prompts_per_image)

            for box_idx in range(n_boxes):
                # Image index for this prompt
                img_ids_list.append(img_idx)

                # Determine text prompt based on query type or batch-level setting
                if query_types is not None and box_idx < len(query_types):
                    # Per-sample query type from SAM3DPromptSampler
                    query_type = query_types[box_idx]
                    use_text_this_prompt = query_type in ["TEXT", "TEXT_GEOMETRIC"]
                    use_geo_this_prompt = query_type in ["GEOMETRIC", "TEXT_GEOMETRIC"]

                    if text_prompts_from_sampler is not None and box_idx < len(text_prompts_from_sampler):
                        text = text_prompts_from_sampler[box_idx]
                    else:
                        text = self.default_text
                else:
                    # Batch-level settings (original behavior)
                    use_text_this_prompt = self.use_text_prompts
                    use_geo_this_prompt = self.use_box_prompts or self.use_point_prompts

                    if use_text_this_prompt and class_names is not None:
                        text = class_names[box_idx] if box_idx < len(class_names) else self.default_text
                    else:
                        text = self.default_text

                if text not in text_to_id:
                    text_to_id[text] = len(unique_texts)
                    unique_texts.append(text)
                text_ids_list.append(text_to_id[text])

                # Get box in pixel xyxy
                box_xyxy = boxes2d[box_idx]  # (4,) pixel xyxy
                box_xyxy_np = box_xyxy.cpu().numpy() if isinstance(box_xyxy, torch.Tensor) else box_xyxy

                # Check if geometric prompts come from SAM3DPromptSampler
                use_box_from_sampler = (
                    geo_boxes_from_sampler is not None
                    and box_idx < len(geo_boxes_from_sampler)
                    and geo_boxes_from_sampler[box_idx].abs().sum() > 0
                )
                use_point_from_sampler = (
                    geo_points_from_sampler is not None
                    and box_idx < len(geo_points_from_sampler)
                    and geo_points_from_sampler[box_idx].abs().sum() > 0
                )

                # Box prompt handling
                if query_types is not None and box_idx < len(query_types):
                    # Per-sample decision from PromptSampler
                    if use_geo_this_prompt and use_box_from_sampler:
                        # Use box from SAM3DPromptSampler (already in normalized cxcywh)
                        geo_box = geo_boxes_from_sampler[box_idx, 0]  # (4,) normalized cxcywh
                        geo_boxes_list.append(geo_box.to(device))
                    elif use_geo_this_prompt and not use_box_from_sampler and not use_point_from_sampler:
                        # Fallback: generate box prompt from GT
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
                        cx = (box_norm_xyxy[0] + box_norm_xyxy[2]) / 2
                        cy = (box_norm_xyxy[1] + box_norm_xyxy[3]) / 2
                        w = box_norm_xyxy[2] - box_norm_xyxy[0]
                        h = box_norm_xyxy[3] - box_norm_xyxy[1]
                        geo_boxes_list.append(torch.tensor([cx, cy, w, h], device=device))
                else:
                    # Batch-level setting (original behavior)
                    if self.use_box_prompts:
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
                        cx = (box_norm_xyxy[0] + box_norm_xyxy[2]) / 2
                        cy = (box_norm_xyxy[1] + box_norm_xyxy[3]) / 2
                        w = box_norm_xyxy[2] - box_norm_xyxy[0]
                        h = box_norm_xyxy[3] - box_norm_xyxy[1]
                        geo_boxes_list.append(torch.tensor([cx, cy, w, h], device=device))

                # Point prompt handling
                if query_types is not None and box_idx < len(query_types):
                    # Per-sample decision from PromptSampler
                    if use_geo_this_prompt and use_point_from_sampler:
                        # Use points from SAM3DPromptSampler (already normalized)
                        geo_pts = geo_points_from_sampler[box_idx]  # (P, 2) normalized xy
                        # Add labels (all positive)
                        labels = torch.ones(geo_pts.shape[0], 1, dtype=torch.float32, device=device)
                        pts_with_labels = torch.cat([geo_pts.to(device), labels], dim=1)  # (P, 3)
                        geo_points_list.append(pts_with_labels)
                else:
                    # Batch-level setting (original behavior)
                    if self.use_point_prompts:
                        # Get mask for this box if available
                        mask = None
                        if masks2d is not None and box_idx < len(masks2d):
                            mask = masks2d[box_idx]
                            if isinstance(mask, torch.Tensor):
                                mask = mask.cpu().numpy()

                        # Sample points (in pixel coords)
                        points = self._sample_points_for_box(
                            boxes2d[box_idx].cpu().numpy() if isinstance(boxes2d[box_idx], torch.Tensor) else boxes2d[box_idx],
                            mask, H, W
                        )  # (N_pts, 3) with (x, y, label)

                        # Normalize to [0, 1]
                        points_normalized = points.copy()
                        points_normalized[:, 0] /= W
                        points_normalized[:, 1] /= H

                        geo_points_list.append(torch.tensor(points_normalized, dtype=torch.float32, device=device))

                # GT boxes for loss (use original GT, not jittered)
                original_box_xyxy = boxes2d[box_idx]
                if isinstance(original_box_xyxy, torch.Tensor):
                    gt_box_norm = original_box_xyxy.clone().float()
                else:
                    gt_box_norm = torch.tensor(original_box_xyxy, dtype=torch.float32)
                gt_box_norm[0::2] /= W
                gt_box_norm[1::2] /= H
                gt_boxes2d_list.append(gt_box_norm.to(device))

                if boxes3d is not None and box_idx < len(boxes3d):
                    gt_boxes3d_list.append(boxes3d[box_idx])
                if class_ids is not None and box_idx < len(class_ids):
                    gt_category_ids_list.append(class_ids[box_idx])

        N_prompts = len(img_ids_list)

        if N_prompts == 0:
            # Handle empty batch
            return SAM3_3DBatchedInputs(
                images=images,
                intrinsics=intrinsics,
                img_ids=torch.zeros(0, dtype=torch.long, device=device),
                text_ids=torch.zeros(0, dtype=torch.long, device=device),
                unique_texts=[self.default_text],
            )

        # Stack tensors
        img_ids = torch.tensor(img_ids_list, dtype=torch.long, device=device)
        text_ids = torch.tensor(text_ids_list, dtype=torch.long, device=device)

        # Box prompts: (N_prompts, 1, 4) - one box per prompt
        geo_boxes = None
        geo_boxes_mask = None
        geo_box_labels = None
        if self.use_box_prompts and geo_boxes_list:
            geo_boxes = torch.stack(geo_boxes_list).unsqueeze(1)  # (N, 1, 4)
            geo_boxes_mask = torch.zeros(N_prompts, 1, dtype=torch.bool, device=device)
            geo_box_labels = torch.ones(N_prompts, 1, dtype=torch.long, device=device)

        # Point prompts: (N_prompts, max_P, 2) with labels
        geo_points = None
        geo_points_mask = None
        geo_point_labels = None
        if self.use_point_prompts and geo_points_list:
            # Pad to same length
            max_points = max(p.shape[0] for p in geo_points_list)
            padded_points = []
            padded_labels = []
            padded_mask = []

            for pts in geo_points_list:
                n_pts = pts.shape[0]
                # Points (x, y)
                pts_xy = pts[:, :2]
                pts_labels = pts[:, 2].long()

                if n_pts < max_points:
                    # Pad with zeros
                    pad_size = max_points - n_pts
                    pts_xy = torch.cat([pts_xy, torch.zeros(pad_size, 2, device=device)], dim=0)
                    pts_labels = torch.cat([pts_labels, torch.zeros(pad_size, dtype=torch.long, device=device)], dim=0)
                    mask = torch.cat([
                        torch.zeros(n_pts, dtype=torch.bool, device=device),
                        torch.ones(pad_size, dtype=torch.bool, device=device)
                    ], dim=0)
                else:
                    mask = torch.zeros(n_pts, dtype=torch.bool, device=device)

                padded_points.append(pts_xy)
                padded_labels.append(pts_labels)
                padded_mask.append(mask)

            geo_points = torch.stack(padded_points)  # (N, max_P, 2)
            geo_point_labels = torch.stack(padded_labels)  # (N, max_P)
            geo_points_mask = torch.stack(padded_mask)  # (N, max_P)

        # GT for loss
        gt_boxes2d = torch.stack(gt_boxes2d_list) if gt_boxes2d_list else None
        gt_boxes3d = torch.stack(gt_boxes3d_list) if gt_boxes3d_list else None
        gt_category_ids = torch.tensor(gt_category_ids_list, dtype=torch.long, device=device) if gt_category_ids_list else None

        # Count GTs per image
        num_gts = torch.zeros(B, dtype=torch.long, device=device)
        for idx in img_ids_list:
            num_gts[idx] += 1

        return SAM3_3DBatchedInputs(
            images=images,
            intrinsics=intrinsics,
            img_ids=img_ids,
            text_ids=text_ids,
            unique_texts=unique_texts,
            geo_boxes=geo_boxes,
            geo_boxes_mask=geo_boxes_mask,
            geo_box_labels=geo_box_labels,
            geo_points=geo_points,
            geo_points_mask=geo_points_mask,
            geo_point_labels=geo_point_labels,
            gt_boxes2d=gt_boxes2d,
            gt_boxes3d=gt_boxes3d,
            num_gts=num_gts,
            gt_category_ids=gt_category_ids,
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
    },
    # Camera
    "intrinsics": data_key(K.intrinsics),
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


def get_sam3_3d_data_connector_cfg() -> tuple[ConfigDict, ConfigDict]:
    """Get SAM3_3D data connector configuration.

    Returns:
        Tuple of (train_connector, test_connector).
    """
    train_data_connector = class_config(
        DataConnector, key_mapping=CONN_SAM3_3D_TRAIN
    )

    test_data_connector = class_config(
        DataConnector, key_mapping=CONN_SAM3_3D_TEST
    )

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
) -> ConfigDict:
    """Get SAM3_3D collator configuration.

    The collator converts per-image DataLoader output to SAM3_3DBatchedInputs.

    Args:
        max_prompts_per_image: Max prompts (GT boxes) per image
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
    )
