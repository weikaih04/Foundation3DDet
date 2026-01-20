"""SAM3_3D Data Types.

Data structures for SAM3_3D model aligned with SAM3's internal formats.

Key Design Decisions:
1. SAM3_3DBatchedInputs: per-prompt batch format (N_prompts samples)
2. Coordinate formats:
   - geo_boxes: normalized cxcywh (SAM3 geometry encoder input)
   - gt_boxes2d: normalized xyxy (for loss computation)
   - pred_boxes_2d: normalized xyxy (SAM3 output)
3. Tensor dimensions follow SAM3 conventions (sequence-first for Prompt)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, NamedTuple

import torch
from torch import Tensor


@dataclass
class SAM3_3DBatchedInputs:
    """SAM3_3D batched inputs format (per-category query batch).

    This dataclass aligns with SAM3's BatchedDatapoint and FindStage.
    Following SAM3 original design: per-category queries with multi-instance targets.

    Key Design (SAM3 original):
    - Each CATEGORY creates ONE query (not each box!)
    - N_prompts = number of unique categories across batch
    - Each query can have multiple GT boxes (multi-instance targets)
    - Visual queries: randomly select one target as geo_box prompt

    Attributes:
        images: (B_images, 3, H, W) - Input images (unique, shared across queries)
        intrinsics: (B_images, 3, 3) - Camera intrinsics per image

        img_ids: (N_prompts,) - Which image each query belongs to
        text_ids: (N_prompts,) - Which unique_text each query uses
        unique_texts: List of unique text prompts (includes "visual" for geo-only)

        geo_boxes: (N_prompts, 1, 4) - Geometry prompts, normalized cxcywh
                   Only visual queries have valid geo_boxes
                   Text-only queries have geo_boxes_mask=True
        geo_boxes_mask: (N_prompts, 1) - True = no valid box (text-only query)
        geo_box_labels: (N_prompts, 1) - 1 for valid boxes, 0 for text-only

        gt_boxes2d: (N_prompts, max_gt, 4) - GT 2D boxes, normalized xyxy
                    Multi-instance: each query can have multiple targets
        gt_boxes3d: (N_prompts, max_gt, 12) - GT 3D boxes (encoded)
        num_gts: (N_prompts,) - Number of valid GTs per query (can be > 1!)
    """
    
    # ========== Image-level (Backbone processes these) ==========
    images: Tensor                      # (B_images, 3, H, W)
    intrinsics: Tensor                  # (B_images, 3, 3)
    
    # ========== Prompt-level (Expanded per-prompt batch) ==========
    img_ids: Tensor                     # (N_prompts,) long
    text_ids: Tensor                    # (N_prompts,) long
    unique_texts: List[str]             # len = num_unique_texts
    
    # Geometry prompts - sequence-first: (max_K, N_prompts, 4)
    geo_boxes: Tensor | None = None     # normalized cxcywh
    geo_boxes_mask: Tensor | None = None  # (N_prompts, max_K), True=padding
    geo_box_labels: Tensor | None = None  # (max_K, N_prompts), 0/1
    
    # Ground Truth for training - batch-first
    gt_boxes2d: Tensor | None = None    # (N_prompts, max_gt, 4) normalized xyxy
    gt_boxes3d: Tensor | None = None    # (N_prompts, max_gt, 12)
    num_gts: Tensor | None = None       # (N_prompts,) int
    gt_category_ids: Tensor | None = None  # (N_prompts,) category ids

    # Query type tracking (following SAM3 TEXT_ID convention)
    # 0 = TEXT (one-to-many), 2 = GEOMETRY (one-to-one)
    query_types: Tensor | None = None   # (N_prompts,) int

    # Point prompts (optional)
    geo_points: Tensor | None = None    # (N_prompts, max_P, 2) normalized xy
    geo_points_mask: Tensor | None = None  # (N_prompts, max_P) True=padding
    geo_point_labels: Tensor | None = None  # (N_prompts, max_P) 0/1

    # Metadata for evaluation/visualization
    sample_names: List[str] | None = None  # Image identifiers
    dataset_name: List[str] | None = None  # Dataset names
    original_hw: List | Tensor | None = None  # (B_images, 2) or list
    original_images: Tensor | None = None  # (B_images, 3, H, W) unresized
    original_intrinsics: Tensor | None = None  # (B_images, 3, 3)

    def to(self, device: torch.device) -> "SAM3_3DBatchedInputs":
        """Move all tensors to specified device."""
        def move(t):
            return t.to(device) if isinstance(t, Tensor) else t

        return SAM3_3DBatchedInputs(
            images=move(self.images),
            intrinsics=move(self.intrinsics),
            img_ids=move(self.img_ids),
            text_ids=move(self.text_ids),
            unique_texts=self.unique_texts,  # List, no move
            geo_boxes=move(self.geo_boxes),
            geo_boxes_mask=move(self.geo_boxes_mask),
            geo_box_labels=move(self.geo_box_labels),
            gt_boxes2d=move(self.gt_boxes2d),
            gt_boxes3d=move(self.gt_boxes3d),
            num_gts=move(self.num_gts),
            gt_category_ids=move(self.gt_category_ids),
            query_types=move(self.query_types),
            geo_points=move(self.geo_points),
            geo_points_mask=move(self.geo_points_mask),
            geo_point_labels=move(self.geo_point_labels),
            sample_names=self.sample_names,  # List, no move
            dataset_name=self.dataset_name,  # List, no move
            original_hw=self.original_hw,  # List or Tensor
            original_images=move(self.original_images),
            original_intrinsics=move(self.original_intrinsics),
        )
    
    @property
    def num_images(self) -> int:
        """Number of unique images."""
        return self.images.shape[0]
    
    @property
    def num_prompts(self) -> int:
        """Number of prompts (batch size for decoder)."""
        return self.img_ids.shape[0]
    
    @property
    def device(self) -> torch.device:
        """Device of the batch."""
        return self.images.device


class SAM3_3DOut(NamedTuple):
    """Output of SAM3_3D model.
    
    All tensors use batch-first format: (N_prompts, num_queries, dim)
    where N_prompts is the number of prompts (per-prompt batch size).
    
    Coordinate formats:
    - pred_boxes_2d: normalized xyxy [0, 1]
    - pred_boxes_3d: encoded 3D params (delta_center, log_depth, log_dims, rot_6d)
    """
    # 2D Detection (from SAM3 decoder)
    pred_logits: Tensor           # (N_prompts, num_queries, 1)
    pred_boxes_2d: Tensor         # (N_prompts, num_queries, 4) normalized xyxy
    
    # 3D Detection (from 3D head)
    pred_boxes_3d: Tensor | None  # (N_prompts, num_queries, 12)
    
    # Auxiliary outputs for each decoder layer (for deep supervision)
    aux_outputs: list[dict] | None
    
    # Geometry backend losses (SILog depth, phi, theta)
    geom_losses: dict[str, Tensor] | None
    
    # SAM3 specific outputs
    presence_logits: Tensor | None  # (N_prompts, num_queries, 1)
    
    # Hidden states for downstream tasks (e.g., segmentation)
    hidden_states: Tensor | None    # (num_layers, N_prompts, num_queries, d_model)
    
    # Encoder hidden states for cross-attention
    encoder_hidden_states: Tensor | None  # (H*W, N_prompts, d_model)

