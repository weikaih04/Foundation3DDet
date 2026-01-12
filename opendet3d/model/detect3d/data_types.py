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
    """SAM3_3D batched inputs format (per-prompt batch).
    
    This dataclass aligns with SAM3's BatchedDatapoint and FindStage.
    All prompt-level tensors have shape (N_prompts, ...) where N_prompts
    is the total number of prompts across all images.
    
    Attributes:
        images: (B_images, 3, H, W) - Input images (unique, shared across prompts)
        intrinsics: (B_images, 3, 3) - Camera intrinsics per image
        
        img_ids: (N_prompts,) - Which image each prompt belongs to
        text_ids: (N_prompts,) - Which unique_text each prompt uses
        unique_texts: List of unique text prompts (includes "visual" for geo-only)
        
        geo_boxes: (max_K, N_prompts, 4) - Geometry prompts, normalized cxcywh
                   Sequence-first format for SAM3's Prompt class
        geo_boxes_mask: (N_prompts, max_K) - True = padding position
        geo_box_labels: (max_K, N_prompts) - 0/1 for negative/positive prompt
        
        gt_boxes2d: (N_prompts, max_gt, 4) - GT 2D boxes, normalized xyxy
        gt_boxes3d: (N_prompts, max_gt, 12) - GT 3D boxes (encoded)
        num_gts: (N_prompts,) - Number of valid GTs per prompt
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
    
    # Optional: original image sizes for visualization
    original_hw: Tensor | None = None   # (B_images, 2)
    
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
            original_hw=move(self.original_hw),
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

