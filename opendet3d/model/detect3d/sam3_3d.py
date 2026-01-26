"""SAM3_3D: SAM3 with 3D Detection Head.

This module combines SAM3 (2D detection with geometric prompting) with
3D-MOOD's 3D detection head and geometry backend.

Key Design Decisions (from Design Doc):
1. Coordinate format: SAM3 uses normalized cxcywh internally,
   model outputs normalized xyxy [0, 1]
2. Tensor format: SAM3 Decoder outputs sequence-first (L, S, B, C),
   3D Head expects batch-first (L, B, S, C) -> need permute
3. Batch strategy: per-prompt batch with img_ids indexing
4. bbox_head: Reuse SAM3 Decoder's internal bbox_embed,
   no external bbox_head needed
5. Forward: Reuse SAM3's forward_grounding() method for 2D detection,
   then add 3D head on top

Data Flow:
1. DataLoader produces per-image data
2. Collator expands to per-prompt batch (SAM3_3DBatchedInputs)
3. Model forward receives expanded data, calls SAM3's forward_grounding
4. 3D head processes SAM3 output
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import List, NamedTuple, Optional

import torch
from torch import Tensor, nn

from opendet3d.utils.profiler import profile_start, profile_stop, profile_step

# SAM3 imports
from sam3.model.sam3_image import Sam3Image
from sam3.model.geometry_encoders import Prompt
from sam3.model.box_ops import box_cxcywh_to_xyxy
from sam3.model.data_misc import FindStage, BatchedFindTarget

# 3D-MOOD imports
from opendet3d.op.detect3d.grounding_dino_3d import (
    GroundingDINO3DHead,
    GroundingDINO3DCoder,
    RoI2Det3D,
)
from opendet3d.model.detect3d.grounding_dino_3d import Det3DOut
from opendet3d.op.detect3d.geometry import GeometryBackendBase


class SAM3_3DOut(NamedTuple):
    """Output of SAM3_3D model.

    All tensors use batch-first format: (B, num_queries, dim)
    where B = N_prompts (per-prompt batch).

    Coordinate formats:
    - pred_boxes_2d: normalized xyxy [0, 1]
    - pred_boxes_3d: encoded 3D params (delta_center, log_depth, log_dims, rot_6d)
    """
    # 2D Detection (from SAM3 decoder) - O2O outputs
    pred_logits: Tensor  # (N_prompts, num_queries, 1) - objectness
    pred_boxes_2d: Tensor  # (N_prompts, num_queries, 4) - normalized xyxy

    # 3D Detection (from 3D head) - O2O outputs
    pred_boxes_3d: Tensor | None  # (N_prompts, num_queries, 12) - encoded 3D params

    # Auxiliary outputs for each decoder layer (for deep supervision)
    aux_outputs: list[dict] | None

    # Geometry backend losses (SILog depth, phi, theta)
    geom_losses: dict[str, Tensor] | None

    # SAM3 specific outputs
    presence_logits: Tensor | None  # (N_prompts, num_queries, 1)
    queries: Tensor | None  # (N_prompts, num_queries, d_model) - for segmentation

    # Encoder hidden states (for depth head if needed)
    encoder_hidden_states: Tensor | None  # (H*W, N_prompts, d_model)

    # Matching indices from SAM3 (for loss computation)
    # Format: (batch_idx, src_idx, tgt_idx) from Hungarian matching
    indices: tuple | None = None

    # O2M (One-to-Many) outputs from SAM3 DAC mechanism
    # These are separate outputs from the second half of queries in DAC mode
    pred_logits_o2m: Tensor | None = None  # (N_prompts, num_queries, 1)
    pred_boxes_2d_o2m: Tensor | None = None  # (N_prompts, num_queries, 4) - normalized xyxy
    pred_boxes_3d_o2m: Tensor | None = None  # (N_prompts, num_queries, 12) - encoded 3D params

    def __getitem__(self, key: str):
        """Support dict-like access for vis4d data connector compatibility."""
        return getattr(self, key)

    def keys(self):
        """Return field names for dict-like iteration."""
        return [f.name for f in fields(self)]

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for dict-like access."""
        return hasattr(self, key)


@dataclass
class SAM3_3DBatchedInputs:
    """SAM3_3D batched input format (per-prompt batch).

    Design Principles:
    1. Aligned with SAM3's BatchedDatapoint
    2. Added 3D-MOOD required fields (intrinsics, gt_boxes3d)
    3. Supports three modes: TEXT / GEOMETRIC / TEXT_GEOMETRIC

    Coordinate Format Convention:
    - geo_boxes: normalized [0,1] cxcywh (SAM3 Geometry Encoder input)
    - gt_boxes2d: normalized [0,1] xyxy (for loss computation)
    - Model output pred_boxes_2d: normalized xyxy [0,1]
    """

    # ========== Image-level (Backbone processing) ==========
    images: Tensor                    # (B_images, 3, H, W)
    intrinsics: Tensor                # (B_images, 3, 3)

    # ========== Prompt-level (expanded) ==========
    img_ids: Tensor                   # (N_prompts,) - which image each prompt belongs to
    text_ids: Tensor                  # (N_prompts,) - text index for each prompt
    unique_texts: List[str]           # deduplicated texts (including "visual" placeholder)

    # Geometry input - batch-first: (N_prompts, max_K, 4) - normalized cxcywh
    # Converted to sequence-first when passed to SAM3 Prompt class
    geo_boxes: Tensor | None = None          # (N_prompts, max_K, 4)
    geo_boxes_mask: Tensor | None = None     # (N_prompts, max_K) - True=padding
    geo_box_labels: Tensor | None = None     # (N_prompts, max_K) - 0/1 for neg/pos

    # Point prompts (optional)
    geo_points: Tensor | None = None         # (N_prompts, max_P, 2) - (x, y)
    geo_points_mask: Tensor | None = None    # (N_prompts, max_P) - True=padding
    geo_point_labels: Tensor | None = None   # (N_prompts, max_P) - 0/1 for neg/pos

    # Ground Truth - normalized xyxy (training)
    gt_boxes2d: Tensor | None = None         # (N_prompts, max_gt, 4) - xyxy
    gt_boxes3d: Tensor | None = None         # (N_prompts, max_gt, 12) - 3D params
    num_gts: Tensor | None = None            # (N_prompts,) - number of GTs per prompt
    gt_category_ids: Tensor | None = None    # (N_prompts, max_gt)

    # Query type tracking (following SAM3 TEXT_ID convention)
    # 0 = TEXT (one-to-many), 2 = GEOMETRY (one-to-one)
    query_types: Tensor | None = None        # (N_prompts,) int

    # Metadata for evaluation/visualization
    sample_names: List[str] | None = None    # (B_images,) - image identifiers
    dataset_name: List[str] | None = None    # (B_images,) - dataset names for evaluator
    original_hw: List[tuple] | None = None   # (B_images,) - original (H, W) per image
    original_images: Tensor | None = None    # (B_images, 3, H_orig, W_orig) - unresized
    original_intrinsics: Tensor | None = None  # (B_images, 3, 3) - intrinsics before resize

    # Depth Ground Truth (for geometry backend supervision)
    depth_gt: Tensor | None = None            # (B_images, 1, H, W) depth map
    depth_mask: Tensor | None = None          # (B_images, H, W) valid depth mask

    # Key aliases for vis4d DataConnector compatibility
    # Maps expected DataLoader keys to actual dataclass field names
    _KEY_ALIASES = {
        # Target boxes (for loss computation)
        "boxes2d": "gt_boxes2d",
        "boxes3d": "gt_boxes3d",
        "boxes2d_classes": "gt_category_ids",
        # Geometric prompts (for SAM3 input)
        "prompt_boxes": "geo_boxes",
        "prompt_box_labels": "geo_box_labels",
        # Not available in per-prompt batch
        "depth_maps": None,
        "original_hw": None,
        "original_images": None,
        "padding": None,
    }

    def __getitem__(self, key: str):
        """Support dict-like access for vis4d data connector compatibility.

        Supports both actual field names and aliased keys from raw DataLoader.
        """
        # Check alias first
        if key in self._KEY_ALIASES:
            aliased_key = self._KEY_ALIASES[key]
            if aliased_key is None:
                return None  # Field not available
            return getattr(self, aliased_key)

        # Handle special computed fields
        if key == "input_hw":
            # Return (H, W) from images shape
            return (self.images.shape[2], self.images.shape[3])

        # Direct field access
        if hasattr(self, key):
            return getattr(self, key)

        # Return None for unknown keys instead of raising error
        return None

    def keys(self):
        """Return field names for dict-like iteration."""
        return [f.name for f in fields(self)]

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for dict-like access."""
        return hasattr(self, key)

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


class SAM3_3D(nn.Module):
    """SAM3 with 3D Detection Head.
    
    This model combines:
    1. SAM3's backbone, encoder, decoder (for 2D detection with geometric prompting)
    2. 3D-MOOD's geometry backend (depth estimation)
    3. 3D-MOOD's 3D head (3D box regression)
    
    Architecture:
    ```
    Image + Prompts
         │
         ▼
    ┌─────────────────────────────────────────┐
    │  SAM3 (backbone + encoder + decoder)    │
    │  - ViT backbone with SimpleFPN          │
    │  - Geometry Encoder for prompts         │
    │  - Transformer Encoder/Decoder          │
    │  - Internal bbox_embed for 2D boxes     │
    └────────────┬────────────────────────────┘
                 │ hidden_states, pred_boxes (cxcywh)
                 │
         ┌───────┴───────┐
         ▼               ▼
    ┌──────────┐   ┌──────────────┐
    │ cxcywh   │   │ Geometry     │
    │ → xyxy   │   │ Backend      │
    └────┬─────┘   │ (depth)      │
         │         └──────┬───────┘
         │                │ depth_latents
         ▼                ▼
    ┌──────────────────────────────┐
    │  3D Head                      │
    │  (depth + ray cross-attention)│
    └──────────────┬───────────────┘
                   │
                   ▼
              pred_boxes_3d
    ```
    """
    
    def __init__(
        self,
        # ========== SAM3 Components ==========
        sam3_model: Sam3Image | None = None,
        sam3_checkpoint: str | None = None,

        # ========== 3D-MOOD Components ==========
        bbox3d_head: GroundingDINO3DHead | None = None,
        box_coder: GroundingDINO3DCoder | None = None,
        geometry_backend: GeometryBackendBase | None = None,
        roi2det3d: RoI2Det3D | None = None,

        # ========== Depth-Memory Fusion ==========
        early_depth_fusion: nn.Module | None = None,
    ) -> None:
        """Initialize SAM3_3D.

        Args:
            sam3_model: Complete SAM3 model (backbone + encoder + decoder).
                If None, will be built from sam3_checkpoint.
            sam3_checkpoint: Path to SAM3 checkpoint. Only used if sam3_model is None.
            bbox3d_head: 3D box regression head. If None, creates default.
            box_coder: 3D box encoder/decoder. If None, creates default.
            geometry_backend: Depth estimation backend. If None, no depth.
            roi2det3d: Inference post-processor. If None, creates default.
            early_depth_fusion: Early fusion module (Backbone后、Encoder前).
                If None, no early fusion is performed.

        Note:
            Learning rate control is handled by param_groups in optimizer config,
            not by freezing parameters here. See opendet3d/zoo/sam3_3d/base/optim.py
            for SAM3_3D-specific param_groups.
        """
        super().__init__()

        # SAM3 model - build if not provided
        if sam3_model is None:
            import os
            from sam3.model_builder import build_sam3_image_model

            # Check if torch.compile should be enabled for SAM3
            use_compile = os.environ.get("SAM3_COMPILE", "0") == "1"
            if use_compile:
                print("[SAM3_3D] torch.compile ENABLED for SAM3 backbone (SAM3_COMPILE=1)")
            else:
                print("[SAM3_3D] torch.compile disabled (set SAM3_COMPILE=1 to enable)")

            print(f"Building SAM3 model from checkpoint: {sam3_checkpoint}")
            sam3_model = build_sam3_image_model(
                checkpoint_path=sam3_checkpoint,
                load_from_HF=(sam3_checkpoint is None),  # Only load from HF if no checkpoint provided
                device="cpu",  # Will be moved to correct device later
                eval_mode=False,  # Must be False to enable matcher for training
                enable_segmentation=False,  # Skip seg head for 3D detection (saves ~4GB memory)
                compile=use_compile,  # Enable torch.compile for backbone
            )
            # Store checkpoint path for logging in on_load_checkpoint
            self._sam3_checkpoint_path = sam3_checkpoint
        else:
            self._sam3_checkpoint_path = "provided_model"

        self.sam3 = sam3_model
        self.hidden_dim = sam3_model.hidden_dim

        # 3D-MOOD components
        self.box_coder = box_coder or GroundingDINO3DCoder()
        self.geometry_backend = geometry_backend
        self.roi2det3d = roi2det3d
        self.early_depth_fusion = early_depth_fusion

        # Determine use_camera_prompt based on geometry_backend.is_ray_aware
        # Ray-aware backends (UniDepthV2, DetAny3D) already fuse ray info into depth_latents,
        # so we don't need the separate ray_embeddings (camera prompt) branch.
        if self.geometry_backend is not None and hasattr(self.geometry_backend, 'is_ray_aware'):
            use_camera_prompt = not self.geometry_backend.is_ray_aware
            print(f"[SAM3_3D] geometry_backend.is_ray_aware={self.geometry_backend.is_ray_aware}, use_camera_prompt={use_camera_prompt}")
        else:
            use_camera_prompt = True  # Default to True for safety
            print(f"[SAM3_3D] No geometry_backend or is_ray_aware attr, defaulting use_camera_prompt=True")

        # Get depth_latent_dim from geometry_backend (for 3D head)
        if self.geometry_backend is not None and hasattr(self.geometry_backend, 'target_latent_dim'):
            depth_latent_dim = self.geometry_backend.target_latent_dim
        else:
            depth_latent_dim = 256  # Default

        # Create or validate bbox3d_head with correct use_camera_prompt setting
        if bbox3d_head is not None:
            self.bbox3d_head = bbox3d_head
            # Warn if provided head has mismatched use_camera_prompt
            if hasattr(bbox3d_head, 'use_camera_prompt') and bbox3d_head.use_camera_prompt != use_camera_prompt:
                print(f"[SAM3_3D] Warning: bbox3d_head.use_camera_prompt={bbox3d_head.use_camera_prompt} "
                      f"but geometry_backend suggests use_camera_prompt={use_camera_prompt}")
        else:
            self.bbox3d_head = GroundingDINO3DHead(
                embed_dims=self.hidden_dim,
                box_coder=self.box_coder,
                use_camera_prompt=use_camera_prompt,
                depth_latent_dim=depth_latent_dim,
            )
            print(f"[SAM3_3D] Created bbox3d_head with use_camera_prompt={use_camera_prompt}, depth_latent_dim={depth_latent_dim}")

        # Load geometry backend pretrained weights
        # This is called during __init__ to ensure weights are loaded for first training
        # (on_load_checkpoint is only called when resuming from checkpoint)
        if self.geometry_backend is not None and hasattr(self.geometry_backend, 'load_pretrained_weights'):
            print("[SAM3_3D] Loading geometry backend pretrained weights...")
            self.geometry_backend.load_pretrained_weights()

        # Ensure SAM3 has a matcher for training
        # SAM3 built with eval_mode=True doesn't have a matcher, so we create one
        # Using BinaryHungarianMatcherV2 with focal=True to match SAM3 original config
        if self.sam3.matcher is None:
            from sam3.train.matcher import BinaryHungarianMatcherV2
            print("[SAM3_3D] Creating BinaryHungarianMatcherV2 for training...")
            self.sam3.matcher = BinaryHungarianMatcherV2(
                cost_class=2.0,  # SAM3 original
                cost_bbox=5.0,   # SAM3 original
                cost_giou=2.0,   # SAM3 original
                focal=True,      # SAM3 original
                alpha=0.25,      # SAM3 original
                gamma=2.0,       # SAM3 original
            )

    def _xyxy_to_cxcywh(self, boxes: Tensor) -> Tensor:
        """Convert boxes from xyxy to cxcywh format.

        Args:
            boxes: Tensor of shape (..., 4) in xyxy format

        Returns:
            Tensor of shape (..., 4) in cxcywh format
        """
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack([cx, cy, w, h], dim=-1)

    def _build_find_target(self, batch: SAM3_3DBatchedInputs) -> BatchedFindTarget:
        """Convert SAM3_3DBatchedInputs GT to SAM3's BatchedFindTarget format.

        This is used for SAM3's internal matching during training.

        SAM3 expects:
        - boxes: (N_total, 4) packed cxcywh normalized
        - boxes_padded: (N_prompts, max_gt, 4) padded cxcywh
        - num_boxes: (N_prompts,) number of GT per prompt
        - is_exhaustive: (N_prompts,) bool

        Note: In SAM3_3D, each prompt corresponds to exactly one GT box,
        so gt_boxes2d has shape (N_prompts, 4) not (N_prompts, max_gt, 4).

        Args:
            batch: SAM3_3DBatchedInputs with gt_boxes2d in normalized xyxy

        Returns:
            BatchedFindTarget for SAM3's _compute_matching
        """
        device = batch.gt_boxes2d.device
        gt_boxes_xyxy = batch.gt_boxes2d

        # Handle different input shapes
        # Case 1: (N_prompts, 4) - one GT per prompt (SAM3_3D design)
        # Case 2: (N_prompts, max_gt, 4) - multiple GTs per prompt (general case)
        if gt_boxes_xyxy.dim() == 2:
            # Shape: (N_prompts, 4) - one GT per prompt
            N_prompts = gt_boxes_xyxy.shape[0]
            max_gt = 1

            # Convert xyxy -> cxcywh
            gt_boxes_cxcywh = self._xyxy_to_cxcywh(gt_boxes_xyxy)  # (N_prompts, 4)

            # Each prompt has exactly 1 GT box
            num_boxes = torch.ones(N_prompts, dtype=torch.long, device=device)

            # Packed boxes = all boxes (no padding)
            boxes_packed = gt_boxes_cxcywh  # (N_prompts, 4)

            # Padded format: add max_gt dimension
            gt_boxes_cxcywh_padded = gt_boxes_cxcywh.unsqueeze(1)  # (N_prompts, 1, 4)

            # Object IDs: sequential
            object_ids = torch.arange(N_prompts, device=device)
            object_ids_padded = torch.arange(N_prompts, device=device).unsqueeze(1)  # (N_prompts, 1)

        else:
            # Shape: (N_prompts, max_gt, 4) - multiple GTs per prompt
            N_prompts = gt_boxes_xyxy.shape[0]
            max_gt = gt_boxes_xyxy.shape[1]

            # Convert xyxy -> cxcywh
            gt_boxes_cxcywh = self._xyxy_to_cxcywh(gt_boxes_xyxy)

            # Compute num_boxes per prompt (count non-zero boxes)
            valid_mask = (gt_boxes_xyxy.abs().sum(dim=-1) > 1e-6)  # (N_prompts, max_gt)
            num_boxes = valid_mask.sum(dim=-1)  # (N_prompts,)

            # Pack boxes (remove padding)
            boxes_list = []
            for i in range(N_prompts):
                n = int(num_boxes[i].item())
                if n > 0:
                    boxes_list.append(gt_boxes_cxcywh[i, :n])
            if boxes_list:
                boxes_packed = torch.cat(boxes_list, dim=0)  # (N_total, 4)
            else:
                boxes_packed = torch.zeros(0, 4, device=device)

            gt_boxes_cxcywh_padded = gt_boxes_cxcywh

            # Object IDs (placeholder - just sequential)
            object_ids = torch.arange(len(boxes_packed), device=device)
            object_ids_padded = torch.full(
                (N_prompts, max_gt), -1, device=device, dtype=torch.long
            )
            offset = 0
            for i in range(N_prompts):
                n = int(num_boxes[i].item())
                if n > 0:
                    object_ids_padded[i, :n] = torch.arange(
                        offset, offset + n, device=device
                    )
                    offset += n

        return BatchedFindTarget(
            num_boxes=num_boxes,
            boxes=boxes_packed,
            boxes_padded=gt_boxes_cxcywh_padded,
            repeated_boxes=None,
            segments=None,
            semantic_segments=None,
            is_valid_segment=None,
            is_exhaustive=torch.ones(N_prompts, dtype=torch.bool, device=device),
            object_ids=object_ids,
            object_ids_padded=object_ids_padded,
        )

    def on_load_checkpoint(self, checkpoint):
        """
        PyTorch Lightning hook called when loading a checkpoint.

        This is called BEFORE load_state_dict, so we can:
        1. Load SAM3 pretrained weights first (if first training)
        2. Load geometry backend pretrained weights first (if first training)
        3. Filter out incompatible keys from the checkpoint
        4. Let PyTorch Lightning load the filtered checkpoint
        """
        print("\n" + "="*80)
        print("📦 SAM3_3D CHECKPOINT LOADING (PyTorch Lightning Hook)")
        print("="*80)

        # Get the state_dict from checkpoint
        state_dict = checkpoint.get('state_dict', {})

        # Analyze checkpoint content
        has_sam3 = any('sam3.' in key for key in state_dict.keys())
        has_geometry_backend = any('geometry_backend' in key for key in state_dict.keys())
        has_bbox3d_head = any('bbox3d_head' in key for key in state_dict.keys())

        # Determine if this is resume training or first training
        is_resume = has_sam3 and has_geometry_backend

        if is_resume:
            # Resume training: load everything from checkpoint
            print("\n📌 Mode: Resume Training")
            print("Loading complete checkpoint (all components)")
            print(f"  Resuming from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"  Resuming from global_step {checkpoint.get('global_step', 'unknown')}")

        else:
            # First training: load pretrained weights
            print("\n📌 Mode: First Training (Fine-tuning)")

            # Step 1: Load SAM3 pretrained weights (if not already loaded in __init__)
            if not has_sam3 and self.sam3 is not None:
                print("\n[Step 1/3] SAM3 weights already loaded in __init__")
                print(f"  SAM3 checkpoint: {getattr(self, '_sam3_checkpoint_path', 'unknown')}")

            # Step 2: Load geometry backend pretrained weights
            if self.geometry_backend is not None and hasattr(self.geometry_backend, 'load_pretrained_weights'):
                print("\n[Step 2/3] Loading geometry backend pretrained weights...")
                self.geometry_backend.load_pretrained_weights()

            # Step 3: Filter checkpoint if needed
            print("\n[Step 3/3] Processing checkpoint...")
            if not has_sam3:
                print("  No SAM3 weights in checkpoint (will use pretrained SAM3)")
            if not has_geometry_backend:
                print("  No geometry_backend weights in checkpoint (will use pretrained)")
            if not has_bbox3d_head:
                print("  No bbox3d_head weights in checkpoint (will initialize randomly)")

            # Step 4: Reset training state (epoch, step, optimizer)
            print("\n[Step 4/4] Resetting training state for fine-tuning...")
            if 'epoch' in checkpoint:
                old_epoch = checkpoint['epoch']
                checkpoint['epoch'] = 0
                print(f"  Reset epoch: {old_epoch} → 0")

            if 'global_step' in checkpoint:
                old_step = checkpoint['global_step']
                checkpoint['global_step'] = 0
                print(f"  Reset global_step: {old_step} → 0")

            # Remove optimizer states (they won't match our new optimizer config)
            if 'optimizer_states' in checkpoint:
                del checkpoint['optimizer_states']
                print(f"  Removed optimizer_states (will initialize fresh)")

            # Remove lr_scheduler states
            if 'lr_schedulers' in checkpoint:
                del checkpoint['lr_schedulers']
                print(f"  Removed lr_schedulers (will initialize fresh)")

        # Store resume status for later use
        self._is_resume_training = is_resume

        print("\n" + "="*80)
        print("✅ Checkpoint loading hook completed")
        print("="*80 + "\n")

    def forward(
        self,
        batch: SAM3_3DBatchedInputs,
        targets: dict | None = None,
    ) -> SAM3_3DOut:
        """Forward pass of SAM3_3D using SAM3's forward_grounding.

        This method reuses SAM3's complete 2D detection pipeline and adds
        3D detection on top.

        Args:
            batch: SAM3_3DBatchedInputs containing:
                - images: (B_images, 3, H, W)
                - intrinsics: (B_images, 3, 3)
                - img_ids: (N_prompts,) - which image each prompt belongs to
                - text_ids: (N_prompts,) - text index per prompt
                - unique_texts: List[str] - all unique texts
                - geo_boxes: (N_prompts, max_K, 4) - normalized cxcywh
                - geo_boxes_mask: (N_prompts, max_K) - True=padding
                - geo_box_labels: (N_prompts, max_K) - 0/1 for neg/pos
            targets: Training targets (optional)

        Returns:
            SAM3_3DOut with 2D and 3D predictions
        """
        B_images = batch.images.shape[0]
        N_prompts = len(batch.img_ids)
        _, _, H, W = batch.images.shape
        device = batch.images.device

        profile_start("forward_total")

        # Sync SAM3 training mode with parent module
        # This is important because SAM3's forward_grounding only computes
        # matching indices when self.training is True
        if self.sam3.training != self.training:
            self.sam3.train(self.training)

        # Handle empty batch (no prompts)
        if N_prompts == 0:
            if self.training:
                # Create dummy output connected to ALL model parameters for DDP backward
                # DDP requires all parameters to participate in backward across all ranks
                # Using only one parameter causes deadlock when other ranks use all params
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                print(f"[SAM3_3D] Empty batch (N_prompts=0) on rank {rank}, using all-param dummy")
                dummy_grad = sum(p.sum() * 0 for p in self.parameters() if p.requires_grad)
                dummy_logits = torch.zeros(1, 1, 1, device=device) + dummy_grad
                return SAM3_3DOut(
                    pred_logits=dummy_logits,
                    pred_boxes_2d=torch.zeros(1, 1, 4, device=device),
                    pred_boxes_3d=None,
                    aux_outputs=None,
                    geom_losses=None,
                    presence_logits=None,
                    queries=None,
                    encoder_hidden_states=None,
                    indices=None,
                )
            else:
                # Test mode: return empty Det3DOut
                return Det3DOut(
                    boxes=[torch.zeros(0, 4, device=device) for _ in range(B_images)],
                    boxes3d=[torch.zeros(0, 10, device=device) for _ in range(B_images)],
                    scores=[torch.zeros(0, device=device) for _ in range(B_images)],
                    class_ids=[torch.zeros(0, dtype=torch.long, device=device) for _ in range(B_images)],
                    depth_maps=None,
                    categories=None,
                )

        # ========== Step 1 & 2: SAM3 Backbone + Geometry Backend (PARALLEL) ==========
        # These two operations are independent - run them in parallel using CUDA streams
        profile_start("  backbone+geom_parallel")

        # Convert images for SAM3 (needed by backbone)
        images_for_sam3 = self._convert_imagenet_to_sam3_norm(batch.images)

        # Prepare geometry backend inputs
        geom_losses = None
        depth_latents = None
        geom_out = None
        _, _, H, W = batch.images.shape

        if self.geometry_backend is not None:
            # Create CUDA streams for parallel execution
            backbone_stream = torch.cuda.Stream()
            geom_stream = torch.cuda.Stream()

            # Prepare inputs for geometry backend (before streams)
            intrinsics_per_image = batch.intrinsics
            depth_gt = None
            depth_mask = None
            if self.training:
                depth_gt = getattr(batch, 'depth_gt', None)
                depth_mask = getattr(batch, 'depth_mask', None)

            # Run backbone on stream 1
            profile_start("  backbone")
            with torch.cuda.stream(backbone_stream):
                backbone_out = {"img_batch_all_stages": batch.images}
                backbone_out.update(self.sam3.backbone.forward_image(images_for_sam3))
                text_out = self.sam3.backbone.forward_text(
                    batch.unique_texts, device=device
                )
                backbone_out.update(text_out)

            # Run geometry backend on stream 2 (parallel with backbone)
            profile_start("  geometry_backend")
            with torch.cuda.stream(geom_stream):
                geom_out = self.geometry_backend(
                    images=batch.images,
                    depth_feats=None,  # Not using backbone features
                    intrinsics=intrinsics_per_image,
                    image_hw=(H, W),
                    depth_gt=depth_gt,
                    depth_mask=depth_mask,
                )

            # Wait for both streams to complete
            backbone_stream.synchronize()
            profile_stop("  backbone")
            geom_stream.synchronize()
            profile_stop("  geometry_backend")

            # Extract geometry outputs
            depth_latents = geom_out.get("depth_latents")
            if self.training:
                geom_losses = geom_out.get("losses", {})
        else:
            # No geometry backend - just run backbone
            profile_start("  backbone")
            backbone_out = {"img_batch_all_stages": batch.images}
            backbone_out.update(self.sam3.backbone.forward_image(images_for_sam3))
            text_out = self.sam3.backbone.forward_text(
                batch.unique_texts, device=device
            )
            backbone_out.update(text_out)
            profile_stop("  backbone")

        profile_stop("  backbone+geom_parallel")

        # ========== Step 2.5: Early Depth Fusion (Backbone后、Encoder前) ==========
        # Fuse depth_latents into backbone visual features before encoder
        # This allows depth information to participate in encoder's self-attention
        # and text cross-attention
        if self.early_depth_fusion is not None and depth_latents is not None:
            # Get depth_latents spatial dimensions from geometry backend output
            aux = geom_out.get("aux", {})
            depth_latents_hw = aux.get("depth_latents_hw")

            if depth_latents_hw is not None and "backbone_fpn" in backbone_out:
                # Fuse depth into visual features
                backbone_fpn = backbone_out["backbone_fpn"]

                # early_depth_fusion expects list of visual features
                if not isinstance(backbone_fpn, list):
                    backbone_fpn = [backbone_fpn]

                # Perform fusion
                fused_fpn = self.early_depth_fusion(
                    visual_feats=backbone_fpn,
                    depth_latents=depth_latents,
                    depth_latents_hw=depth_latents_hw,
                )

                # Update backbone_out with fused features
                # SAM3 will use these fused features in encoder
                if len(fused_fpn) == 1:
                    backbone_out["backbone_fpn"] = fused_fpn[0]
                else:
                    backbone_out["backbone_fpn"] = fused_fpn
            else:
                # Warn user that early depth fusion is configured but cannot run
                import warnings
                if depth_latents_hw is None:
                    warnings.warn(
                        "EarlyDepthFusion is configured but depth_latents_hw not "
                        "provided by geometry backend. Skipping depth fusion. "
                        "Check geometry backend outputs include 'aux.depth_latents_hw'.",
                        UserWarning,
                    )
                elif "backbone_fpn" not in backbone_out:
                    warnings.warn(
                        "EarlyDepthFusion is configured but backbone_fpn not found "
                        "in backbone outputs. Skipping depth fusion.",
                        UserWarning,
                    )

        # ========== Step 3: Build SAM3 inputs ==========
        find_input = self._build_find_stage(batch, device)
        geometric_prompt = self._build_geometric_prompt(batch, device)

        # ========== Step 4: SAM3 forward_grounding ==========
        # This does: encode_prompt -> encoder -> decoder -> score/box prediction
        #
        # In training mode, we build find_target from batch GT boxes so that
        # SAM3's internal _compute_matching can compute matching indices.
        # These indices are then used by our loss function.
        find_target = None
        if self.training:
            assert batch.gt_boxes2d is not None, \
                "Training requires GT boxes (batch.gt_boxes2d)"
            find_target = self._build_find_target(batch)

        profile_start("  sam3_grounding")
        sam3_out = self.sam3.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=find_target,
            geometric_prompt=geometric_prompt,
        )
        profile_stop("  sam3_grounding")

        # ========== Step 5: Extract SAM3 outputs ==========
        # SAM3 output format (after _update_scores_and_boxes):
        # - pred_logits: (N_prompts, num_queries, 1) - final layer
        # - pred_boxes: (N_prompts, num_queries, 4) - normalized cxcywh
        # - pred_boxes_xyxy: (N_prompts, num_queries, 4) - normalized xyxy
        # - queries: (N_prompts, num_queries, d_model) - last layer hidden states
        # - aux_outputs: list of dicts for each decoder layer (for deep supervision)
        # O2O outputs (one-to-one matching)
        pred_logits = sam3_out["pred_logits"]  # (N_prompts, S, 1)
        pred_boxes_xyxy = sam3_out["pred_boxes_xyxy"]  # (N_prompts, S, 4)
        queries = sam3_out.get("queries")  # (N_prompts, S, d_model)
        encoder_hidden_states = sam3_out.get("encoder_hidden_states")
        presence_logits = sam3_out.get("presence_logit_dec")

        # O2M outputs (one-to-many matching) from SAM3 DAC mechanism
        # These are separate outputs from the second half of queries in DAC mode
        pred_logits_o2m = sam3_out.get("pred_logits_o2m")  # (N_prompts, S, 1)
        pred_boxes_xyxy_o2m = sam3_out.get("pred_boxes_xyxy_o2m")  # (N_prompts, S, 4)
        queries_o2m = sam3_out.get("queries_o2m")  # (N_prompts, S, d_model)

        # Extract auxiliary outputs from SAM3 for deep supervision
        sam3_aux_outputs = sam3_out.get("aux_outputs", [])

        # ========== Step 6: 3D Head ==========
        profile_start("  3d_head")
        pred_boxes_3d = None
        aux_outputs = None

        if self.bbox3d_head is not None and queries is not None:
            # Generate ray embeddings if camera prompt is enabled
            # For ray-aware backends (UniDepthV2, DetAny3D), depth_latents already
            # contain ray info, so we can either use camera prompt or skip it
            ray_embeddings = None
            if self.bbox3d_head.use_camera_prompt:
                # Get ray parameters from geometry backend output
                if geom_out is not None:
                    # Use backend's ray parameters for consistent space
                    ray_intrinsics = geom_out.get("ray_intrinsics", batch.intrinsics)
                    ray_image_hw = geom_out.get("ray_image_hw", (H, W))
                    ray_downsample = geom_out.get("ray_downsample", 16)
                else:
                    # Fallback: use image-level intrinsics with default downsample
                    # Note: This will broadcast to all prompts, not per-prompt
                    ray_intrinsics = batch.intrinsics
                    ray_image_hw = (H, W)
                    ray_downsample = 16  # Default

                ray_embeddings = self.bbox3d_head.get_camera_embeddings(
                    ray_intrinsics, ray_image_hw, ray_downsample
                )

            # Align depth_latents and ray_embeddings spatial resolution (if needed)
            #
            # Note: This code only runs when use_camera_prompt=True (i.e., for non-ray-aware
            # backends like UniDepthHead v1). For ray-aware backends (UniDepthV2, DetAny3D),
            # use_camera_prompt=False and ray_embeddings=None, so this block is skipped.
            #
            # When this does run, depth_latents and ray_embeddings may have different spatial
            # resolutions that need to be aligned for the 3D head's cross-attention.
            if depth_latents is not None and ray_embeddings is not None:
                # depth_latents: [B_images, N_depth, C_depth]
                # ray_embeddings: [B_images, N_ray, C_ray]
                B_depth, N_depth, C_depth = depth_latents.shape
                B_ray, N_ray, C_ray = ray_embeddings.shape

                if N_depth != N_ray:
                    # Resize depth_latents to match ray spatial size
                    # Infer spatial dimensions (assuming square)
                    H_depth = int(N_depth ** 0.5)
                    W_depth = H_depth
                    H_ray = int(N_ray ** 0.5)
                    W_ray = H_ray

                    # Reshape depth_latents: [B, N, C] -> [B, C, H, W]
                    depth_latents_2d = depth_latents.permute(0, 2, 1).reshape(
                        B_depth, C_depth, H_depth, W_depth
                    )

                    # Adaptive pool to ray size
                    depth_latents_resized = torch.nn.functional.adaptive_avg_pool2d(
                        depth_latents_2d, (H_ray, W_ray)
                    )

                    # Reshape back: [B, C, H, W] -> [B, N, C]
                    depth_latents = depth_latents_resized.reshape(
                        B_depth, C_depth, H_ray * W_ray
                    ).permute(0, 2, 1)

            # Index ray_embeddings and depth_latents from per-image to per-prompt
            # ray_embeddings and depth_latents are per-image [B_images, N, C]
            # But 3D head expects them to be per-prompt [N_prompts, N, C]
            # Use batch.img_ids to correctly map prompts to their corresponding images
            if ray_embeddings is not None:
                # batch.img_ids: [N_prompts] - which image each prompt belongs to
                # ray_embeddings: [B_images, N, C]
                # Index to get: [N_prompts, N, C]
                ray_embeddings = ray_embeddings[batch.img_ids]

            if depth_latents is not None:
                # depth_latents: [B_images, N, C]
                # Index to get: [N_prompts, N, C]
                depth_latents = depth_latents[batch.img_ids]

            # ========== Deep Supervision: Process all decoder layers ==========
            # Following SAM3's design, we process auxiliary outputs from all decoder layers
            # for deep supervision during training
            #
            # SAM3's output structure:
            # - aux_outputs[0..L-2]: intermediate decoder layers (layer 0 to layer L-2)
            # - final output (pred_logits, queries, etc.): final decoder layer (layer L-1)

            # Collect all layers' queries in correct order: [layer0, layer1, ..., layerL-1]
            # Track which aux_outputs have queries for building aux_outputs later
            all_layers_queries = []
            aux_indices_with_queries = []  # Track original indices of aux_outputs with queries
            for i, aux_out in enumerate(sam3_aux_outputs):
                aux_queries = aux_out.get("queries")
                if aux_queries is not None:
                    all_layers_queries.append(aux_queries)
                    aux_indices_with_queries.append(i)
            all_layers_queries.append(queries)  # Final layer at the end

            # Stack to (L, N_prompts, S, C) format expected by 3D head
            if len(all_layers_queries) > 1:
                # Have auxiliary outputs - stack all layers
                hidden_states = torch.stack(all_layers_queries, dim=0)  # (L, N_prompts, S, C)
            else:
                # No auxiliary outputs - just expand final layer
                hidden_states = queries.unsqueeze(0)  # (1, N_prompts, S, C)

            # Call 3D head with all layers
            # Returns: (L, N_prompts, S, 12) if L > 1, else (N_prompts, S, 12)
            all_layers_boxes_3d = self.bbox3d_head(
                hidden_states=hidden_states,
                ray_embeddings=ray_embeddings,
                depth_latents=depth_latents,
            )

            # Extract final layer output
            if len(all_layers_queries) > 1:
                pred_boxes_3d = all_layers_boxes_3d[-1]  # (N_prompts, S, 12)
            else:
                # bbox3d_head always returns (L, N_prompts, S, 12), squeeze L dim when L=1
                pred_boxes_3d = all_layers_boxes_3d.squeeze(0)  # (N_prompts, S, 12)

            # Build auxiliary outputs for deep supervision
            # Only include layers that have queries (tracked by aux_indices_with_queries)
            if len(aux_indices_with_queries) > 0 and self.training:
                aux_outputs = []
                for layer_idx, orig_idx in enumerate(aux_indices_with_queries):
                    aux_out = sam3_aux_outputs[orig_idx]
                    aux_dict = {
                        "pred_logits": aux_out["pred_logits"],
                        "pred_boxes_2d": aux_out["pred_boxes_xyxy"],
                        "pred_boxes_3d": all_layers_boxes_3d[layer_idx],  # 3D predictions for this layer
                    }
                    # Include presence logits if available
                    if "presence_logit_dec" in aux_out:
                        aux_dict["presence_logits"] = aux_out["presence_logit_dec"]
                    aux_outputs.append(aux_dict)

        # Compute 3D boxes for O2M queries (if available, only during training)
        pred_boxes_3d_o2m = None
        if self.bbox3d_head is not None and queries_o2m is not None and self.training:
            # O2M queries use the same 3D head but only compute final layer (no aux)
            o2m_hidden_states = queries_o2m.unsqueeze(0)  # (1, N_prompts, S, C)
            o2m_boxes_3d = self.bbox3d_head(
                hidden_states=o2m_hidden_states,
                ray_embeddings=ray_embeddings,
                depth_latents=depth_latents,
            )
            pred_boxes_3d_o2m = o2m_boxes_3d.squeeze(0)  # (N_prompts, S, 12)

        profile_stop("  3d_head")

        # Training mode: return raw outputs for loss computation
        if self.training:
            # Extract matching indices from SAM3 output (computed by _compute_matching)
            sam3_indices = sam3_out.get("indices", None)

            profile_stop("forward_total")

            # Record profiling step (will print summary every N steps if enabled)
            profile_step()

            return SAM3_3DOut(
                pred_logits=pred_logits,
                pred_boxes_2d=pred_boxes_xyxy,
                pred_boxes_3d=pred_boxes_3d,
                aux_outputs=aux_outputs,
                geom_losses=geom_losses,
                presence_logits=presence_logits,
                queries=queries,
                encoder_hidden_states=encoder_hidden_states,
                indices=sam3_indices,
                # O2M outputs from SAM3 DAC mechanism
                pred_logits_o2m=pred_logits_o2m,
                pred_boxes_2d_o2m=pred_boxes_xyxy_o2m,
                pred_boxes_3d_o2m=pred_boxes_3d_o2m,
            )

        # Test mode: forward_test returns Det3DOut for evaluation
        return self._forward_test(
            pred_logits=pred_logits,
            pred_boxes_2d=pred_boxes_xyxy,
            pred_boxes_3d=pred_boxes_3d,
            presence_logits=presence_logits,
            batch=batch,
            geom_out=geom_out,
        )

    def _convert_imagenet_to_sam3_norm(self, images: Tensor) -> Tensor:
        """Convert ImageNet normalized images to SAM3 normalization.

        vis4d/3D-MOOD uses ImageNet normalization:
            ImageNet: (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            Output range: ~[-2.5, 2.5]

        SAM3 expects custom normalization:
            SAM3: (x - 0.5) / 0.5
            Output range: [-1, 1]

        This function converts from ImageNet normalized to SAM3 normalized:
            1. Denormalize ImageNet -> [0, 1]
            2. Normalize SAM3 -> [-1, 1]

        Args:
            images: ImageNet normalized images (B, 3, H, W)

        Returns:
            SAM3 normalized images (B, 3, H, W)
        """
        # ImageNet constants
        imagenet_mean = torch.tensor(
            [0.485, 0.456, 0.406], device=images.device, dtype=images.dtype
        ).view(1, 3, 1, 1)
        imagenet_std = torch.tensor(
            [0.229, 0.224, 0.225], device=images.device, dtype=images.dtype
        ).view(1, 3, 1, 1)

        # Denormalize: ImageNet normalized -> [0, 1]
        images_01 = images * imagenet_std + imagenet_mean

        # Normalize: [0, 1] -> SAM3 [-1, 1]
        images_sam3 = (images_01 - 0.5) / 0.5

        return images_sam3

    def _forward_test(
        self,
        pred_logits: Tensor,
        pred_boxes_2d: Tensor,
        pred_boxes_3d: Tensor | None,
        presence_logits: Tensor | None,
        batch: SAM3_3DBatchedInputs,
        geom_out: dict | None,
    ) -> Det3DOut:
        """Forward pass for test/inference mode.

        Postprocesses model outputs to Det3DOut format for evaluation.
        Converts per-prompt outputs to per-image outputs with:
        - Pixel coordinate boxes (scaled from normalized)
        - Decoded 3D boxes
        - Score thresholding (optional)

        Args:
            pred_logits: (N_prompts, S, 1) objectness logits
            pred_boxes_2d: (N_prompts, S, 4) normalized xyxy boxes
            pred_boxes_3d: (N_prompts, S, 12) encoded 3D params or None
            presence_logits: (N_prompts, 1) presence logits (category exists in image)
            batch: Input batch with img_ids, intrinsics, etc.
            geom_out: Geometry backend output (may contain depth_maps)

        Returns:
            Det3DOut with per-image detection results
        """
        H, W = batch.images.shape[2:]
        device = pred_logits.device
        B_images = batch.images.shape[0]

        # Get scores from logits
        scores_all = pred_logits.sigmoid().squeeze(-1)  # (N_prompts, S)

        # Apply presence score if available (following SAM3 original postprocessors.py)
        # Presence score indicates whether a category has objects in the image
        # This suppresses all proposals for categories that don't exist in the image
        # SAM3 original: presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        if presence_logits is not None:
            presence_score = presence_logits.sigmoid()
            # Ensure correct shape for broadcasting: (N_prompts, 1) or (N_prompts,) -> (N_prompts, 1)
            if presence_score.dim() == 1:
                presence_score = presence_score.unsqueeze(-1)
            scores_all = scores_all * presence_score  # (N_prompts, S) * (N_prompts, 1)

        # Scale boxes to pixel coordinates
        # pred_boxes_2d is normalized xyxy [0, 1]
        boxes_pixel = pred_boxes_2d.clone()
        boxes_pixel[..., 0::2] *= W
        boxes_pixel[..., 1::2] *= H

        # Group by image
        boxes_list = []
        boxes3d_list = []
        scores_list = []
        class_ids_list = []

        # Get parameters from roi2det3d if available
        # Default values match SAM3 ODinW config: max_dets_per_img=-1, detection_threshold=-1.0
        max_per_img = getattr(self.roi2det3d, 'max_per_img', -1) if self.roi2det3d else -1
        score_threshold = getattr(self.roi2det3d, 'score_threshold', -1.0) if self.roi2det3d else -1.0

        S = scores_all.shape[1]  # predictions per prompt

        for img_idx in range(B_images):
            # Find prompts belonging to this image
            prompt_mask = batch.img_ids == img_idx
            n_prompts_this_img = prompt_mask.sum().item()

            if n_prompts_this_img == 0:
                # No prompts for this image
                boxes_list.append(torch.zeros(0, 4, device=device))
                boxes3d_list.append(torch.zeros(0, 10, device=device))
                scores_list.append(torch.zeros(0, device=device))
                class_ids_list.append(torch.zeros(0, dtype=torch.long, device=device))
                continue

            # Get predictions for this image's prompts
            img_scores = scores_all[prompt_mask]  # (n_prompts, S)
            img_boxes = boxes_pixel[prompt_mask]  # (n_prompts, S, 4)

            # Get class IDs for each prompt
            if batch.gt_category_ids is not None:
                img_class_ids = batch.gt_category_ids[prompt_mask]  # (n_prompts,) or (n_prompts, max_gt)
                if img_class_ids.dim() > 1:
                    img_class_ids = img_class_ids[:, 0]  # Take first if multiple
            else:
                img_class_ids = torch.zeros(n_prompts_this_img, dtype=torch.long, device=device)

            # Flatten all predictions: (n_prompts, S) -> (n_prompts * S,)
            # Each prompt outputs S=100 predictions, we want ALL of them for multi-instance detection
            img_scores_flat = img_scores.flatten()  # (n_prompts * S,)
            img_boxes_flat = img_boxes.reshape(-1, 4)  # (n_prompts * S, 4)

            # Expand class_ids to match flattened shape
            # Each prompt has S predictions, all with the same class_id
            img_class_ids_flat = img_class_ids.unsqueeze(1).expand(-1, S).flatten()  # (n_prompts * S,)

            # Get 3D boxes if available (flattened)
            if pred_boxes_3d is not None:
                img_boxes3d = pred_boxes_3d[prompt_mask]  # (n_prompts, S, 12)
                img_boxes3d_flat = img_boxes3d.reshape(-1, 12)  # (n_prompts * S, 12)
            else:
                img_boxes3d_flat = None

            # No topk filtering here - keep all 100 proposals per category
            # SAM3_3D outputs S=100 proposals per prompt, each prompt is one category
            # Evaluator does per-category matching anyway, so no need to limit here
            # This ensures fair evaluation across all categories

            # Score threshold filter
            if score_threshold > 0:
                keep = img_scores_flat > score_threshold
                img_scores_flat = img_scores_flat[keep]
                img_boxes_flat = img_boxes_flat[keep]
                img_class_ids_flat = img_class_ids_flat[keep]
                if img_boxes3d_flat is not None:
                    img_boxes3d_flat = img_boxes3d_flat[keep]

            # Rescale 2D boxes from resize space (H, W) to original image space
            # GDino3D does this in RoI2Det3D.__call__ (line 396): det_bboxes /= scales
            # This is CRITICAL for correct IoU computation in evaluation!
            if batch.original_hw is not None and batch.original_hw[img_idx] is not None:
                orig_h, orig_w = batch.original_hw[img_idx]
                scale_x = W / orig_w
                scale_y = H / orig_h
                img_boxes_flat = img_boxes_flat.clone()  # Don't modify in-place
                img_boxes_flat[:, 0::2] /= scale_x  # x coordinates
                img_boxes_flat[:, 1::2] /= scale_y  # y coordinates

            # Decode 3D boxes AFTER filtering (more efficient)
            # Use original intrinsics for 3D decoding if available
            if img_boxes3d_flat is not None and self.box_coder is not None and len(img_boxes_flat) > 0:
                if batch.original_intrinsics is not None:
                    intrinsics_this_img = batch.original_intrinsics[img_idx]  # (3, 3)
                else:
                    intrinsics_this_img = batch.intrinsics[img_idx]  # (3, 3)
                decoded_boxes3d = self.box_coder.decode(
                    img_boxes_flat,  # pixel xyxy in original image space
                    img_boxes3d_flat,
                    intrinsics_this_img,
                )
            else:
                decoded_boxes3d = torch.zeros(len(img_boxes_flat), 10, device=device)

            boxes_list.append(img_boxes_flat)
            boxes3d_list.append(decoded_boxes3d)
            scores_list.append(img_scores_flat)
            class_ids_list.append(img_class_ids_flat)

        # Get depth maps if available
        depth_maps = None
        if geom_out is not None and "depth" in geom_out:
            depth_maps = [geom_out["depth"][i] for i in range(B_images)]

        # Get predicted intrinsics if available
        predicted_intrinsics = None
        if geom_out is not None and "K_pred" in geom_out:
            predicted_intrinsics = geom_out["K_pred"]

        return Det3DOut(
            boxes=boxes_list,
            boxes3d=boxes3d_list,
            scores=scores_list,
            class_ids=class_ids_list,
            depth_maps=depth_maps,
            categories=None,
            predicted_intrinsics=predicted_intrinsics,
        )

    def _build_find_stage(
        self,
        batch: SAM3_3DBatchedInputs,
        device: torch.device,
    ) -> FindStage:
        """Convert SAM3_3DBatchedInputs to SAM3's FindStage format.

        FindStage is SAM3's internal representation for per-prompt batch,
        containing img_ids, text_ids, and geometry inputs.
        """
        N_prompts = len(batch.img_ids)

        # Prepare geometry inputs - need to convert to sequence-first
        # FindStage expects (max_K, N_prompts, 4) for boxes
        if batch.geo_boxes is not None:
            # (N_prompts, max_K, 4) -> (max_K, N_prompts, 4)
            input_boxes = batch.geo_boxes.permute(1, 0, 2)
            input_boxes_mask = batch.geo_boxes_mask  # (N_prompts, max_K)
            input_boxes_label = (
                batch.geo_box_labels.permute(1, 0)
                if batch.geo_box_labels is not None
                else torch.ones(
                    input_boxes.shape[0], N_prompts, dtype=torch.long, device=device
                )
            )
        else:
            # No geometry input - create empty tensors
            input_boxes = torch.zeros(0, N_prompts, 4, device=device)
            input_boxes_mask = torch.ones(N_prompts, 0, dtype=torch.bool, device=device)
            input_boxes_label = torch.zeros(0, N_prompts, dtype=torch.long, device=device)

        # Points (if any)
        if batch.geo_points is not None:
            input_points = batch.geo_points.permute(1, 0, 2)  # (max_P, N, 2)
            input_points_mask = batch.geo_points_mask
        else:
            input_points = torch.zeros(0, N_prompts, 2, device=device)
            input_points_mask = torch.ones(N_prompts, 0, dtype=torch.bool, device=device)

        return FindStage(
            img_ids=batch.img_ids,
            text_ids=batch.text_ids,
            input_boxes=input_boxes,
            input_boxes_mask=input_boxes_mask,
            input_boxes_label=input_boxes_label,
            input_points=input_points,
            input_points_mask=input_points_mask,
            object_ids=None,
        )

    def _build_geometric_prompt(
        self,
        batch: SAM3_3DBatchedInputs,
        device: torch.device,
    ) -> Prompt:
        """Build SAM3 Prompt object from batch.

        SAM3's Prompt class expects sequence-first format: (K, N_prompts, dim)
        """
        N_prompts = len(batch.img_ids)

        # Box prompts
        if batch.geo_boxes is not None and batch.geo_boxes.shape[1] > 0:
            # (N_prompts, max_K, 4) -> (max_K, N_prompts, 4)
            box_embeddings = batch.geo_boxes.permute(1, 0, 2)
            box_mask = batch.geo_boxes_mask  # (N_prompts, max_K)
            box_labels = (
                batch.geo_box_labels.permute(1, 0)
                if batch.geo_box_labels is not None
                else torch.ones(
                    box_embeddings.shape[0], N_prompts, dtype=torch.long, device=device
                )
            )
        else:
            box_embeddings = None
            box_mask = None
            box_labels = None

        # Point prompts
        if batch.geo_points is not None and batch.geo_points.shape[1] > 0:
            point_embeddings = batch.geo_points.permute(1, 0, 2)  # (max_P, N, 2)
            point_mask = batch.geo_points_mask
            point_labels = (
                batch.geo_point_labels.permute(1, 0)
                if batch.geo_point_labels is not None
                else torch.ones(
                    point_embeddings.shape[0], N_prompts, dtype=torch.long, device=device
                )
            )
        else:
            # For text-only mode: create empty tensors instead of None
            # SAM3's geometry encoder cannot handle None for points
            point_embeddings = torch.zeros(0, N_prompts, 2, device=device)
            point_mask = torch.ones(N_prompts, 0, dtype=torch.bool, device=device)
            point_labels = torch.zeros(0, N_prompts, dtype=torch.long, device=device)

        # Ensure box prompts also have empty tensors if None
        if box_embeddings is None:
            box_embeddings = torch.zeros(0, N_prompts, 4, device=device)
            box_mask = torch.ones(N_prompts, 0, dtype=torch.bool, device=device)
            box_labels = torch.zeros(0, N_prompts, dtype=torch.long, device=device)

        return Prompt(
            box_embeddings=box_embeddings,
            box_mask=box_mask,
            box_labels=box_labels,
            point_embeddings=point_embeddings,
            point_mask=point_mask,
            point_labels=point_labels,
        )

    @torch.no_grad()
    def inference(
        self,
        batch: SAM3_3DBatchedInputs,
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
    ) -> list[dict]:
        """Run inference and decode 3D boxes.

        Args:
            batch: SAM3_3DBatchedInputs with images and prompts
            score_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold

        Returns:
            List of dicts per image with decoded 3D boxes
        """
        self.eval()

        out = self.forward(batch)

        if self.roi2det3d is None or out.pred_boxes_3d is None:
            return self._decode_2d_only(out, batch.img_ids, score_threshold)

        # Decode 3D boxes using roi2det3d
        H, W = batch.images.shape[2:]
        intrinsics_per_prompt = batch.intrinsics[batch.img_ids]
        results = self.roi2det3d(
            pred_logits=out.pred_logits,
            pred_boxes_2d=out.pred_boxes_2d,
            pred_boxes_3d=out.pred_boxes_3d,
            intrinsics=intrinsics_per_prompt,
            image_size=(H, W),
            img_ids=batch.img_ids,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        )
        return results

    def _decode_2d_only(
        self,
        out: SAM3_3DOut,
        img_ids: Tensor,
        score_threshold: float,
    ) -> list[dict]:
        """Decode 2D-only results when 3D head is not available."""
        scores = out.pred_logits.sigmoid().squeeze(-1)  # (N_prompts, S)
        boxes = out.pred_boxes_2d  # (N_prompts, S, 4) normalized xyxy

        results = []
        unique_img_ids = img_ids.unique()

        for img_id in unique_img_ids:
            mask = img_ids == img_id
            img_scores = scores[mask].flatten()
            img_boxes = boxes[mask].reshape(-1, 4)

            keep = img_scores > score_threshold
            results.append({
                "scores": img_scores[keep],
                "boxes_2d": img_boxes[keep],
                "boxes_3d": None,
            })

        return results


def build_sam3_3d(
    sam3_checkpoint: str | None = None,
    geometry_backend_type: str = "unidepth_v2",
    hidden_dim: int = 256,
    num_decoder_layers: int = 6,
    device: str = "cuda",
) -> SAM3_3D:
    """Factory function to build SAM3_3D model.

    Args:
        sam3_checkpoint: Path to SAM3 checkpoint
        geometry_backend_type: Type of geometry backend
        hidden_dim: Hidden dimension for 3D head
        num_decoder_layers: Number of decoder layers
        device: Device to load model on

    Returns:
        Initialized SAM3_3D model

    Note:
        Learning rate control is handled by param_groups in optimizer config,
        not by freezing parameters. See opendet3d/zoo/sam3_3d/base/optim.py.
    """
    from sam3.model.sam3_image import build_sam3_image
    from opendet3d.op.detect3d.geometry import build_geometry_backend

    # Build SAM3 model
    sam3_model = build_sam3_image(checkpoint=sam3_checkpoint)
    sam3_model = sam3_model.to(device)

    # Build geometry backend
    geometry_backend = build_geometry_backend(
        backend_type=geometry_backend_type,
        device=device,
    )

    # Build 3D head
    bbox3d_head = GroundingDINO3DHead(
        hidden_dim=hidden_dim,
        num_layers=num_decoder_layers,
    )

    # Build box coder
    box_coder = GroundingDINO3DCoder()

    # Build inference post-processor
    roi2det3d = RoI2Det3D(box_coder=box_coder)

    model = SAM3_3D(
        sam3_model=sam3_model,
        bbox3d_head=bbox3d_head,
        box_coder=box_coder,
        geometry_backend=geometry_backend,
        roi2det3d=roi2det3d,
    )

    return model.to(device)
