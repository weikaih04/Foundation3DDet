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

from dataclasses import dataclass
from typing import List, NamedTuple, Optional

import torch
from torch import Tensor, nn

# SAM3 imports
from sam3.model.sam3_image import Sam3Image
from sam3.model.geometry_encoders import Prompt
from sam3.model.box_ops import box_cxcywh_to_xyxy
from sam3.model.data_misc import FindStage

# 3D-MOOD imports
from opendet3d.op.detect3d.grounding_dino_3d import (
    GroundingDINO3DHead,
    GroundingDINO3DCoder,
    RoI2Det3D,
)
from opendet3d.op.detect3d.geometry import GeometryBackendBase


class SAM3_3DOut(NamedTuple):
    """Output of SAM3_3D model.

    All tensors use batch-first format: (B, num_queries, dim)
    where B = N_prompts (per-prompt batch).

    Coordinate formats:
    - pred_boxes_2d: normalized xyxy [0, 1]
    - pred_boxes_3d: encoded 3D params (delta_center, log_depth, log_dims, rot_6d)
    """
    # 2D Detection (from SAM3 decoder)
    pred_logits: Tensor  # (N_prompts, num_queries, 1) - objectness
    pred_boxes_2d: Tensor  # (N_prompts, num_queries, 4) - normalized xyxy

    # 3D Detection (from 3D head)
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

        # ========== Freeze Settings ==========
        freeze_sam3_backbone: bool = True,
        freeze_geometry_backend_encoder: bool = True,
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
            freeze_sam3_backbone: Whether to freeze SAM3 ViT backbone.
            freeze_geometry_backend_encoder: Whether to freeze depth encoder.
        """
        super().__init__()

        # SAM3 model - build if not provided
        if sam3_model is None:
            from sam3.model_builder import build_sam3_image_model
            print(f"Building SAM3 model from checkpoint: {sam3_checkpoint}")
            sam3_model = build_sam3_image_model(
                checkpoint_path=sam3_checkpoint,
                load_from_HF=(sam3_checkpoint is None),  # Only load from HF if no checkpoint provided
                device="cpu",  # Will be moved to correct device later
                eval_mode=True,
            )
            # Store checkpoint path for logging in on_load_checkpoint
            self._sam3_checkpoint_path = sam3_checkpoint
        else:
            self._sam3_checkpoint_path = "provided_model"

        self.sam3 = sam3_model
        self.hidden_dim = sam3_model.hidden_dim

        # 3D-MOOD components
        self.box_coder = box_coder or GroundingDINO3DCoder()
        self.bbox3d_head = bbox3d_head
        self.geometry_backend = geometry_backend
        self.roi2det3d = roi2det3d

        # Freeze settings
        if freeze_sam3_backbone:
            self._freeze_sam3_backbone()
        if freeze_geometry_backend_encoder and geometry_backend is not None:
            self._freeze_geometry_encoder()

    def _freeze_sam3_backbone(self) -> None:
        """Freeze SAM3 ViT backbone parameters."""
        for param in self.sam3.backbone.parameters():
            param.requires_grad = False

    def _freeze_geometry_encoder(self) -> None:
        """Freeze geometry backend encoder parameters."""
        if hasattr(self.geometry_backend, 'encoder'):
            for param in self.geometry_backend.encoder.parameters():
                param.requires_grad = False

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

        # ========== Step 1: SAM3 Backbone ==========
        # Get image and text features
        backbone_out = {"img_batch_all_stages": batch.images}
        backbone_out.update(self.sam3.backbone.forward_image(batch.images))
        text_out = self.sam3.backbone.forward_text(
            batch.unique_texts, device=device
        )
        backbone_out.update(text_out)

        # ========== Step 2: Geometry Backend (Depth) ==========
        geom_losses = None
        depth_latents = None
        geom_out = None
        if self.geometry_backend is not None:
            # Extract depth features from SAM3 backbone
            # SAM3's backbone_fpn is a list of multi-scale features, similar to FPN
            depth_feats = backbone_out.get("backbone_fpn", None)

            # Get image dimensions
            _, _, H, W = batch.images.shape

            # Index intrinsics per image (NOT per prompt) for geometry backend
            # Geometry backend processes per-image, not per-prompt
            intrinsics_per_image = batch.intrinsics

            # Prepare depth GT if in training mode
            depth_gt = None
            depth_mask = None
            if self.training and targets is not None:
                depth_gt = targets.get("depth_gt")
                depth_mask = targets.get("depth_mask")

            # Call geometry backend with correct interface
            geom_out = self.geometry_backend(
                images=batch.images,
                depth_feats=depth_feats,
                intrinsics=intrinsics_per_image,
                image_hw=(H, W),
                depth_gt=depth_gt,
                depth_mask=depth_mask,
            )
            depth_latents = geom_out.get("depth_latents")
            if self.training and targets is not None:
                geom_losses = geom_out.get("losses", {})

        # ========== Step 3: Build SAM3 inputs ==========
        find_input = self._build_find_stage(batch, device)
        geometric_prompt = self._build_geometric_prompt(batch, device)

        # ========== Step 4: SAM3 forward_grounding ==========
        # This does: encode_prompt → encoder → decoder → score/box prediction
        sam3_out = self.sam3.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,  # We'll compute loss externally
            geometric_prompt=geometric_prompt,
        )

        # ========== Step 5: Extract SAM3 outputs ==========
        # SAM3 output format (after _update_scores_and_boxes):
        # - pred_logits: (N_prompts, num_queries, 1)
        # - pred_boxes: (N_prompts, num_queries, 4) - normalized cxcywh
        # - pred_boxes_xyxy: (N_prompts, num_queries, 4) - normalized xyxy
        # - queries: (N_prompts, num_queries, d_model) - last layer hidden states
        pred_logits = sam3_out["pred_logits"]  # (N_prompts, S, 1)
        pred_boxes_xyxy = sam3_out["pred_boxes_xyxy"]  # (N_prompts, S, 4)
        queries = sam3_out.get("queries")  # (N_prompts, S, d_model)
        encoder_hidden_states = sam3_out.get("encoder_hidden_states")
        presence_logits = sam3_out.get("presence_logit_dec")

        # ========== Step 6: 3D Head ==========
        pred_boxes_3d = None
        aux_outputs = None

        if self.bbox3d_head is not None and queries is not None:
            # 3D head expects hidden_states in (L, B, S, C) format
            # queries is (B, S, C), expand to (1, B, S, C) for single layer
            hidden_states = queries.unsqueeze(0)  # (1, N_prompts, S, C)

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

            # Align depth_latents and ray_embeddings spatial resolution
            # This is required when using DINOv2-based geometry backends (UniDepthV2, DetAny3D).
            #
            # DINOv2 uses patch_size=14, producing features at resolutions that don't align
            # with standard 1/8 or 1/16 downsampling. For example:
            #   - Input: 1008x1008
            #   - DINOv2 patches: 1008/14 = 72 per side
            #   - UniDepthV2 decoder output (2x upsample): 72*2 = 144x144
            #   - Ray embeddings (1/8 downsample): 1008/8 = 126x126
            #
            # The 3D head requires spatial alignment for cross-attention, so we resize
            # depth_latents to match ray_embeddings resolution.
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

            # Call 3D head with correct signature
            pred_boxes_3d = self.bbox3d_head(
                hidden_states=hidden_states,
                ray_embeddings=ray_embeddings,
                depth_latents=depth_latents,
            )

            # Note: GroundingDINO3DHead.forward() returns Tensor directly, not dict
            # aux_outputs handling would be done differently if needed

        return SAM3_3DOut(
            pred_logits=pred_logits,
            pred_boxes_2d=pred_boxes_xyxy,
            pred_boxes_3d=pred_boxes_3d,
            aux_outputs=aux_outputs,
            geom_losses=geom_losses,
            presence_logits=presence_logits,
            queries=queries,
            encoder_hidden_states=encoder_hidden_states,
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
    freeze_sam3_backbone: bool = True,
    freeze_geometry_backend_encoder: bool = True,
    device: str = "cuda",
) -> SAM3_3D:
    """Factory function to build SAM3_3D model.

    Args:
        sam3_checkpoint: Path to SAM3 checkpoint
        geometry_backend_type: Type of geometry backend
        hidden_dim: Hidden dimension for 3D head
        num_decoder_layers: Number of decoder layers
        freeze_sam3_backbone: Whether to freeze SAM3 backbone
        freeze_geometry_backend_encoder: Whether to freeze depth encoder
        device: Device to load model on

    Returns:
        Initialized SAM3_3D model
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
        freeze_sam3_backbone=freeze_sam3_backbone,
        freeze_geometry_backend_encoder=freeze_geometry_backend_encoder,
    )

    return model.to(device)
