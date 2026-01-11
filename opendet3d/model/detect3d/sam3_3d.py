"""SAM3_3D: SAM3 with 3D Detection Head.

This module combines SAM3 (2D detection with geometric prompting) with
3D-MOOD's 3D detection head and geometry backend.

Key Design Decisions (from Design Doc):
1. Coordinate format: SAM3 uses normalized cxcywh internally, 
   3D-MOOD expects xyxy pixel coordinates for box_coder
2. Tensor format: SAM3 Decoder outputs sequence-first (L, S, B, C),
   3D Head expects batch-first (L, B, S, C) -> need permute
3. Batch strategy: per-prompt batch with img_ids indexing
4. bbox_head: Reuse SAM3 Decoder's internal bbox_embed, 
   no external bbox_head needed
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn

# SAM3 imports
from sam3.model.sam3_image import Sam3Image
from sam3.model.geometry_encoders import SequenceGeometryEncoder, Prompt
from sam3.model.box_ops import box_cxcywh_to_xyxy

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
    pred_boxes_3d: Tensor  # (N_prompts, num_queries, 12) - encoded 3D params
    
    # Auxiliary outputs for each decoder layer (for deep supervision)
    aux_outputs: list[dict] | None
    
    # Geometry backend losses (SILog depth, phi, theta)
    geom_losses: dict[str, Tensor] | None
    
    # SAM3 specific outputs
    presence_logits: Tensor | None  # (N_prompts, num_queries, 1)
    queries: Tensor | None  # (N_prompts, num_queries, d_model) - for segmentation


@dataclass
class GeometricQueryBatch:
    """Batched geometric queries for SAM3_3D.
    
    All tensors follow SAM3's sequence-first convention for Prompt class,
    but are stored in batch-first here and converted when needed.
    
    Coordinate format: normalized [0, 1] cxcywh (SAM3 convention)
    """
    img_ids: Tensor  # (N_prompts,) - which image each prompt belongs to
    
    # Box prompts: (N_prompts, max_K, 4) - normalized cxcywh
    boxes: Tensor | None = None
    boxes_mask: Tensor | None = None  # (N_prompts, max_K) - True = padding
    box_labels: Tensor | None = None  # (N_prompts, max_K) - 0/1 for neg/pos
    
    # Point prompts: (N_prompts, max_P, 3) - (x, y, label)
    points: Tensor | None = None
    points_mask: Tensor | None = None  # (N_prompts, max_P) - True = padding


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
        sam3_model: Sam3Image,
        
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
            sam3_model: Complete SAM3 model (backbone + encoder + decoder)
            bbox3d_head: 3D box regression head. If None, creates default.
            box_coder: 3D box encoder/decoder. If None, creates default.
            geometry_backend: Depth estimation backend. If None, no depth.
            roi2det3d: Inference post-processor. If None, creates default.
            freeze_sam3_backbone: Whether to freeze SAM3 ViT backbone.
            freeze_geometry_backend_encoder: Whether to freeze depth encoder.
        """
        super().__init__()
        
        # SAM3 model
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

    def forward(
        self,
        images: Tensor,  # (B, 3, H, W)
        query_batch: GeometricQueryBatch,
        intrinsics: Tensor | None = None,  # (B, 3, 3)
        targets: dict | None = None,  # For training
    ) -> SAM3_3DOut:
        """Forward pass of SAM3_3D.

        Args:
            images: Input images, shape (B, 3, H, W)
            query_batch: Batched geometric queries with img_ids
            intrinsics: Camera intrinsics, shape (B, 3, 3)
            targets: Training targets (optional)

        Returns:
            SAM3_3DOut with 2D and 3D predictions
        """
        B, _, H, W = images.shape
        device = images.device

        # ========== Step 1: SAM3 Backbone + Encoder ==========
        # Get image features from SAM3 backbone
        backbone_out = self.sam3.backbone(images)

        # Get encoder features (for decoder cross-attention)
        encoder_out = self.sam3.encoder(backbone_out)

        # ========== Step 2: Geometry Backend (Depth) ==========
        geom_losses = None
        depth_latents = None
        if self.geometry_backend is not None:
            # Geometry backend expects backbone features
            geom_out = self.geometry_backend(
                backbone_out,
                intrinsics=intrinsics,
                targets=targets,
            )
            depth_latents = geom_out.get("depth_latents")
            if self.training and targets is not None:
                geom_losses = geom_out.get("losses", {})

        # ========== Step 3: Prepare Prompts for SAM3 Decoder ==========
        # Convert GeometricQueryBatch to SAM3's Prompt format
        prompts = self._prepare_prompts(query_batch, device)

        # ========== Step 4: SAM3 Decoder ==========
        # SAM3 decoder outputs: hidden_states, pred_boxes (cxcywh), pred_logits
        decoder_out = self.sam3.decoder(
            encoder_out=encoder_out,
            prompts=prompts,
            img_ids=query_batch.img_ids,
        )

        # Extract decoder outputs
        # SAM3 decoder outputs are sequence-first: (L, S, B, C)
        # We need batch-first: (L, B, S, C) for 3D head
        hidden_states = decoder_out["hidden_states"]  # (L, S, B, C)
        pred_boxes_cxcywh = decoder_out["pred_boxes"]  # (B, S, 4) normalized cxcywh
        pred_logits = decoder_out["pred_logits"]  # (B, S, 1)

        # Convert to batch-first for 3D head
        # (L, S, B, C) -> (L, B, S, C)
        hidden_states_bf = hidden_states.permute(0, 2, 1, 3)

        # ========== Step 5: Convert 2D Boxes to xyxy ==========
        # SAM3 uses normalized cxcywh, 3D head needs normalized xyxy
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_cxcywh)

        # ========== Step 6: 3D Head ==========
        pred_boxes_3d = None
        aux_outputs = None

        if self.bbox3d_head is not None:
            # 3D head expects:
            # - hidden_states: (L, B, S, C) batch-first
            # - pred_boxes: (B, S, 4) normalized xyxy
            # - depth_latents: from geometry backend
            # - intrinsics: (B, 3, 3)
            # - image_size: (H, W)

            head_out = self.bbox3d_head(
                hidden_states=hidden_states_bf,
                pred_boxes=pred_boxes_xyxy,
                depth_latents=depth_latents,
                intrinsics=intrinsics,
                image_size=(H, W),
            )
            pred_boxes_3d = head_out["pred_boxes_3d"]  # (B, S, 12)

            # Auxiliary outputs for deep supervision
            if "aux_outputs" in head_out:
                aux_outputs = head_out["aux_outputs"]

        # ========== Step 7: Presence Logits (optional) ==========
        presence_logits = decoder_out.get("presence_logits")
        queries = decoder_out.get("queries")

        return SAM3_3DOut(
            pred_logits=pred_logits,
            pred_boxes_2d=pred_boxes_xyxy,
            pred_boxes_3d=pred_boxes_3d,
            aux_outputs=aux_outputs,
            geom_losses=geom_losses,
            presence_logits=presence_logits,
            queries=queries,
        )

    def _prepare_prompts(
        self,
        query_batch: GeometricQueryBatch,
        device: torch.device,
    ) -> Prompt:
        """Convert GeometricQueryBatch to SAM3's Prompt format.

        SAM3's Prompt expects sequence-first format: (K, N_prompts, dim)
        GeometricQueryBatch stores batch-first: (N_prompts, K, dim)
        """
        # Box prompts: (N_prompts, K, 4) -> (K, N_prompts, 4)
        boxes = None
        box_labels = None
        if query_batch.boxes is not None:
            boxes = query_batch.boxes.permute(1, 0, 2)  # (K, N, 4)
            if query_batch.box_labels is not None:
                box_labels = query_batch.box_labels.permute(1, 0)  # (K, N)

        # Point prompts: (N_prompts, P, 3) -> (P, N_prompts, 3)
        points = None
        if query_batch.points is not None:
            points = query_batch.points.permute(1, 0, 2)  # (P, N, 3)

        return Prompt(
            boxes=boxes,
            box_labels=box_labels,
            points=points,
        )

    @torch.no_grad()
    def inference(
        self,
        images: Tensor,
        query_batch: GeometricQueryBatch,
        intrinsics: Tensor,
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
    ) -> list[dict]:
        """Run inference and decode 3D boxes.

        Args:
            images: Input images (B, 3, H, W)
            query_batch: Geometric queries
            intrinsics: Camera intrinsics (B, 3, 3)
            score_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold

        Returns:
            List of dicts per image with decoded 3D boxes
        """
        self.eval()

        out = self.forward(images, query_batch, intrinsics)

        if self.roi2det3d is None or out.pred_boxes_3d is None:
            # Return 2D only
            return self._decode_2d_only(out, query_batch, score_threshold)

        # Decode 3D boxes using roi2det3d
        B, _, H, W = images.shape
        results = self.roi2det3d(
            pred_logits=out.pred_logits,
            pred_boxes_2d=out.pred_boxes_2d,
            pred_boxes_3d=out.pred_boxes_3d,
            intrinsics=intrinsics,
            image_size=(H, W),
            img_ids=query_batch.img_ids,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
        )
        return results

    def _decode_2d_only(
        self,
        out: SAM3_3DOut,
        query_batch: GeometricQueryBatch,
        score_threshold: float,
    ) -> list[dict]:
        """Decode 2D-only results when 3D head is not available."""
        scores = out.pred_logits.sigmoid().squeeze(-1)  # (N_prompts, S)
        boxes = out.pred_boxes_2d  # (N_prompts, S, 4) normalized xyxy

        results = []
        unique_img_ids = query_batch.img_ids.unique()

        for img_id in unique_img_ids:
            mask = query_batch.img_ids == img_id
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
