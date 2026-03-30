"""GLEE_3D: GLEE (MIT-licensed) + 3D Detection Head.

Follows the same composition pattern as SAM3_3D:
- GLEE handles 2D detection (backbone + pixel decoder + transformer decoder)
- 3D head, geometry backend, and early depth fusion from opendet3d
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# GLEE imports
from glee.models.glee_model import GLEE_Model, agg_lang_feat, rand_sample
from glee.models.vos_utils import masks_to_boxes

# opendet3d 3D components (same as SAM3_3D)
from opendet3d.op.detect3d.geometry import GeometryBackendBase
from torchvision.ops import batched_nms, nms

from opendet3d.op.detect3d.grounding_dino_3d import (
    GroundingDINO3DCoder,
    GroundingDINO3DHead,
)
class Det3DOut(NamedTuple):
    """Standard detection output for evaluator compatibility."""

    boxes: list[Tensor]
    boxes3d: list[Tensor]
    scores: list[Tensor]
    class_ids: list[Tensor]
    depth_maps: list[Tensor] | None = None
    categories: list[list[str]] | None = None
    predicted_intrinsics: Tensor | None = None
    scores_3d: list[Tensor] | None = None
    scores_2d: list[Tensor] | None = None


def _cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert cxcywh to xyxy format."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


# ---------------------------------------------------------------------------
# Output / Input data structures
# ---------------------------------------------------------------------------

class GLEE_3DOut(NamedTuple):
    """Output of GLEE_3D model. Per-image batching (B, Q, dim)."""

    # 2D detection (from GLEE)
    pred_logits: Tensor  # (B, Q, C) class logits
    pred_scores: Tensor  # (B, Q, 1) confidence
    pred_boxes_2d: Tensor  # (B, Q, 4) normalized cxcywh

    # 3D detection (from 3D head)
    pred_boxes_3d: Tensor | None  # (B, Q, 12) encoded 3D
    pred_conf_3d: Tensor | None  # (B, Q, 1)

    # DN 3D predictions (for DN 3D loss)
    pred_boxes_3d_dn: Tensor | None = None  # (B, DN_Q, 12)

    # Auxiliary (for loss)
    aux_outputs: list[dict] | None = None
    geom_losses: dict[str, Tensor] | None = None
    mask_dict: dict | None = None  # GLEE DN info

    def __getitem__(self, key: str):
        return getattr(self, key)


@dataclass
class GLEE_3DBatchedInputs:
    """Batched inputs for GLEE_3D. Per-image batching (simpler than SAM3_3D)."""

    # Images & camera
    images: Tensor  # (B, 3, H, W) ImageNet-normalized
    intrinsics: Tensor  # (B, 3, 3)

    # Text prompts
    class_names: list[str]  # all category names for this batch

    # Visual prompts (box/point) - in PADDED image coordinate space
    # spatial_masks: list of (1, H, W) binary masks, one per image
    #   For box prompt: filled rectangle mask
    #   For point prompt: small rectangle around the point
    # If None, only text prompts are used.
    spatial_masks: list[Tensor] | None = None
    visual_prompt_type: str = "box"  # "box", "point", or "mask"

    # GLEE-format targets (training only)
    # targets[i] = {"labels": LongTensor, "boxes": FloatTensor cxcywh norm}
    targets: list[dict] | None = None

    # 3D ground truth (training only, per-image)
    gt_boxes_3d: list[Tensor] | None = None  # [N_i, 10] per image
    gt_boxes_2d_pixel: list[Tensor] | None = None  # [N_i, 4] xyxy pixel

    # Ignore boxes (for ignore suppress in loss)
    ignore_boxes_2d: list[Tensor] | None = None  # per-image [M_i, 4] xyxy pixel

    # Depth supervision
    depth_gt: Tensor | None = None  # (B, 1, H, W)
    depth_mask: Tensor | None = None  # (B, 1, H, W)

    # Metadata
    sample_names: list[str] | None = None
    dataset_name: list[str] | None = None
    original_hw: list[tuple] | None = None
    padding: list | None = None

    @property
    def num_images(self) -> int:
        return self.images.shape[0]

    @property
    def device(self) -> torch.device:
        return self.images.device

    def __contains__(self, key: str) -> bool:
        """Support 'key in batch' for vis4d connector compatibility."""
        return hasattr(self, key)

    def __getitem__(self, key: str):
        """Support batch[key] for vis4d connector compatibility."""
        return getattr(self, key)


# ---------------------------------------------------------------------------
# GLEE_3D Model
# ---------------------------------------------------------------------------

class GLEE_3D(nn.Module):
    """GLEE + 3D Detection Head.

    Architecture:
        GLEE backbone (Swin-L) + LingbotDepth → EarlyDepthFusion
        → GLEE pixel decoder + transformer decoder → 2D boxes + hidden states
        → GroundingDINO3DHead → 3D boxes
    """

    def __init__(
        self,
        # GLEE model (pre-built from detectron2 cfg)
        glee_model: GLEE_Model | None = None,
        # 3D components (same as SAM3_3D)
        geometry_backend: GeometryBackendBase | None = None,
        bbox3d_head: GroundingDINO3DHead | None = None,
        box_coder: GroundingDINO3DCoder | None = None,
        early_depth_fusion: nn.Module | None = None,
        # Freeze settings
        backbone_freeze_blocks: int = 0,
        freeze_text_encoder: bool = True,
        # Task config
        task: str = "glee3d",
        # Eval options
        use_depth_input_test: bool = False,
    ) -> None:
        super().__init__()

        assert glee_model is not None, "glee_model must be provided"
        self.glee = glee_model
        self.hidden_dim = 256  # GLEE's HIDDEN_DIM is always 256

        # 3D components
        self.box_coder = box_coder or GroundingDINO3DCoder()
        self.geometry_backend = geometry_backend
        self.early_depth_fusion = early_depth_fusion
        self.task = task
        self.use_depth_input_test = use_depth_input_test

        # Determine camera prompt usage from geometry backend
        use_camera_prompt = True
        depth_latent_dim = 256
        if self.geometry_backend is not None:
            if hasattr(self.geometry_backend, "is_ray_aware"):
                use_camera_prompt = not self.geometry_backend.is_ray_aware
            if hasattr(self.geometry_backend, "target_latent_dim"):
                depth_latent_dim = self.geometry_backend.target_latent_dim

        # Build 3D head if not provided
        if bbox3d_head is not None:
            self.bbox3d_head = bbox3d_head
        else:
            self.bbox3d_head = GroundingDINO3DHead(
                embed_dims=self.hidden_dim,
                box_coder=self.box_coder,
                use_camera_prompt=use_camera_prompt,
                depth_latent_dim=depth_latent_dim,
            )

        # Freeze settings
        if freeze_text_encoder:
            for p in self.glee.text_encoder.parameters():
                p.requires_grad = False

        if backbone_freeze_blocks > 0:
            self._freeze_backbone(backbone_freeze_blocks)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, batch: GLEE_3DBatchedInputs
    ):
        if not self.training:
            return self.forward_test(batch)
        return self.forward_train(batch)

    def forward_train(
        self, batch: GLEE_3DBatchedInputs
    ) -> GLEE_3DOut:
        images = batch.images  # (B, 3, H, W) ImageNet normalized
        B, _, H, W = images.shape
        device = batch.device

        # ============================================================
        # Step 1: Text encoding (reuse GLEE's text path)
        # ============================================================
        extra, early_semantic, dist_loss = self._encode_text(
            batch.class_names, images
        )

        # ============================================================
        # Step 2: GLEE backbone
        # ============================================================
        # Convert vis4d ImageNet norm → GLEE's expected BGR mean-subtracted format
        # vis4d: (RGB_255 - mean_rgb_255) / (std_rgb * 255), range ~[-2, 3]
        # GLEE: BGR_255 - mean_bgr_255, range ~[-124, 152]
        # Since means are identical (just RGB/BGR order), undo std and flip channels
        _std_255 = torch.tensor(
            [58.395, 57.12, 57.375], device=device
        ).view(1, 3, 1, 1)
        images_glee = images * _std_255          # undo std → mean-subtracted RGB_255
        images_glee = images_glee[:, [2, 1, 0]]  # RGB → BGR
        features = self.glee.backbone(images_glee)

        # ============================================================
        # Step 2.5: Visual prompts (box/point) processing
        # ============================================================
        # spatial_masks tracks the (possibly modified) masks for spatial query step
        spatial_masks = batch.spatial_masks
        if spatial_masks is not None:
            early_semantic, spatial_masks = self._process_visual_prompts(
                images, spatial_masks, batch.visual_prompt_type,
                features, early_semantic,
            )

        # ============================================================
        # Step 3: Geometry backend (parallel depth estimation)
        # ============================================================
        depth_latents = None
        geom_losses = None
        geom_out = {}

        if self.geometry_backend is not None:
            intrinsics = batch.intrinsics
            depth_gt = None
            if self.training or self.use_depth_input_test:
                depth_gt = batch.depth_gt
            depth_mask = batch.depth_mask if self.training else None

            if self.training:
                geom_out = self.geometry_backend.forward_train(
                    images=images,
                    depth_feats=None,
                    intrinsics=intrinsics,
                    image_hw=(H, W),
                    depth_gt=depth_gt,
                    depth_mask=depth_mask,
                )
            else:
                geom_out = self.geometry_backend.forward_test(
                    images=images,
                    depth_feats=None,
                    intrinsics=intrinsics,
                    image_hw=(H, W),
                )

            depth_latents = geom_out.get("depth_latents")
            geom_losses = geom_out.get("losses")

        # ============================================================
        # Step 4: GLEE pixel decoder (with depth fusion + VLFuse)
        # ============================================================
        # Depth fusion happens INSIDE pixel_decoder, after input_proj
        # (256-dim projection) but before MSDeformAttn encoder.
        # This matches SAM3_3D's design: FPN(256-dim) → depth fusion → encoder.
        # The encoder's self-attention then operates on depth-enhanced features,
        # and VLFuse naturally creates language+depth interaction.
        depth_fusion_args = None
        if (
            self.early_depth_fusion is not None
            and depth_latents is not None
        ):
            aux = geom_out.get("aux", {})
            depth_latents_hw = aux.get("depth_latents_hw")
            if depth_latents_hw is not None:
                depth_fusion_args = (
                    self.early_depth_fusion,
                    depth_latents,
                    depth_latents_hw,
                )

        mask_features, _, multi_scale_features, zero_loss = (
            self.glee.pixel_decoder.forward_features(
                features,
                masks=None,
                early_fusion=early_semantic,
                depth_fusion=depth_fusion_args,
            )
        )

        # ============================================================
        # Step 4.5: Build spatial query tokens for visual prompts
        # ============================================================
        if spatial_masks is not None:
            extra = self._build_spatial_queries(
                spatial_masks, mask_features, multi_scale_features, extra
            )

        # ============================================================
        # Step 5: GLEE transformer decoder (9 layers + DN)
        # ============================================================
        targets_for_decoder = batch.targets if self.training else None
        outputs, mask_dict = self.glee.predictor(
            multi_scale_features,
            mask_features,
            extra=extra,
            task=self.task,
            masks=None,
            targets=targets_for_decoder,
        )

        # Extract 2D predictions
        pred_logits = outputs["pred_logits"]  # (B, Q, C)
        pred_scores = outputs["pred_scores"]  # (B, Q, 1)
        pred_boxes_2d = outputs["pred_boxes"]  # (B, Q, 4) cxcywh
        aux_outputs_2d = outputs.get("aux_outputs", [])

        # Extract hidden states (exposed by our patch)
        hs = outputs["hs"]  # list of (B, Q, 256), one per decoder layer
        hs_dn = outputs.get("hs_dn")  # list of (B, DN_Q, 256) or None

        # ============================================================
        # Step 7: 3D detection head
        # ============================================================
        pred_boxes_3d = None
        pred_conf_3d = None
        pred_boxes_3d_dn = None
        aux_outputs = None

        if self.bbox3d_head is not None and hs is not None:
            # Stack hidden states: (num_layers, B, Q, 256)
            hidden_states = torch.stack(hs, dim=0)

            # Generate ray embeddings for camera-aware 3D
            ray_embeddings = None
            if self.bbox3d_head.use_camera_prompt:
                ray_intrinsics = geom_out.get(
                    "ray_intrinsics", batch.intrinsics
                )
                ray_image_hw = geom_out.get("ray_image_hw", (H, W))
                ray_downsample = geom_out.get("ray_downsample", 16)
                ray_embeddings = self.bbox3d_head.get_camera_embeddings(
                    ray_intrinsics, ray_image_hw, ray_downsample
                )

            # Align depth_latents and ray_embeddings spatial resolution
            depth_latents_for_3d = depth_latents
            if (
                depth_latents_for_3d is not None
                and ray_embeddings is not None
            ):
                _, N_d, C_d = depth_latents_for_3d.shape
                _, N_r, _ = ray_embeddings.shape
                if N_d != N_r:
                    H_d = int(N_d**0.5)
                    H_r = int(N_r**0.5)
                    d2d = depth_latents_for_3d.permute(0, 2, 1).reshape(
                        B, C_d, H_d, H_d
                    )
                    d2d = F.adaptive_avg_pool2d(d2d, (H_r, H_r))
                    depth_latents_for_3d = d2d.reshape(
                        B, C_d, H_r * H_r
                    ).permute(0, 2, 1)

            # 3D head forward on matching queries (all decoder layers)
            all_layers_boxes_3d, all_layers_conf_3d = self.bbox3d_head(
                hidden_states=hidden_states,
                ray_embeddings=ray_embeddings,
                depth_latents=depth_latents_for_3d,
            )

            # Final layer
            pred_boxes_3d = all_layers_boxes_3d[-1]  # (B, Q, 12)
            pred_conf_3d = all_layers_conf_3d[-1]  # (B, Q, 1)

            # DN queries: also pass through 3D head (last layer only)
            if self.training and hs_dn is not None and len(hs_dn) > 0:
                # Use last decoder layer's DN hidden states
                dn_hidden = hs_dn[-1].unsqueeze(0)  # (1, B, DN_Q, 256)
                dn_boxes_3d, _ = self.bbox3d_head(
                    hidden_states=dn_hidden,
                    ray_embeddings=ray_embeddings,
                    depth_latents=depth_latents_for_3d,
                )
                pred_boxes_3d_dn = dn_boxes_3d[0]  # (B, DN_Q, 12)

            # Build aux_outputs with both 2D and 3D for deep supervision
            if self.training and len(aux_outputs_2d) > 0:
                num_aux = min(len(aux_outputs_2d), len(hs) - 1)
                aux_outputs = []
                for i in range(num_aux):
                    aux_dict = dict(aux_outputs_2d[i])
                    aux_dict["pred_boxes_3d"] = all_layers_boxes_3d[i]
                    aux_dict["pred_conf_3d"] = all_layers_conf_3d[i]
                    aux_outputs.append(aux_dict)

        return GLEE_3DOut(
            pred_logits=pred_logits,
            pred_scores=pred_scores,
            pred_boxes_2d=pred_boxes_2d,
            pred_boxes_3d=pred_boxes_3d,
            pred_conf_3d=pred_conf_3d,
            pred_boxes_3d_dn=pred_boxes_3d_dn,
            aux_outputs=aux_outputs,
            geom_losses=geom_losses,
            mask_dict=mask_dict,
        )

    # ------------------------------------------------------------------
    # Test-time forward (produces Det3DOut for evaluator)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_test(self, batch: GLEE_3DBatchedInputs):
        """Test-time forward: runs model + postprocessing → Det3DOut.

        Follows GDino3D's RoI2Det3D pattern:
        1. logits sigmoid → per-class scores
        2. top-K + score threshold
        3. cxcywh norm → pixel xyxy
        4. NMS (optional)
        5. box_coder.decode → 3D boxes
        6. remove padding, rescale to original image
        """
        # Move spatial_masks to device if present (oracle eval)
        if batch.spatial_masks is not None:
            batch.spatial_masks = [
                m.to(batch.images.device) for m in batch.spatial_masks
            ]
        out = self.forward_train(batch)

        B, _, H, W = batch.images.shape
        device = batch.device

        # Config
        score_threshold = getattr(self, "score_threshold", 0.0)
        max_per_img = getattr(self, "max_per_img", 300)
        use_nms = getattr(self, "use_nms", False)
        iou_threshold = getattr(self, "iou_threshold", 0.5)
        eval_3d_conf_weight = getattr(self, "eval_3d_conf_weight", 0.0)

        boxes_list = []
        boxes3d_list = []
        scores_list = []
        class_ids_list = []

        for b in range(B):
            cls_score = out.pred_logits[b].sigmoid()  # (Q, C)
            bbox_pred = out.pred_boxes_2d[b]  # (Q, 4) cxcywh norm
            bbox_3d_pred = (
                out.pred_boxes_3d[b]
                if out.pred_boxes_3d is not None
                else None
            )

            # Step 1: Convert ALL boxes cxcywh norm → pixel xyxy
            # (same order as GDino3D RoI2Det3D: convert first, then top-K)
            det_bboxes_all = _cxcywh_to_xyxy(bbox_pred)  # (Q, 4)
            det_bboxes_all[:, 0::2] *= W
            det_bboxes_all[:, 1::2] *= H
            det_bboxes_all[:, 0::2].clamp_(min=0, max=W)
            det_bboxes_all[:, 1::2].clamp_(min=0, max=H)

            # Step 2: Top-K across all queries x classes
            k = min(max_per_img, cls_score.numel())
            if k == 0:
                boxes_list.append(torch.zeros(0, 4, device=device))
                boxes3d_list.append(torch.zeros(0, 10, device=device))
                scores_list.append(torch.zeros(0, device=device))
                class_ids_list.append(
                    torch.zeros(0, dtype=torch.long, device=device)
                )
                continue

            scores_flat, indexes = cls_score.view(-1).topk(k)
            num_classes = cls_score.shape[-1]
            det_labels = indexes % num_classes
            query_idx = indexes // num_classes

            det_bboxes = det_bboxes_all[query_idx]
            det_3d = (
                bbox_3d_pred[query_idx]
                if bbox_3d_pred is not None
                else None
            )

            # Step 3: 3D confidence weighting (optional)
            if eval_3d_conf_weight > 0 and out.pred_conf_3d is not None:
                conf_3d = (
                    out.pred_conf_3d[b, query_idx].sigmoid().squeeze(-1)
                )
                scores_flat = scores_flat + eval_3d_conf_weight * conf_3d

            # Step 4: Score threshold
            if score_threshold > 0:
                mask = scores_flat > score_threshold
                det_bboxes = det_bboxes[mask]
                det_labels = det_labels[mask]
                scores_flat = scores_flat[mask]
                if det_3d is not None:
                    det_3d = det_3d[mask]

            # Step 5: NMS (optional, same as GDino3D)
            if use_nms and len(det_bboxes) > 0:
                keep = batched_nms(
                    det_bboxes, scores_flat, det_labels, iou_threshold
                )
                det_bboxes = det_bboxes[keep]
                det_labels = det_labels[keep]
                scores_flat = scores_flat[keep]
                if det_3d is not None:
                    det_3d = det_3d[keep]

            # Step 6: Decode 3D boxes (BEFORE rescale, same as GDino3D)
            if det_3d is not None and det_3d.numel() > 0:
                intrinsics_b = batch.intrinsics[b]
                det_bboxes3d = self.box_coder.decode(
                    det_bboxes, det_3d, intrinsics_b
                )
            else:
                det_bboxes3d = torch.zeros(
                    len(det_bboxes), 10, device=device
                )

            # Step 7: Remove padding + rescale to original image space
            # (exact same logic as GDino3D RoI2Det3D lines 411-427)
            if batch.padding is not None:
                pad = batch.padding[b]
                det_bboxes[:, 0] -= pad[0]  # left
                det_bboxes[:, 1] -= pad[2]  # top
                det_bboxes[:, 2] -= pad[0]
                det_bboxes[:, 3] -= pad[2]

                ori_h, ori_w = batch.original_hw[b]
                scales = det_bboxes.new_tensor([
                    (W - pad[0] - pad[1]) / ori_w,
                    (H - pad[2] - pad[3]) / ori_h,
                ])
            elif batch.original_hw is not None:
                ori_h, ori_w = batch.original_hw[b]
                scales = det_bboxes.new_tensor([W / ori_w, H / ori_h])
            else:
                scales = det_bboxes.new_tensor([1.0, 1.0])

            det_bboxes /= scales.repeat(2).unsqueeze(0)

            boxes_list.append(det_bboxes)
            boxes3d_list.append(det_bboxes3d)
            scores_list.append(scores_flat)
            class_ids_list.append(det_labels)

        return Det3DOut(
            boxes=boxes_list,
            boxes3d=boxes3d_list,
            scores=scores_list,
            class_ids=class_ids_list,
            depth_maps=None,
        )

    # ------------------------------------------------------------------
    # Text encoding helper (extracted from GLEE_Model.forward)
    # ------------------------------------------------------------------

    def _encode_text(
        self, class_names: list[str], images: Tensor
    ) -> tuple[dict, dict | None, Tensor]:
        """Encode text prompts using GLEE's CLIP text encoder.

        Returns:
            extra: dict with 'class_embeddings'
            early_semantic: dict for VLFuse (or None)
            dist_loss: distillation loss (zero if no teacher)
        """
        extra = {}
        early_semantic = None
        device = images.device
        B = images.shape[0]

        tokenized = self.glee.tokenizer.batch_encode_plus(
            class_names,
            max_length=self.glee.cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN,
            padding="max_length",
            return_special_tokens_mask=True,
            return_tensors="pt",
            truncation=True,
        ).to(device)

        texts = (tokenized["input_ids"], tokenized["attention_mask"])

        with torch.no_grad():
            token_x = self.glee.text_encoder(*texts)[
                "last_hidden_state"
            ]

        token_x = token_x @ self.glee.lang_projection
        lang_feat_pool = agg_lang_feat(
            token_x, tokenized["attention_mask"], pool_type="average"
        )
        extra["class_embeddings"] = lang_feat_pool

        # Distillation loss (zero if not using teacher mode)
        dist_loss = (lang_feat_pool * 0).sum()

        # Early fusion: flatten text tokens for VLFuse in pixel decoder
        if self.glee.early_fusion:
            mask_flat = tokenized["attention_mask"].flatten(0, 1)
            gather_tokens = token_x.flatten(0, 1)[mask_flat > 0]
            gather_tokens = (
                gather_tokens.unsqueeze(0).repeat(B, 1, 1)
            )
            gather_mask = (
                torch.ones_like(gather_tokens[:, :, 0]) > 0
            )
            early_semantic = {
                "hidden": gather_tokens.float(),
                "masks": gather_mask,
            }

        return extra, early_semantic, dist_loss

    # ------------------------------------------------------------------
    # Freeze helpers
    # ------------------------------------------------------------------

    def _freeze_backbone(self, num_blocks: int) -> None:
        """Freeze first `num_blocks` blocks of GLEE's Swin backbone."""
        backbone = self.glee.backbone

        # Swin-L has: patch_embed, layers (4 stages), norm
        # Freeze patch_embed
        if num_blocks > 0:
            if hasattr(backbone, "patch_embed"):
                for p in backbone.patch_embed.parameters():
                    p.requires_grad = False

        # Freeze stages progressively
        frozen = 0
        if hasattr(backbone, "layers"):
            for stage in backbone.layers:
                if hasattr(stage, "blocks"):
                    for block in stage.blocks:
                        if frozen >= num_blocks:
                            return
                        for p in block.parameters():
                            p.requires_grad = False
                        frozen += 1

    # ------------------------------------------------------------------
    # Visual prompt helpers
    # ------------------------------------------------------------------

    def _process_visual_prompts(
        self,
        images: Tensor,
        spatial_masks: list[Tensor],
        prompt_type: str,
        features: dict,
        early_semantic: dict | None,
    ) -> tuple[dict | None, list[Tensor]]:
        """Process visual prompts (box/point) through GLEE's template pipeline.

        Follows GLEE_Model.forward() exactly:
        - Inference: use prompt_type directly
        - Training: randomly switch between mask/box/point modes (data aug)

        The spatial_masks may be MODIFIED in-place (for spatial query step later).

        Args:
            images: (B, 3, H, W) input images
            spatial_masks: list of (1, H, W) binary masks per image (PADDED space)
            prompt_type: "box", "point", or "mask"
            features: backbone features (not used directly)
            early_semantic: existing text early fusion dict (or None)

        Returns:
            (updated early_semantic, updated spatial_masks for spatial queries)
        """
        import numpy as np

        key_images = [images[i:i+1] for i in range(images.shape[0])]
        key_promptmasks = [m.unsqueeze(0) for m in spatial_masks]  # list of (1, 1, H, W)

        if self.training:
            # Training: randomly switch between box and point modes
            # (no mask mode - GLEE_3D only does detection, not segmentation)
            if np.random.rand() > 0.5:
                prompt_mode = "box"
            else:
                prompt_mode = "point"
                # Convert mask to small rectangle around a sampled point
                _, h, w = spatial_masks[0].shape
                point_h, point_w = h // 40, w // 40
                new_promptmasks = []
                for pmask in key_promptmasks:
                    nonzero_pts = pmask[0].nonzero()[:, 1:]  # (N, 2)
                    zeromask = torch.zeros_like(pmask)
                    if nonzero_pts.shape[0] > 0:
                        pts = rand_sample(nonzero_pts.t(), 1).t()
                        py, px = pts[0, 0].long(), pts[0, 1].long()
                        zeromask[:, :,
                                 max(0, py - point_h):py + point_h,
                                 max(0, px - point_w):px + point_w] = True
                    # Empty mask: keep zeromask (no point prompt for this image)
                    new_promptmasks.append(zeromask)
                key_promptmasks = new_promptmasks

            # Update spatial_masks to box version (for spatial query step)
            updated_masks = []
            for ori_mask, pmask in zip(spatial_masks, key_promptmasks):
                zeromask = torch.zeros_like(ori_mask)
                x1, y1, x2, y2 = masks_to_boxes(pmask[0])[0].long().tolist()
                zeromask[:, y1:y2, x1:x2] = True
                updated_masks.append(zeromask)
            spatial_masks = updated_masks
        else:
            # Inference: use prompt_type directly (box or point)
            prompt_mode = prompt_type

        # get_template: crop around mask, resize to 256x256, backbone → features
        ref_feats, ref_masks = self.glee.get_template(
            key_images, key_promptmasks, prompt_mode
        )

        early_fusion = {"hidden": ref_feats, "masks": ref_masks}

        if early_semantic is None:
            early_semantic = early_fusion
        else:
            early_semantic["hidden"] = torch.cat(
                [early_semantic["hidden"], early_fusion["hidden"]], dim=1
            )
            early_semantic["masks"] = torch.cat(
                [early_semantic["masks"], early_fusion["masks"]], dim=1
            )

        return early_semantic, spatial_masks

    def _build_spatial_queries(
        self,
        spatial_masks: list[Tensor],
        mask_features: Tensor,
        multi_scale_features: list[Tensor],
        extra: dict,
    ) -> dict:
        """Build spatial query tokens from visual prompts for decoder.

        These tokens are injected into decoder self-attention layers,
        allowing the decoder to attend to the prompted spatial regions.
        Follows GLEE_Model.forward() spatial query logic exactly.

        Args:
            spatial_masks: list of (1, H, W) binary masks per image
            mask_features: (B, 256, H/4, W/4) from pixel decoder
            multi_scale_features: list of (B, 256, H_i, W_i)
            extra: existing extra dict

        Returns:
            Updated extra dict with visual_prompt_tokens and masks.
        """
        device = mask_features.device

        pos_masks = [m.bool() for m in spatial_masks]
        neg_masks = [torch.zeros_like(m, dtype=torch.bool) for m in spatial_masks]
        extra["spatial_query_pos_mask"] = pos_masks
        extra["spatial_query_neg_mask"] = neg_masks

        _, h, w = pos_masks[0].shape
        divisor = torch.tensor([h, w], device=device)[None,]

        # Build per-level spatial query tokens
        src_spatial_queries = []
        src_spatial_maskings = []
        max_spatial_len = [512, 512, 512, 512]

        for i in range(len(multi_scale_features)):
            bs, dc, fh, fw = multi_scale_features[i].shape
            src_mask_features = multi_scale_features[i].permute(2, 3, 0, 1)
            src_mask_features = src_mask_features @ self.glee.mask_sptial_embed[i]

            non_zero_query_point_pos = [
                rand_sample((m.nonzero()[:, 1:] / divisor).t(), max_spatial_len[i]).t()
                for m in pos_masks
            ]
            non_zero_query_point_neg = [
                rand_sample((m.nonzero()[:, 1:] / divisor).t(), max_spatial_len[i]).t()
                for m in neg_masks
            ]
            non_zero_query_point = [
                torch.cat([x, y], dim=0)
                for x, y in zip(non_zero_query_point_pos, non_zero_query_point_neg)
            ]
            pos_neg_indicator = [
                torch.cat([
                    torch.ones(x.shape[0], device=device),
                    -torch.ones(y.shape[0], device=device),
                ])
                for x, y in zip(non_zero_query_point_pos, non_zero_query_point_neg)
            ]
            pos_neg_indicator = nn.utils.rnn.pad_sequence(
                pos_neg_indicator, padding_value=0
            )
            non_zero_query_point = nn.utils.rnn.pad_sequence(
                non_zero_query_point, padding_value=-1
            ).permute(1, 0, 2)
            non_zero_query_mask = non_zero_query_point.sum(dim=-1) < 0
            non_zero_query_point[non_zero_query_mask] = 0

            from glee.modules.point_features import point_sample

            spatial_tokens = point_sample(
                src_mask_features.permute(2, 3, 0, 1),
                non_zero_query_point.flip(dims=(2,)).type(src_mask_features.dtype),
                align_corners=True,
            ).permute(2, 0, 1)
            spatial_tokens[pos_neg_indicator == 1] += self.glee.pn_indicator.weight[0:1]
            spatial_tokens[pos_neg_indicator == -1] += self.glee.pn_indicator.weight[1:2]

            src_spatial_queries.append(spatial_tokens)
            src_spatial_maskings.append(non_zero_query_mask)

        extra["visual_prompt_tokens"] = src_spatial_queries
        extra["visual_prompt_nonzero_mask"] = src_spatial_maskings

        return extra

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def inference(self, batch: GLEE_3DBatchedInputs) -> list[dict]:
        """Run inference and decode 3D boxes."""
        self.eval()
        out = self.forward(batch)

        results = []
        B = batch.num_images
        H, W = batch.images.shape[-2:]

        for b in range(B):
            logits_b = out.pred_logits[b]  # (Q, C)
            scores_b = out.pred_scores[b].squeeze(-1)  # (Q,)
            boxes_2d_b = out.pred_boxes_2d[b]  # (Q, 4) cxcywh
            boxes_3d_b = out.pred_boxes_3d[b] if out.pred_boxes_3d is not None else None

            # Convert cxcywh to xyxy for decoding
            cx, cy, w, h = boxes_2d_b.unbind(-1)
            boxes_xyxy = torch.stack([
                (cx - 0.5 * w) * W,
                (cy - 0.5 * h) * H,
                (cx + 0.5 * w) * W,
                (cy + 0.5 * h) * H,
            ], dim=-1)

            # Decode 3D boxes
            boxes_3d_decoded = None
            if boxes_3d_b is not None:
                intrinsics_b = batch.intrinsics[b]
                boxes_3d_decoded = self.box_coder.decode(
                    boxes_xyxy, boxes_3d_b, intrinsics_b
                )

            results.append({
                "pred_logits": logits_b,
                "pred_scores": scores_b,
                "pred_boxes_2d": boxes_xyxy,
                "pred_boxes_3d": boxes_3d_decoded,
            })

        return results
