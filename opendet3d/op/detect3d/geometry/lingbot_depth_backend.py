"""LingbotDepthBackend: LingBot-Depth geometry backend for 3D-MOOD.

Uses DINOv2 RGB-D encoder with mixed depth input strategy (per-sample):
- 70% monocular: zero depth input
- 20% patch-masked: patch-level random masking (60-90% ratio, following
  the Masked Depth Modeling paper) for depth completion training
- 10% copy-through: full depth_gt as input
- Inference: always zero depth (monocular mode)

Intrinsic prediction: MLP on cls_token predicts camera K.
is_ray_aware = False so the 3D head's camera prompt branch is active.

Depth loss: L1 + MoGe2 affine-invariant losses (global, local, edge)
  + confidence mask BCE on all valid pixels.
Camera loss: ray-based MSE (same approach as UniDepthV2).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import GeometryBackendBase, GeometryBackendOutput
from opendet3d.op.geometric.ray import generate_rays
from opendet3d.op.loss.silog_loss import SILogLoss

from moge.train.losses import (
    affine_invariant_global_loss,
    affine_invariant_local_loss,
    edge_loss,
    mask_bce_loss,
)


def backproject_depth_to_points(
    depth: Tensor, K: Tensor, H: int, W: int
) -> Tensor:
    """Back-project depth map to 3D points using camera intrinsics.

    Args:
        depth: [B, 1, H, W] or [B, H, W] metric depth.
        K: [B, 3, 3] camera intrinsics.
        H: Image height.
        W: Image width.

    Returns:
        points: [B, H, W, 3] 3D points in camera space (x, y, z).
    """
    u = torch.arange(W, device=depth.device, dtype=depth.dtype)
    v = torch.arange(H, device=depth.device, dtype=depth.dtype)
    u, v = torch.meshgrid(u, v, indexing="xy")  # [H, W]

    z = depth.squeeze(1) if depth.ndim == 4 else depth  # [B, H, W]
    fx = K[:, 0, 0]  # [B]
    fy = K[:, 1, 1]  # [B]
    cx = K[:, 0, 2]  # [B]
    cy = K[:, 1, 2]  # [B]

    x = (u[None] - cx[:, None, None]) / fx[:, None, None] * z
    y = (v[None] - cy[:, None, None]) / fy[:, None, None] * z
    return torch.stack([x, y, z], dim=-1)  # [B, H, W, 3]


class LingbotDepthBackend(GeometryBackendBase):
    """Backend using LingBot-Depth (DINOv2 RGB-D encoder + ConvStack decoder).

    Loads a pretrained MDMModel and decomposes it into:
    - encoder: DINOv2_RGBD_Encoder (RGB-D feature extraction)
    - neck: ConvStack (multiscale refinement)
    - depth_head: ConvStack (depth regression)

    depth_latents are extracted from neck level 1 output (after 2 ResBlocks,
    256-dim, 2x encoder resolution) and pooled to encoder grid size.
    This matches UniDepthV2's approach of using decoder intermediate features.

    During training, each sample independently gets one of three modes:
    - monocular (zero depth): prob = monocular_prob (default 0.7)
    - patch-masked depth: prob = masked_prob (default 0.2)
    - copy-through (full depth): prob = 1 - monocular - masked (0.1)
    During inference, always zero depth.

    Args:
        pretrained_model: Path or HuggingFace repo ID for MDMModel.
        num_tokens: Number of base tokens for the encoder.
        target_latent_dim: Target dimension for depth_latents.
            Neck level 1 outputs 256-dim; if target != 256, a Linear
            projection is applied. Use 256 to avoid projection.
        depth_loss_weight: Weight for L1 depth loss.
        silog_loss_weight: Weight for SILog depth loss (scale-invariant).
        affine_global_weight: Weight for MoGe2 affine-invariant global loss.
        affine_local_weight: Weight for MoGe2 affine-invariant local loss.
        edge_loss_weight: Weight for MoGe2 edge loss.
        mask_loss_weight: Weight for confidence mask BCE loss.
        monocular_prob: Probability of zero depth input (training).
        masked_prob: Probability of patch-masked depth input (training).
        mask_ratio_range: (min, max) masking ratio for patch-masked mode.
        mask_patch_size: Patch size for depth masking grid.
        camera_loss_weight: Weight for ray-based L2 camera loss.
        detach_depth_latents: Whether to detach depth_latents from graph.
        encoder_freeze_blocks: Number of encoder transformer blocks to
            freeze (from the beginning). ViT-L has 24 blocks; e.g. 20
            freezes blocks[0..19], only training the last 4.
    """

    # Encoder does not fuse camera rays; 3D head needs camera prompt
    is_ray_aware: bool = False

    def __init__(
        self,
        pretrained_model: str = (
            "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"
        ),
        num_tokens: int = 2400,
        target_latent_dim: int = 128,
        depth_loss_weight: float = 1.0,
        silog_loss_weight: float = 0.5,
        affine_global_weight: float = 1.0,
        affine_local_weight: float = 1.0,
        edge_loss_weight: float = 1.0,
        mask_loss_weight: float = 0.1,
        monocular_prob: float = 0.7,
        masked_prob: float = 0.2,
        mask_ratio_range: tuple[float, float] = (0.6, 0.9),
        mask_patch_size: int = 14,
        camera_loss_weight: float = 1.0,
        detach_depth_latents: bool = True,
        encoder_freeze_blocks: int = 0,
    ) -> None:
        """Initialize the LingbotDepthBackend."""
        super().__init__(detach_depth_latents=detach_depth_latents)

        self.num_tokens = num_tokens
        self.target_latent_dim = target_latent_dim
        self.depth_loss_weight = depth_loss_weight
        self.silog_loss_weight = silog_loss_weight
        self.affine_global_weight = affine_global_weight
        self.affine_local_weight = affine_local_weight
        self.edge_loss_weight = edge_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.monocular_prob = monocular_prob
        self.masked_prob = masked_prob
        self.mask_ratio_range = mask_ratio_range
        self.mask_patch_size = mask_patch_size
        self.camera_loss_weight = camera_loss_weight

        # SILog loss (scale-invariant)
        self.silog_loss = SILogLoss(
            scale_pred_weight=0.15,
        ) if silog_loss_weight > 0 else None

        # Load pretrained MDMModel and decompose into sub-modules
        from mdm.model.v2 import MDMModel

        print(
            f"[LingbotDepth] Loading pretrained model: "
            f"{pretrained_model}"
        )
        mdm_model = MDMModel.from_pretrained(pretrained_model)

        self.encoder = mdm_model.encoder
        self.neck = mdm_model.neck
        self.depth_head = mdm_model.depth_head
        self.remap_depth_in = mdm_model.remap_depth_in
        self.remap_depth_out = mdm_model.remap_depth_out

        # Load mask_head from pretrained model (confidence prediction)
        if hasattr(mdm_model, "mask_head"):
            self.mask_head = mdm_model.mask_head
            print("[LingbotDepth] mask_head loaded from checkpoint")
        else:
            self.mask_head = None
            print(
                "[LingbotDepth] WARNING: mask_head not found in "
                "checkpoint, confidence prediction disabled"
            )

        # Get dimensions from loaded model
        cls_dim = self.encoder.dim_features

        # Neck level 1 outputs 256-dim features.
        # If target_latent_dim != 256, project; otherwise Identity.
        self._neck_latent_dim = 256
        if target_latent_dim != self._neck_latent_dim:
            self.latent_proj = nn.Linear(
                self._neck_latent_dim, target_latent_dim
            )
        else:
            self.latent_proj = nn.Identity()

        # Intrinsic prediction head: cls_token -> camera K
        # Same parameterization as UniDepthV2 CameraHead:
        # exp(raw_f) * 0.7 * diagonal for focal length,
        # sigmoid(raw_c) * W/H for principal point.
        # Init: exp(0)=1.0 gives fx ~ 0.7*diag, sigmoid(0)=0.5 gives cx=W/2
        self.intrinsic_head = nn.Sequential(
            nn.LayerNorm(cls_dim),
            nn.Linear(cls_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )
        nn.init.zeros_(self.intrinsic_head[-1].weight)
        nn.init.zeros_(self.intrinsic_head[-1].bias)

        # De-normalization buffers: convert 3D-MOOD normalized images
        # back to [0,1] for the encoder (which does its own ImageNet norm)
        self.register_buffer(
            "denorm_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "denorm_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        # Delete reference to full model (sub-modules survive via self)
        del mdm_model

        # torch.compile for encoder (controlled by SAM3_COMPILE env var)
        import os
        if os.environ.get("SAM3_COMPILE", "0") == "1":
            self.encoder = torch.compile(self.encoder)
            print("[LingbotDepth] torch.compile ENABLED for encoder")

        # Freeze the first N transformer blocks of the encoder backbone.
        # ViT-L has 24 blocks; e.g. encoder_freeze_blocks=20 freezes
        # blocks[0..19] and only trains blocks[20..23] + patch_embed +
        # norm + output_projections + neck + depth_head + new heads.
        num_blocks = len(self.encoder.backbone.blocks)
        encoder_freeze_blocks = min(encoder_freeze_blocks, num_blocks)
        if encoder_freeze_blocks > 0:
            bb = self.encoder.backbone
            # Freeze everything in backbone first
            for p in bb.parameters():
                p.requires_grad = False
            # Unfreeze the last (num_blocks - freeze_blocks) blocks
            for i in range(encoder_freeze_blocks, num_blocks):
                for p in bb.blocks[i].parameters():
                    p.requires_grad = True
            # Unfreeze final norm (after all blocks)
            for p in bb.norm.parameters():
                p.requires_grad = True

        copythrough_prob = 1.0 - monocular_prob - masked_prob
        freeze_msg = (
            f"  encoder freeze: {encoder_freeze_blocks}/{num_blocks}"
            f" blocks frozen"
        )
        print(
            f"[LingbotDepth] Initialized: "
            f"cls_dim={cls_dim}, num_tokens={num_tokens}, "
            f"depth_latents=neck[1] (256-dim, pooled)\n"
            f"  remap_depth_in={self.remap_depth_in}, "
            f"remap_depth_out={self.remap_depth_out}\n"
            f"  depth strategy: {monocular_prob:.0%} monocular / "
            f"{masked_prob:.0%} patch-masked / "
            f"{copythrough_prob:.0%} copy-through\n"
            f"  mask_ratio_range={mask_ratio_range}, "
            f"mask_patch_size={mask_patch_size}\n"
            f"  losses: L1={depth_loss_weight}, "
            f"affine_global={affine_global_weight}, "
            f"affine_local={affine_local_weight}, "
            f"edge={edge_loss_weight}, "
            f"mask_bce={mask_loss_weight}, "
            f"camera_ray={camera_loss_weight}\n"
            f"  mask_head={'loaded' if self.mask_head is not None else 'none'}\n"
            f"{freeze_msg}"
        )

    def load_pretrained_weights(self) -> None:
        """No-op: weights already loaded in __init__ via from_pretrained."""
        pass

    def _compute_token_grid(
        self, H: int, W: int
    ) -> tuple[int, int]:
        """Compute token grid dimensions from image aspect ratio.

        Same formula as MDMModel.forward lines 110-115.

        Args:
            H: Image height.
            W: Image width.

        Returns:
            (base_h, base_w) token grid dimensions.
        """
        aspect_ratio = W / H
        base_h = round(math.sqrt(self.num_tokens / aspect_ratio))
        base_w = round(math.sqrt(self.num_tokens * aspect_ratio))
        return base_h, base_w

    def _prepare_depth_input(
        self,
        depth_gt: Tensor | None,
        depth_mask: Tensor | None,
        B: int,
        H: int,
        W: int,
        device: torch.device,
    ) -> Tensor | None:
        """Prepare depth input with mixed strategy for training.

        Per-sample mode selection:
        - [0, monocular_prob): zero depth (monocular)
        - [monocular_prob, monocular_prob + masked_prob): patch-masked
        - [monocular_prob + masked_prob, 1.0): copy-through (full depth)

        Args:
            depth_gt: Ground truth depth [B, H, W] or [B, 1, H, W].
            depth_mask: Valid depth mask [B, H, W] or [B, 1, H, W].
            B: Batch size.
            H: Image height.
            W: Image width.
            device: Tensor device.

        Returns:
            depth_input [B, 1, H, W] or None if no depth_gt.
        """
        if depth_gt is None:
            return None

        if depth_gt.ndim == 3:
            depth_gt = depth_gt.unsqueeze(1)  # [B, 1, H, W]

        # Apply depth_mask if provided
        if depth_mask is not None:
            if depth_mask.ndim == 3:
                depth_mask = depth_mask.unsqueeze(1)
            depth_gt = depth_gt * depth_mask.float()

        depth_input = torch.zeros_like(depth_gt)
        rand_vals = torch.rand(B, device=device)
        masked_threshold = self.monocular_prob + self.masked_prob

        for i in range(B):
            if rand_vals[i] < self.monocular_prob:
                # Monocular: keep zeros
                pass
            elif rand_vals[i] < masked_threshold:
                # Patch-level random masking
                depth_input[i] = self._patch_mask_depth(
                    depth_gt[i], H, W, device
                )
            else:
                # Copy-through: full depth
                depth_input[i] = depth_gt[i]

        return depth_input

    def _patch_mask_depth(
        self,
        depth: Tensor,
        H: int,
        W: int,
        device: torch.device,
    ) -> Tensor:
        """Apply patch-level random masking to depth map.

        Following the MDM paper: randomly mask 60-90% of patches,
        zeroing out entire patch regions.

        Args:
            depth: [1, H, W] single-sample depth map.
            H: Image height.
            W: Image width.
            device: Tensor device.

        Returns:
            Masked depth [1, H, W] with some patches zeroed out.
        """
        ps = self.mask_patch_size
        grid_h = H // ps
        grid_w = W // ps
        num_patches = grid_h * grid_w

        # Random masking ratio in [min, max]
        lo, hi = self.mask_ratio_range
        mask_ratio = torch.rand(1, device=device).item() * (hi - lo) + lo
        num_masked = int(num_patches * mask_ratio)

        # Random permutation: first num_masked patches are masked (0)
        perm = torch.randperm(num_patches, device=device)
        keep = torch.ones(num_patches, device=device)
        keep[perm[:num_masked]] = 0.0

        # Reshape to spatial grid and upsample to image size
        keep = keep.view(1, 1, grid_h, grid_w)
        keep = F.interpolate(
            keep, size=(grid_h * ps, grid_w * ps), mode="nearest"
        )  # [1, 1, grid_h*ps, grid_w*ps]

        # Pad if image size not divisible by patch size
        pad_h = H - grid_h * ps
        pad_w = W - grid_w * ps
        if pad_h > 0 or pad_w > 0:
            keep = F.pad(keep, (0, pad_w, 0, pad_h), value=1.0)

        return depth * keep.squeeze(0)  # [1, H, W]

    def _predict_intrinsics(
        self, cls_token: Tensor, H: int, W: int
    ) -> Tensor:
        """Predict camera intrinsics from cls_token.

        Same parameterization as UniDepthV2 CameraHead.fill_intrinsics:
        - fx = exp(raw) * 0.7 * diagonal
        - fy = exp(raw) * 0.7 * diagonal
        - cx = sigmoid(raw) * W
        - cy = sigmoid(raw) * H

        Args:
            cls_token: [B, cls_dim] class token from encoder.
            H: Image height (original pixel space).
            W: Image width (original pixel space).

        Returns:
            K_pred: [B, 3, 3] predicted intrinsics in pixel coords.
        """
        params = self.intrinsic_head(cls_token)  # [B, 4]

        diagonal = (H**2 + W**2) ** 0.5
        fx = torch.exp(params[:, 0].clamp(-10, 10)) * 0.7 * diagonal
        fy = torch.exp(params[:, 1].clamp(-10, 10)) * 0.7 * diagonal
        cx = torch.sigmoid(params[:, 2]) * W
        cy = torch.sigmoid(params[:, 3]) * H

        B = cls_token.shape[0]
        K_pred = torch.zeros(
            B, 3, 3, device=cls_token.device, dtype=cls_token.dtype
        )
        K_pred[:, 0, 0] = fx
        K_pred[:, 1, 1] = fy
        K_pred[:, 0, 2] = cx
        K_pred[:, 1, 2] = cy
        K_pred[:, 2, 2] = 1.0

        return K_pred

    def _run_encoder_and_decoder(
        self,
        images: Tensor,
        depth_input: Tensor | None,
        image_hw: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor, int, int, list[Tensor]]:
        """Run encoder + neck + depth_head pipeline.

        Replicates MDMModel.forward() logic (lines 98-168 of v2.py).

        Args:
            images: [B, 3, H, W] 3D-MOOD normalized images.
            depth_input: [B, 1, H, W] depth for encoder, or None.
            image_hw: Original (H, W) dimensions.

        Returns:
            depth_map: [B, 1, H, W] metric depth in meters.
            depth_latents: [B, N, target_latent_dim].
            cls_token: [B, cls_dim].
            base_h: Token grid height.
            base_w: Token grid width.
            neck_out: List of neck feature maps for mask_head.
        """
        from mdm.utils.geo import normalized_view_plane_uv

        B = images.shape[0]
        H, W = image_hw
        device, dtype = images.device, images.dtype

        # De-normalize from 3D-MOOD normalization to [0, 1]
        # 3D-MOOD: norm_img = (img_255 - mean_255) / std_255
        # Reverse: img_01 = norm_img * (std_255/255) + (mean_255/255)
        #        = norm_img * imagenet_std + imagenet_mean
        images_01 = images * self.denorm_std + self.denorm_mean

        # Compute token grid
        base_h, base_w = self._compute_token_grid(H, W)

        # Prepare depth: zeros if None (monocular mode)
        if depth_input is None:
            depth_for_encoder = torch.zeros(
                B, 1, H, W, device=device, dtype=dtype
            )
        else:
            depth_for_encoder = depth_input

        # Encoder forward: expects [0,1] images
        # (encoder internally normalizes with ImageNet stats and resizes
        # to (base_h*14, base_w*14))
        # enable_depth_mask=False avoids xformers BlockDiagonalMask
        # dependency and uses standard attention instead
        features, cls_token, _, _ = self.encoder(
            images_01,
            depth_for_encoder,
            base_h,
            base_w,
            return_class_token=True,
            remap_depth_in=self.remap_depth_in,
            enable_depth_mask=False,
        )
        # features: [B, encoder_dim, base_h, base_w]
        # cls_token: [B, cls_dim]

        # Run neck + depth_head (MDMModel.forward lines 120-148)
        aspect_ratio = W / H

        # Add cls_token to features
        feat_with_cls = features + cls_token[..., None, None]
        feat_list = [feat_with_cls, None, None, None, None]

        # Concat UV coordinates at 5 pyramid levels
        for level in range(5):
            uv = normalized_view_plane_uv(
                width=base_w * 2**level,
                height=base_h * 2**level,
                aspect_ratio=aspect_ratio,
                dtype=dtype,
                device=device,
            )
            uv = (
                uv.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
            )
            if feat_list[level] is None:
                feat_list[level] = uv
            else:
                feat_list[level] = torch.cat(
                    [feat_list[level], uv], dim=1
                )

        # Shared neck
        neck_out = self.neck(feat_list)

        # Extract depth_latents from neck level 1 (after 2 ResBlocks)
        # neck_out[1]: [B, 256, base_h*2, base_w*2]
        # Pool to (base_h, base_w) to keep N = base_h * base_w
        neck_feat = neck_out[1]  # [B, 256, base_h*2, base_w*2]
        neck_feat_pooled = F.adaptive_avg_pool2d(
            neck_feat, (base_h, base_w)
        )  # [B, 256, base_h, base_w]
        depth_latents = neck_feat_pooled.flatten(2).permute(
            0, 2, 1
        )  # [B, N, 256]
        depth_latents = self.latent_proj(depth_latents)

        # Depth head: take last output
        depth_reg = self.depth_head(neck_out)[-1]  # [B, 1, h, w]

        # Resize to original image dimensions
        depth_reg = F.interpolate(
            depth_reg,
            (H, W),
            mode="bilinear",
            align_corners=False,
        )

        # Apply output remapping
        # Clamp before exp to prevent overflow (float16 overflows at ~11,
        # float32 at ~88). Range [-10, 10] maps to depth [4.5e-5, 22026] m.
        if self.remap_depth_out == "exp":
            depth_map = depth_reg.clamp(-10, 10).exp()  # [B, 1, H, W]
        elif self.remap_depth_out == "linear":
            # Linear output can be negative; clamp to positive for
            # downstream log-based losses and 3D head depth usage.
            depth_map = depth_reg.clamp(min=1e-3)
        else:
            raise ValueError(
                f"Invalid remap_depth_out: {self.remap_depth_out}"
            )

        return depth_map, depth_latents, cls_token, base_h, base_w, neck_out

    def _run_mask_head(
        self,
        neck_out: list[Tensor],
        H: int,
        W: int,
    ) -> Tensor | None:
        """Run mask_head to produce confidence map.

        Args:
            neck_out: List of neck feature maps.
            H: Target height.
            W: Target width.

        Returns:
            confidence_map: [B, 1, H, W] sigmoid probabilities, or None.
        """
        if self.mask_head is None:
            return None
        confidence_raw = self.mask_head(neck_out)[-1]  # [B, 1, h, w]
        confidence_map = F.interpolate(
            confidence_raw,
            (H, W),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()
        return confidence_map

    @torch.autocast(device_type="cuda", enabled=False)
    def _compute_losses(
        self,
        depth_map: Tensor,
        depth_gt: Tensor | None,
        depth_mask: Tensor | None,
        K_pred: Tensor,
        intrinsics: Tensor,
        image_hw: tuple[int, int],
        confidence_map: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute depth and camera losses.

        Depth loss: masked L1 + MoGe2 affine-invariant losses (global,
        local level 4 & 16, edge) + confidence mask BCE.
        Camera loss: ray-based L2 RMSE (same as UniDepthV2).

        Args:
            depth_map: [B, 1, H, W] predicted metric depth.
            depth_gt: [B, H, W] or [B, 1, H, W] ground truth depth.
            depth_mask: [B, H, W] or [B, 1, H, W] valid depth mask.
            K_pred: [B, 3, 3] predicted intrinsics.
            intrinsics: [B, 3, 3] ground truth intrinsics.
            image_hw: (H, W) image dimensions.
            confidence_map: [B, 1, H, W] confidence from mask_head.

        Returns:
            Dictionary of loss tensors.
        """
        losses = {}
        H, W = image_hw

        # Cast to float32 for numerical stability under mixed precision
        depth_map = depth_map.float()
        K_pred = K_pred.float()
        intrinsics = intrinsics.float()
        if depth_gt is not None:
            depth_gt = depth_gt.float()
        if depth_mask is not None:
            depth_mask = depth_mask.float()
        if confidence_map is not None:
            confidence_map = confidence_map.float()

        # Depth losses
        if depth_gt is not None:
            depth_pred = depth_map.squeeze(1)  # [B, H, W]

            if depth_gt.ndim == 4:
                depth_gt = depth_gt.squeeze(1)  # [B, H, W]

            valid_mask = depth_gt > 0
            if depth_mask is not None:
                if depth_mask.ndim == 4:
                    depth_mask = depth_mask.squeeze(1)
                valid_mask = valid_mask & depth_mask.bool()

            B = depth_pred.shape[0]

            # L1 metric depth loss
            if valid_mask.any():
                depth_loss = F.l1_loss(
                    depth_pred[valid_mask], depth_gt[valid_mask]
                )
            else:
                depth_loss = depth_pred.new_tensor(0.0)

            losses["depth_l1"] = (
                depth_loss.clamp(max=10.0) * self.depth_loss_weight
            )

            # SILog loss (scale-invariant)
            if self.silog_loss is not None and valid_mask.any():
                silog_val = self.silog_loss(
                    depth_pred, depth_gt, mask=valid_mask
                )
                losses["depth_silog"] = (
                    silog_val.clamp(max=10.0)
                    * self.silog_loss_weight
                )

            # Back-project to 3D points for MoGe2 losses
            pred_points = backproject_depth_to_points(
                depth_pred, K_pred, H, W
            )  # [B, H, W, 3]
            gt_points = backproject_depth_to_points(
                depth_gt, intrinsics, H, W
            )  # [B, H, W, 3]
            # MoGe2 convention: invalid GT -> inf
            gt_points[~valid_mask] = float("inf")

            # Per-image MoGe2 losses (alignment is per-image)
            aff_global_sum = depth_pred.new_tensor(0.0)
            aff_local4_sum = depth_pred.new_tensor(0.0)
            aff_local16_sum = depth_pred.new_tensor(0.0)
            edge_sum = depth_pred.new_tensor(0.0)

            for i in range(B):
                # Affine-invariant global
                loss_g, _, scale_i = affine_invariant_global_loss(
                    pred_points[i],
                    gt_points[i],
                    align_resolution=48,
                )
                aff_global_sum = aff_global_sum + loss_g

                focal_i = K_pred[i, 0, 0]

                # Affine-invariant local level 4 (large patches)
                loss_l4, _ = affine_invariant_local_loss(
                    pred_points[i],
                    gt_points[i],
                    focal_i,
                    scale_i,
                    level=4,
                    align_resolution=24,
                    num_patches=16,
                )
                aff_local4_sum = aff_local4_sum + loss_l4

                # Affine-invariant local level 16 (medium patches)
                loss_l16, _ = affine_invariant_local_loss(
                    pred_points[i],
                    gt_points[i],
                    focal_i,
                    scale_i,
                    level=16,
                    align_resolution=24,
                    num_patches=256,
                )
                aff_local16_sum = aff_local16_sum + loss_l16

                # Edge loss (per-image)
                loss_e, _ = edge_loss(
                    pred_points[i], gt_points[i]
                )
                edge_sum = edge_sum + loss_e

            losses["affine_global"] = (
                (aff_global_sum / B).clamp(max=10.0)
                * self.affine_global_weight
            )
            losses["affine_local_4"] = (
                (aff_local4_sum / B).clamp(max=10.0)
                * self.affine_local_weight
            )
            losses["affine_local_16"] = (
                (aff_local16_sum / B).clamp(max=10.0)
                * self.affine_local_weight
            )
            losses["edge"] = (
                (edge_sum / B).clamp(max=10.0)
                * self.edge_loss_weight
            )

            # Mask BCE loss (confidence map)
            if (
                confidence_map is not None
                and self.mask_loss_weight > 0
            ):
                conf = confidence_map.squeeze(1)  # [B, H, W]
                gt_mask_fin = valid_mask  # positive: valid depth
                gt_mask_inf = ~valid_mask  # negative: invalid depth
                loss_mask, _ = mask_bce_loss(
                    conf, gt_mask_fin, gt_mask_inf
                )
                losses["mask_bce"] = (
                    loss_mask.mean().clamp(max=10.0)
                    * self.mask_loss_weight
                )

        # Camera loss: ray-based MSE (same as UniDepthV2)
        rays_pred, _ = generate_rays(K_pred, image_hw)
        rays_gt, _ = generate_rays(intrinsics, image_hw)
        camera_loss = F.mse_loss(rays_pred, rays_gt)
        losses["camera_ray"] = (
            camera_loss.clamp(max=10.0) * self.camera_loss_weight
        )

        return losses

    def _scale_intrinsics(
        self,
        intrinsics: Tensor,
        from_hw: tuple[int, int],
        to_hw: tuple[int, int],
    ) -> Tensor:
        """Scale intrinsics from one image space to another.

        Args:
            intrinsics: [B, 3, 3] intrinsics in from_hw space.
            from_hw: Source (H, W).
            to_hw: Target (H, W).

        Returns:
            Scaled intrinsics [B, 3, 3] in to_hw space.
        """
        scale_x = to_hw[1] / from_hw[1]
        scale_y = to_hw[0] / from_hw[0]

        K_scaled = intrinsics.clone()
        K_scaled[:, 0, 0] *= scale_x  # fx
        K_scaled[:, 0, 2] *= scale_x  # cx
        K_scaled[:, 1, 1] *= scale_y  # fy
        K_scaled[:, 1, 2] *= scale_y  # cy

        return K_scaled

    def _has_valid_padding(self, padding: list | None) -> bool:
        """Check if padding info is valid and non-zero."""
        if padding is None:
            return False
        return any(
            p is not None and any(v > 0 for v in p) for p in padding
        )

    def _crop_padding_single(
        self,
        image: Tensor,
        intrinsics: Tensor,
        pad_info: list[int],
        H_pad: int,
        W_pad: int,
        depth_gt: Tensor | None = None,
        depth_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, int, int, Tensor | None, Tensor | None]:
        """Crop padding from a single image and adjust intrinsics.

        Args:
            image: [1, 3, H_pad, W_pad] padded image.
            intrinsics: [1, 3, 3] padded-space intrinsics.
            pad_info: [pad_left, pad_right, pad_top, pad_bottom].
            H_pad: Padded height.
            W_pad: Padded width.
            depth_gt: [1, 1, H_pad, W_pad] or None.
            depth_mask: [1, 1, H_pad, W_pad] or [1, H_pad, W_pad] or None.

        Returns:
            (cropped_image, adjusted_intrinsics, H_orig, W_orig,
             cropped_depth_gt, cropped_depth_mask)
        """
        pad_left, pad_right, pad_top, pad_bottom = pad_info
        H_orig = H_pad - pad_top - pad_bottom
        W_orig = W_pad - pad_left - pad_right

        # Crop image
        img_cropped = image[
            :, :, pad_top : pad_top + H_orig, pad_left : pad_left + W_orig
        ]

        # Adjust intrinsics: reverse CenterPadIntrinsics
        K_cropped = intrinsics.clone()
        K_cropped[0, 0, 2] -= pad_left  # cx
        K_cropped[0, 1, 2] -= pad_top  # cy

        # Crop depth_gt
        dgt_cropped = None
        if depth_gt is not None:
            dgt_cropped = depth_gt[
                :, :,
                pad_top : pad_top + H_orig,
                pad_left : pad_left + W_orig,
            ]

        # Crop depth_mask
        dm_cropped = None
        if depth_mask is not None:
            if depth_mask.ndim == 3:
                dm_cropped = depth_mask[
                    :,
                    pad_top : pad_top + H_orig,
                    pad_left : pad_left + W_orig,
                ]
            else:
                dm_cropped = depth_mask[
                    :, :,
                    pad_top : pad_top + H_orig,
                    pad_left : pad_left + W_orig,
                ]

        return (
            img_cropped,
            K_cropped,
            H_orig,
            W_orig,
            dgt_cropped,
            dm_cropped,
        )

    def _repad_depth_latents(
        self,
        depth_latents: Tensor,
        base_h_orig: int,
        base_w_orig: int,
        base_h_pad: int,
        base_w_pad: int,
        pad_top: int,
        pad_left: int,
        H_pad: int,
        W_pad: int,
    ) -> Tensor:
        """Repad depth latents from original to padded token grid.

        Places original-resolution tokens at the correct position within
        the padded token grid, with zeros filling the padding regions.

        Args:
            depth_latents: [1, N_orig, C] original-resolution latents.
            base_h_orig: Original token grid height.
            base_w_orig: Original token grid width.
            base_h_pad: Padded token grid height.
            base_w_pad: Padded token grid width.
            pad_top: Pixel-space top padding.
            pad_left: Pixel-space left padding.
            H_pad: Padded image height.
            W_pad: Padded image width.

        Returns:
            [1, N_pad, C] depth latents in padded token grid.
        """
        if (
            base_h_orig == base_h_pad
            and base_w_orig == base_w_pad
        ):
            return depth_latents

        _, N_orig, C = depth_latents.shape

        # Reshape to spatial: [1, C, base_h_orig, base_w_orig]
        dl_2d = depth_latents.permute(0, 2, 1).reshape(
            1, C, base_h_orig, base_w_orig
        )

        # Compute token-space offsets
        pad_top_tok = round(pad_top * base_h_pad / H_pad)
        pad_left_tok = round(pad_left * base_w_pad / W_pad)

        # Clamp to valid range
        pad_top_tok = min(pad_top_tok, base_h_pad - 1)
        pad_left_tok = min(pad_left_tok, base_w_pad - 1)

        # How many original tokens fit
        h_fit = min(base_h_orig, base_h_pad - pad_top_tok)
        w_fit = min(base_w_orig, base_w_pad - pad_left_tok)

        # Create padded output with zeros
        dl_padded = torch.zeros(
            1,
            C,
            base_h_pad,
            base_w_pad,
            device=depth_latents.device,
            dtype=depth_latents.dtype,
        )
        dl_padded[
            :,
            :,
            pad_top_tok : pad_top_tok + h_fit,
            pad_left_tok : pad_left_tok + w_fit,
        ] = dl_2d[:, :, :h_fit, :w_fit]

        # Flatten back: [1, N_pad, C]
        return dl_padded.flatten(2).permute(0, 2, 1)

    def _repad_depth_map(
        self,
        depth_map: Tensor,
        pad_left: int,
        pad_right: int,
        pad_top: int,
        pad_bottom: int,
    ) -> Tensor:
        """Repad depth map from original to padded resolution.

        Args:
            depth_map: [1, 1, H_orig, W_orig].
            pad_left, pad_right, pad_top, pad_bottom: Pixel padding.

        Returns:
            [1, 1, H_pad, W_pad] with zeros in padding region.
        """
        return F.pad(
            depth_map,
            (pad_left, pad_right, pad_top, pad_bottom),
            value=0.0,
        )

    def forward_train(
        self,
        images: Tensor,
        depth_feats: list[Tensor] | None,
        intrinsics: Tensor,
        image_hw: tuple[int, int],
        depth_gt: Tensor | None = None,
        depth_mask: Tensor | None = None,
        **kwargs,
    ) -> GeometryBackendOutput:
        """Forward pass for training.

        Uses mixed depth input strategy: each sample independently
        gets monocular / patch-masked / copy-through depth input.

        When padding info is provided, crops padding before the encoder
        so LingBot-Depth processes at original resolution with correct
        aspect ratio, then repads outputs back to padded space.

        Args:
            images: [B, 3, H, W] 3D-MOOD normalized images.
            depth_feats: Ignored (we use our own encoder).
            intrinsics: [B, 3, 3] camera intrinsics.
            image_hw: (H, W) image dimensions.
            depth_gt: [B, H, W] ground truth depth.
            depth_mask: [B, H, W] valid depth mask.
            **kwargs: May contain 'padding' (list of [L,R,T,B] per image).

        Returns:
            GeometryBackendOutput.
        """
        B = images.shape[0]
        H_pad, W_pad = image_hw
        padding = kwargs.get("padding", None)

        # If no valid padding, use original batched code path
        if not self._has_valid_padding(padding):
            return self._forward_train_batched(
                images, intrinsics, image_hw, depth_gt, depth_mask
            )

        # Per-image processing at original (unpadded) resolution
        # Padded token grid (target for repadding depth_latents)
        base_h_pad, base_w_pad = self._compute_token_grid(
            H_pad, W_pad
        )

        depth_maps_list = []
        depth_latents_list = []
        K_pred_list = []
        confidence_maps_list = []
        losses_accum = {}

        for i in range(B):
            pad_info = padding[i]
            if pad_info is None or all(v == 0 for v in pad_info):
                # No padding for this image
                pad_left = pad_right = pad_top = pad_bottom = 0
                img_i = images[i : i + 1]
                K_i = intrinsics[i : i + 1]
                H_orig, W_orig = H_pad, W_pad
                dgt_i = (
                    depth_gt[i : i + 1] if depth_gt is not None
                    else None
                )
                dm_i = (
                    depth_mask[i : i + 1]
                    if depth_mask is not None
                    else None
                )
            else:
                pad_left, pad_right, pad_top, pad_bottom = pad_info
                (
                    img_i,
                    K_i,
                    H_orig,
                    W_orig,
                    dgt_i,
                    dm_i,
                ) = self._crop_padding_single(
                    images[i : i + 1],
                    intrinsics[i : i + 1],
                    pad_info,
                    H_pad,
                    W_pad,
                    (
                        depth_gt[i : i + 1]
                        if depth_gt is not None
                        else None
                    ),
                    (
                        depth_mask[i : i + 1]
                        if depth_mask is not None
                        else None
                    ),
                )

            orig_hw = (H_orig, W_orig)

            # Prepare depth input with mixed strategy (per-image)
            depth_input_i = self._prepare_depth_input(
                dgt_i, dm_i, 1, H_orig, W_orig, images.device
            )

            # Run encoder at ORIGINAL resolution (correct aspect ratio)
            (
                depth_map_i,
                depth_latents_i,
                cls_token_i,
                base_h_i,
                base_w_i,
                neck_out_i,
            ) = self._run_encoder_and_decoder(
                img_i, depth_input_i, orig_hw
            )

            # Predict intrinsics at original resolution
            K_pred_i = self._predict_intrinsics(
                cls_token_i, H_orig, W_orig
            )

            # Run mask_head for confidence map
            confidence_map_i = self._run_mask_head(
                neck_out_i, H_orig, W_orig
            )

            # Compute losses at original resolution
            losses_i = self._compute_losses(
                depth_map_i,
                dgt_i,
                dm_i,
                K_pred_i,
                K_i,
                orig_hw,
                confidence_map=confidence_map_i,
            )

            # Accumulate losses
            for key, val in losses_i.items():
                if key not in losses_accum:
                    losses_accum[key] = val
                else:
                    losses_accum[key] = losses_accum[key] + val

            # Repad depth_map back to padded resolution
            depth_map_padded_i = self._repad_depth_map(
                depth_map_i,
                pad_left,
                pad_right,
                pad_top,
                pad_bottom,
            )
            depth_maps_list.append(depth_map_padded_i)

            # Repad confidence_map back to padded resolution
            if confidence_map_i is not None:
                confidence_maps_list.append(
                    self._repad_depth_map(
                        confidence_map_i,
                        pad_left,
                        pad_right,
                        pad_top,
                        pad_bottom,
                    )
                )

            # Repad depth_latents to padded token grid
            depth_latents_padded_i = self._repad_depth_latents(
                depth_latents_i,
                base_h_i,
                base_w_i,
                base_h_pad,
                base_w_pad,
                pad_top,
                pad_left,
                H_pad,
                W_pad,
            )
            depth_latents_list.append(depth_latents_padded_i)

            # K_pred: restore to padded space (add padding offset)
            # fx, fy unchanged (padding doesn't change focal length)
            # Use non-inplace ops to preserve autograd graph
            K_pred_padded_i = K_pred_i.clone()
            K_pred_padded_i[:, 0, 2] = K_pred_i[:, 0, 2] + pad_left
            K_pred_padded_i[:, 1, 2] = K_pred_i[:, 1, 2] + pad_top
            K_pred_list.append(K_pred_padded_i)

        # Average losses across batch
        for key in losses_accum:
            losses_accum[key] = losses_accum[key] / B

        # Stack results
        depth_map = torch.cat(depth_maps_list, dim=0)
        depth_latents = torch.cat(depth_latents_list, dim=0)
        K_pred = torch.cat(K_pred_list, dim=0)
        confidence_map = (
            torch.cat(confidence_maps_list, dim=0)
            if confidence_maps_list
            else None
        )

        depth_latents = self._maybe_detach_latents(depth_latents)

        # Ray intrinsics: padded intrinsics scaled to padded token grid
        # (consistent with padded depth_latents space)
        internal_hw = (base_h_pad * 14, base_w_pad * 14)
        ray_intrinsics = self._scale_intrinsics(
            intrinsics, (H_pad, W_pad), internal_hw
        )

        return GeometryBackendOutput(
            depth_map=depth_map,
            depth_latents=depth_latents,
            K_pred=K_pred,
            ray_intrinsics=ray_intrinsics,
            ray_image_hw=internal_hw,
            ray_downsample=14,
            aux={
                "depth_latents_hw": (base_h_pad, base_w_pad),
                "confidence_map": confidence_map,
            },
            losses=losses_accum,
        )

    def _forward_train_batched(
        self,
        images: Tensor,
        intrinsics: Tensor,
        image_hw: tuple[int, int],
        depth_gt: Tensor | None,
        depth_mask: Tensor | None,
    ) -> GeometryBackendOutput:
        """Original batched forward_train path (no unpadding)."""
        B = images.shape[0]
        H, W = image_hw

        depth_input = self._prepare_depth_input(
            depth_gt, depth_mask, B, H, W, images.device
        )

        (
            depth_map, depth_latents, cls_token,
            base_h, base_w, neck_out,
        ) = self._run_encoder_and_decoder(
            images, depth_input, image_hw
        )

        depth_latents = self._maybe_detach_latents(depth_latents)
        K_pred = self._predict_intrinsics(cls_token, H, W)

        # Run mask_head for confidence map
        confidence_map = self._run_mask_head(neck_out, H, W)

        losses = self._compute_losses(
            depth_map, depth_gt, depth_mask, K_pred, intrinsics,
            image_hw, confidence_map=confidence_map,
        )

        internal_hw = (base_h * 14, base_w * 14)
        ray_intrinsics = self._scale_intrinsics(
            intrinsics, (H, W), internal_hw
        )

        return GeometryBackendOutput(
            depth_map=depth_map,
            depth_latents=depth_latents,
            K_pred=K_pred,
            ray_intrinsics=ray_intrinsics,
            ray_image_hw=internal_hw,
            ray_downsample=14,
            aux={
                "depth_latents_hw": (base_h, base_w),
                "confidence_map": confidence_map,
            },
            losses=losses,
        )

    @torch.no_grad()
    def forward_test(
        self,
        images: Tensor,
        depth_feats: list[Tensor] | None,
        intrinsics: Tensor,
        image_hw: tuple[int, int],
        depth_gt: Tensor | None = None,
        **kwargs,
    ) -> GeometryBackendOutput:
        """Forward pass for inference.

        When padding info is provided, crops padding before the encoder
        so LingBot-Depth processes at original resolution, then repads.

        Args:
            images: [B, 3, H, W] 3D-MOOD normalized images.
            depth_feats: Ignored.
            intrinsics: [B, 3, 3] camera intrinsics.
            image_hw: (H, W) image dimensions.
            depth_gt: [B, 1, H, W] depth map input (optional).
            **kwargs: May contain 'padding' (list of [L,R,T,B]).

        Returns:
            GeometryBackendOutput.
        """
        H_pad, W_pad = image_hw
        padding = kwargs.get("padding", None)

        # If no valid padding, use original batched code path
        if not self._has_valid_padding(padding):
            return self._forward_test_batched(
                images, intrinsics, image_hw, depth_gt
            )

        # Per-image processing at original resolution
        B = images.shape[0]
        base_h_pad, base_w_pad = self._compute_token_grid(
            H_pad, W_pad
        )

        depth_maps_list = []
        depth_latents_list = []
        K_pred_list = []
        confidence_maps_list = []

        for i in range(B):
            pad_info = padding[i]
            if pad_info is None or all(v == 0 for v in pad_info):
                pad_left = pad_right = pad_top = pad_bottom = 0
                img_i = images[i : i + 1]
                K_i = intrinsics[i : i + 1]
                H_orig, W_orig = H_pad, W_pad
                dgt_i = (
                    depth_gt[i : i + 1]
                    if depth_gt is not None
                    else None
                )
            else:
                pad_left, pad_right, pad_top, pad_bottom = pad_info
                (
                    img_i,
                    K_i,
                    H_orig,
                    W_orig,
                    dgt_i,
                    _,
                ) = self._crop_padding_single(
                    images[i : i + 1],
                    intrinsics[i : i + 1],
                    pad_info,
                    H_pad,
                    W_pad,
                    (
                        depth_gt[i : i + 1]
                        if depth_gt is not None
                        else None
                    ),
                )

            orig_hw = (H_orig, W_orig)

            # Use depth_gt as input if available, otherwise monocular
            depth_input_i = dgt_i if dgt_i is not None else None

            (
                depth_map_i,
                depth_latents_i,
                cls_token_i,
                base_h_i,
                base_w_i,
                neck_out_i,
            ) = self._run_encoder_and_decoder(
                img_i, depth_input_i, orig_hw
            )

            K_pred_i = self._predict_intrinsics(
                cls_token_i, H_orig, W_orig
            )

            # Run mask_head for confidence map
            confidence_map_i = self._run_mask_head(
                neck_out_i, H_orig, W_orig
            )

            # Repad depth_map
            depth_maps_list.append(
                self._repad_depth_map(
                    depth_map_i,
                    pad_left,
                    pad_right,
                    pad_top,
                    pad_bottom,
                )
            )

            # Repad confidence_map
            if confidence_map_i is not None:
                confidence_maps_list.append(
                    self._repad_depth_map(
                        confidence_map_i,
                        pad_left,
                        pad_right,
                        pad_top,
                        pad_bottom,
                    )
                )

            # Repad depth_latents
            depth_latents_list.append(
                self._repad_depth_latents(
                    depth_latents_i,
                    base_h_i,
                    base_w_i,
                    base_h_pad,
                    base_w_pad,
                    pad_top,
                    pad_left,
                    H_pad,
                    W_pad,
                )
            )

            # K_pred: restore to padded space (non-inplace for autograd)
            K_pred_padded_i = K_pred_i.clone()
            K_pred_padded_i[:, 0, 2] = K_pred_i[:, 0, 2] + pad_left
            K_pred_padded_i[:, 1, 2] = K_pred_i[:, 1, 2] + pad_top
            K_pred_list.append(K_pred_padded_i)

        depth_map = torch.cat(depth_maps_list, dim=0)
        depth_latents = torch.cat(depth_latents_list, dim=0)
        K_pred = torch.cat(K_pred_list, dim=0)
        confidence_map = (
            torch.cat(confidence_maps_list, dim=0)
            if confidence_maps_list
            else None
        )

        depth_latents = self._maybe_detach_latents(depth_latents)

        internal_hw = (base_h_pad * 14, base_w_pad * 14)
        ray_intrinsics = self._scale_intrinsics(
            intrinsics, (H_pad, W_pad), internal_hw
        )

        return GeometryBackendOutput(
            depth_map=depth_map,
            depth_latents=depth_latents,
            K_pred=K_pred,
            ray_intrinsics=ray_intrinsics,
            ray_image_hw=internal_hw,
            ray_downsample=14,
            aux={
                "depth_latents_hw": (base_h_pad, base_w_pad),
                "confidence_map": confidence_map,
            },
            losses={},
        )

    def _forward_test_batched(
        self,
        images: Tensor,
        intrinsics: Tensor,
        image_hw: tuple[int, int],
        depth_gt: Tensor | None,
    ) -> GeometryBackendOutput:
        """Original batched forward_test path (no unpadding)."""
        H, W = image_hw

        depth_input = depth_gt if depth_gt is not None else None
        (
            depth_map, depth_latents, cls_token,
            base_h, base_w, neck_out,
        ) = self._run_encoder_and_decoder(
            images, depth_input, image_hw
        )

        depth_latents = self._maybe_detach_latents(depth_latents)
        K_pred = self._predict_intrinsics(cls_token, H, W)

        # Run mask_head for confidence map
        confidence_map = self._run_mask_head(neck_out, H, W)

        internal_hw = (base_h * 14, base_w * 14)
        ray_intrinsics = self._scale_intrinsics(
            intrinsics, (H, W), internal_hw
        )

        return GeometryBackendOutput(
            depth_map=depth_map,
            depth_latents=depth_latents,
            K_pred=K_pred,
            ray_intrinsics=ray_intrinsics,
            ray_image_hw=internal_hw,
            ray_downsample=14,
            aux={
                "depth_latents_hw": (base_h, base_w),
                "confidence_map": confidence_map,
            },
            losses={},
        )
