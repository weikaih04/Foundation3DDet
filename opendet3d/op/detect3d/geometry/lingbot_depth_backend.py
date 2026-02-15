"""LingbotDepthBackend: LingBot-Depth geometry backend for 3D-MOOD.

Uses DINOv2 RGB-D encoder with mixed depth input strategy (per-sample):
- 70% monocular: zero depth input
- 20% patch-masked: patch-level random masking (60-90% ratio, following
  the Masked Depth Modeling paper) for depth completion training
- 10% copy-through: full depth_gt as input
- Inference: always zero depth (monocular mode)

Intrinsic prediction: MLP on cls_token predicts camera K.
is_ray_aware = False so the 3D head's camera prompt branch is active.

Depth loss: L1 + SILog on all valid pixels of full depth_gt (all samples).
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


class LingbotDepthBackend(GeometryBackendBase):
    """Backend using LingBot-Depth (DINOv2 RGB-D encoder + ConvStack decoder).

    Loads a pretrained MDMModel and decomposes it into:
    - encoder: DINOv2_RGBD_Encoder (RGB-D feature extraction)
    - neck: ConvStack (multiscale refinement)
    - depth_head: ConvStack (depth regression)

    depth_latents are extracted from encoder features (before neck)
    and projected to target_latent_dim via a learned linear layer.

    During training, each sample independently gets one of three modes:
    - monocular (zero depth): prob = monocular_prob (default 0.7)
    - patch-masked depth: prob = masked_prob (default 0.2)
    - copy-through (full depth): prob = 1 - monocular - masked (0.1)
    During inference, always zero depth.

    Args:
        pretrained_model: Path or HuggingFace repo ID for MDMModel.
        num_tokens: Number of base tokens for the encoder.
        target_latent_dim: Target dimension for depth_latents.
        depth_loss_weight: Weight for L1 depth loss.
        silog_loss_weight: Weight for SILog depth loss (scale-invariant).
        monocular_prob: Probability of zero depth input (training).
        masked_prob: Probability of patch-masked depth input (training).
        mask_ratio_range: (min, max) masking ratio for patch-masked mode.
        mask_patch_size: Patch size for depth masking grid.
        camera_loss_weight: Weight for ray-based L2 camera loss.
        ssi_loss_weight: Weight for EdgeGuidedLocalSSI depth loss.
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
            "robbyant/lingbot-depth-postrain-dc-vitl14"
        ),
        num_tokens: int = 2400,
        target_latent_dim: int = 128,
        depth_loss_weight: float = 1.0,
        silog_loss_weight: float = 0.5,
        monocular_prob: float = 0.7,
        masked_prob: float = 0.2,
        mask_ratio_range: tuple[float, float] = (0.6, 0.9),
        mask_patch_size: int = 14,
        camera_loss_weight: float = 1.0,
        ssi_loss_weight: float = 0.5,
        detach_depth_latents: bool = True,
        encoder_freeze_blocks: int = 0,
    ) -> None:
        """Initialize the LingbotDepthBackend."""
        super().__init__(detach_depth_latents=detach_depth_latents)

        self.num_tokens = num_tokens
        self.target_latent_dim = target_latent_dim
        self.depth_loss_weight = depth_loss_weight
        self.silog_loss_weight = silog_loss_weight
        self.monocular_prob = monocular_prob
        self.masked_prob = masked_prob
        self.mask_ratio_range = mask_ratio_range
        self.mask_patch_size = mask_patch_size
        self.camera_loss_weight = camera_loss_weight
        self.ssi_loss_weight = ssi_loss_weight

        # SILog loss (scale-invariant, same as UniDepthV2)
        self.silog_loss = SILogLoss(
            scale_pred_weight=0.15,
        ) if silog_loss_weight > 0 else None

        # EdgeGuidedLocalSSI loss (edge-guided scale-shift-invariant)
        if ssi_loss_weight > 0:
            from unidepth.ops.losses import EdgeGuidedLocalSSI

            self.ssi_loss = EdgeGuidedLocalSSI.build({
                "name": "EdgeGuidedLocalSSI",
                "weight": 1.0,
                "output_fn": "sqrt",
                "input_fn": "log1i",
                "use_global": True,
                "min_samples": 6,
            })
        else:
            self.ssi_loss = None

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

        # Get dimensions from loaded model
        encoder_dim = self.encoder.output_projections[0].out_channels
        cls_dim = self.encoder.dim_features

        # Latent projection: encoder features -> target_latent_dim
        self.latent_proj = nn.Linear(encoder_dim, target_latent_dim)

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
            f"[LingbotDepth] Initialized: encoder_dim={encoder_dim}, "
            f"cls_dim={cls_dim}, num_tokens={num_tokens}, "
            f"target_latent_dim={target_latent_dim}\n"
            f"  remap_depth_in={self.remap_depth_in}, "
            f"remap_depth_out={self.remap_depth_out}\n"
            f"  depth strategy: {monocular_prob:.0%} monocular / "
            f"{masked_prob:.0%} patch-masked / "
            f"{copythrough_prob:.0%} copy-through\n"
            f"  mask_ratio_range={mask_ratio_range}, "
            f"mask_patch_size={mask_patch_size}\n"
            f"  depth_loss_weight={depth_loss_weight}, "
            f"silog_loss_weight={silog_loss_weight}, "
            f"ssi_loss_weight={ssi_loss_weight}, "
            f"camera_loss_weight={camera_loss_weight}\n"
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
        fx = torch.exp(params[:, 0]) * 0.7 * diagonal
        fy = torch.exp(params[:, 1]) * 0.7 * diagonal
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
    ) -> tuple[Tensor, Tensor, Tensor, int, int]:
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

        # Extract depth_latents from encoder features (BEFORE neck)
        depth_latents = features.flatten(2).permute(
            0, 2, 1
        )  # [B, N, encoder_dim]
        depth_latents = self.latent_proj(
            depth_latents
        )  # [B, N, target_latent_dim]

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
        if self.remap_depth_out == "exp":
            depth_map = depth_reg.exp()  # [B, 1, H, W]
        elif self.remap_depth_out == "linear":
            depth_map = depth_reg
        else:
            raise ValueError(
                f"Invalid remap_depth_out: {self.remap_depth_out}"
            )

        return depth_map, depth_latents, cls_token, base_h, base_w

    def _compute_losses(
        self,
        depth_map: Tensor,
        depth_gt: Tensor | None,
        depth_mask: Tensor | None,
        K_pred: Tensor,
        intrinsics: Tensor,
        image_hw: tuple[int, int],
        images: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute depth and camera losses.

        Depth loss: masked L1 on raw metric depth (LingBot-Depth paper
        Section 2.3).
        Camera loss: ray-based L2 RMSE (same as UniDepthV2).

        Args:
            depth_map: [B, 1, H, W] predicted metric depth.
            depth_gt: [B, H, W] or [B, 1, H, W] ground truth depth.
            depth_mask: [B, H, W] or [B, 1, H, W] valid depth mask.
            K_pred: [B, 3, 3] predicted intrinsics.
            intrinsics: [B, 3, 3] ground truth intrinsics.
            image_hw: (H, W) image dimensions.
            images: [B, 3, H, W] normalized images for SSI edge detection.

        Returns:
            Dictionary of loss tensors.
        """
        losses = {}

        # Depth loss: masked L1 on raw metric depth
        if depth_gt is not None:
            depth_pred = depth_map.squeeze(1)  # [B, H, W]

            if depth_gt.ndim == 4:
                depth_gt = depth_gt.squeeze(1)  # [B, H, W]

            valid_mask = depth_gt > 0
            if depth_mask is not None:
                if depth_mask.ndim == 4:
                    depth_mask = depth_mask.squeeze(1)
                valid_mask = valid_mask & depth_mask.bool()

            if valid_mask.any():
                depth_loss = F.l1_loss(
                    depth_pred[valid_mask], depth_gt[valid_mask]
                )
            else:
                depth_loss = torch.tensor(
                    0.0,
                    device=depth_map.device,
                    dtype=depth_map.dtype,
                )

            losses["depth_l1"] = depth_loss * self.depth_loss_weight

            # SILog loss: scale-invariant log loss (same as UniDepthV2)
            if self.silog_loss is not None:
                silog_loss = self.silog_loss(
                    depth_pred, depth_gt, mask=valid_mask
                )
                losses["depth_silog"] = silog_loss * self.silog_loss_weight

            # EdgeGuidedLocalSSI loss (same as UniDepthV2)
            if self.ssi_loss is not None and images is not None:
                depth_pred_4d = depth_map  # [B, 1, H, W]
                depth_gt_4d = depth_gt.unsqueeze(1)  # [B, 1, H, W]
                mask_4d = valid_mask.unsqueeze(1)  # [B, 1, H, W]
                images_01 = (
                    images * self.denorm_std + self.denorm_mean
                )
                ssi_loss = self.ssi_loss(
                    input=depth_pred_4d,
                    target=depth_gt_4d,
                    mask=mask_4d.float(),
                    image=images_01,
                    validity_mask=mask_4d.float(),
                )
                losses["depth_ssi"] = (
                    ssi_loss.mean() * self.ssi_loss_weight
                )

        # Camera loss: ray-based MSE (same as UniDepthV2 Regression loss
        # on rays, see unidepthv2.py compute_losses "camera" section)
        rays_pred, _ = generate_rays(K_pred, image_hw)
        rays_gt, _ = generate_rays(intrinsics, image_hw)
        camera_loss = F.mse_loss(rays_pred, rays_gt)
        losses["camera_ray"] = camera_loss * self.camera_loss_weight

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

        Args:
            images: [B, 3, H, W] 3D-MOOD normalized images.
            depth_feats: Ignored (we use our own encoder).
            intrinsics: [B, 3, 3] camera intrinsics.
            image_hw: (H, W) image dimensions.
            depth_gt: [B, H, W] ground truth depth.
            depth_mask: [B, H, W] valid depth mask.

        Returns:
            GeometryBackendOutput.
        """
        B = images.shape[0]
        H, W = image_hw

        # Prepare depth input with mixed strategy
        depth_input = self._prepare_depth_input(
            depth_gt, depth_mask, B, H, W, images.device
        )

        # Run encoder + decoder pipeline
        depth_map, depth_latents, cls_token, base_h, base_w = (
            self._run_encoder_and_decoder(images, depth_input, image_hw)
        )

        depth_latents = self._maybe_detach_latents(depth_latents)

        # Predict intrinsics from cls_token
        K_pred = self._predict_intrinsics(cls_token, H, W)

        # Compute losses
        losses = self._compute_losses(
            depth_map, depth_gt, depth_mask, K_pred, intrinsics, image_hw,
            images=images,
        )

        # Ray intrinsics: GT K scaled to encoder internal resolution
        # The 3D head generates ray_embeddings at ray_image_hw, then
        # flat_interpolate downsamples by ray_downsample to match
        # depth_latents spatial grid (base_h, base_w)
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
            aux={"depth_latents_hw": (base_h, base_w)},
            losses=losses,
        )

    @torch.no_grad()
    def forward_test(
        self,
        images: Tensor,
        depth_feats: list[Tensor] | None,
        intrinsics: Tensor,
        image_hw: tuple[int, int],
        **kwargs,
    ) -> GeometryBackendOutput:
        """Forward pass for inference (monocular mode, zero depth input).

        Args:
            images: [B, 3, H, W] 3D-MOOD normalized images.
            depth_feats: Ignored.
            intrinsics: [B, 3, 3] camera intrinsics.
            image_hw: (H, W) image dimensions.

        Returns:
            GeometryBackendOutput.
        """
        H, W = image_hw

        # Always zero depth input for inference
        depth_map, depth_latents, cls_token, base_h, base_w = (
            self._run_encoder_and_decoder(images, None, image_hw)
        )

        depth_latents = self._maybe_detach_latents(depth_latents)

        K_pred = self._predict_intrinsics(cls_token, H, W)

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
            aux={"depth_latents_hw": (base_h, base_w)},
            losses={},
        )
