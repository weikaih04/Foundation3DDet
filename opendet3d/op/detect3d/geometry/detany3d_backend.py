"""DetAny3DGeometryBackend: Wraps DetAny3D's depth estimation system.

This backend uses:
- DINOv2 for feature extraction (DINO-only, no SAM backbone)
- Unidepth_Decoder (DINO-only version) for depth prediction
- SILogLoss on depth + intrinsic angles (phi/theta) - directly from DetAny3D

The backend encapsulates the complete DetAny3D geometry pipeline.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .base import GeometryBackendBase, GeometryBackendOutput

# Import loss directly from DetAny3D - DO NOT reimplement
from DetAny3D.train_utils import SILogLoss

# Import utilities from DetAny3D
from DetAny3D.detect_anything.modeling.depth_predictor.unidepth_utils import (
    generate_rays,
)

# ImageNet normalization constants (same as DetAny3D uses for DINO)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DetAny3DGeometryBackend(GeometryBackendBase):
    """Backend using DINOv2 + DINO-only Unidepth_Decoder.

    This backend:
    - Uses DINOv2 (vit_large) for feature extraction
    - Uses a DINO-only version of Unidepth_Decoder (no SAM backbone)
    - Computes SILogLoss on depth and intrinsic angles (phi/theta)
    - Uses the EXACT same loss function from DetAny3D/train_utils.py

    The depth_latents output is extracted from out_features at the selected
    resolution (1/8, 1/4, or 1/2) and projected to target_latent_dim.

    Args:
        dino_encoder: DINOv2 encoder (from DetAny3D.detect_anything.modeling.backbones)
        depth_decoder: DINO-only Unidepth_Decoder
        depth_loss_weight: Weight for depth SILog loss.
        phi_loss_weight: Weight for phi angle loss.
        theta_loss_weight: Weight for theta angle loss.
        depth_coefficient: SILog coefficient for depth loss.
        phi_coefficient: SILog coefficient for phi angle loss.
        theta_coefficient: SILog coefficient for theta angle loss.
        freeze_dino: Whether to freeze DINOv2 weights.
        detach_depth_latents: Whether to detach depth_latents from graph.
            If True, gradients from 3D head won't flow back to depth head.
    """

    def __init__(
        self,
        dino_encoder: nn.Module,
        depth_decoder: nn.Module,
        depth_loss_weight: float = 10.0,
        phi_loss_weight: float = 2.5,
        theta_loss_weight: float = 2.5,
        depth_coefficient: float = 0.15,
        phi_coefficient: float = 1.0,
        theta_coefficient: float = 1.0,
        freeze_dino: bool = False,
        detach_depth_latents: bool = False,
    ) -> None:
        """Initialize the DetAny3DGeometryBackend."""
        super().__init__(detach_depth_latents=detach_depth_latents)
        self.dino_encoder = dino_encoder
        self.depth_decoder = depth_decoder

        # Loss weights (applied after loss computation)
        self.depth_loss_weight = depth_loss_weight
        self.phi_loss_weight = phi_loss_weight
        self.theta_loss_weight = theta_loss_weight

        # SILog coefficients (used inside SILogLoss function)
        self.depth_coefficient = depth_coefficient
        self.phi_coefficient = phi_coefficient
        self.theta_coefficient = theta_coefficient

        if freeze_dino:
            self.dino_encoder.freeze()

        # Register normalization constants
        self.register_buffer(
            "pixel_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def _normalize_image(self, images: Tensor) -> Tensor:
        """Normalize images to ImageNet format for DINOv2.

        Args:
            images: Input images in [0, 255] range, shape [B, 3, H, W]

        Returns:
            Normalized images for DINOv2
        """
        # Normalize to [0, 1] then apply ImageNet normalization
        images = images.float() / 255.0
        images = (images - self.pixel_mean) / self.pixel_std
        return images

    def _prepare_input_dict(
        self,
        images: Tensor,
        dino_features: list[Tensor],
        dino_tokens: list[Tensor],
        intrinsics: Tensor | None,
        image_hw: tuple[int, int],
    ) -> dict:
        """Prepare input dict for DINO-only Unidepth_Decoder.

        This matches the format expected by Unidepth_Decoder.forward(),
        but only with DINO features (no SAM features).
        """
        B = images.shape[0]
        H, W = image_hw
        device = images.device

        input_dict = {
            "image_for_dino": images,  # Already normalized
            "dino_feature": dino_features,  # List of 4 feature maps
            "dino_token": dino_tokens,  # List of 4 cls tokens
            "vit_pad_size": torch.tensor(
                [[H // 14, W // 14]] * B,  # DINOv2 uses patch_size=14
                device=device,
            ),
        }

        if intrinsics is not None:
            input_dict["gt_intrinsic"] = intrinsics

        return input_dict

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

        Args:
            images: Input images [B, 3, H, W] in [0, 255] range
            depth_feats: Ignored (we use our own DINO features)
            intrinsics: Camera intrinsics [B, 3, 3]
            image_hw: Image height and width tuple
            depth_gt: Ground truth depth [B, H, W] or [B, 1, H, W]
            depth_mask: Valid depth mask [B, H, W] or [B, 1, H, W]

        Returns:
            GeometryBackendOutput with depth_map, depth_latents, K_pred, losses
        """
        # 1. Normalize images for DINOv2 (ImageNet normalization)
        images_normalized = self._normalize_image(images)

        # 2. Extract DINO features
        dino_features, dino_tokens = self.dino_encoder(images_normalized)

        # 3. Prepare input dict for decoder
        input_dict = self._prepare_input_dict(
            images=images_normalized,
            dino_features=dino_features,
            dino_tokens=dino_tokens,
            intrinsics=intrinsics,
            image_hw=image_hw,
        )

        # 4. Run DINO-only decoder
        outputs = self.depth_decoder(input_dict)

        depth_map = outputs["depth_maps"]  # [B, 1, H, W]
        depth_latents = outputs.get("depth_latents")  # [B, N, C] - already flattened
        pred_K = outputs.get("pred_K")  # [B, 3, 3]

        # Apply optional detach
        depth_latents = self._maybe_detach_latents(depth_latents)

        # 5. Compute losses using DetAny3D's SILogLoss
        losses = self._compute_losses(
            depth_map=depth_map,
            depth_gt=depth_gt,
            depth_mask=depth_mask,
            pred_K=pred_K,
            gt_K=intrinsics,
            image_hw=image_hw,
        )

        return GeometryBackendOutput(
            depth_map=depth_map,
            depth_latents=depth_latents,
            K_pred=pred_K,
            aux={
                "confidence": outputs.get("confidence"),
                "rays": outputs.get("rays"),
                "depth_latents_hw": outputs.get("depth_latents_hw"),
            },
            losses=losses,
        )

    def _compute_losses(
        self,
        depth_map: Tensor,
        depth_gt: Tensor | None,
        depth_mask: Tensor | None,
        pred_K: Tensor | None,
        gt_K: Tensor,
        image_hw: tuple[int, int],
    ) -> dict[str, Tensor]:
        """Compute DetAny3D geometry losses.

        Uses the EXACT SILogLoss function from DetAny3D/train_utils.py
        """
        losses = {}

        # Depth loss
        if depth_gt is not None:
            # Ensure depth_gt is 2D [B, H, W]
            if depth_gt.ndim == 4:
                depth_gt = depth_gt.squeeze(1)
            if depth_mask is not None and depth_mask.ndim == 4:
                depth_mask = depth_mask.squeeze(1)

            depth_pred = depth_map.squeeze(1)  # [B, H, W]
            H, W = depth_gt.shape[-2:]

            # Interpolate if sizes don't match
            if depth_pred.shape[-2:] != (H, W):
                depth_pred = F.interpolate(
                    depth_pred.unsqueeze(1),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

            # Use DetAny3D's SILogLoss directly
            depth_loss = SILogLoss(
                depth_pred,
                depth_gt,
                coefficient=self.depth_coefficient,
                masks=depth_mask,
                log_mode=True,
            )
            if depth_loss is not None:
                losses["depth_loss"] = depth_loss * self.depth_loss_weight

        # Intrinsic loss (phi/theta angles)
        if pred_K is not None and gt_K is not None:
            _, gt_angles = generate_rays(gt_K, image_hw)
            _, pred_angles = generate_rays(pred_K, image_hw)

            phi_gt, theta_gt = gt_angles[..., 0], gt_angles[..., 1]
            phi_pred, theta_pred = pred_angles[..., 0], pred_angles[..., 1]

            # Use DetAny3D's SILogLoss directly
            loss_phi = SILogLoss(
                phi_pred, phi_gt, coefficient=self.phi_coefficient
            )
            loss_theta = SILogLoss(
                theta_pred, theta_gt, coefficient=self.theta_coefficient
            )

            if loss_phi is not None:
                losses["loss_phi"] = loss_phi * self.phi_loss_weight
            if loss_theta is not None:
                losses["loss_theta"] = loss_theta * self.theta_loss_weight

        return losses

    @torch.no_grad()
    def forward_test(
        self,
        images: Tensor,
        depth_feats: list[Tensor] | None,
        intrinsics: Tensor,
        image_hw: tuple[int, int],
        **kwargs,
    ) -> GeometryBackendOutput:
        """Forward pass for inference."""
        # 1. Normalize images for DINOv2
        images_normalized = self._normalize_image(images)

        # 2. Extract DINO features
        dino_features, dino_tokens = self.dino_encoder(images_normalized)

        # 3. Prepare input dict
        input_dict = self._prepare_input_dict(
            images=images_normalized,
            dino_features=dino_features,
            dino_tokens=dino_tokens,
            intrinsics=intrinsics,
            image_hw=image_hw,
        )

        # 4. Run decoder
        outputs = self.depth_decoder(input_dict)

        depth_map = outputs["depth_maps"]
        depth_latents = outputs.get("depth_latents")  # [B, N, C] - already flattened
        pred_K = outputs.get("pred_K")

        # Apply optional detach
        depth_latents = self._maybe_detach_latents(depth_latents)

        return GeometryBackendOutput(
            depth_map=depth_map,
            depth_latents=depth_latents,
            K_pred=pred_K,
            aux={
                "confidence": outputs.get("confidence"),
                "rays": outputs.get("rays"),
                "depth_latents_hw": outputs.get("depth_latents_hw"),
            },
            losses={},
        )

