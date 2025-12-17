"""UniDepthHeadBackend: Wraps the existing 3D-MOOD UniDepthHead.

This backend uses:
- External features from Swin + FPN (passed in as depth_feats)
- UniDepthHead for depth prediction
- SILogLoss for depth supervision

This is the default backend that maintains backward compatibility with
the original 3D-MOOD implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from opendet3d.op.loss.silog_loss import SILogLoss

from .base import GeometryBackendBase, GeometryBackendOutput

if TYPE_CHECKING:
    from opendet3d.op.detect3d.grounding_dino_3d.depth import UniDepthHead


class UniDepthHeadBackend(GeometryBackendBase):
    """Backend wrapping the original 3D-MOOD UniDepthHead.

    This backend:
    - Takes FPN features from the main model (Swin backbone + FPN)
    - Runs UniDepthHead to predict depth
    - Computes SILogLoss when depth_gt is provided

    Args:
        depth_head: The UniDepthHead module.
        depth_loss: The SILogLoss module (optional, created if not provided).
        depth_loss_weight: Weight for the depth loss.
    """

    def __init__(
        self,
        depth_head: UniDepthHead,
        depth_loss: SILogLoss | None = None,
        depth_loss_weight: float = 10.0,
        detach_depth_latents: bool = False,
    ) -> None:
        """Initialize the UniDepthHeadBackend."""
        super().__init__(detach_depth_latents=detach_depth_latents)
        self.depth_head = depth_head
        self.depth_loss = depth_loss if depth_loss is not None else SILogLoss()
        self.depth_loss_weight = depth_loss_weight

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
            images: Input images [B, 3, H, W] (not used, features come from FPN).
            depth_feats: Multi-scale features from FPN [B, C, H_i, W_i].
            intrinsics: Camera intrinsics [B, 3, 3].
            image_hw: Input image size (H, W).
            depth_gt: Ground truth depth [B, H, W] (optional).
            depth_mask: Valid depth mask [B, H, W] (optional).

        Returns:
            GeometryBackendOutput with depth_map, depth_latents, and losses.
        """
        if depth_feats is None:
            raise ValueError(
                "UniDepthHeadBackend requires depth_feats from FPN. "
                "Make sure the model provides FPN features."
            )

        # Run UniDepthHead
        depth_preds, depth_latent = self.depth_head(
            depth_feats, intrinsics, image_hw
        )

        # Apply optional detach
        depth_latent = self._maybe_detach_latents(depth_latent)

        # Compute losses
        losses: dict[str, Tensor] = {}
        if depth_gt is not None:
            depth_loss = self.depth_loss(depth_preds, depth_gt, mask=depth_mask)
            losses["depth_loss"] = depth_loss * self.depth_loss_weight

        return GeometryBackendOutput(
            depth_map=depth_preds.unsqueeze(1),  # [B, 1, H, W]
            depth_latents=depth_latent,  # [B, N, C]
            K_pred=None,  # UniDepthHead doesn't predict intrinsics
            aux={},
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
        """Forward pass for inference.

        Args:
            images: Input images [B, 3, H, W] (not used).
            depth_feats: Multi-scale features from FPN.
            intrinsics: Camera intrinsics [B, 3, 3].
            image_hw: Input image size (H, W).

        Returns:
            GeometryBackendOutput with depth_map and depth_latents.
        """
        if depth_feats is None:
            raise ValueError(
                "UniDepthHeadBackend requires depth_feats from FPN."
            )

        # Run UniDepthHead
        depth_preds, depth_latent = self.depth_head(
            depth_feats, intrinsics, image_hw
        )

        # Apply optional detach
        depth_latent = self._maybe_detach_latents(depth_latent)

        return GeometryBackendOutput(
            depth_map=depth_preds.unsqueeze(1),  # [B, 1, H, W]
            depth_latents=depth_latent,  # [B, N, C]
            K_pred=None,
            aux={},
            losses={},
        )

