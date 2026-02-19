"""GeometryBackendBase: Abstract interface for depth/geometry backends.

Each backend is a self-contained geometry module that:
- Extracts features using its own method (DINO, Swin+FPN, etc.)
- Runs its own depth head
- Computes its own geometry losses

The interface provides a unified way to plug different geometry systems
into the 3D-MOOD / GroundingDINO3D framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict

import torch
from torch import Tensor, nn


class GeometryBackendOutput(TypedDict, total=False):
    """Output dictionary from GeometryBackend.

    Attributes:
        depth_map: Predicted depth map [B, 1, H, W] in metric scale.
        depth_latents: Depth latent tokens [B, N, C] for 3D head.
            Dimension C is aligned to target_latent_dim (default: 128).
        K_pred: Predicted camera intrinsics [B, 3, 3] (optional).
        ray_intrinsics: Intrinsics to use for ray_embeddings generation [B, 3, 3].
            This may be adjusted intrinsics for DINOv2-based backends.
        ray_image_hw: Image (H, W) to use for ray_embeddings generation.
            This corresponds to the space where depth_latents were computed.
        ray_downsample: Downsample factor for ray_embeddings (8 or 16).
            Must match the spatial resolution of depth_latents.
        aux: Auxiliary outputs (rays, points, confidence, etc.).
        losses: Dictionary of geometry losses (only in training).
    """

    depth_map: Tensor
    depth_latents: Tensor
    K_pred: Tensor | None
    ray_intrinsics: Tensor
    ray_image_hw: tuple[int, int]
    ray_downsample: int
    aux: dict[str, Tensor]
    losses: dict[str, Tensor]


class GeometryBackendBase(nn.Module, ABC):
    """Abstract base class for geometry backends.

    Each concrete implementation wraps a complete geometry pipeline:
    - Feature extraction (backbone + neck specific to this backend)
    - Depth head
    - Loss computation

    This allows switching between different depth systems (UniDepthHead,
    DetAny3D, UniDepthV2) without changing the main GroundingDINO3D code.

    Args:
        detach_depth_latents: If True, detach depth_latents before returning.
            This prevents gradients from the 3D head from flowing back to
            the depth head. Useful when you want to freeze depth training
            but still use its features for 3D detection.
    """

    # Whether this backend's depth decoder already incorporates ray/camera info.
    # If True, the 3D head does NOT need a separate camera prompt branch,
    # because the depth_latents are already ray-aware.
    # - UniDepthV2 / DetAny3D: True (decoder fuses rays internally)
    # - UniDepthHead (v1): False (no ray info in decoder)
    is_ray_aware: bool = False

    def __init__(self, detach_depth_latents: bool = False) -> None:
        """Initialize the geometry backend.

        Args:
            detach_depth_latents: Whether to detach depth_latents from the graph.
        """
        super().__init__()
        self.detach_depth_latents = detach_depth_latents

    def _maybe_detach_latents(self, depth_latents: Tensor | None) -> Tensor | None:
        """Optionally detach depth latents from computation graph.

        Args:
            depth_latents: Depth latents [B, N, C] or None

        Returns:
            Detached latents if detach_depth_latents is True, otherwise unchanged
        """
        if depth_latents is not None and self.detach_depth_latents:
            return depth_latents.detach()
        return depth_latents

    @abstractmethod
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
            images: Input images [B, 3, H, W].
            depth_feats: Multi-scale features from FPN [B, C, H_i, W_i] (for
                backends that use external features like UniDepthHead).
                Can be None for backends with their own encoder (e.g., UniDepthV2).
            intrinsics: Camera intrinsics [B, 3, 3].
            image_hw: Input image size (H, W).
            depth_gt: Ground truth depth [B, H, W] (optional).
            depth_mask: Valid depth mask [B, H, W] (optional).
            **kwargs: Additional backend-specific arguments.

        Returns:
            GeometryBackendOutput containing:
                - depth_map: [B, 1, H, W]
                - depth_latents: [B, N, C]
                - K_pred: [B, 3, 3] or None
                - aux: dict of auxiliary outputs
                - losses: dict of loss tensors
        """
        raise NotImplementedError

    @torch.no_grad()
    @abstractmethod
    def forward_test(
        self,
        images: Tensor,
        depth_feats: list[Tensor] | None,
        intrinsics: Tensor,
        image_hw: tuple[int, int],
        **kwargs,
    ) -> GeometryBackendOutput:
        """Forward pass for inference (no loss computation).

        Args:
            images: Input images [B, 3, H, W].
            depth_feats: Multi-scale features from FPN (optional).
            intrinsics: Camera intrinsics [B, 3, 3].
            image_hw: Input image size (H, W).
            **kwargs: Additional backend-specific arguments.

        Returns:
            GeometryBackendOutput containing:
                - depth_map: [B, 1, H, W]
                - depth_latents: [B, N, C]
                - K_pred: [B, 3, 3] or None
                - aux: dict of auxiliary outputs
                - losses: empty dict
        """
        raise NotImplementedError

    def forward(
        self,
        images: Tensor,
        depth_feats: list[Tensor] | None,
        intrinsics: Tensor,
        image_hw: tuple[int, int],
        depth_gt: Tensor | None = None,
        depth_mask: Tensor | None = None,
        **kwargs,
    ) -> GeometryBackendOutput:
        """Forward pass (dispatches to train or test based on mode)."""
        if self.training:
            return self.forward_train(
                images=images,
                depth_feats=depth_feats,
                intrinsics=intrinsics,
                image_hw=image_hw,
                depth_gt=depth_gt,
                depth_mask=depth_mask,
                **kwargs,
            )
        return self.forward_test(
            images=images,
            depth_feats=depth_feats,
            intrinsics=intrinsics,
            image_hw=image_hw,
            depth_gt=depth_gt,
            **kwargs,
        )

