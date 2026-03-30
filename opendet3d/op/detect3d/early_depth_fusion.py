"""Early Depth Fusion Modules.

Two variants for fusing depth latents into visual features before the encoder:

1. EarlyDepthFusionUniDepthV2 (Concat-Add):
   Concatenate visual + depth, project back, residual add.
       delta = W * [P; D]
       output = P + delta

2. EarlyDepthFusionLingbot (ControlNet-style):
   LayerNorm depth, project depth only, residual add.
       delta = W_d @ LayerNorm(D)
       output = P + delta
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class EarlyDepthFusionUniDepthV2(nn.Module):
    """Concat-Add fusion for UniDepthV2 backend.

    Concatenates visual and depth features, projects back to visual dim,
    then adds as residual. More expressive than depth-only projection:
        delta = W_P * P + W_D * D  (from concat projection)
        output = P + delta = (I + W_P) * P + W_D * D

    Args:
        visual_dim: Dimension of visual features (e.g., 256).
        depth_dim: Dimension of depth latents (e.g., 256).
        fusion_type: Kept for config compatibility, ignored.
        zero_init: Whether to zero-initialize the projection layer.
    """

    def __init__(
        self,
        visual_dim: int = 256,
        depth_dim: int = 256,
        fusion_type: str = "concat_add",
        zero_init: bool = True,
    ):
        super().__init__()

        self.visual_dim = visual_dim
        self.depth_dim = depth_dim

        # Projection: [C + C_depth] -> [C]
        self.proj = nn.Conv2d(
            visual_dim + depth_dim,
            visual_dim,
            kernel_size=1,
            bias=True,
        )

        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        visual_feats: list[Tensor],
        depth_latents: Tensor,
        depth_latents_hw: tuple[int, int],
    ) -> list[Tensor]:
        """Fuse depth latents into visual features.

        Args:
            visual_feats: List of visual features [[B, C, H, W]].
            depth_latents: Depth features [B, N, C_depth].
            depth_latents_hw: (H_d, W_d) spatial dims of depth latents.

        Returns:
            List of fused visual features with same shapes as input.
        """
        if depth_latents is None or len(visual_feats) == 0:
            return visual_feats

        B, N, C_depth = depth_latents.shape
        H_d, W_d = depth_latents_hw

        assert N == H_d * W_d, f"depth_latents N={N} != H_d*W_d={H_d * W_d}"

        # Reshape: [B, N, C_depth] -> [B, C_depth, H_d, W_d]
        depth_2d = depth_latents.permute(0, 2, 1).reshape(
            B, C_depth, H_d, W_d
        )

        fused_feats = []
        for visual_feat in visual_feats:
            B_v, C_v, H_v, W_v = visual_feat.shape
            assert C_v == self.visual_dim

            # Interpolate depth to match visual spatial size
            if (H_d, W_d) != (H_v, W_v):
                depth_resized = torch.nn.functional.interpolate(
                    depth_2d,
                    size=(H_v, W_v),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                depth_resized = depth_2d

            # Concat + project + residual
            concat_feat = torch.cat([visual_feat, depth_resized], dim=1)
            proj_feat = self.proj(concat_feat)
            fused_feat = visual_feat + proj_feat

            fused_feats.append(fused_feat)

        return fused_feats


class EarlyDepthFusionLingbot(nn.Module):
    """ControlNet-style fusion for Lingbot depth backend.

    LayerNorm on depth latents, project depth only, residual add.
    Visual features never pass through any trainable layer, preserving
    the pretrained distribution.

    Args:
        visual_dim: Dimension of visual features (e.g., 256).
        depth_dim: Dimension of depth latents (e.g., 256).
        fusion_type: Kept for config compatibility, ignored.
        zero_init: Whether to zero-initialize the projection layer.
    """

    def __init__(
        self,
        visual_dim: int = 256,
        depth_dim: int = 256,
        fusion_type: str = "concat_add",
        zero_init: bool = True,
    ):
        super().__init__()

        self.visual_dim = visual_dim
        self.depth_dim = depth_dim

        # Normalize depth_latents to unit scale before projection.
        # depth_latents (raw neck output, std~4.0) and visual features
        # (SAM3 FPN, std~0.017) differ by ~230x. LayerNorm brings depth
        # to mean=0, std=1 so the projection sees consistent input scale.
        self.depth_norm = nn.LayerNorm(depth_dim)

        # Projection: depth_dim -> visual_dim (depth only)
        self.proj = nn.Conv2d(
            depth_dim,
            visual_dim,
            kernel_size=1,
            bias=True,
        )

        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        visual_feats: list[Tensor],
        depth_latents: Tensor,
        depth_latents_hw: tuple[int, int],
    ) -> list[Tensor]:
        """Fuse depth latents into visual features.

        Args:
            visual_feats: List of visual features [[B, C, H, W]].
            depth_latents: Depth features [B, N, C_depth].
            depth_latents_hw: (H_d, W_d) spatial dims of depth latents.

        Returns:
            List of fused visual features with same shapes as input.
        """
        if depth_latents is None or len(visual_feats) == 0:
            return visual_feats

        B, N, C_depth = depth_latents.shape
        H_d, W_d = depth_latents_hw

        assert N == H_d * W_d, f"depth_latents N={N} != H_d*W_d={H_d * W_d}"

        # Normalize depth_latents to unit scale
        # Cast to match LayerNorm dtype (AMP bf16 compatibility)
        depth_latents = depth_latents.to(self.depth_norm.weight.dtype)
        depth_latents = self.depth_norm(depth_latents)

        # Reshape: [B, N, C_depth] -> [B, C_depth, H_d, W_d]
        depth_2d = depth_latents.permute(0, 2, 1).reshape(
            B, C_depth, H_d, W_d
        )

        fused_feats = []
        for visual_feat in visual_feats:
            B_v, C_v, H_v, W_v = visual_feat.shape
            assert C_v == self.visual_dim

            # Interpolate depth to match visual spatial size
            if (H_d, W_d) != (H_v, W_v):
                depth_resized = torch.nn.functional.interpolate(
                    depth_2d,
                    size=(H_v, W_v),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                depth_resized = depth_2d

            # Project depth only + residual add
            delta = self.proj(depth_resized)
            fused_feat = visual_feat + delta

            self._last_delta_mean_abs = delta.detach().abs().mean().item()

            fused_feats.append(fused_feat)

        return fused_feats


# Backward compatibility alias
EarlyDepthFusion = EarlyDepthFusionUniDepthV2
