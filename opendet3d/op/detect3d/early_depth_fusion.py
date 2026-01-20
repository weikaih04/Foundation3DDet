"""Early Depth Fusion Module.

Fuses depth latents into visual features before the encoder.
This allows depth information to participate in encoder's self-attention
and text cross-attention, enabling better 3D-aware feature learning.

Design follows Concat-Add with Zero-Init:
- Fusion location: After backbone, before encoder
- Fusion method: Concat + Zero-Init projection + Residual add
- Goal: Let depth info participate in full encoder processing

Math (more expressive than pure ControlNet):
    delta = W_P * P + W_D * D  (from concat projection)
    output = P + delta = (I + W_P) * P + W_D * D
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class EarlyDepthFusion(nn.Module):
    """Early fusion of depth latents into visual features (before encoder).

    Fuses depth features from geometry backend into visual features from backbone,
    before they enter the transformer encoder. This allows depth information to
    participate in the full encoder processing (self-attention + text cross-attention).

    Architecture:
        visual_feats [B, C, H, W] ──┐
                                     ├─→ Concat [B, C+depth_dim, H, W]
        depth_latents [B, N, C_d] ──┤        ↓
            └─ reshape & interpolate─┘   Conv1x1 (zero-init) [B, C, H, W]
                                              ↓
        visual_feats [B, C, H, W] ──→ Add ───┘
                                       ↓
                                  fused [B, C, H, W]

    Args:
        visual_dim: Dimension of visual features (e.g., 256)
        depth_dim: Dimension of depth latents (e.g., 256)
        fusion_type: Type of fusion, "concat_add" (default)
        zero_init: Whether to zero-initialize the projection layer

    Forward Args:
        visual_feats: List of visual features from backbone FPN
            For SAM3 single-scale: [Tensor of shape [B, C, H, W]]
        depth_latents: Depth features [B, N, C_depth]
        depth_latents_hw: Spatial dimensions (H_d, W_d) of depth latents

    Returns:
        List of fused visual features with same shapes as input
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
        self.fusion_type = fusion_type

        # Projection layer: [C+depth_dim] -> [C]
        self.proj = nn.Conv2d(
            visual_dim + depth_dim,
            visual_dim,
            kernel_size=1,
            bias=True,
        )

        # Zero initialization - delta=0 at start, preserves pretrained features
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
            visual_feats: List of visual features from FPN
                For SAM3 (single-scale): [[B, C, H, W]]
            depth_latents: Depth features [B, N, C_depth] where N = H_d * W_d
            depth_latents_hw: (H_d, W_d) spatial dimensions of depth latents

        Returns:
            List of fused visual features with same structure as input
        """
        if depth_latents is None or len(visual_feats) == 0:
            return visual_feats

        B, N, C_depth = depth_latents.shape
        H_d, W_d = depth_latents_hw

        assert N == H_d * W_d, f"depth_latents N={N} != H_d*W_d={H_d * W_d}"

        # Reshape depth_latents: [B, N, C_depth] -> [B, C_depth, H_d, W_d]
        depth_2d = depth_latents.permute(0, 2, 1).reshape(B, C_depth, H_d, W_d)

        # Fuse with each level of visual features
        fused_feats = []
        for visual_feat in visual_feats:
            # visual_feat: [B, C, H, W]
            B_v, C_v, H_v, W_v = visual_feat.shape
            assert C_v == self.visual_dim, f"visual_dim mismatch: {C_v} != {self.visual_dim}"

            # Interpolate depth to match visual feature's spatial size
            if (H_d, W_d) != (H_v, W_v):
                depth_resized = torch.nn.functional.interpolate(
                    depth_2d,
                    size=(H_v, W_v),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                depth_resized = depth_2d

            # Concat: [B, C, H, W] + [B, C_depth, H, W] -> [B, C+C_depth, H, W]
            concat_feat = torch.cat([visual_feat, depth_resized], dim=1)

            # Project back: [B, C+C_depth, H, W] -> [B, C, H, W]
            proj_feat = self.proj(concat_feat)

            # Residual add
            fused_feat = visual_feat + proj_feat

            fused_feats.append(fused_feat)

        return fused_feats
