"""Geometry backend modules for 3D-MOOD.

This package provides a unified interface (GeometryBackendBase) for different
depth estimation systems:

1. UniDepthHeadBackend: Current 3D-MOOD UniDepthHead (Swin + FPN + UniDepthHead)
2. DetAny3DGeometryBackend: DetAny3D's DINOv2 + Unidepth_Decoder (DINO-only)
3. UniDepthV2GeometryBackend: Original UniDepthV2 pixel_encoder + Decoder

Each backend encapsulates:
- Its own feature extraction (DINO / Swin / etc.)
- Its own depth head architecture
- Its own loss functions

The unified interface outputs:
- depth_map: [B, 1, H, W] metric depth
- depth_latents: [B, N, C] latents for 3D head (configurable resolution)
- K_pred: [B, 3, 3] predicted intrinsics (optional)
- losses: dict of geometry losses

All backends support:
- output_scales: 1=1/8, 2=1/4, 3=1/2 resolution for depth_latents
- target_latent_dim: Target dimension for depth_latents (default 128)
- detach_depth_latents: Whether to detach latents from gradient graph
"""

from .base import GeometryBackendBase, GeometryBackendOutput
from .unidepth_head_backend import UniDepthHeadBackend
from .detany3d_backend import DetAny3DGeometryBackend
from .unidepth_v2_backend import UniDepthV2GeometryBackend
from .unidepth_decoder_dino_only import UnidepthDecoderDinoOnly
from .lingbot_depth_backend import LingbotDepthBackend

__all__ = [
    "GeometryBackendBase",
    "GeometryBackendOutput",
    "UniDepthHeadBackend",
    "DetAny3DGeometryBackend",
    "UniDepthV2GeometryBackend",
    "UnidepthDecoderDinoOnly",
    "LingbotDepthBackend",
]

