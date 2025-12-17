"""UniDepthHeadDinoBackend: UniDepth V1 Head + DINOv2 backbone.

This backend uses:
- DINOv2 for feature extraction (all 24 blocks, grouped into 4 scales)
- Multi-scale feature aggregation (like UniDepthV2)
- UniDepthHead for depth prediction (like original 3D-MOOD)
- SILogLoss for depth supervision

This is a hybrid approach combining:
- UniDepthV2's feature extraction strategy (all DINOv2 blocks with grouping)
- 3D-MOOD's depth head architecture
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from opendet3d.op.loss.silog_loss import SILogLoss

from .base import GeometryBackendBase, GeometryBackendOutput

# Import DINOv2 model factory from DetAny3D
from DetAny3D.detect_anything.modeling.backbones import _make_dinov2_model

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

if TYPE_CHECKING:
    from opendet3d.op.detect3d.grounding_dino_3d.depth import UniDepthHead


class UniDepthHeadDinoBackend(GeometryBackendBase):
    """Backend using DINOv2 (all blocks, grouped) + UniDepthHead.

    This backend:
    - Uses DINOv2 for feature extraction (all 24 blocks for vit_large)
    - Groups blocks into 4 scales using mean aggregation (like UniDepthV2)
    - Converts grouped features to FPN-like format [B, C, H, W]
    - Feeds to UniDepthHead for depth prediction
    - Computes SILogLoss for depth supervision

    Args:
        dino_model: DINOv2 model variant ("vit_large", "vit_base", "vit_small").
        dino_pretrained: Path to pretrained DINO weights, or "" for default hub weights.
        depth_head: The UniDepthHead module.
        depth_loss_weight: Weight for the depth loss.
        freeze_dino: Whether to freeze DINOv2 weights.
        detach_depth_latents: Whether to detach depth_latents from graph.
        stacking_mode: How to aggregate blocks ("mean", "max", "last").
    """

    # DINOv2 model configurations
    DINO_CONFIGS = {
        "vit_small": {"embed_dim": 384, "depth": 12},
        "vit_base": {"embed_dim": 768, "depth": 12},
        "vit_large": {"embed_dim": 1024, "depth": 24},
        "vit_giant2": {"embed_dim": 1536, "depth": 40},
    }

    def __init__(
        self,
        dino_model: str = "vit_large",
        dino_pretrained: str = "",
        depth_head: UniDepthHead | None = None,
        depth_loss_weight: float = 10.0,
        freeze_dino: bool = True,
        detach_depth_latents: bool = False,
        stacking_mode: str = "mean",
        target_embed_dims: int = 256,
    ) -> None:
        """Initialize the UniDepthHeadDinoBackend."""
        super().__init__(detach_depth_latents=detach_depth_latents)

        # Get DINOv2 config
        if dino_model not in self.DINO_CONFIGS:
            raise ValueError(
                f"Unknown dino_model: {dino_model}. "
                f"Available: {list(self.DINO_CONFIGS.keys())}"
            )
        dino_config = self.DINO_CONFIGS[dino_model]
        self.dino_depth = dino_config["depth"]
        self.dino_embed_dim = dino_config["embed_dim"]

        # Build DINOv2 encoder - extract ALL blocks
        self.dino_encoder = _make_dinov2_model(
            arch_name=dino_model,
            pretrained=dino_pretrained,
            output_idx=list(range(1, self.dino_depth + 1)),  # All blocks
            drop_path_rate=0.0,
            num_register_tokens=0,
            use_norm=True,
        )

        # Freeze DINOv2 if requested
        if freeze_dino:
            for param in self.dino_encoder.parameters():
                param.requires_grad = False

        # Define slicing ranges for grouping blocks (like UniDepthV2)
        # For vit_large (24 blocks): [(0, 6), (6, 12), (12, 18), (18, 24)]
        # For vit_small/base (12 blocks): [(0, 3), (3, 6), (6, 9), (9, 12)]
        num_groups = 4
        blocks_per_group = self.dino_depth // num_groups
        self.slices_encoder_range = [
            (i * blocks_per_group, (i + 1) * blocks_per_group)
            for i in range(num_groups)
        ]

        # Stacking function for aggregation
        self.stacking_mode = stacking_mode
        if stacking_mode == "mean":
            self.stacking_fn = self._mean_stack
        elif stacking_mode == "max":
            self.stacking_fn = self._max_stack
        elif stacking_mode == "last":
            self.stacking_fn = self._last_stack
        else:
            raise ValueError(f"Unknown stacking_mode: {stacking_mode}")

        # UniDepthHead
        if depth_head is None:
            raise ValueError("depth_head must be provided")
        self.depth_head = depth_head

        # Feature projection: DINOv2 embed_dim -> target_embed_dims
        # UniDepthHead expects all input_dims to be equal to embed_dims
        self.target_embed_dims = target_embed_dims
        if self.dino_embed_dim != target_embed_dims:
            # Need projection layers for each scale
            self.feature_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.dino_embed_dim, target_embed_dims),
                    nn.LayerNorm(target_embed_dims),
                )
                for _ in range(num_groups)
            ])
        else:
            self.feature_projections = None

        # Loss
        self.depth_loss = SILogLoss()
        self.depth_loss_weight = depth_loss_weight

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

    @staticmethod
    def _mean_stack(tensors: list[Tensor]) -> Tensor:
        """Average stack of tensors."""
        if len(tensors) == 1:
            return tensors[0]
        return torch.stack(tensors, dim=-1).mean(dim=-1)

    @staticmethod
    def _max_stack(tensors: list[Tensor]) -> Tensor:
        """Max stack of tensors."""
        if len(tensors) == 1:
            return tensors[0]
        return torch.stack(tensors, dim=-1).max(dim=-1).values

    @staticmethod
    def _last_stack(tensors: list[Tensor]) -> Tensor:
        """Take last tensor."""
        return tensors[-1]

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

    def _extract_dino_features(self, images: Tensor) -> list[Tensor]:
        """Extract and group DINOv2 features.

        Args:
            images: Normalized images [B, 3, H, W]

        Returns:
            List of 4 grouped features [B, target_dim, H/14, W/14]
        """
        # Extract all DINOv2 blocks
        dino_features, dino_tokens = self.dino_encoder(images)
        # dino_features: List[depth] of [B, H/14, W/14, embed_dim]
        # dino_tokens: List[depth] of [B, 1, embed_dim]

        # Group blocks and aggregate (like UniDepthV2)
        grouped_features = []
        for idx, (i, j) in enumerate(self.slices_encoder_range):
            # Aggregate blocks [i:j] using stacking function
            feat = self.stacking_fn(dino_features[i:j])
            # feat: [B, H/14, W/14, dino_embed_dim]

            # Project to target dimension if needed
            if self.feature_projections is not None:
                # feat: [B, H/14, W/14, dino_embed_dim]
                B, H, W, C = feat.shape
                feat = feat.reshape(B, H * W, C)  # [B, H*W, C]
                feat = self.feature_projections[idx](feat)  # [B, H*W, target_dim]
                feat = feat.reshape(B, H, W, -1)  # [B, H, W, target_dim]

            # Convert to [B, C, H, W] format for UniDepthHead
            feat = feat.permute(0, 3, 1, 2)  # → [B, target_dim, H/14, W/14]
            grouped_features.append(feat)

        return grouped_features

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
            images: Input images [B, 3, H, W] in [0, 255] range.
            depth_feats: Ignored (we use our own DINO features).
            intrinsics: Camera intrinsics [B, 3, 3].
            image_hw: Input image size (H, W).
            depth_gt: Ground truth depth [B, H, W] (optional).
            depth_mask: Valid depth mask [B, H, W] (optional).

        Returns:
            GeometryBackendOutput with depth_map, depth_latents, and losses.
        """
        # 1. Normalize images for DINOv2
        images_normalized = self._normalize_image(images)

        # 2. Extract and group DINOv2 features
        depth_feats = self._extract_dino_features(images_normalized)
        # depth_feats: List[4] of [B, embed_dim, H/14, W/14]

        # 3. Run UniDepthHead
        depth_preds, depth_latent = self.depth_head(
            depth_feats, intrinsics, image_hw
        )

        # Apply optional detach
        depth_latent = self._maybe_detach_latents(depth_latent)

        # 4. Compute losses
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
            images: Input images [B, 3, H, W] in [0, 255] range.
            depth_feats: Ignored (we use our own DINO features).
            intrinsics: Camera intrinsics [B, 3, 3].
            image_hw: Input image size (H, W).

        Returns:
            GeometryBackendOutput with depth_map and depth_latents.
        """
        # 1. Normalize images for DINOv2
        images_normalized = self._normalize_image(images)

        # 2. Extract and group DINOv2 features
        depth_feats = self._extract_dino_features(images_normalized)

        # 3. Run UniDepthHead
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

