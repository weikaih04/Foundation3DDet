"""DINO-only version of DetAny3D's Unidepth_Decoder.

This module is a modified version of DetAny3D's Unidepth_Decoder that only uses
DINOv2 features, removing all SAM backbone dependencies.

The original Unidepth_Decoder fuses features from both:
1. DINOv2 encoder
2. SAM's ViTDet adapter

This DINO-only version removes SAM adapters and only uses DINOv2 features.

Additionally, it supports `output_scales` parameter to select which resolution
of depth latents to return (matching UniDepthHead's behavior):
- output_scales=1: Return 1/8 resolution latents
- output_scales=2: Return 1/4 resolution latents
- output_scales=3: Return 1/2 resolution latents
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Import components from DetAny3D
from DetAny3D.detect_anything.modeling.depth_predictor.unidepth import (
    CameraHead,
    DepthHead,
    GlobalHead,
    ListAdapter,
    PositionEmbeddingSine,
    flat_interpolate,
)
from DetAny3D.detect_anything.modeling.depth_predictor.unidepth_utils import (
    generate_rays,
)


class UnidepthDecoderDinoOnly(nn.Module):
    """DINO-only version of Unidepth_Decoder.

    This decoder only uses DINOv2 features without SAM backbone features.
    It removes the SAM-related adapters from the original Unidepth_Decoder.

    The decoder predicts:
    - depth_maps: Dense depth prediction [B, 1, H, W]
    - pred_K: Predicted camera intrinsics [B, 3, 3]
    - depth_latents: Features for downstream 3D head [B, N, C]
    - confidence: Depth confidence map
    - rays: Camera rays

    Supports output_scales parameter to select latent resolution:
    - output_scales=1: Return 1/8 resolution latents
    - output_scales=2: Return 1/4 resolution latents
    - output_scales=3: Return 1/2 resolution latents
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        expansion: int = 4,
        dropout: float = 0.0,
        depth_layers: list[int] = [6, 0, 0],
        num_resolutions: int = 4,
        dino_embed_dim: int = 1024,  # DINOv2 ViT-L embed dim
        output_scales: int = 1,  # 1=1/8, 2=1/4, 3=1/2
        target_latent_dim: int = 128,  # Target dim for 3D head (matches UniDepthHead)
    ) -> None:
        """Initialize UnidepthDecoderDinoOnly.

        Args:
            hidden_dim: Hidden dimension for decoder layers.
            num_heads: Number of attention heads.
            expansion: MLP expansion ratio.
            dropout: Dropout rate.
            depth_layers: Depth of each decoder stage.
            num_resolutions: Number of feature resolutions.
            dino_embed_dim: DINOv2 embedding dimension (1024 for ViT-L).
            output_scales: Which resolution latents to return (1=1/8, 2=1/4, 3=1/2).
            target_latent_dim: Target dimension for depth_latents (for 3D head).
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_resolutions = num_resolutions
        self.downsample = 4
        self.output_scales = output_scales
        self.target_latent_dim = target_latent_dim

        assert output_scales >= 1 and output_scales <= 3, "output_scales must be 1, 2, or 3"

        # Input dimensions from DINOv2 (4 layers, all same dim for ViT)
        input_dims_dino = [dino_embed_dim] * 4  # [1024, 1024, 1024, 1024]

        # Camera layer - predicts intrinsics
        self.camera_layer = CameraHead(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
        )

        # Global layer - predicts scale and shift
        self.global_layer = GlobalHead(
            hidden_dim=hidden_dim,
            camera_dim=96,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
        )

        # DINO feature adapters only (no SAM adapters)
        self.camera_token_adapter = ListAdapter(input_dims_dino, hidden_dim)
        self.global_token_adapter = ListAdapter(input_dims_dino[:2], hidden_dim)
        self.input_adapter = ListAdapter(input_dims_dino, hidden_dim)

        # Depth layer - main depth prediction
        self.depth_layer = DepthHead(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            depths=depth_layers,
            dropout=dropout,
            camera_dim=96,
            num_resolutions=num_resolutions,
        )

        # Positional embeddings
        self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.level_embeds = nn.Parameter(
            torch.randn(num_resolutions, hidden_dim), requires_grad=True
        )
        self.level_embed_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Projection layer to align latent dimension to target_latent_dim
        # out_features dimensions: [hidden_dim//2, hidden_dim//4, hidden_dim//8]
        # for output_scales=1: hidden_dim//2 = 256 -> target_latent_dim
        # for output_scales=2: hidden_dim//4 = 128 -> target_latent_dim
        # for output_scales=3: hidden_dim//8 = 64 -> target_latent_dim
        source_dim = hidden_dim // (2 ** output_scales)
        if source_dim != target_latent_dim:
            self.latent_proj = nn.Linear(source_dim, target_latent_dim)
        else:
            self.latent_proj = nn.Identity()

    def get_adapted_features(
        self, features: list[torch.Tensor], splits: torch.Tensor
    ) -> list[torch.Tensor]:
        """Adapt DINO features to hidden dimension."""
        features = torch.cat(features, dim=-1)
        features = self.input_adapter(features, splits)
        features = torch.chunk(features, splits.shape[0], dim=-1)
        return list(features)

    def run_camera(self, cls_tokens, features, pos_embed, original_shapes, rays_gt):
        """Run camera head to predict intrinsics."""
        intrinsics, cls_tokens = self.camera_layer(
            features=features, cls_tokens=cls_tokens, pos_embed=pos_embed
        )

        # Scale intrinsics to image size
        intrinsics[:, 0, 0] = max(original_shapes) / 2 * intrinsics[:, 0, 0]
        intrinsics[:, 1, 1] = max(original_shapes) / 2 * intrinsics[:, 1, 1]
        intrinsics[:, 0, 2] = intrinsics[:, 0, 2] * original_shapes[1]
        intrinsics[:, 1, 2] = intrinsics[:, 1, 2] * original_shapes[0]

        rays = (
            rays_gt
            if rays_gt is not None
            else generate_rays(intrinsics, original_shapes)[0]
        )
        return intrinsics, rays, cls_tokens

    def run_global(self, cls_tokens, features, rays):
        """Run global head to predict scale and shift."""
        scale, shift, metric_feature = self.global_layer(
            features=features, rays=rays, cls_tokens=cls_tokens
        )
        return scale, shift, metric_feature

    def decode_depth_with_latents(
        self,
        latents_16: torch.Tensor,
        rays_embeddings: list[torch.Tensor],
        shapes: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Decode depth and return intermediate out_features.

        This is a modified version of DepthHead.decode_depth that also returns
        the intermediate out_features at different resolutions.

        Args:
            latents_16: Input latents at 1/16 resolution [B, N, C]
            rays_embeddings: List of ray embeddings at different resolutions
            shapes: Base shape (H//16, W//16)

        Returns:
            logdepth: Log depth prediction [B, 1, H, W]
            confidence: Confidence map [B, 1, H, W]
            out_features: List of features at 1/8, 1/4, 1/2 resolutions
                [B, H, W, C] for each resolution
        """
        depth_layer = self.depth_layer
        latents = latents_16
        out_features, depths, confidences = [], [], []

        for i, (up, layers, rays_embedding) in enumerate(
            zip(depth_layer.ups, depth_layer.process_layers, rays_embeddings)
        ):
            for layer in layers:
                latents = layer(latents, pos_embed=rays_embedding)
            latents = up(
                rearrange(
                    latents + rays_embedding,
                    "b (h w) c -> b c h w",
                    h=shapes[0] * int(2**i),
                    w=shapes[1] * int(2**i),
                ).contiguous()
            )
            out = rearrange(
                latents,
                "b (h w) c -> b h w c",
                h=shapes[0] * int(2 ** (1 + i)),
                w=shapes[1] * int(2 ** (1 + i)),
            )
            out_features.append(out)

        # Aggregate output and project to depth
        for i, (layer, features) in enumerate(
            zip(depth_layer.depth_mlp[::-1], out_features[::-1])
        ):
            out_depth_features = layer(features).permute(0, 3, 1, 2)
            out_depth_features = F.interpolate(
                out_depth_features, size=depth_layer.original_shapes, mode="bilinear"
            )
            depths.append(out_depth_features)
        logdepth = depth_layer.to_depth(torch.cat(depths, dim=1))

        # Aggregate output and project to confidences
        for i, (layer, features) in enumerate(
            zip(depth_layer.confidence_mlp[::-1], out_features[::-1])
        ):
            out_conf_features = layer(features).permute(0, 3, 1, 2)
            out_conf_features = F.interpolate(
                out_conf_features, size=depth_layer.original_shapes, mode="bilinear"
            )
            confidences.append(out_conf_features)
        confidence = depth_layer.to_confidence(torch.cat(confidences, dim=1))

        # Apply sigmoid to get conf in [0, 1]
        confidence = torch.sigmoid(confidence)

        return logdepth, confidence, out_features

    def forward(self, input_dict: dict) -> dict:
        """Forward pass.

        Args:
            input_dict: Dictionary containing:
                - image_for_dino: Normalized images [B, 3, H, W]
                - dino_feature: List of 4 DINO feature maps [B, h, w, C]
                - dino_token: List of 4 DINO cls tokens [B, C]
                - vit_pad_size: Patch grid size [B, 2]
                - gt_intrinsic (optional): Ground truth intrinsics [B, 3, 3]

        Returns:
            Dictionary containing:
                - depth_maps: Predicted depth [B, 1, H, W]
                - pred_K: Predicted intrinsics [B, 3, 3]
                - depth_latents: Depth latents [B, N, target_latent_dim]
                - depth_latents_hw: Tuple (H, W) of latent spatial dims
                - confidence: Confidence map
                - rays: Camera rays
        """
        B, C, H, W = input_dict["image_for_dino"].shape
        device = input_dict["image_for_dino"].device
        dtype = input_dict["image_for_dino"].dtype

        # Generate rays from GT intrinsics if provided
        if input_dict.get("gt_intrinsic") is not None:
            rays, angles = generate_rays(input_dict["gt_intrinsic"].to(dtype), (H, W))
            input_dict["rays"] = rays
            input_dict["angles"] = angles
            input_dict["gt_K"] = input_dict["gt_intrinsic"].to(dtype)

        # Get DINO tokens for camera and global heads
        global_tokens_dino = [input_dict["dino_token"][i] for i in [-2, -1]]
        camera_tokens_dino = [input_dict["dino_token"][i] for i in [-3, -2, -1, -2]]
        features_dino = [input_dict["dino_feature"][i] for i in range(4)]

        # Get patch shape from vit_pad_size
        patch_shape_H = int(input_dict["vit_pad_size"][0, 0])
        patch_shape_W = int(input_dict["vit_pad_size"][0, 1])

        # Interpolate DINO features to patch shape
        features_dino = [
            F.interpolate(
                feature.permute(0, 3, 1, 2),
                size=(patch_shape_H, patch_shape_W),
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            for feature in features_dino
        ]

        # Get level shapes
        level_shapes = sorted(
            list(set([tuple([x.shape[1], x.shape[2]]) for x in features_dino]))
        )[::-1]

        if len(level_shapes) == 1:
            level_shapes = level_shapes * self.num_resolutions

        input_shapes = level_shapes
        common_shape = level_shapes[-2]
        patch_H, patch_W = common_shape

        # Flatten and interpolate DINO features
        features_flat_dino = [
            flat_interpolate(
                rearrange(x, "b h w c -> b (h w) c"), old=input_shape, new=common_shape
            )
            for x, input_shape in zip(features_dino, input_shapes)
        ]
        features_splits = torch.tensor(
            [x.shape[-1] for x in features_flat_dino],
            device=device,
            requires_grad=False,
            dtype=torch.float32,
        )

        # Adapt DINO features
        features_dino = self.get_adapted_features(features_flat_dino, features_splits)
        features_dino = torch.stack(features_dino, dim=-1)

        # Adapt camera tokens
        cls_tokens_splits = torch.tensor(
            [x.shape[-1] for x in camera_tokens_dino],
            device=device,
            requires_grad=False,
            dtype=features_dino.dtype,
        )
        camera_tokens_dino = torch.cat(camera_tokens_dino, dim=-1)
        camera_tokens_dino = self.camera_token_adapter(
            camera_tokens_dino, cls_tokens_splits
        )
        camera_tokens_dino = torch.cat(
            torch.chunk(camera_tokens_dino, cls_tokens_splits.shape[0], dim=-1), dim=1
        )

        # Adapt global tokens
        cls_tokens_splits = torch.tensor(
            [x.shape[-1] for x in global_tokens_dino],
            device=device,
            requires_grad=False,
            dtype=torch.float32,
        )
        global_tokens_dino = torch.cat(global_tokens_dino, dim=-1)
        global_tokens_dino = self.global_token_adapter(
            global_tokens_dino, cls_tokens_splits
        )
        global_tokens_dino = torch.cat(
            torch.chunk(global_tokens_dino, cls_tokens_splits.shape[0], dim=-1), dim=1
        )

        # DINO-only: no SAM features to add
        global_tokens = global_tokens_dino
        camera_tokens = camera_tokens_dino
        features = features_dino

        # Positional embeddings
        level_embed = torch.cat(
            [
                self.level_embed_layer(self.level_embeds)[i : i + 1]
                .unsqueeze(0)
                .repeat(B, common_shape[0] * common_shape[1], 1)
                for i in range(self.num_resolutions)
            ],
            dim=1,
        )
        dummy_tensor = torch.zeros(
            B, 1, common_shape[0], common_shape[1], device=device, requires_grad=False
        )
        pos_embed = self.pos_embed(dummy_tensor)
        pos_embed = rearrange(pos_embed, "b c h w -> b (h w) c").repeat(
            1, self.num_resolutions, 1
        )

        # Run camera head
        self.camera_layer.set_shapes(common_shape)
        intrinsics, rays, camera_feature = self.run_camera(
            camera_tokens,
            features=features,
            pos_embed=pos_embed + level_embed,
            original_shapes=(H, W),
            rays_gt=input_dict.get("rays"),
        )

        # Run global head
        self.global_layer.set_shapes(common_shape)
        self.global_layer.set_original_shapes((H, W))
        scale, shift, metric_feature = self.run_global(
            global_tokens, features=features, rays=rays
        )

        # Run depth head - get intermediate latents using custom decode
        self.depth_layer.set_shapes(common_shape)
        self.depth_layer.set_original_shapes((H, W))

        # Get rays embeddings and initial latents (same as DepthHead.forward)
        features_unbind = features.unbind(dim=-1)
        rays_embeddings = self.depth_layer.project_rays(rays, common_shape)
        init_latents_16 = self.depth_layer.init_latents(features_unbind, common_shape)

        # Aggregate features: F -> D
        latents_16_0 = self.depth_layer.aggregate_16(
            init_latents_16,
            context=torch.cat(features_unbind, dim=1),
            pos_embed_context=pos_embed + level_embed,
        )
        # Aggregate camera: D -> D|E
        latents_16 = self.depth_layer.prompt_camera(
            latents_16_0, context=rays_embeddings[0]
        )

        # Decode depth with our custom method that returns out_features
        logdepth, confidence, out_features = self.decode_depth_with_latents(
            latents_16, rays_embeddings, common_shape
        )
        logdepth = logdepth.to(torch.float32, non_blocking=True)

        # Normalize in log space
        shapes_logdepth = [int(x) for x in logdepth.shape[-2:]]
        depth_normalized = F.layer_norm(logdepth, shapes_logdepth).exp()

        # Apply scale and shift
        depth = (depth_normalized + shift) * scale
        depth = F.softplus(depth, beta=10.0).to(dtype, non_blocking=True)

        # Select depth_latents based on output_scales
        # out_features: [1/8, 1/4, 1/2] -> index = output_scales - 1
        latent_idx = self.output_scales - 1
        selected_latents = out_features[latent_idx]  # [B, H, W, C]

        # Apply projection to match target_latent_dim
        depth_latents = self.latent_proj(selected_latents)  # [B, H, W, target_dim]

        # Flatten to [B, N, C] format for 3D head
        B_lat, H_lat, W_lat, C_lat = depth_latents.shape
        depth_latents = depth_latents.reshape(B_lat, H_lat * W_lat, C_lat)

        outputs = {
            "depth_maps": depth,
            "confidence": confidence,
            "depth_latents": depth_latents,  # [B, N, target_latent_dim]
            "depth_latents_hw": (H_lat, W_lat),  # For reshaping if needed
            "metric_features": metric_feature,
            "camera_features": camera_feature,
            "scale": scale,
            "shift": shift,
            "pred_K": intrinsics,
            "rays": rays,
        }
        return outputs

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"latents_pos", "level_embeds"}

