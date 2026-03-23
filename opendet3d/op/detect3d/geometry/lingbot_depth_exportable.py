"""Export-friendly LingbotDepthBackend wrapper.

Wraps the original LingbotDepthBackend with static shapes for 1008x1008 input.
No retraining needed - same weights, same math, just export-compatible ops.

Changes from original:
  - adaptive_avg_pool2d(98->49) -> avg_pool2d(kernel=2, stride=2) [exact]
  - Dynamic base_h/base_w -> constants (49, 49)
  - DINOv2 xformers -> F.scaled_dot_product_attention [~1e-7 diff]
  - GeometryBackendOutput dict -> plain tuple return
  - No .item() calls, no dynamic Python int conversions
  - Accepts depth + intrinsics as inputs (not hardcoded)
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LingbotDepthExportable(nn.Module):
    """Static-shape wrapper around LingbotDepthBackend for ONNX export.

    For 1008x1008 input:
      base_h = base_w = 49
      neck[1] shape = (B, 256, 98, 98)
      depth_latents shape = (B, 2401, target_latent_dim)

    Inputs:
      images: (B, 3, 1008, 1008) ImageNet-normalized
      depth: (B, 1, 1008, 1008) metric depth in meters (0=monocular)
      intrinsics: (B, 3, 3) camera intrinsics in padded pixel space
    Outputs:
      depth_latents: (B, 2401, C)
      depth_map: (B, 1, 1008, 1008) metric depth
      K_pred: (B, 3, 3) predicted intrinsics
      ray_intrinsics: (B, 3, 3) intrinsics scaled to internal depth space
      confidence_map: (B, 1, 1008, 1008) depth confidence
    """

    # Pre-computed constants for 1008x1008 input
    INPUT_H = 1008
    INPUT_W = 1008
    BASE_H = 49  # round(sqrt(2400 / 1.0))
    BASE_W = 49
    PATCH_SIZE = 14
    INTERNAL_H = 49 * 14  # 686
    INTERNAL_W = 49 * 14  # 686

    def __init__(self, backend):
        """Wrap an existing LingbotDepthBackend.

        Args:
            backend: Trained LingbotDepthBackend instance
        """
        super().__init__()
        # Copy all sub-modules (shares weights, no copy)
        self.encoder = backend.encoder
        self.neck = backend.neck
        self.depth_head = backend.depth_head
        self.mask_head = getattr(backend, 'mask_head', None)
        self.intrinsic_head = getattr(backend, 'intrinsic_head', None)

        # Enable ONNX-compatible mode in encoder + DINOv2
        # This disables antialias and problematic pos embed interpolation
        self.encoder.onnx_compatible_mode = True

        # Copy config
        self.target_latent_dim = backend.target_latent_dim
        self.remap_depth_out = backend.remap_depth_out
        self.detach_depth_latents = backend.detach_depth_latents
        self.remap_depth_in = backend.remap_depth_in

        # Depth latent projection (may be Identity)
        self.latent_proj = backend.latent_proj

        # ImageNet normalization constants
        self.register_buffer('imagenet_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('imagenet_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(
        self, images: Tensor, depth: Tensor, intrinsics: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward pass with static shapes.

        Args:
            images: (B, 3, 1008, 1008) ImageNet-normalized
            depth: (B, 1, 1008, 1008) metric depth in meters (0=invalid/monocular)
            intrinsics: (B, 3, 3) camera intrinsics in padded pixel space

        Returns:
            depth_latents: (B, 2401, target_latent_dim)
            depth_map: (B, 1, 1008, 1008) predicted metric depth in meters
            K_pred: (B, 3, 3) predicted camera intrinsics (padded space)
            ray_intrinsics: (B, 3, 3) intrinsics scaled to internal depth space
            confidence_map: (B, 1, 1008, 1008) depth confidence
        """
        B = images.shape[0]
        device = images.device

        # Step 1: De-normalize from ImageNet to [0,1]
        images_01 = images * self.imagenet_std + self.imagenet_mean

        # Step 2: Use provided depth (0 values = monocular fallback, handled by encoder)
        depth_input = depth

        # Step 3: Run encoder (DINOv2 with onnx_compatible_mode=True)
        features, cls_token, _, _ = self.encoder(
            images_01, depth_input,
            self.BASE_H, self.BASE_W,  # token_rows, token_cols (static constants)
            return_class_token=True,
            remap_depth_in=self.remap_depth_in,
            enable_depth_mask=False,
        )
        # features: (B, encoder_dim, 49, 49)
        # cls_token: (B, cls_dim)

        # Step 4: Prepare features for neck
        feat_with_cls = features + cls_token[..., None, None]
        feat_list = [feat_with_cls, None, None, None, None]

        # Add UV coordinates at 5 pyramid levels
        from mdm.utils.geo import normalized_view_plane_uv
        aspect_ratio = 1.0  # 1008/1008
        for level in range(5):
            uv = normalized_view_plane_uv(
                width=self.BASE_W * 2**level,
                height=self.BASE_H * 2**level,
                aspect_ratio=aspect_ratio,
                dtype=images.dtype, device=device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)
            if feat_list[level] is None:
                feat_list[level] = uv
            else:
                feat_list[level] = torch.cat([feat_list[level], uv], dim=1)

        # Step 5: Neck forward
        neck_out = self.neck(feat_list)

        # Step 6: Extract depth_latents from neck level 1
        # neck[1]: (B, 256, 98, 98)
        neck_feat = neck_out[1]
        # Pool 98->49: exact 2x downsample = avg_pool2d(kernel=2, stride=2)
        neck_feat_pooled = F.avg_pool2d(neck_feat, kernel_size=2, stride=2)
        # (B, 256, 49, 49)

        # Flatten to sequence: (B, 256, 49, 49) -> (B, 2401, 256)
        depth_latents = neck_feat_pooled.flatten(2).permute(0, 2, 1)

        # Project to target dim
        depth_latents = self.latent_proj(depth_latents)

        if self.detach_depth_latents:
            depth_latents = depth_latents.detach()

        # Step 7: Depth head
        depth_reg = self.depth_head(neck_out)
        if isinstance(depth_reg, (list, tuple)):
            depth_reg = depth_reg[-1]
        # Upsample to input size
        depth_reg = F.interpolate(depth_reg, size=(self.INPUT_H, self.INPUT_W),
                                  mode='bilinear', align_corners=False)
        # Remap to metric depth
        depth_map = torch.clamp(depth_reg, -10, 10).exp()

        # Step 8: Predict intrinsics
        K_pred = self._predict_intrinsics(cls_token)

        # Step 9: Scale provided intrinsics to internal depth space
        # This matches _forward_test_batched line 1520-1522
        ray_intrinsics = self._scale_intrinsics(
            intrinsics,
            from_hw=(self.INPUT_H, self.INPUT_W),
            to_hw=(self.INTERNAL_H, self.INTERNAL_W),
        )

        # Step 10: Confidence map (mask head)
        confidence_map = torch.ones(B, 1, self.INPUT_H, self.INPUT_W, device=device)
        if self.mask_head is not None:
            mask_out = self.mask_head(neck_out)
            if isinstance(mask_out, (list, tuple)):
                mask_out = mask_out[-1]
            confidence_map = torch.sigmoid(mask_out)
            if confidence_map.shape[-2:] != (self.INPUT_H, self.INPUT_W):
                confidence_map = F.interpolate(confidence_map,
                    size=(self.INPUT_H, self.INPUT_W), mode='bilinear', align_corners=False)

        return depth_latents, depth_map, K_pred, ray_intrinsics, confidence_map

    def _predict_intrinsics(self, cls_token: Tensor) -> Tensor:
        """Predict camera intrinsics from cls_token."""
        B = cls_token.shape[0]
        device = cls_token.device

        if self.intrinsic_head is None:
            # Default intrinsics
            K = torch.zeros(B, 3, 3, device=device)
            K[:, 0, 0] = 1008.0
            K[:, 1, 1] = 1008.0
            K[:, 0, 2] = 504.0
            K[:, 1, 2] = 504.0
            K[:, 2, 2] = 1.0
            return K

        params = self.intrinsic_head(cls_token)  # (B, 4)
        H, W = self.INPUT_H, self.INPUT_W
        diagonal = math.sqrt(H * H + W * W)

        fx = torch.exp(torch.clamp(params[:, 0], -10, 10)) * 0.7 * diagonal
        fy = torch.exp(torch.clamp(params[:, 1], -10, 10)) * 0.7 * diagonal
        cx = torch.sigmoid(params[:, 2]) * W
        cy = torch.sigmoid(params[:, 3]) * H

        K = torch.zeros(B, 3, 3, device=device)
        K[:, 0, 0] = fx
        K[:, 1, 1] = fy
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy
        K[:, 2, 2] = 1.0
        return K

    def _scale_intrinsics(
        self, intrinsics: Tensor,
        from_hw: tuple, to_hw: tuple,
    ) -> Tensor:
        """Scale intrinsics from one image space to another.

        Copied from LingbotDepthBackend._scale_intrinsics (line 832-857).
        """
        scale_x = to_hw[1] / from_hw[1]
        scale_y = to_hw[0] / from_hw[0]

        K_scaled = intrinsics.clone()
        K_scaled[:, 0, 0] = K_scaled[:, 0, 0] * scale_x  # fx
        K_scaled[:, 0, 2] = K_scaled[:, 0, 2] * scale_x  # cx
        K_scaled[:, 1, 1] = K_scaled[:, 1, 1] * scale_y  # fy
        K_scaled[:, 1, 2] = K_scaled[:, 1, 2] * scale_y  # cy

        return K_scaled
