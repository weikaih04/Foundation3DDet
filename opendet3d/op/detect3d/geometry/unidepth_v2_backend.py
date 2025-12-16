"""UniDepthV2GeometryBackend: Wraps UniDepthV2's depth estimation system.

This backend uses:
- DINOv2 pixel_encoder for feature extraction
- UniDepthV2 Decoder for depth prediction
- Full loss suite: SILog, camera, invariance, SSI, confidence
  (uses UniDepthV2's native compute_losses - NOT reimplemented)

The backend encapsulates the complete UniDepthV2 geometry pipeline.
"""

from __future__ import annotations

import torch
import torchvision.transforms.functional as TF
from torch import Tensor, nn
import torch.nn.functional as F

from .base import GeometryBackendBase, GeometryBackendOutput

# ImageNet normalization constants (used by UniDepthV2)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class UniDepthV2GeometryBackend(GeometryBackendBase):
    """Backend wrapping UniDepthV2's complete depth estimation system.

    This backend:
    - Uses UniDepthV2's DINOv2 pixel_encoder for feature extraction
    - Runs UniDepthV2's Decoder to predict depth, rays, and intrinsics
    - Computes the full UniDepthV2 loss suite using native compute_losses()

    The loss functions are NOT reimplemented - they come directly from
    UniDepthV2's forward_train() and compute_losses() methods.

    Supports output_scales parameter to select latent resolution:
    - output_scales=1: Return 1/8 resolution latents
    - output_scales=2: Return 1/4 resolution latents
    - output_scales=3: Return 1/2 resolution latents

    Args:
        unidepth_model: The UniDepthV2 model instance.
        use_native_losses: Whether to use UniDepthV2's native loss functions.
        depth_loss_weight: Weight multiplier for depth losses.
        output_scales: Which resolution latents to return (1=1/8, 2=1/4, 3=1/2).
        target_latent_dim: Target dimension for depth_latents (for 3D head).
        detach_depth_latents: Whether to detach depth_latents from graph.
    """

    def __init__(
        self,
        unidepth_model: nn.Module,
        use_native_losses: bool = True,
        depth_loss_weight: float = 1.0,
        output_scales: int = 1,
        target_latent_dim: int = 128,
        detach_depth_latents: bool = False,
    ) -> None:
        """Initialize the UniDepthV2GeometryBackend."""
        super().__init__(detach_depth_latents=detach_depth_latents)
        self.unidepth_model = unidepth_model
        self.use_native_losses = use_native_losses
        self.depth_loss_weight = depth_loss_weight
        self.output_scales = output_scales
        self.target_latent_dim = target_latent_dim

        assert output_scales >= 1 and output_scales <= 3, "output_scales must be 1, 2, or 3"

        # Projection layer to align latent dimension
        # UniDepthV2 hidden_dim is typically 512
        # out_features dimensions depend on the ups configuration
        # We'll create the projection lazily in forward since we don't know dims yet
        self.latent_proj = None
        self._latent_proj_initialized = False

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
        """Normalize images to ImageNet format for UniDepthV2.

        Args:
            images: Input images in [0, 255] range, shape [B, 3, H, W]

        Returns:
            Normalized images for UniDepthV2
        """
        # Normalize to [0, 1] then apply ImageNet normalization
        images = images.float() / 255.0
        images = (images - self.pixel_mean) / self.pixel_std
        return images

    def _prepare_inputs(
        self,
        images: Tensor,
        intrinsics: Tensor | None,
        image_hw: tuple[int, int],
        depth_gt: Tensor | None = None,
        depth_mask: Tensor | None = None,
    ) -> tuple[dict, list[dict]]:
        """Prepare inputs and image_metas for UniDepthV2.

        Note: Images should already be normalized before calling this method.
        """
        from unidepth.utils.camera import Pinhole, BatchCamera

        B = images.shape[0]
        device = images.device

        # Build camera object if intrinsics provided
        camera = None
        if intrinsics is not None:
            camera = Pinhole(K=intrinsics)
            camera = BatchCamera.from_camera(camera)
            camera = camera.to(device)

        inputs = {
            "image": images,
            "camera": camera,
        }

        # Add depth GT if provided
        if depth_gt is not None:
            if depth_gt.ndim == 3:
                depth_gt = depth_gt.unsqueeze(1)  # [B, 1, H, W]
            inputs["depth"] = depth_gt

        # Add depth mask
        if depth_mask is not None:
            if depth_mask.ndim == 3:
                depth_mask = depth_mask.unsqueeze(1)
            inputs["depth_mask"] = depth_mask
        elif depth_gt is not None:
            inputs["depth_mask"] = (depth_gt > 0).float()

        # Add validity mask (same as depth_mask for now)
        if "depth_mask" in inputs:
            inputs["validity_mask"] = inputs["depth_mask"].clone()

        # Paddings (no padding by default)
        inputs["paddings"] = torch.zeros(B, 4, device=device, dtype=torch.long)

        # Build image_metas (include depth_paddings for encode_decode)
        image_metas = []
        for i in range(B):
            meta = {
                "paddings": (0, 0, 0, 0),  # (left, right, top, bottom)
                "depth_paddings": (0, 0, 0, 0),  # Required by encode_decode
                "si": False,  # scale-invariant flag
                "flip": False,
            }
            image_metas.append(meta)

        return inputs, image_metas

    def _init_latent_proj(self, source_dim: int) -> None:
        """Initialize the latent projection layer lazily."""
        if source_dim != self.target_latent_dim:
            self.latent_proj = nn.Linear(source_dim, self.target_latent_dim)
            # Move to same device as model
            device = next(self.unidepth_model.parameters()).device
            self.latent_proj = self.latent_proj.to(device)
        else:
            self.latent_proj = nn.Identity()
        self._latent_proj_initialized = True

    def _extract_depth_latents(
        self, out_features: list[Tensor] | None
    ) -> tuple[Tensor | None, tuple[int, int] | None]:
        """Extract and project depth latents from out_features.

        Args:
            out_features: List of features at [1/8, 1/4, 1/2] resolutions
                Each tensor is [B, C, H, W]

        Returns:
            depth_latents: [B, N, target_latent_dim]
            depth_latents_hw: (H, W) of latent spatial dims
        """
        if out_features is None:
            return None, None

        # Select based on output_scales (1=1/8, 2=1/4, 3=1/2)
        latent_idx = self.output_scales - 1
        selected = out_features[latent_idx]  # [B, C, H, W]

        B, C, H, W = selected.shape

        # Initialize projection layer if needed
        if not self._latent_proj_initialized:
            self._init_latent_proj(C)

        # Reshape to [B, H*W, C] and project
        latents = selected.permute(0, 2, 3, 1).reshape(B, H * W, C)
        latents = self.latent_proj(latents)

        return latents, (H, W)

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
            depth_feats: Ignored (we use UniDepthV2's own encoder)
            intrinsics: Camera intrinsics [B, 3, 3]
            image_hw: Image height and width tuple
            depth_gt: Ground truth depth [B, H, W] or [B, 1, H, W]
            depth_mask: Valid depth mask [B, H, W] or [B, 1, H, W]

        Returns:
            GeometryBackendOutput with depth_map, depth_latents, K_pred, losses
        """
        # 1. Normalize images for UniDepthV2 (ImageNet normalization)
        images_normalized = self._normalize_image(images)

        # 2. Prepare inputs
        inputs, image_metas = self._prepare_inputs(
            images=images_normalized,
            intrinsics=intrinsics,
            image_hw=image_hw,
            depth_gt=depth_gt,
            depth_mask=depth_mask,
        )

        # 3. Run UniDepthV2 forward_train (uses native compute_losses)
        if self.use_native_losses and depth_gt is not None:
            outputs, losses_dict = self.unidepth_model.forward_train(
                inputs, image_metas
            )
            # Flatten losses from {"opt": {...}, "stat": {...}} format
            losses = {}
            for loss_type in ["opt", "stat"]:
                for name, value in losses_dict.get(loss_type, {}).items():
                    losses[f"{loss_type}_{name}"] = value * self.depth_loss_weight
        else:
            # Just encode-decode without loss computation
            inputs, outputs = self.unidepth_model.encode_decode(inputs, image_metas)
            losses = {}

        # 4. Extract outputs
        depth_map = outputs.get("depth")  # [B, 1, H, W]
        out_features = outputs.get("out_features")  # [1/8, 1/4, 1/2] resolution latents
        pred_K = outputs.get("intrinsics")  # [B, 3, 3]

        # 5. Extract and project depth latents
        depth_latents, depth_latents_hw = self._extract_depth_latents(out_features)
        depth_latents = self._maybe_detach_latents(depth_latents)

        return GeometryBackendOutput(
            depth_map=depth_map,
            depth_latents=depth_latents,
            K_pred=pred_K,
            aux={
                "confidence": outputs.get("confidence"),
                "rays": outputs.get("rays"),
                "points": outputs.get("points"),
                "depth_latents_hw": depth_latents_hw,
            },
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
            images: Input images [B, 3, H, W] in [0, 255] range
            depth_feats: Ignored (we use UniDepthV2's own encoder)
            intrinsics: Camera intrinsics [B, 3, 3]
            image_hw: Image height and width tuple

        Returns:
            GeometryBackendOutput with depth_map, depth_latents, K_pred
        """
        # 1. Normalize images for UniDepthV2
        images_normalized = self._normalize_image(images)

        # 2. Prepare inputs (no depth GT)
        inputs, image_metas = self._prepare_inputs(
            images=images_normalized,
            intrinsics=intrinsics,
            image_hw=image_hw,
        )

        # 3. Run UniDepthV2 encode_decode
        inputs, outputs = self.unidepth_model.encode_decode(inputs, image_metas)

        # 4. Extract outputs
        depth_map = outputs.get("depth")
        out_features = outputs.get("out_features")
        pred_K = outputs.get("intrinsics")

        # 5. Extract and project depth latents
        depth_latents, depth_latents_hw = self._extract_depth_latents(out_features)
        depth_latents = self._maybe_detach_latents(depth_latents)

        return GeometryBackendOutput(
            depth_map=depth_map,
            depth_latents=depth_latents,
            K_pred=pred_K,
            aux={
                "confidence": outputs.get("confidence"),
                "rays": outputs.get("rays"),
                "points": outputs.get("points"),
                "depth_latents_hw": depth_latents_hw,
            },
            losses={},
        )

    def infer(
        self,
        images: Tensor,
        intrinsics: Tensor | None = None,
        normalize: bool = True,
    ) -> GeometryBackendOutput:
        """High-level inference API (wraps UniDepthV2.infer).

        This method handles preprocessing (padding, resizing) automatically.

        Note: This method may not return high-resolution latents since it uses
        UniDepthV2's native infer() which may not expose out_features.

        Args:
            images: Input images [B, 3, H, W] in [0, 255] range.
            intrinsics: Camera intrinsics [B, 3, 3] (optional).
            normalize: Whether to normalize images.

        Returns:
            GeometryBackendOutput with depth_map, K_pred, etc.
        """
        outputs = self.unidepth_model.infer(
            rgb=images,
            camera=intrinsics,
            normalize=normalize,
        )

        depth_map = outputs.get("depth")
        out_features = outputs.get("out_features")
        pred_K = outputs.get("intrinsics")

        # Extract and project depth latents if available
        depth_latents, depth_latents_hw = self._extract_depth_latents(out_features)
        depth_latents = self._maybe_detach_latents(depth_latents)

        return GeometryBackendOutput(
            depth_map=depth_map,
            depth_latents=depth_latents,
            K_pred=pred_K,
            aux={
                "confidence": outputs.get("confidence"),
                "rays": outputs.get("rays"),
                "points": outputs.get("points"),
                "depth_latents_hw": depth_latents_hw,
                "radius": outputs.get("radius"),
            },
            losses={},
        )
