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
from .dinov2_mixin import DINOv2Mixin

# Import UniDepthV2 model
from unidepth.models import UniDepthV2

# Map version string to HuggingFace model name
VERSION_TO_HF_NAME = {
    "v2-vitl14": "lpiccinelli/unidepth-v2-vitl14",
    "v2-vitb14": "lpiccinelli/unidepth-v2-vitb14",
    "v2-vits14": "lpiccinelli/unidepth-v2-vits14",
}


class UniDepthV2GeometryBackend(GeometryBackendBase, DINOv2Mixin):
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
        version: UniDepthV2 version string ("v2-vitl14", "v2-vitb14", "v2-vits14").
        pretrained_path: Path to local full checkpoint, or None to load from HuggingFace.
        encoder_pretrained: Path to encoder-only weights (DINOv2 backbone).
        decoder_pretrained: Path to decoder-only weights.
        use_native_losses: Whether to use UniDepthV2's native loss functions.
        depth_loss_weight: Weight multiplier for depth losses.
        output_scales: Which resolution latents to return (1=1/8, 2=1/4, 3=1/2).
        target_latent_dim: Target dimension for depth_latents (for 3D head).
        freeze_encoder: Whether to freeze pixel encoder weights.
        detach_depth_latents: Whether to detach depth_latents from graph.
    """

    def __init__(
        self,
        version: str = "v2-vitl14",
        pretrained_path: str | None = None,
        encoder_pretrained: str | None = None,
        decoder_pretrained: str | None = None,
        use_native_losses: bool = True,
        depth_loss_weight: float = 10.0,  # Scale to match 3D-MOOD SILog weight
        output_scales: int = 1,
        target_latent_dim: int = 128,
        freeze_encoder: bool = True,
        detach_depth_latents: bool = False,
    ) -> None:
        """Initialize the UniDepthV2GeometryBackend."""
        super().__init__(detach_depth_latents=detach_depth_latents)

        # Store pretrained paths for logging
        self._pretrained_path = pretrained_path
        self._encoder_pretrained_path = encoder_pretrained
        self._decoder_pretrained_path = decoder_pretrained

        # Store version for loading weights later
        self._version = version

        # Build UniDepthV2 model architecture (without loading pretrained weights yet)
        # Get HuggingFace model name
        hf_name = VERSION_TO_HF_NAME.get(version)
        if hf_name is None:
            raise ValueError(
                f"Unknown version: {version}. "
                f"Available versions: {list(VERSION_TO_HF_NAME.keys())}"
            )

        # Download config from HuggingFace
        print(f"[UniDepthV2] Loading config from HuggingFace: {hf_name}")
        from huggingface_hub import hf_hub_download
        import json

        config_file = hf_hub_download(
            repo_id=hf_name,
            filename="config.json"
        )
        with open(config_file, "r") as f:
            config = json.load(f)

        # IMPORTANT: Ensure pretrained is None to avoid auto-downloading weights
        # The HuggingFace config should have "pretrained": null, but we verify here
        if "model" in config and "pixel_encoder" in config["model"]:
            encoder_pretrained = config["model"]["pixel_encoder"].get("pretrained")
            if encoder_pretrained is not None and encoder_pretrained != "":
                print(f"  WARNING: Config has pretrained={encoder_pretrained}, setting to None")
                config["model"]["pixel_encoder"]["pretrained"] = None
            else:
                print(f"  ✅ Config pretrained={encoder_pretrained} (no auto-download)")

        # Create model from config (without loading pretrained weights)
        print(f"[UniDepthV2] Creating model architecture (no pretrained weights yet)")
        self.unidepth_model = UniDepthV2(config=config)

        self.use_native_losses = use_native_losses
        self.depth_loss_weight = depth_loss_weight
        self.output_scales = output_scales
        self.target_latent_dim = target_latent_dim

        assert output_scales >= 1 and output_scales <= 3, "output_scales must be 1, 2, or 3"

        # Fix HuggingFace config bug: confidence loss is incorrectly set to Regression
        # The HuggingFace config has "name": "Regression" for confidence loss,
        # but UniDepthV2.compute_losses() calls it with Confidence signature:
        #   loss(input, target_gt=..., target_pred=..., mask=...)
        # Regression only accepts: loss(input, target, mask)
        # This causes runtime errors, so we replace with the correct Confidence class.
        if "confidence" in self.unidepth_model.losses:
            loss_obj = self.unidepth_model.losses["confidence"]
            if loss_obj.name == "Regression":
                print(f"  [UniDepthV2] Fixing HuggingFace config bug: replacing Regression with Confidence")
                from unidepth.ops.losses.confidence import Confidence
                # Use the same config as in official UniDepthV2 training
                confidence_config = {
                    "weight": 0.1,
                    "output_fn": "sqrt",
                    "input_fn": "linear",
                    "rescale": True,
                }
                self.unidepth_model.losses["confidence"] = Confidence.build(confidence_config)

        # Note: Freezing is now handled by optimizer's lr_mult=0.0 instead of requires_grad=False
        # This avoids DDP's find_unused_parameters overhead
        # if freeze_encoder:
        #     for param in self.unidepth_model.pixel_encoder.parameters():
        #         param.requires_grad = False

        # Projection layer to align latent dimension
        # UniDepthV2 hidden_dim is typically 512
        # out_features dimensions depend on the ups configuration
        # We'll create the projection lazily in forward since we don't know dims yet
        self.latent_proj = None
        self._latent_proj_initialized = False

    def load_pretrained_weights(self) -> None:
        """Load pretrained weights for UniDepthV2 encoder and decoder.

        This should be called BEFORE loading the full model checkpoint.
        It's called from GroundingDINO3D.load_state_dict().
        """
        # Determine which weights to load
        if self._pretrained_path is not None:
            # Load from local full checkpoint (encoder + decoder)
            print(f"[UniDepthV2] Loading full model from: {self._pretrained_path}")
            state_dict = torch.load(self._pretrained_path, map_location="cpu")
            missing, unexpected = self.unidepth_model.load_state_dict(state_dict, strict=False)

            if missing:
                print(f"  Warning: Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Warning: Unexpected keys: {len(unexpected)}")

            print(f"  ✅ Loaded UniDepthV2 full model successfully!")

        elif self._encoder_pretrained_path is not None and self._decoder_pretrained_path is not None:
            # Load encoder and decoder separately
            print(f"[UniDepthV2] Loading encoder and decoder separately:")

            # Load encoder weights
            # IMPORTANT: The encoder checkpoint has 'pixel_encoder.' prefix, need to remove it
            print(f"  Loading encoder from: {self._encoder_pretrained_path}")
            encoder_state = torch.load(self._encoder_pretrained_path, map_location="cpu")

            # Remove 'pixel_encoder.' prefix from keys
            encoder_state_clean = {}
            for key, value in encoder_state.items():
                if key.startswith('pixel_encoder.'):
                    new_key = key.replace('pixel_encoder.', '', 1)
                    encoder_state_clean[new_key] = value
                else:
                    encoder_state_clean[key] = value

            print(f"    Original keys: {len(encoder_state)}, Cleaned keys: {len(encoder_state_clean)}")
            missing, unexpected = self.unidepth_model.pixel_encoder.load_state_dict(
                encoder_state_clean, strict=False
            )
            if missing:
                print(f"    ⚠️ Missing encoder keys: {len(missing)}")
                if len(missing) <= 10:
                    for k in missing:
                        print(f"      - {k}")
            if unexpected:
                print(f"    ⚠️ Unexpected encoder keys: {len(unexpected)}")
                if len(unexpected) <= 10:
                    for k in unexpected:
                        print(f"      - {k}")

            # Load decoder weights
            # The decoder checkpoint does NOT have 'pixel_decoder.' prefix, load directly
            print(f"  Loading decoder from: {self._decoder_pretrained_path}")
            decoder_state = torch.load(self._decoder_pretrained_path, map_location="cpu")
            missing, unexpected = self.unidepth_model.pixel_decoder.load_state_dict(
                decoder_state, strict=False
            )
            if missing:
                print(f"    ⚠️ Missing decoder keys: {len(missing)}")
                if len(missing) <= 10:
                    for k in missing:
                        print(f"      - {k}")
            if unexpected:
                print(f"    ⚠️ Unexpected decoder keys: {len(unexpected)}")
                if len(unexpected) <= 10:
                    for k in unexpected:
                        print(f"      - {k}")

            print(f"  ✅ Loaded encoder and decoder weights successfully!")

        else:
            # Load from HuggingFace (full pretrained model: encoder + decoder)
            hf_name = VERSION_TO_HF_NAME.get(self._version)
            print(f"[UniDepthV2] Loading pretrained weights from HuggingFace: {hf_name}")

            # Download weights file
            from huggingface_hub import hf_hub_download
            try:
                # Try safetensors first
                weights_file = hf_hub_download(
                    repo_id=hf_name,
                    filename="model.safetensors"
                )
                from safetensors.torch import load_file
                state_dict = load_file(weights_file)
            except:
                # Fall back to pytorch_model.bin
                weights_file = hf_hub_download(
                    repo_id=hf_name,
                    filename="pytorch_model.bin"
                )
                state_dict = torch.load(weights_file, map_location="cpu")

            # Load weights
            missing, unexpected = self.unidepth_model.load_state_dict(state_dict, strict=False)

            if missing:
                print(f"  Warning: Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Warning: Unexpected keys: {len(unexpected)}")

            print(f"  ✅ Loaded UniDepthV2 pretrained weights from HuggingFace!")
            print(f"     Encoder (pixel_encoder): ✅ Pretrained")
            print(f"     Decoder (pixel_decoder): ✅ Pretrained")

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
            # Create individual Pinhole cameras for each batch element
            cameras = [Pinhole(K=intrinsics[i:i+1]) for i in range(B)]
            # Stack all camera parameters
            params_list = [cam.params for cam in cameras]
            K_list = [cam.K for cam in cameras]
            params_stacked = torch.cat(params_list, dim=0)
            K_stacked = torch.cat(K_list, dim=0)
            # Create BatchCamera manually
            camera = BatchCamera(
                params=params_stacked,
                K=K_stacked,
                original_class=[Pinhole.__name__] * B,
                cameras=cameras
            )
            camera = camera.to(device)

        inputs = {
            "image": images,
            "camera": camera,
        }

        # Add depth GT if provided (should already be [B, 1, H, W])
        if depth_gt is not None:
            inputs["depth"] = depth_gt

        # Add depth mask (should already be [B, 1, H, W])
        if depth_mask is not None:
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
        # Store original dimensions
        orig_H, orig_W = image_hw

        # 1. Resize images to be divisible by DINOv2 patch size (14) and adjust intrinsics
        images_resized, intrinsics_adjusted = self._make_divisible_by_dinov2_patch(
            images, intrinsics
        )

        # Update image_hw to match resized dimensions
        _, _, new_H, new_W = images_resized.shape
        image_hw_adjusted = (new_H, new_W)

        # 2. Resize depth_gt and depth_mask to match resized image dimensions
        depth_gt_resized = None
        depth_mask_resized = None
        if depth_gt is not None:
            if depth_gt.ndim == 3:
                depth_gt = depth_gt.unsqueeze(1)  # [B, 1, H, W]
            depth_gt_resized = F.interpolate(
                depth_gt, size=(new_H, new_W), mode='nearest'
            )
        if depth_mask is not None:
            if depth_mask.ndim == 3:
                depth_mask = depth_mask.unsqueeze(1)  # [B, 1, H, W]
            depth_mask_resized = F.interpolate(
                depth_mask.float(), size=(new_H, new_W), mode='nearest'
            ).bool()

        # 3. Normalize images for UniDepthV2 (ImageNet normalization)
        images_normalized = self._normalize_image_for_dinov2(images_resized)

        # 4. Prepare inputs with adjusted intrinsics and image_hw
        inputs, image_metas = self._prepare_inputs(
            images=images_normalized,
            intrinsics=intrinsics_adjusted,
            image_hw=image_hw_adjusted,
            depth_gt=depth_gt_resized,
            depth_mask=depth_mask_resized,
        )

        # DEBUG: Check inputs for nan/inf
        if torch.isnan(inputs["image"]).any() or torch.isinf(inputs["image"]).any():
            print(f"[DEBUG NaN/Inf] inputs['image'] has nan={torch.isnan(inputs['image']).any().item()}, inf={torch.isinf(inputs['image']).any().item()}")
        if "camera" in inputs and inputs["camera"] is not None:
            K = inputs["camera"].K
            if torch.isnan(K).any() or torch.isinf(K).any():
                print(f"[DEBUG NaN/Inf] camera.K has nan={torch.isnan(K).any().item()}, inf={torch.isinf(K).any().item()}")

        # 3. Run UniDepthV2 forward_train (uses native compute_losses)
        if self.use_native_losses and depth_gt is not None:
            # DEBUG: Manually run encode_decode to check intermediate values
            B, _, H, W = inputs["image"].shape

            # Check if camera rays have nan
            if inputs.get("camera", None) is not None:
                rays = inputs["camera"].get_rays(shapes=(B, H, W))
                if torch.isnan(rays).any() or torch.isinf(rays).any():
                    print(f"[DEBUG NaN/Inf] camera rays has nan={torch.isnan(rays).any().item()}, inf={torch.isinf(rays).any().item()}")
                    print(f"[DEBUG NaN/Inf] camera.K = {inputs['camera'].K}")

            outputs, losses_dict = self.unidepth_model.forward_train(
                inputs, image_metas
            )

            # DEBUG: Check decoder outputs
            if "radius" in outputs:
                radius = outputs["radius"]
                if torch.isnan(radius).any() or torch.isinf(radius).any():
                    print(f"[DEBUG NaN/Inf] decoder radius has nan={torch.isnan(radius).any().item()}, inf={torch.isinf(radius).any().item()}")
                    print(f"[DEBUG NaN/Inf] radius stats: min={radius.min().item():.4f}, max={radius.max().item():.4f}, mean={radius.mean().item():.4f}")

            if "rays" in outputs:
                rays_out = outputs["rays"]
                if torch.isnan(rays_out).any() or torch.isinf(rays_out).any():
                    print(f"[DEBUG NaN/Inf] decoder rays has nan={torch.isnan(rays_out).any().item()}, inf={torch.isinf(rays_out).any().item()}")

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
        depth_map = outputs.get("depth")  # [B, 1, H', W']
        out_features = outputs.get("out_features")  # [1/8, 1/4, 1/2] resolution latents
        pred_K = outputs.get("intrinsics")  # [B, 3, 3]

        # DEBUG: Check for nan/inf in depth predictions
        if depth_map is not None:
            has_nan = torch.isnan(depth_map).any().item()
            has_inf = torch.isinf(depth_map).any().item()
            if has_nan or has_inf:
                print(f"[DEBUG NaN/Inf] depth_map has nan={has_nan}, inf={has_inf}")
                print(f"[DEBUG NaN/Inf] depth_map stats: min={depth_map.min().item():.4f}, max={depth_map.max().item():.4f}, mean={depth_map.mean().item():.4f}")
                # Check which batch elements have nan/inf
                for i in range(depth_map.shape[0]):
                    batch_has_nan = torch.isnan(depth_map[i]).any().item()
                    batch_has_inf = torch.isinf(depth_map[i]).any().item()
                    if batch_has_nan or batch_has_inf:
                        print(f"[DEBUG NaN/Inf] Batch {i}: nan={batch_has_nan}, inf={batch_has_inf}")

        # DEBUG: Check for nan/inf in losses
        if losses:
            for loss_name, loss_value in losses.items():
                if torch.isnan(loss_value).any().item() or torch.isinf(loss_value).any().item():
                    print(f"[DEBUG NaN/Inf] Loss {loss_name} has nan/inf: {loss_value.item()}")
                    # Print depth_gt stats to see if GT has issues
                    if depth_gt_resized is not None:
                        print(f"[DEBUG NaN/Inf] depth_gt_resized stats: min={depth_gt_resized.min().item():.4f}, max={depth_gt_resized.max().item():.4f}, mean={depth_gt_resized.mean().item():.4f}")
                        print(f"[DEBUG NaN/Inf] depth_gt_resized has nan={torch.isnan(depth_gt_resized).any().item()}, inf={torch.isinf(depth_gt_resized).any().item()}")

        # 5. Resize depth_map back to original dimensions
        depth_map_resized = self._resize_depth_to_original(depth_map, (orig_H, orig_W))

        # 6. Extract and project depth latents
        depth_latents, depth_latents_hw = self._extract_depth_latents(out_features)
        depth_latents = self._maybe_detach_latents(depth_latents)

        # Compute downsample factor based on output_scales (1=1/8, 2=1/4, 3=1/2)
        ray_downsample = 8 // (2 ** (self.output_scales - 1))

        return GeometryBackendOutput(
            depth_map=depth_map_resized,
            depth_latents=depth_latents,
            K_pred=pred_K,
            ray_intrinsics=intrinsics_adjusted,  # Adjusted intrinsics for DINOv2 space
            ray_image_hw=image_hw_adjusted,  # Adjusted image size
            ray_downsample=ray_downsample,  # Based on output_scales
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
        # Store original dimensions
        orig_H, orig_W = image_hw

        # 1. Resize images to be divisible by DINOv2 patch size (14) and adjust intrinsics
        images_resized, intrinsics_adjusted = self._make_divisible_by_dinov2_patch(
            images, intrinsics
        )

        # Update image_hw to match resized dimensions
        _, _, new_H, new_W = images_resized.shape
        image_hw_adjusted = (new_H, new_W)

        # 2. Normalize images for UniDepthV2
        images_normalized = self._normalize_image_for_dinov2(images_resized)

        # 3. Prepare inputs (no depth GT) with adjusted intrinsics and image_hw
        inputs, image_metas = self._prepare_inputs(
            images=images_normalized,
            intrinsics=intrinsics_adjusted,
            image_hw=image_hw_adjusted,
        )

        # 4. Run UniDepthV2 encode_decode
        inputs, outputs = self.unidepth_model.encode_decode(inputs, image_metas)

        # 5. Extract outputs
        depth_map = outputs.get("depth")  # [B, 1, H', W']
        out_features = outputs.get("out_features")
        pred_K = outputs.get("intrinsics")

        # 6. Resize depth_map back to original dimensions
        depth_map_resized = self._resize_depth_to_original(depth_map, (orig_H, orig_W))

        # 7. Extract and project depth latents
        depth_latents, depth_latents_hw = self._extract_depth_latents(out_features)
        depth_latents = self._maybe_detach_latents(depth_latents)

        # Compute downsample factor based on output_scales (1=1/8, 2=1/4, 3=1/2)
        ray_downsample = 8 // (2 ** (self.output_scales - 1))

        return GeometryBackendOutput(
            depth_map=depth_map_resized,
            depth_latents=depth_latents,
            K_pred=pred_K,
            ray_intrinsics=intrinsics_adjusted,  # Adjusted intrinsics for DINOv2 space
            ray_image_hw=image_hw_adjusted,  # Adjusted image size
            ray_downsample=ray_downsample,  # Based on output_scales
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
