"""UniDepthV2GeometryBackend: Wraps UniDepthV2's depth estimation system.

This backend uses:
- DINOv2 pixel_encoder for feature extraction
- UniDepthV2 Decoder for depth prediction
- Full loss suite: SILog, camera, invariance, SSI, confidence
  (uses UniDepthV2's native compute_losses - NOT reimplemented)

The backend encapsulates the complete UniDepthV2 geometry pipeline.
"""

from __future__ import annotations

import os
import torch
import torchvision.transforms.functional as TF
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from .base import GeometryBackendBase, GeometryBackendOutput
from .dinov2_mixin import DINOv2Mixin
from opendet3d.utils.profiler import profile_start, profile_stop

# Global counter for depth visualization
_DEPTH_VIS_COUNTER = 0
_DEPTH_VIS_SAVE_DIR = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/Foundation3DDet/sam3_da3/Foundation3DDet/mmdetection_exp/training/depth_ablation/unidepth_v2_depth_vis"
_DEPTH_VIS_MAX_SAVE = 0  # Disabled for training (set > 0 for debugging)

# Debug prints (intrinsics/rays sanity checks). Keep off by default.
_DEBUG_PREPARE_INPUTS = False

# Import UniDepthV2 model
from unidepth.models import UniDepthV2

# Map version string to local config file path (relative to repo root)
VERSION_TO_LOCAL_CONFIG = {
    "v2-vitl14": "UniDepth/configs/config_v2_vitl14.json",
    "v2-vitb14": "UniDepth/configs/config_v2_vitb14.json",
    "v2-vits14": "UniDepth/configs/config_v2_vits14.json",
}

# Map version string to HuggingFace repo name for pretrained weights
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

    The decoder internally fuses ray embeddings at each scale, so depth_latents
    are already ray-aware. The 3D head does NOT need a separate camera prompt.

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
        silog_only: If True, only use SILog loss (disable camera, ssi, confidence losses).
    """

    # UniDepthV2's decoder fuses rays internally via embed_rays + multi-scale fusion
    is_ray_aware: bool = True

    def __init__(
        self,
        version: str = "v2-vitl14",
        pretrained_path: str | None = None,
        encoder_pretrained: str | None = None,
        decoder_pretrained: str | None = None,
        use_native_losses: bool = True,
        depth_loss_weight: float = 1.0,  # Raw loss, scaling done in SAM3_3DLoss via loss_geom_scale
        output_scales: int = 1,
        target_latent_dim: int = 256,
        freeze_encoder: bool = True,
        detach_depth_latents: bool = False,
        silog_only: bool = False,  # If True, only use SILog loss (disable camera, ssi, confidence)
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
        # Get local config file path
        import json
        import os

        config_path = VERSION_TO_LOCAL_CONFIG.get(version)
        if config_path is None:
            raise ValueError(
                f"Unknown version: {version}. "
                f"Available versions: {list(VERSION_TO_LOCAL_CONFIG.keys())}"
            )

        # Find the config file relative to this file's location
        # unidepth_v2_backend.py is in opendet3d/op/detect3d/geometry/
        # UniDepth/configs/ is at repo root
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(backend_dir))))
        config_file = os.path.join(repo_root, config_path)

        print(f"[UniDepthV2] Loading config from local file: {config_file}")
        with open(config_file, "r") as f:
            config = json.load(f)

        # Create model from config (without loading pretrained weights)
        # Note: Local config already has "pretrained": null, no need to override
        print(f"[UniDepthV2] Creating model architecture (no pretrained weights yet)")
        self.unidepth_model = UniDepthV2(config=config)

        self.use_native_losses = use_native_losses
        self.depth_loss_weight = depth_loss_weight
        self.output_scales = output_scales
        self.target_latent_dim = target_latent_dim

        assert output_scales >= 1 and output_scales <= 3, "output_scales must be 1, 2, or 3"

        # Disable invariance loss (SelfDistill) for 3D-MOOD training
        # The invariance loss expects paired images (same scene with augmentations)
        # but 3D-MOOD batches contain independent images from different scenes.
        # SelfDistill.forward() does: chunks = batch_size // 2, then input.chunk(chunks)
        # This causes errors when batch_size < 2 or when images aren't paired.
        if "invariance" in self.unidepth_model.losses:
            print(f"  [UniDepthV2] Disabling invariance loss (SelfDistill) - not compatible with 3D-MOOD data")
            del self.unidepth_model.losses["invariance"]

        # If silog_only, disable all losses except depth (SILog)
        if silog_only:
            print(f"  [UniDepthV2] silog_only=True: Keeping only SILog loss")
            losses_to_remove = [k for k in self.unidepth_model.losses.keys() if k != "depth"]
            for loss_name in losses_to_remove:
                print(f"    - Disabling {loss_name} loss")
                del self.unidepth_model.losses[loss_name]

        # Temporarily disable confidence loss
        if "confidence" in self.unidepth_model.losses:
            print(f"  [UniDepthV2] DEBUG: Disabling confidence loss")
            del self.unidepth_model.losses["confidence"]

        # Print remaining losses
        print(f"  [UniDepthV2] Active losses: {list(self.unidepth_model.losses.keys())}")

        # Note: Freezing is now handled by optimizer's lr_mult=0.0 instead of requires_grad=False
        # This avoids DDP's find_unused_parameters overhead
        # if freeze_encoder:
        #     for param in self.unidepth_model.pixel_encoder.parameters():
        #         param.requires_grad = False

        # Projection layer to align latent dimension
        # Create at init time to ensure checkpoint loading works correctly
        # DINOv2 encoder dimensions: vits=384, vitb=768, vitl=1024
        # But out_features uses ups which reduces dims. For output_scales=1 (1/8 res):
        # - v2-vits14: 256 (from ups config)
        # - v2-vitb14: 512
        # - v2-vitl14: 1024
        version_to_latent_dim = {
            "v2-vits14": 256,
            "v2-vitb14": 512,
            "v2-vitl14": 1024,
        }
        source_dim = version_to_latent_dim.get(version, 256)
        if source_dim != target_latent_dim:
            self.latent_proj = nn.Linear(source_dim, target_latent_dim)
        else:
            self.latent_proj = nn.Identity()
        self._latent_proj_initialized = True

        # Auto-load pretrained weights during initialization
        # This ensures weights are loaded even for first training (no checkpoint)
        if (self._pretrained_path is not None or
            (self._encoder_pretrained_path is not None and self._decoder_pretrained_path is not None)):
            print(f"[UniDepthV2] Auto-loading pretrained weights during __init__...")
            self.load_pretrained_weights()

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

            # DEBUG: Check if loaded weights have NaN
            has_nan = False
            for name, param in self.unidepth_model.pixel_encoder.named_parameters():
                if torch.isnan(param).any():
                    print(f"    ❌ ENCODER WEIGHT HAS NaN: {name}")
                    has_nan = True
            if not has_nan:
                print(f"    ✅ All encoder weights are clean (no NaN)")

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
            # DEBUG: Check intrinsics shape
            if _DEBUG_PREPARE_INPUTS:
                print(f"[DEBUG _prepare_inputs] intrinsics shape: {intrinsics.shape}")
                print(f"[DEBUG _prepare_inputs] intrinsics[0]:\n{intrinsics[0]}")

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

            # DEBUG: Check get_rays output
            H, W = image_hw
            rays_test = camera.get_rays(shapes=(B, H, W))
            if _DEBUG_PREPARE_INPUTS:
                print(f"[DEBUG _prepare_inputs] BatchCamera.get_rays output shape: {rays_test.shape}")
                print(f"[DEBUG _prepare_inputs] Expected shape: [{B}, 3, {H}, {W}]")
                if rays_test.shape != (B, 3, H, W):
                    print(f"[ERROR] rays_test shape mismatch!")
                    print(f"[DEBUG] cameras[0] type: {type(cameras[0])}")
                    print(f"[DEBUG] cameras[0].K shape: {cameras[0].K.shape}")
                    # Test individual camera
                    test_uv = torch.randn(1, 2, H, W, device=device)
                    test_rays_0 = cameras[0].unproject(test_uv)
                    print(f"[DEBUG] cameras[0].unproject([1, 2, {H}, {W}]) shape: {test_rays_0.shape}")

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
            images: Input images [B, 3, H, W] already normalized by 3D-MOOD's NormalizeImages
                    (mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
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
        profile_start("    geom_preprocess")
        images_resized, intrinsics_adjusted = self._make_divisible_by_dinov2_patch(
            images, intrinsics
        )

        # Update image_hw to match resized dimensions
        _, _, new_H, new_W = images_resized.shape
        image_hw_adjusted = (new_H, new_W)
        profile_stop("    geom_preprocess")

        # 2. Resize depth_gt and depth_mask to match resized image dimensions
        depth_gt_resized = None
        depth_mask_resized = None
        if depth_gt is not None:
            if depth_gt.ndim == 3:
                depth_gt = depth_gt.unsqueeze(1)  # [B, 1, H, W]
            depth_gt_resized = F.interpolate(
                depth_gt, size=(new_H, new_W), mode='nearest'
            )

            # DEBUG: Shape and intrinsic information (commented out for production)
            # print(f"[DEBUG UniDepthV2 forward_train]")
            # print(f"  image_hw (original): {image_hw}")
            # print(f"  image_hw_adjusted: {image_hw_adjusted}")
            # print(f"  depth_gt (original) shape: {depth_gt.shape}")
            # print(f"  depth_gt_resized shape: {depth_gt_resized.shape}")
            # if depth_gt.shape[-2:] != image_hw:
            #     print(f"  [WARNING] depth_gt {depth_gt.shape[-2:]} != image_hw {image_hw}!")
            # print(f"  intrinsics (original)[0]: fx={intrinsics[0, 0, 0].item():.2f}, ...")
            # print(f"  intrinsics_adjusted[0]: fx={intrinsics_adjusted[0, 0, 0].item():.2f}, ...")

        if depth_mask is not None:
            if depth_mask.ndim == 3:
                depth_mask = depth_mask.unsqueeze(1)  # [B, 1, H, W]
            depth_mask_resized = F.interpolate(
                depth_mask.float(), size=(new_H, new_W), mode='nearest'
            ).bool()

        # 3. Skip normalization - images are already normalized by 3D-MOOD's NormalizeImages
        # The 3D-MOOD normalization (mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        # is mathematically equivalent to ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # because 123.675/255 = 0.485, 58.395/255 = 0.229, etc.
        images_normalized = images_resized

        # 4. Prepare inputs with adjusted intrinsics and image_hw
        inputs, image_metas = self._prepare_inputs(
            images=images_normalized,
            intrinsics=intrinsics_adjusted,
            image_hw=image_hw_adjusted,
            depth_gt=depth_gt_resized,
            depth_mask=depth_mask_resized,
        )

        # 3. Run UniDepthV2 forward_train (uses native compute_losses)
        # Profile encoder separately (just to measure time, not used in computation)
        profile_start("    geom_encoder_only")
        with torch.no_grad():
            _ = self.unidepth_model.pixel_encoder(inputs["image"])
        profile_stop("    geom_encoder_only")

        # Profile full forward (encoder + decoder + loss)
        profile_start("    geom_full_forward")
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
        profile_stop("    geom_full_forward")

        # 4. Extract outputs
        depth_map = outputs.get("depth")  # [B, 1, H', W']
        out_features = outputs.get("out_features")  # [1/8, 1/4, 1/2] resolution latents
        pred_K = outputs.get("intrinsics")  # [B, 3, 3]

        # 5. Resize depth_map back to original dimensions
        profile_start("    geom_postprocess")
        depth_map_resized = self._resize_depth_to_original(depth_map, (orig_H, orig_W))

        # 5.5 Save depth visualization (first N samples only)
        profile_start("    geom_save_vis")
        self._save_depth_visualization(
            depth_map_resized, depth_gt, images, image_hw
        )
        profile_stop("    geom_save_vis")

        # 6. Scale pred_K back to original dimensions
        # Intrinsic matrix scaling: fx, cx scale with width; fy, cy scale with height
        # pred_K was predicted for adjusted dimensions, need to scale to original
        pred_K_scaled = None
        if pred_K is not None:
            scale_x = orig_W / image_hw_adjusted[1]  # orig_W / adjusted_W
            scale_y = orig_H / image_hw_adjusted[0]  # orig_H / adjusted_H
            pred_K_scaled = pred_K.clone()
            pred_K_scaled[:, 0, 0] = pred_K[:, 0, 0] * scale_x  # fx
            pred_K_scaled[:, 1, 1] = pred_K[:, 1, 1] * scale_y  # fy
            pred_K_scaled[:, 0, 2] = pred_K[:, 0, 2] * scale_x  # cx
            pred_K_scaled[:, 1, 2] = pred_K[:, 1, 2] * scale_y  # cy

            # DEBUG: Verify intrinsic scaling (commented out for cleaner logs)
            # print(f"[DEBUG UniDepthV2 Intrinsic Scaling]")
            # print(f"  original_hw: ({orig_H}, {orig_W})")
            # print(f"  adjusted_hw: {image_hw_adjusted}")
            # print(f"  scale_x: {scale_x:.4f}, scale_y: {scale_y:.4f}")
            # print(f"  intrinsics (original gt)[0]: fx={intrinsics[0, 0, 0].item():.2f}, fy={intrinsics[0, 1, 1].item():.2f}, cx={intrinsics[0, 0, 2].item():.2f}, cy={intrinsics[0, 1, 2].item():.2f}")
            # print(f"  intrinsics_adjusted[0]: fx={intrinsics_adjusted[0, 0, 0].item():.2f}, fy={intrinsics_adjusted[0, 1, 1].item():.2f}, cx={intrinsics_adjusted[0, 0, 2].item():.2f}, cy={intrinsics_adjusted[0, 1, 2].item():.2f}")
            # print(f"  pred_K (adjusted space)[0]: fx={pred_K[0, 0, 0].item():.2f}, fy={pred_K[0, 1, 1].item():.2f}, cx={pred_K[0, 0, 2].item():.2f}, cy={pred_K[0, 1, 2].item():.2f}")
            # print(f"  pred_K_scaled (original space)[0]: fx={pred_K_scaled[0, 0, 0].item():.2f}, fy={pred_K_scaled[0, 1, 1].item():.2f}, cx={pred_K_scaled[0, 0, 2].item():.2f}, cy={pred_K_scaled[0, 1, 2].item():.2f}")

        # 7. Extract and project depth latents
        depth_latents, depth_latents_hw = self._extract_depth_latents(out_features)
        depth_latents = self._maybe_detach_latents(depth_latents)
        profile_stop("    geom_postprocess")

        # Compute downsample factor based on output_scales (1=1/8, 2=1/4, 3=1/2)
        ray_downsample = 8 // (2 ** (self.output_scales - 1))

        return GeometryBackendOutput(
            depth_map=depth_map_resized,
            depth_latents=depth_latents,
            K_pred=pred_K_scaled,  # Use scaled K_pred (in original image space)
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
            images: Input images [B, 3, H, W] already normalized by 3D-MOOD's NormalizeImages
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

        # 2. Skip normalization - images are already normalized by 3D-MOOD's NormalizeImages
        images_normalized = images_resized

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

        # 7. Scale pred_K back to original dimensions
        # pred_K was predicted for adjusted dimensions (e.g., 798x798),
        # need to scale to original dimensions (e.g., 800x800)
        pred_K_scaled = None
        if pred_K is not None:
            scale_x = orig_W / image_hw_adjusted[1]  # orig_W / adjusted_W
            scale_y = orig_H / image_hw_adjusted[0]  # orig_H / adjusted_H
            pred_K_scaled = pred_K.clone()
            pred_K_scaled[:, 0, 0] = pred_K[:, 0, 0] * scale_x  # fx
            pred_K_scaled[:, 1, 1] = pred_K[:, 1, 1] * scale_y  # fy
            pred_K_scaled[:, 0, 2] = pred_K[:, 0, 2] * scale_x  # cx
            pred_K_scaled[:, 1, 2] = pred_K[:, 1, 2] * scale_y  # cy

        # 8. Extract and project depth latents
        depth_latents, depth_latents_hw = self._extract_depth_latents(out_features)
        depth_latents = self._maybe_detach_latents(depth_latents)

        # Compute downsample factor based on output_scales (1=1/8, 2=1/4, 3=1/2)
        ray_downsample = 8 // (2 ** (self.output_scales - 1))

        return GeometryBackendOutput(
            depth_map=depth_map_resized,
            depth_latents=depth_latents,
            K_pred=pred_K_scaled,  # Scaled to original image space
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

    def _save_depth_visualization(
        self,
        depth_pred: Tensor | None,
        depth_gt: Tensor | None,
        images: Tensor,
        image_hw: tuple[int, int],
    ) -> None:
        """Save depth map visualization as PNG.

        Args:
            depth_pred: Predicted depth [B, 1, H, W] or [B, H, W]
            depth_gt: Ground truth depth [B, 1, H, W] or [B, H, W]
            images: Input images [B, 3, H, W]
            image_hw: Original image size (H, W)
        """
        global _DEPTH_VIS_COUNTER

        if _DEPTH_VIS_COUNTER >= _DEPTH_VIS_MAX_SAVE:
            return

        if depth_pred is None:
            return

        # Ensure save directory exists
        os.makedirs(_DEPTH_VIS_SAVE_DIR, exist_ok=True)

        # Process each sample in batch
        batch_size = depth_pred.shape[0]
        for i in range(batch_size):
            if _DEPTH_VIS_COUNTER >= _DEPTH_VIS_MAX_SAVE:
                break

            try:
                # Get depth prediction
                d_pred = depth_pred[i]
                if d_pred.dim() == 3:
                    d_pred = d_pred.squeeze(0)  # [H, W]
                d_pred_np = d_pred.detach().cpu().numpy()

                # Get depth GT if available
                d_gt_np = None
                if depth_gt is not None:
                    d_gt = depth_gt[i]
                    if d_gt.dim() == 3:
                        d_gt = d_gt.squeeze(0)  # [H, W]
                    d_gt_np = d_gt.detach().cpu().numpy()

                # Get input image (denormalize)
                img = images[i].detach().cpu()
                # Denormalize from ImageNet normalization
                mean = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
                std = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
                img = img * std + mean
                img = img.permute(1, 2, 0).numpy().astype(np.uint8)
                img = np.clip(img, 0, 255)

                # Create visualization figure
                if d_gt_np is not None:
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                else:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes = [axes[0], axes[1], None]

                # Plot input image
                axes[0].imshow(img)
                axes[0].set_title("Input Image")
                axes[0].axis("off")

                # Plot predicted depth with colormap
                vmin = np.percentile(d_pred_np[d_pred_np > 0], 5) if (d_pred_np > 0).any() else 0
                vmax = np.percentile(d_pred_np[d_pred_np > 0], 95) if (d_pred_np > 0).any() else 1
                im = axes[1].imshow(d_pred_np, cmap="magma", vmin=vmin, vmax=vmax)
                axes[1].set_title(f"Pred Depth (min={d_pred_np.min():.2f}, max={d_pred_np.max():.2f})")
                axes[1].axis("off")
                plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

                # Plot GT depth if available
                if d_gt_np is not None and axes[2] is not None:
                    valid_mask = d_gt_np > 0
                    if valid_mask.any():
                        vmin_gt = np.percentile(d_gt_np[valid_mask], 5)
                        vmax_gt = np.percentile(d_gt_np[valid_mask], 95)
                    else:
                        vmin_gt, vmax_gt = 0, 1
                    im_gt = axes[2].imshow(d_gt_np, cmap="magma", vmin=vmin_gt, vmax=vmax_gt)
                    axes[2].set_title(f"GT Depth (min={d_gt_np.min():.2f}, max={d_gt_np.max():.2f})")
                    axes[2].axis("off")
                    plt.colorbar(im_gt, ax=axes[2], fraction=0.046, pad=0.04)

                plt.tight_layout()

                # Save figure
                save_path = os.path.join(_DEPTH_VIS_SAVE_DIR, f"depth_{_DEPTH_VIS_COUNTER:04d}.png")
                plt.savefig(save_path, dpi=100, bbox_inches="tight")
                plt.close(fig)

                # print(f"[Depth Vis] Saved {save_path}")
                _DEPTH_VIS_COUNTER += 1

            except Exception as e:
                print(f"[Depth Vis] Error saving depth visualization: {e}")
                continue
