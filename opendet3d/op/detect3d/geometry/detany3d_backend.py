"""DetAny3DGeometryBackend: Wraps DetAny3D's depth estimation system.

This backend uses:
- DINOv2 for feature extraction (DINO-only, no SAM backbone)
- Unidepth_Decoder (DINO-only version) for depth prediction
- SILogLoss on depth + intrinsic angles (phi/theta) - directly from DetAny3D

The backend encapsulates the complete DetAny3D geometry pipeline.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .base import GeometryBackendBase, GeometryBackendOutput
from .unidepth_decoder_dino_only import UnidepthDecoderDinoOnly
from .dinov2_mixin import DINOv2Mixin

# Import utilities from DetAny3D
from DetAny3D.detect_anything.modeling.depth_predictor.unidepth_utils import (
    generate_rays,
)

# Import DINOv2 model factory from DetAny3D
from DetAny3D.detect_anything.modeling.backbones import _make_dinov2_model

# ImageNet normalization constants (same as DetAny3D uses for DINO)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def SILogLoss(
    input: Tensor,
    target: Tensor,
    coefficient: float,
    masks: Tensor | None = None,
    log_mode: bool = False,
) -> Tensor | None:
    """Scale-Invariant Logarithmic Loss (SILog).

    This is the EXACT implementation from DetAny3D/train_utils.py.

    Args:
        input: Predicted values [...]
        target: Ground truth values [...]
        coefficient: Variance coefficient (lambda in the paper)
        masks: Optional mask to filter valid pixels [...]
        log_mode: If True, compute loss in log space

    Returns:
        SILog loss value, or None if no valid pixels
    """
    if masks is not None:
        input = input[masks > 0]
        target = target[masks > 0]
    if input.numel() == 0:
        return None
    if log_mode:
        alpha = 1e-7
        g = torch.log(input + alpha) - torch.log(target + alpha)
    else:
        g = input - target

    Dg = torch.pow(g, 2).mean() - coefficient * torch.pow(torch.mean(g), 2)
    loss = torch.sqrt(Dg)

    if torch.isnan(loss):
        print("Nan SILog loss")
        print("Input min max", torch.min(input), torch.max(input))
    return loss

# DINO model depths (number of blocks)
DINO_MODEL_DEPTHS = {
    "vit_small": 12,
    "vit_base": 12,
    "vit_large": 24,
    "vit_giant2": 40,
}


def get_dino_output_idx(dino_model: str) -> list[int]:
    """Get the output indices for the last 4 blocks of a DINO model.

    Args:
        dino_model: DINOv2 model variant ("vit_small", "vit_base", "vit_large", "vit_giant2")

    Returns:
        List of 4 block indices (1-indexed) for the last 4 blocks

    Examples:
        vit_small (12 blocks) → [9, 10, 11, 12]
        vit_large (24 blocks) → [21, 22, 23, 24]
    """
    if dino_model not in DINO_MODEL_DEPTHS:
        raise ValueError(
            f"Unknown dino_model: {dino_model}. "
            f"Available: {list(DINO_MODEL_DEPTHS.keys())}"
        )

    depth = DINO_MODEL_DEPTHS[dino_model]
    # Return last 4 blocks (1-indexed)
    return [depth - 3, depth - 2, depth - 1, depth]


class DetAny3DGeometryBackend(GeometryBackendBase, DINOv2Mixin):
    """Backend using DINOv2 + DINO-only Unidepth_Decoder.

    This backend:
    - Uses DINOv2 (vit_large) for feature extraction
    - Uses a DINO-only version of Unidepth_Decoder (no SAM backbone)
    - Computes SILogLoss on depth and intrinsic angles (phi/theta)
    - Uses the EXACT same loss function from DetAny3D/train_utils.py

    The depth_latents output is extracted from out_features at the selected
    resolution (1/8, 1/4, or 1/2) and projected to target_latent_dim.

    Args:
        dino_model: DINOv2 model variant ("vit_large", "vit_base", "vit_small").
        dino_pretrained: Path to pretrained DINO weights, or "" for default hub weights.
        output_scales: Which resolution latents to return (1=1/8, 2=1/4, 3=1/2).
        target_latent_dim: Target dimension for depth_latents (for 3D head).
        depth_loss_weight: Weight for depth SILog loss.
        phi_loss_weight: Weight for phi angle loss.
        theta_loss_weight: Weight for theta angle loss.
        depth_coefficient: SILog coefficient for depth loss.
        phi_coefficient: SILog coefficient for phi angle loss.
        theta_coefficient: SILog coefficient for theta angle loss.
        freeze_dino: Whether to freeze DINOv2 weights.
        detach_depth_latents: Whether to detach depth_latents from graph.
            If True, gradients from 3D head won't flow back to depth head.
    """

    # DINOv2 embedding dimensions
    DINO_EMBED_DIMS = {
        "vit_small": 384,
        "vit_base": 768,
        "vit_large": 1024,
        "vit_giant2": 1536,
    }

    def __init__(
        self,
        dino_model: str = "vit_large",
        dino_pretrained: str = "",
        decoder_pretrained: str = "",
        decoder_dino_model: str | None = None,  # NEW: decoder's expected DINO model
        output_scales: int = 1,
        target_latent_dim: int = 128,
        depth_loss_weight: float = 10.0,  # Original DetAny3D default
        phi_loss_weight: float = 2.5,  # Original DetAny3D default
        theta_loss_weight: float = 2.5,  # Original DetAny3D default
        depth_coefficient: float = 0.15,
        phi_coefficient: float = 1.0,
        theta_coefficient: float = 1.0,
        freeze_dino: bool = True,
        detach_depth_latents: bool = False,
    ) -> None:
        """Initialize the DetAny3DGeometryBackend."""
        super().__init__(detach_depth_latents=detach_depth_latents)

        # Store pretrained paths for logging
        self.dino_model = dino_model
        self._dino_pretrained_path = dino_pretrained if dino_pretrained else None
        self._decoder_pretrained_path = decoder_pretrained if decoder_pretrained else None

        # Get DINO embed dim based on model variant
        if dino_model not in self.DINO_EMBED_DIMS:
            raise ValueError(
                f"Unknown dino_model: {dino_model}. "
                f"Available: {list(self.DINO_EMBED_DIMS.keys())}"
            )
        dino_embed_dim = self.DINO_EMBED_DIMS[dino_model]

        # Determine decoder's expected DINO embed dim
        # If decoder_dino_model is specified, use that; otherwise use current dino_model
        if decoder_dino_model is not None:
            if decoder_dino_model not in self.DINO_EMBED_DIMS:
                raise ValueError(
                    f"Unknown decoder_dino_model: {decoder_dino_model}. "
                    f"Available: {list(self.DINO_EMBED_DIMS.keys())}"
                )
            decoder_embed_dim = self.DINO_EMBED_DIMS[decoder_dino_model]
        else:
            decoder_embed_dim = dino_embed_dim

        # Store pretrained paths for later loading
        self._dino_pretrained_path = dino_pretrained if dino_pretrained else None
        self._decoder_pretrained_path = decoder_pretrained if decoder_pretrained else None
        self.dino_model = dino_model

        # Get output indices for the last 4 blocks of this DINO model
        output_idx = get_dino_output_idx(dino_model)
        print(f"[DetAny3D] Using DINOv2 {dino_model} with output_idx={output_idx}")

        # Build DINOv2 encoder (without loading pretrained weights yet)
        # IMPORTANT: Use pretrained=None (not "") to avoid auto-downloading from torch hub
        self.dino_encoder = _make_dinov2_model(
            arch_name=dino_model,
            pretrained=None,  # None = no loading; "" = download from torch hub!
            output_idx=output_idx,
            drop_path_rate=0.0,
            num_register_tokens=0,
            use_norm=True,
        )

        # Add feature projector if encoder and decoder have different dimensions
        if dino_embed_dim != decoder_embed_dim:
            print(f"Adding feature projector: {dino_embed_dim} → {decoder_embed_dim}")
            self.feature_projector = nn.Linear(dino_embed_dim, decoder_embed_dim)
            self._has_projector = True
        else:
            self.feature_projector = nn.Identity()
            self._has_projector = False

        # Build DINO-only Unidepth_Decoder with decoder's expected embed dim
        self.depth_decoder = UnidepthDecoderDinoOnly(
            hidden_dim=512,
            num_heads=8,
            expansion=4,
            dropout=0.0,
            depth_layers=[6, 0, 0],
            num_resolutions=4,
            dino_embed_dim=decoder_embed_dim,  # Use decoder's expected dim
            output_scales=output_scales,
            target_latent_dim=target_latent_dim,
        )

        # Note: Pretrained weights will be loaded in load_pretrained_weights()
        # which is called from GroundingDINO3D.load_state_dict()

        # Loss weights (applied after loss computation)
        self.depth_loss_weight = depth_loss_weight
        self.phi_loss_weight = phi_loss_weight
        self.theta_loss_weight = theta_loss_weight

        # SILog coefficients (used inside SILogLoss function)
        self.depth_coefficient = depth_coefficient
        self.phi_coefficient = phi_coefficient
        self.theta_coefficient = theta_coefficient

        # Note: Freezing is now handled by optimizer's lr_mult=0.0 instead of requires_grad=False
        # This avoids DDP's find_unused_parameters overhead
        # if freeze_dino:
        #     for param in self.dino_encoder.parameters():
        #         param.requires_grad = False

        # Store detach flag
        self.detach_depth_latents = detach_depth_latents
        self.target_latent_dim = target_latent_dim

    def load_pretrained_weights(self) -> None:
        """Load pretrained weights for DINOv2 encoder and decoder.

        This should be called BEFORE loading the full model checkpoint.
        It's called from GroundingDINO3D.load_state_dict().
        """
        # Load DINOv2 encoder weights
        if self._dino_pretrained_path:
            print(f"[DetAny3D] Loading DINOv2 encoder from: {self._dino_pretrained_path}")
            dino_state = torch.load(self._dino_pretrained_path, map_location="cpu")

            # Handle different checkpoint formats
            if "model" in dino_state:
                dino_state = dino_state["model"]

            missing, unexpected = self.dino_encoder.load_state_dict(dino_state, strict=False)

            if missing:
                print(f"  Warning: Missing DINOv2 keys: {len(missing)}")
            if unexpected:
                print(f"  Warning: Unexpected DINOv2 keys: {len(unexpected)}")

            print(f"  ✅ Loaded DINOv2 encoder weights successfully!")

        # Load decoder weights
        if self._decoder_pretrained_path:
            print(f"[DetAny3D] Loading decoder from: {self._decoder_pretrained_path}")
            decoder_state = torch.load(self._decoder_pretrained_path, map_location="cpu")

            # Filter out SAM-related keys (we only want DINO-related keys)
            filtered_state = {
                k: v for k, v in decoder_state.items()
                if "_for_sam" not in k
            }

            print(f"  Total keys in checkpoint: {len(decoder_state)}")
            print(f"  Filtered keys (DINO-only): {len(filtered_state)}")

            # Load with strict=False to allow missing/unexpected keys
            missing, unexpected = self.depth_decoder.load_state_dict(
                filtered_state, strict=False
            )

            if missing:
                print(f"  Warning: Missing decoder keys: {len(missing)}")
                if len(missing) <= 10:
                    for key in missing:
                        print(f"    - {key}")
            if unexpected:
                print(f"  Warning: Unexpected decoder keys: {len(unexpected)}")
                if len(unexpected) <= 10:
                    for key in unexpected:
                        print(f"    - {key}")

            print(f"  ✅ Loaded DetAny3D decoder weights successfully!")



    def _prepare_input_dict(
        self,
        images: Tensor,
        dino_features: list[Tensor],
        dino_tokens: list[Tensor],
        intrinsics: Tensor | None,
        image_hw: tuple[int, int],
    ) -> dict:
        """Prepare input dict for DINO-only Unidepth_Decoder.

        This matches the format expected by Unidepth_Decoder.forward(),
        but only with DINO features (no SAM features).
        """
        B = images.shape[0]
        H, W = image_hw
        device = images.device

        input_dict = {
            "image_for_dino": images,  # Already normalized
            "dino_feature": dino_features,  # List of 4 feature maps
            "dino_token": dino_tokens,  # List of 4 cls tokens
            "vit_pad_size": torch.tensor(
                [[H // 14, W // 14]] * B,  # DINOv2 uses patch_size=14
                device=device,
            ),
        }

        if intrinsics is not None:
            input_dict["gt_intrinsic"] = intrinsics

        return input_dict

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
            depth_feats: Ignored (we use our own DINO features)
            intrinsics: Camera intrinsics [B, 3, 3]
            image_hw: Image height and width tuple
            depth_gt: Ground truth depth [B, H, W] or [B, 1, H, W]
            depth_mask: Valid depth mask [B, H, W] or [B, 1, H, W]

        Returns:
            GeometryBackendOutput with depth_map, depth_latents, K_pred, losses
        """
        # Store original dimensions for later
        orig_H, orig_W = image_hw

        # 1. Resize images to be divisible by DINOv2 patch size (14) and adjust intrinsics
        images_resized, intrinsics_adjusted = self._make_divisible_by_dinov2_patch(
            images, intrinsics
        )

        # Update image_hw to match resized dimensions
        _, _, new_H, new_W = images_resized.shape
        image_hw_adjusted = (new_H, new_W)

        # 2. Normalize images for DINOv2 (ImageNet normalization)
        images_normalized = self._normalize_image_for_dinov2(images_resized)

        # 3. Extract DINO features
        dino_features, dino_tokens = self.dino_encoder(images_normalized)

        # 4. Project features if needed (e.g., vit_small → vit_large)
        if self._has_projector:
            # DINOv2 returns features in [B, H, W, C] format
            # Project each feature map: [B, H, W, C_in] → [B, H, W, C_out]
            # Keep in [B, H, W, C] format for decoder
            dino_features = [
                self.feature_projector(feat)
                for feat in dino_features
            ]
            # Project tokens: [B, C_in] → [B, C_out]
            dino_tokens = [self.feature_projector(tok) for tok in dino_tokens]
        # else: Keep features in [B, H, W, C] format (no conversion needed)

        # 5. Prepare input dict for decoder
        input_dict = self._prepare_input_dict(
            images=images_normalized,
            dino_features=dino_features,
            dino_tokens=dino_tokens,
            intrinsics=intrinsics_adjusted,
            image_hw=image_hw_adjusted,
        )

        # 6. Run DINO-only decoder
        outputs = self.depth_decoder(input_dict)

        depth_map = outputs["depth_maps"]  # [B, 1, H_new, W_new]
        depth_latents = outputs.get("depth_latents")  # [B, N, C] - already flattened
        pred_K = outputs.get("pred_K")  # [B, 3, 3]

        # Apply optional detach
        depth_latents = self._maybe_detach_latents(depth_latents)

        # 7. Resize depth_map back to original size
        depth_map_resized = self._resize_depth_to_original(depth_map, (orig_H, orig_W))

        # 8. Compute losses using DetAny3D's SILogLoss
        # Use original intrinsics and image_hw for loss computation
        losses = self._compute_losses(
            depth_map=depth_map_resized,
            depth_gt=depth_gt,
            depth_mask=depth_mask,
            pred_K=pred_K,
            gt_K=intrinsics,
            image_hw=(orig_H, orig_W),
        )

        return GeometryBackendOutput(
            depth_map=depth_map_resized,
            depth_latents=depth_latents,
            K_pred=pred_K,
            ray_intrinsics=intrinsics_adjusted,  # Adjusted intrinsics for DINOv2 space
            ray_image_hw=image_hw_adjusted,  # Adjusted image size
            ray_downsample=8,  # DetAny3D uses 1/8 resolution for depth_latents
            aux={
                "confidence": outputs.get("confidence"),
                "rays": outputs.get("rays"),
                "depth_latents_hw": outputs.get("depth_latents_hw"),
            },
            losses=losses,
        )

    def _compute_losses(
        self,
        depth_map: Tensor,
        depth_gt: Tensor | None,
        depth_mask: Tensor | None,
        pred_K: Tensor | None,
        gt_K: Tensor,
        image_hw: tuple[int, int],
    ) -> dict[str, Tensor]:
        """Compute DetAny3D geometry losses.

        Uses the EXACT SILogLoss function from DetAny3D/train_utils.py
        """
        losses = {}

        # Depth loss
        if depth_gt is not None:
            # Ensure depth_gt is 2D [B, H, W]
            if depth_gt.ndim == 4:
                depth_gt = depth_gt.squeeze(1)
            if depth_mask is not None and depth_mask.ndim == 4:
                depth_mask = depth_mask.squeeze(1)

            depth_pred = depth_map.squeeze(1)  # [B, H, W]
            H, W = depth_gt.shape[-2:]

            # Interpolate if sizes don't match
            if depth_pred.shape[-2:] != (H, W):
                depth_pred = F.interpolate(
                    depth_pred.unsqueeze(1),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

            # Use DetAny3D's SILogLoss directly
            depth_loss = SILogLoss(
                depth_pred,
                depth_gt,
                coefficient=self.depth_coefficient,
                masks=depth_mask,
                log_mode=True,
            )
            if depth_loss is not None:
                losses["depth_loss"] = depth_loss * self.depth_loss_weight

        # Intrinsic loss (phi/theta angles)
        if pred_K is not None and gt_K is not None:
            _, gt_angles = generate_rays(gt_K, image_hw)
            _, pred_angles = generate_rays(pred_K, image_hw)

            phi_gt, theta_gt = gt_angles[..., 0], gt_angles[..., 1]
            phi_pred, theta_pred = pred_angles[..., 0], pred_angles[..., 1]

            # Use DetAny3D's SILogLoss directly
            loss_phi = SILogLoss(
                phi_pred, phi_gt, coefficient=self.phi_coefficient
            )
            loss_theta = SILogLoss(
                theta_pred, theta_gt, coefficient=self.theta_coefficient
            )

            if loss_phi is not None:
                losses["loss_phi"] = loss_phi * self.phi_loss_weight
            if loss_theta is not None:
                losses["loss_theta"] = loss_theta * self.theta_loss_weight

        return losses

    @torch.no_grad()
    def forward_test(
        self,
        images: Tensor,
        depth_feats: list[Tensor] | None,
        intrinsics: Tensor,
        image_hw: tuple[int, int],
        **kwargs,
    ) -> GeometryBackendOutput:
        """Forward pass for inference."""
        # Store original dimensions for resizing depth_map back
        orig_H, orig_W = image_hw

        # 1. Resize images to be divisible by DINOv2 patch size (14) and adjust intrinsics
        images_resized, intrinsics_adjusted = self._make_divisible_by_dinov2_patch(
            images, intrinsics
        )

        # Update image_hw to match resized dimensions
        _, _, new_H, new_W = images_resized.shape
        image_hw_resized = (new_H, new_W)

        # 2. Normalize images for DINOv2
        images_normalized = self._normalize_image_for_dinov2(images_resized)

        # 3. Extract DINO features
        dino_features, dino_tokens = self.dino_encoder(images_normalized)

        # 4. Project features if needed (e.g., vit_small → vit_large)
        if self._has_projector:
            # DINOv2 returns features in [B, H, W, C] format
            # Project each feature map: [B, H, W, C_in] → [B, H, W, C_out]
            # Keep in [B, H, W, C] format for decoder
            dino_features = [
                self.feature_projector(feat)
                for feat in dino_features
            ]
            # Project tokens: [B, C_in] → [B, C_out]
            dino_tokens = [self.feature_projector(tok) for tok in dino_tokens]
        # else: Keep features in [B, H, W, C] format (no conversion needed)

        # 5. Prepare input dict (use adjusted intrinsics and resized image_hw)
        input_dict = self._prepare_input_dict(
            images=images_normalized,
            dino_features=dino_features,
            dino_tokens=dino_tokens,
            intrinsics=intrinsics_adjusted,
            image_hw=image_hw_resized,
        )

        # 6. Run decoder
        outputs = self.depth_decoder(input_dict)

        depth_map = outputs["depth_maps"]
        depth_latents = outputs.get("depth_latents")  # [B, N, C] - already flattened
        pred_K = outputs.get("pred_K")

        # 7. Resize depth_map back to original dimensions
        depth_map = self._resize_depth_to_original(depth_map, (orig_H, orig_W))

        # Apply optional detach
        depth_latents = self._maybe_detach_latents(depth_latents)

        return GeometryBackendOutput(
            depth_map=depth_map,
            depth_latents=depth_latents,
            K_pred=pred_K,
            ray_intrinsics=intrinsics_adjusted,  # Adjusted intrinsics for DINOv2 space
            ray_image_hw=image_hw_resized,  # Adjusted image size
            ray_downsample=8,  # DetAny3D uses 1/8 resolution for depth_latents
            aux={
                "confidence": outputs.get("confidence"),
                "rays": outputs.get("rays"),
                "depth_latents_hw": outputs.get("depth_latents_hw"),
            },
            losses={},
        )

