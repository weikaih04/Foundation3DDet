"""EfficientSAM3_3D: Lightweight 3D Detection with Integrated Depth.

Single TinyViT-11M backbone handles both RGB and depth (4ch input),
replacing SAM3 ViT-H (462M) + LingBot DINOv2 (300M) dual-backbone.

Architecture:
    RGB + Depth (4ch) -> TinyViT-11M -> {
        Stage 2 features -> depth_latents (for 3D head cross-attention)
        Stage 3 features -> adapter -> ConvStack neck -> depth head -> depth_map
        Final features -> SimpleFPN -> SAM3 encoder/decoder -> 2D boxes
    } -> 3D head -> 3D boxes

Total backbone: ~10.55M + MobileCLIP 63.56M = ~74M
vs original: SAM3 ViT-H 462M + LingBot 300M + SAM3 Text 354M = ~1116M
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Reuse SAM3_3D base classes and helpers (no modification to sam3_3d.py)
from opendet3d.model.detect3d.sam3_3d import (
    SAM3_3D,
    SAM3_3DBatchedInputs,
    SAM3_3DOut,
    Det3DOut,
    Fp32LayerNorm,
    _upgrade_layernorms_to_fp32,
)

# SAM3 imports
from sam3.model.sam3_image import Sam3Image

# 3D-MOOD imports
from opendet3d.op.detect3d.grounding_dino_3d import (
    GroundingDINO3DHead,
    GroundingDINO3DCoder,
    RoI2Det3D,
)

# LingBot depth modules (ConvStack architecture)
lingbot_depth_root = Path(__file__).parent.parent.parent.parent / "lingbot-depth"
if str(lingbot_depth_root) not in sys.path:
    sys.path.insert(0, str(lingbot_depth_root))
from mdm.model.modules_decoder import ConvStack


# ============================================================================
# Depth normalization constants
# ============================================================================
# Log depth range: log(0.01m) = -4.6, log(200m) = 5.3
# Map to [0, 1] then scale to [0, 255] then apply ImageNet green channel norm
DEPTH_LOG_MIN = -5.0
DEPTH_LOG_MAX = 5.5
DEPTH_LOG_RANGE = DEPTH_LOG_MAX - DEPTH_LOG_MIN

# ImageNet green channel stats (most neutral for single-channel depth)
DEPTH_IMAGENET_MEAN = 0.456
DEPTH_IMAGENET_STD = 0.224

# SAM3 normalization: (x - 0.5) / 0.5
SAM3_MEAN = 0.5
SAM3_STD = 0.5


def normalize_depth_to_sam3(
    depth: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Normalize metric depth to SAM3 input range [-1, 1].

    Pipeline: metric depth -> log -> [0, 1] -> [0, 255] -> ImageNet norm -> SAM3 norm.

    Args:
        depth: (B, 1, H, W) metric depth in meters.
        valid_mask: (B, 1, H, W) True for valid depth pixels.

    Returns:
        (B, 1, H, W) normalized depth in SAM3 range [-1, 1].
    """
    # Log transform (handles wide dynamic range: 0.01m - 200m)
    depth_log = torch.log(depth.clamp(min=0.01))

    # Map to [0, 1]
    depth_01 = (depth_log - DEPTH_LOG_MIN) / DEPTH_LOG_RANGE
    depth_01 = depth_01.clamp(0.0, 1.0)

    # ImageNet norm on [0, 1] scale
    depth_inet = (depth_01 - DEPTH_IMAGENET_MEAN) / DEPTH_IMAGENET_STD

    # Convert ImageNet norm -> SAM3 norm: denorm to [0,1] then (x-0.5)/0.5
    # Shortcut: SAM3 = (ImageNet * std + mean - 0.5) / 0.5
    depth_sam3 = (depth_inet * DEPTH_IMAGENET_STD + DEPTH_IMAGENET_MEAN - SAM3_MEAN) / SAM3_STD

    # Zero out invalid pixels (monocular mode: all zeros)
    if valid_mask is not None:
        depth_sam3 = depth_sam3 * valid_mask.float()

    return depth_sam3


class EfficientSAM3_3D(SAM3_3D):
    """EfficientSAM3 with integrated depth estimation and 3D detection.

    Inherits helper methods from SAM3_3D:
        _build_find_stage, _build_geometric_prompt, _build_find_target,
        _forward_test, _convert_imagenet_to_sam3_norm, _xyxy_to_cxcywh,
        _get_is_exhaustive

    Key differences from SAM3_3D:
        1. TinyViT-11M backbone (vs SAM3 ViT-H 462M)
        2. MobileCLIP-S1 text encoder (vs SAM3 text encoder 354M)
        3. 4ch input (RGB + depth) with depth conditioning
        4. No separate geometry backend or early depth fusion
        5. LingBot depth head attached to TinyViT intermediate features
    """

    def __init__(
        self,
        # EfficientSAM3 model (pre-built with TinyViT + MobileCLIP)
        sam3_model: Sam3Image,
        # 3D detection components (same as SAM3_3D)
        bbox3d_head: GroundingDINO3DHead | None = None,
        box_coder: GroundingDINO3DCoder | None = None,
        roi2det3d: RoI2Det3D | None = None,
        # Checkpoint paths for weight initialization
        sam3_3d_checkpoint: str | None = None,
        lingbot_pretrained: str = "pretrained/lingbot-depth/postrain-dc-vitl14/model.pt",
        # Depth conditioning (same as LingBot)
        monocular_prob: float = 0.7,
        masked_prob: float = 0.2,
        mask_ratio_range: tuple = (0.6, 0.9),
        mask_patch_size: int = 14,
        remap_depth_out: str = "exp",
        # Depth loss weights
        depth_loss_weight: float = 1.0,
        silog_loss_weight: float = 0.5,
        camera_loss_weight: float = 1.0,
        mask_loss_weight: float = 0.1,
        # Intrinsic prediction
        intrinsic_use_gt_prob: float = 0.5,
        # Depth latent dim for 3D head
        depth_latent_dim: int = 256,
        # Eval settings
        oracle_eval: bool = False,
        eval_3d_conf_weight: float = 0.5,
        use_depth_input_test: bool = False,
    ) -> None:
        # Skip SAM3_3D.__init__, call nn.Module directly
        nn.Module.__init__(self)

        # =====================================================================
        # 1. EfficientSAM3 model
        # =====================================================================
        self.sam3 = sam3_model
        self.hidden_dim = sam3_model.hidden_dim
        self.oracle_eval = oracle_eval
        self.use_depth_input_test = use_depth_input_test
        self.eval_3d_conf_weight = eval_3d_conf_weight

        # These are needed by inherited methods but not used
        self.geometry_backend = None
        self.early_depth_fusion = None

        # =====================================================================
        # 2. Modify TinyViT PatchEmbed for 4ch input
        # =====================================================================
        tinyvit = self._get_tinyvit()
        self._modify_patch_embed_for_4ch(tinyvit)

        # =====================================================================
        # 3. Register hooks to capture intermediate features
        # =====================================================================
        self._stage2_features = None
        self._stage3_features = None
        self._register_feature_hooks(tinyvit)

        # =====================================================================
        # 4. Build depth pipeline (from LingBot architecture)
        # =====================================================================
        # Adapter: TinyViT Stage 3 (448ch) -> 1024ch for ConvStack neck
        self.depth_adapter = nn.Conv2d(448, 1024, kernel_size=1)

        # ConvStack neck (same config as LingBot)
        self.depth_neck = ConvStack(
            dim_in=[1026, 2, 2, 2, 2],
            dim_res_blocks=[1024, 256, 128, 64, 32],
            dim_out=None,
            num_res_blocks=[0, 2, 2, 2, 0],
            res_block_in_norm="none",
            res_block_hidden_norm="none",
            resamplers=["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
        )

        # ConvStack depth head (same config as LingBot)
        self.depth_head = ConvStack(
            dim_in=[1024, 256, 128, 64, 32],
            dim_out=[None, None, None, None, 1],
            dim_res_blocks=[1024, 256, 128, 64, 32],
            num_res_blocks=[0, 1, 1, 1, 0],
            res_block_in_norm="none",
            res_block_hidden_norm="none",
            resamplers=["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
        )

        # ConvStack mask head (confidence prediction, same as LingBot)
        self.mask_head = ConvStack(
            dim_in=[1024, 256, 128, 64, 32],
            dim_out=[None, None, None, None, 1],
            dim_res_blocks=[1024, 256, 128, 64, 32],
            num_res_blocks=[0, 1, 1, 1, 0],
            res_block_in_norm="none",
            res_block_hidden_norm="none",
            resamplers=["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"],
        )

        # Intrinsic head: GAP(stage3, 448) -> MLP -> (log_fx, log_fy, logit_cx, logit_cy)
        self.intrinsic_head = nn.Sequential(
            nn.LayerNorm(448),
            nn.Linear(448, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
        )

        # =====================================================================
        # 5. Depth conditioning config
        # =====================================================================
        self.monocular_prob = monocular_prob
        self.masked_prob = masked_prob
        self.mask_ratio_range = mask_ratio_range
        self.mask_patch_size = mask_patch_size
        self.remap_depth_out = remap_depth_out

        # Loss weights
        self.depth_loss_weight = depth_loss_weight
        self.silog_loss_weight = silog_loss_weight
        self.camera_loss_weight = camera_loss_weight
        self.mask_loss_weight = mask_loss_weight

        # Intrinsic prediction
        self.intrinsic_use_gt_prob = intrinsic_use_gt_prob
        self.depth_latent_dim = depth_latent_dim

        # =====================================================================
        # 6. 3D detection head
        # =====================================================================
        self.box_coder = box_coder or GroundingDINO3DCoder()

        # LingBot is NOT ray-aware -> use camera prompt
        use_camera_prompt = True

        if bbox3d_head is not None:
            self.bbox3d_head = bbox3d_head
        else:
            self.bbox3d_head = GroundingDINO3DHead(
                embed_dims=self.hidden_dim,
                box_coder=self.box_coder,
                use_camera_prompt=use_camera_prompt,
                depth_latent_dim=depth_latent_dim,
            )

        self.roi2det3d = roi2det3d

        # =====================================================================
        # 7. Ensure SAM3 has a matcher for training
        # =====================================================================
        if self.sam3.matcher is None:
            from sam3.train.matcher import BinaryHungarianMatcherV2
            self.sam3.matcher = BinaryHungarianMatcherV2(
                cost_class=2.0, cost_bbox=5.0, cost_giou=2.0,
                focal=True, alpha=0.25, gamma=2.0,
            )

        # =====================================================================
        # 8. Load pretrained weights
        # =====================================================================
        self._load_depth_weights(lingbot_pretrained, sam3_3d_checkpoint)

        # =====================================================================
        # 9. Upgrade LayerNorms to fp32 for bf16 stability
        # =====================================================================
        n_replaced = _upgrade_layernorms_to_fp32(self)
        print(f"[EfficientSAM3_3D] Upgraded {n_replaced} LayerNorm -> Fp32LayerNorm")

        # Print parameter summary
        self._print_param_summary()

    # ========================================================================
    # TinyViT access helpers
    # ========================================================================

    def _get_tinyvit(self):
        """Navigate nested wrappers to get the raw TinyViT model.

        Structure:
            Sam3Image -> SAM3VLBackbone -> Sam3DualViTDetNeck
              -> ListWrapper (trunk)
                -> ImageStudentEncoder (trunk.model)
                  -> TinyViTTrunkWrapper (trunk.model.backbone)
                    -> TinyViT (trunk.model.backbone.model)
        """
        vision_backbone = self.sam3.backbone.vision_backbone
        trunk = vision_backbone.trunk  # ListWrapper

        # ListWrapper.model -> ImageStudentEncoder
        encoder = trunk.model
        # ImageStudentEncoder.backbone -> TinyViTTrunkWrapper
        trunk_wrapper = encoder.backbone
        # TinyViTTrunkWrapper.model -> TinyViT
        tinyvit = trunk_wrapper.model

        return tinyvit

    def _modify_patch_embed_for_4ch(self, tinyvit):
        """Modify TinyViT's first conv from 3ch to 4ch input.

        The first layer is Conv2d_BN(3, 32, k=3, s=2, p=1).
        We replace with Conv2d_BN(4, 32, k=3, s=2, p=1),
        copying RGB weights and zero-initializing the depth channel.
        """
        patch_embed = tinyvit.patch_embed
        # First module in seq is Conv2d_BN(3 -> 32)
        first_conv_bn = patch_embed.seq[0]
        old_conv = first_conv_bn.c  # The Conv2d inside Conv2d_BN

        if old_conv.in_channels == 4:
            print("[EfficientSAM3_3D] PatchEmbed already 4ch, skipping modification")
            return

        # Create new conv with 4 input channels
        new_conv = nn.Conv2d(
            4, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        # Copy RGB weights, zero-init depth channel
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:, :, :] = 0.0
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        first_conv_bn.c = new_conv
        print("[EfficientSAM3_3D] Modified PatchEmbed: 3ch -> 4ch (depth zero-init)")

    def _register_feature_hooks(self, tinyvit):
        """Register forward hooks to capture TinyViT intermediate features.

        Stage 2 (layers[2]): 63x63, 256ch -> depth_latents
        Stage 3 (layers[3]): 32x32, 448ch -> depth head input
        """
        # Capture Stage 2 features BEFORE downsample (PatchMerging)
        # forward_pre_hook on downsample gets input = features before downsampling
        if tinyvit.layers[2].downsample is not None:
            tinyvit.layers[2].downsample.register_forward_pre_hook(
                self._hook_capture_stage2
            )
        else:
            # Fallback: hook on layer output
            tinyvit.layers[2].register_forward_hook(
                self._hook_capture_stage2_output
            )

        # Capture Stage 3 features (final layer, no downsample)
        tinyvit.layers[3].register_forward_hook(
            self._hook_capture_stage3
        )

    def _hook_capture_stage2(self, module, args):
        """Pre-hook on PatchMerging: captures features before downsampling."""
        # args[0] is the input tensor to downsample: (B, H*W, 256)
        self._stage2_features = args[0]

    def _hook_capture_stage2_output(self, module, input, output):
        """Fallback: capture Stage 2 output directly."""
        self._stage2_features = output

    def _hook_capture_stage3(self, module, input, output):
        """Hook on Stage 3: captures final features."""
        # output: (B, H*W, 448)
        self._stage3_features = output

    # ========================================================================
    # Depth conditioning (same logic as LingBot)
    # ========================================================================

    def _prepare_depth_input(
        self,
        depth_gt: Tensor | None,
        depth_mask: Tensor | None,
        B: int, H: int, W: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Prepare depth input with mixed conditioning strategy.

        Returns:
            depth_input: (B, 1, H, W) conditioned depth (may be zeros)
            valid_mask: (B, 1, H, W) True where depth is valid
        """
        if depth_gt is None or not self.training:
            # Inference: always monocular (zero depth)
            if not self.training and self.use_depth_input_test and depth_gt is not None:
                # Test with depth input
                if depth_gt.dim() == 3:
                    depth_gt = depth_gt.unsqueeze(1)
                valid = (depth_gt > 0.01) & torch.isfinite(depth_gt)
                return depth_gt * valid.float(), valid
            return (
                torch.zeros(B, 1, H, W, device=device),
                torch.zeros(B, 1, H, W, dtype=torch.bool, device=device),
            )

        if depth_gt.dim() == 3:
            depth_gt = depth_gt.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        depth_input = torch.zeros(B, 1, H, W, device=device)
        valid_mask = torch.zeros(B, 1, H, W, dtype=torch.bool, device=device)

        for i in range(B):
            r = torch.rand(1).item()
            d = depth_gt[i:i+1]  # (1, 1, H, W)
            m = depth_mask[i:i+1] if depth_mask is not None else (d > 0.01).squeeze(1)
            if m.dim() == 2:
                m = m.unsqueeze(0)  # (1, H, W)

            d_valid = d.clone()
            d_valid[~torch.isfinite(d_valid)] = 0.0
            d_valid[d_valid < 0.01] = 0.0
            pixel_valid = (d_valid > 0.01).squeeze(1) & m  # (1, H, W)

            if r < self.monocular_prob:
                # Monocular: zero depth
                pass
            elif r < self.monocular_prob + self.masked_prob:
                # Patch-masked depth
                masked = self._patch_mask_depth(
                    d_valid.squeeze(0), pixel_valid.squeeze(0),
                    H, W, device,
                )
                depth_input[i] = masked
                valid_mask[i] = masked > 0.01
            else:
                # Copy-through: full GT depth
                depth_input[i] = d_valid.squeeze(0) * pixel_valid.float()
                valid_mask[i] = pixel_valid

        return depth_input, valid_mask

    def _patch_mask_depth(
        self,
        depth: Tensor,
        valid: Tensor,
        H: int, W: int,
        device: torch.device,
    ) -> Tensor:
        """Apply random patch-level masking to depth (60-90% masked)."""
        ps = self.mask_patch_size
        nH = (H + ps - 1) // ps
        nW = (W + ps - 1) // ps

        ratio = (
            self.mask_ratio_range[0]
            + torch.rand(1).item() * (self.mask_ratio_range[1] - self.mask_ratio_range[0])
        )
        n_mask = int(nH * nW * ratio)

        # Random patch indices to mask (set to zero)
        indices = torch.randperm(nH * nW, device=device)[:n_mask]
        mask_2d = torch.ones(nH, nW, device=device)
        rows = indices // nW
        cols = indices % nW
        mask_2d[rows, cols] = 0.0

        # Upsample to pixel level
        mask_pixel = F.interpolate(
            mask_2d.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode="nearest",
        ).squeeze(0)  # (1, H, W)

        return depth * mask_pixel * valid.float().unsqueeze(0)

    # ========================================================================
    # UV coordinate generation (same as LingBot)
    # ========================================================================

    def _generate_uv_coords(self, H: int, W: int, device: torch.device) -> Tensor:
        """Generate normalized UV coordinates (B=1, 2, H, W)."""
        u = torch.linspace(0, 1, W, device=device)
        v = torch.linspace(0, 1, H, device=device)
        grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
        uv = torch.stack([grid_u, grid_v], dim=0).unsqueeze(0)  # (1, 2, H, W)
        return uv

    # ========================================================================
    # Intrinsic prediction
    # ========================================================================

    def _predict_intrinsics(
        self,
        stage3_feat: Tensor,
        image_hw: tuple[int, int],
    ) -> Tensor:
        """Predict camera intrinsics from Stage 3 features via GAP + MLP.

        Args:
            stage3_feat: (B, N, 448) Stage 3 features.
            image_hw: (H, W) original image size.

        Returns:
            K_pred: (B, 3, 3) predicted camera intrinsics.
        """
        H, W = image_hw
        diagonal = math.sqrt(H * H + W * W)

        # Global average pooling
        feat_gap = stage3_feat.mean(dim=1)  # (B, 448)

        # MLP -> 4 raw params
        raw = self.intrinsic_head(feat_gap)  # (B, 4)
        log_fx, log_fy, logit_cx, logit_cy = raw.unbind(-1)

        # Parameterization (same as LingBot)
        fx = log_fx.exp() * 0.7 * diagonal
        fy = log_fy.exp() * 0.7 * diagonal
        cx = torch.sigmoid(logit_cx) * W
        cy = torch.sigmoid(logit_cy) * H

        B = stage3_feat.shape[0]
        K = torch.zeros(B, 3, 3, device=stage3_feat.device, dtype=stage3_feat.dtype)
        K[:, 0, 0] = fx
        K[:, 1, 1] = fy
        K[:, 0, 2] = cx
        K[:, 1, 2] = cy
        K[:, 2, 2] = 1.0

        return K

    # ========================================================================
    # Depth head forward
    # ========================================================================

    def _run_depth_pipeline(
        self,
        stage3_feat: Tensor,
        B: int, H_orig: int, W_orig: int,
    ) -> dict:
        """Run depth estimation from TinyViT Stage 3 features.

        Args:
            stage3_feat: (B, N, 448) TinyViT Stage 3 output.
            B: batch size.
            H_orig, W_orig: original image spatial dims.

        Returns:
            dict with depth_map, confidence_map, K_pred.
        """
        N = stage3_feat.shape[1]
        side = int(N ** 0.5)
        device = stage3_feat.device

        # Reshape to spatial: (B, 448, H_s3, W_s3)
        feat_2d = stage3_feat.permute(0, 2, 1).reshape(B, 448, side, side)

        # Adapter: 448 -> 1024
        feat_1024 = self.depth_adapter(feat_2d)  # (B, 1024, H_s3, W_s3)

        # Generate UV coords at each neck level resolution
        H_s3, W_s3 = side, side
        uv_0 = self._generate_uv_coords(H_s3, W_s3, device).expand(B, -1, -1, -1)
        neck_input_0 = torch.cat([feat_1024, uv_0], dim=1)  # (B, 1026, H, W)

        # Build 5-level input for ConvStack neck
        # Levels 1-4: UV coords at progressively larger resolutions
        resolutions = [
            (H_s3 * 2, W_s3 * 2),
            (H_s3 * 4, W_s3 * 4),
            (H_s3 * 8, W_s3 * 8),
            (H_s3 * 16, W_s3 * 16),
        ]
        neck_inputs = [neck_input_0]
        for h, w in resolutions:
            uv = self._generate_uv_coords(h, w, device).expand(B, -1, -1, -1)
            neck_inputs.append(uv)

        # ConvStack neck
        neck_features = self.depth_neck(neck_inputs)

        # ConvStack depth head
        depth_reg = self.depth_head(neck_features)[-1]  # (B, 1, H_out, W_out)

        # Remap depth output
        if self.remap_depth_out == "exp":
            depth_map = depth_reg.clamp(-10, 10).exp()
        elif self.remap_depth_out == "linear":
            depth_map = depth_reg.clamp(min=1e-3)
        else:
            raise ValueError(f"Invalid remap_depth_out: {self.remap_depth_out}")

        # Resize to original image size
        depth_map = F.interpolate(
            depth_map, size=(H_orig, W_orig), mode="bilinear", align_corners=False,
        )

        # Confidence map
        confidence_map = None
        if self.mask_head is not None:
            mask_logits = self.mask_head(neck_features)[-1]
            confidence_map = F.interpolate(
                mask_logits, size=(H_orig, W_orig), mode="bilinear", align_corners=False,
            )

        # Intrinsic prediction
        K_pred = self._predict_intrinsics(stage3_feat, (H_orig, W_orig))

        return {
            "depth_map": depth_map,
            "confidence_map": confidence_map,
            "K_pred": K_pred,
            "neck_features": neck_features,
        }

    # ========================================================================
    # Depth loss computation
    # ========================================================================

    def _compute_depth_losses(
        self,
        depth_pred: Tensor,
        depth_gt: Tensor | None,
        depth_mask: Tensor | None,
        confidence_map: Tensor | None,
        K_pred: Tensor,
        intrinsics_gt: Tensor,
        image_hw: tuple[int, int],
    ) -> dict[str, Tensor]:
        """Compute depth-related training losses."""
        device = depth_pred.device
        losses = {}

        if depth_gt is None:
            return losses

        if depth_gt.dim() == 3:
            depth_gt = depth_gt.unsqueeze(1)

        H, W = image_hw

        # Build valid mask
        if depth_mask is not None:
            if depth_mask.dim() == 3:
                valid = depth_mask.unsqueeze(1).bool()
            else:
                valid = depth_mask.bool()
        else:
            valid = (depth_gt > 0.01) & torch.isfinite(depth_gt)

        valid = valid & (depth_gt > 0.01) & torch.isfinite(depth_gt)
        n_valid = valid.sum().clamp(min=1)

        # L1 depth loss
        l1_loss = (depth_pred[valid] - depth_gt[valid]).abs().mean()
        losses["depth_l1"] = l1_loss.clamp(max=10.0) * self.depth_loss_weight

        # SILog loss (scale-invariant logarithmic)
        if n_valid > 1:
            log_pred = torch.log(depth_pred[valid].clamp(min=1e-3))
            log_gt = torch.log(depth_gt[valid].clamp(min=1e-3))
            d = log_pred - log_gt
            silog = (d ** 2).mean() - 0.5 * (d.mean() ** 2)
            losses["depth_silog"] = silog.clamp(max=10.0) * self.silog_loss_weight

        # Confidence/mask loss (BCE)
        if confidence_map is not None and self.mask_loss_weight > 0:
            # Target: 1 for valid finite depth, 0 for invalid/infinite
            has_depth = (depth_gt > 0.01) & torch.isfinite(depth_gt)
            mask_target = has_depth.float()
            mask_bce = F.binary_cross_entropy_with_logits(
                confidence_map, mask_target, reduction="mean",
            )
            losses["mask_bce"] = mask_bce * self.mask_loss_weight

        # Camera loss (ray-based MSE)
        if self.camera_loss_weight > 0 and intrinsics_gt is not None:
            from opendet3d.op.geometric.ray import generate_rays
            rays_pred = generate_rays(K_pred, image_hw)
            rays_gt = generate_rays(intrinsics_gt, image_hw)
            # generate_rays returns (rays, _) tuple
            if isinstance(rays_pred, tuple):
                rays_pred = rays_pred[0]
            if isinstance(rays_gt, tuple):
                rays_gt = rays_gt[0]
            camera_loss = F.mse_loss(rays_pred, rays_gt)
            losses["camera_ray"] = camera_loss * self.camera_loss_weight

        return losses

    # ========================================================================
    # Depth latents extraction
    # ========================================================================

    def _extract_depth_latents(self, stage2_feat: Tensor) -> Tensor:
        """Extract depth latents from Stage 2 features for 3D head.

        Args:
            stage2_feat: (B, N, 256) TinyViT Stage 2 features.

        Returns:
            depth_latents: (B, N, 256) for 3D head cross-attention.
        """
        # Stage 2 is already 256-dim, matching depth_latent_dim
        # Detach to prevent depth gradients from flowing back through backbone
        # (same as LingBot's detach_depth_latents=True default)
        return stage2_feat.detach()

    # ========================================================================
    # Forward pass
    # ========================================================================

    def forward(
        self,
        batch: SAM3_3DBatchedInputs,
        targets: dict | None = None,
    ) -> SAM3_3DOut:
        """Forward pass of EfficientSAM3_3D.

        Flow:
            1. Prepare 4ch input (RGB + conditioned depth)
            2. Run backbone (hooks capture Stage 2/3 features)
            3. Run depth pipeline (Stage 3 -> depth head)
            4. Extract depth latents (Stage 2)
            5. SAM3 forward_grounding (encoder + decoder + matching)
            6. 3D head with depth_latents and ray_embeddings
        """
        B_images = batch.images.shape[0]
        N_prompts = len(batch.img_ids)
        _, _, H, W = batch.images.shape
        device = batch.images.device

        # Sync training mode
        if self.sam3.training != self.training:
            self.sam3.train(self.training)

        # Handle empty batch
        if N_prompts == 0:
            if self.training:
                dummy_grad = sum(p.sum() * 0 for p in self.parameters() if p.requires_grad)
                dummy_logits = torch.zeros(1, 1, 1, device=device) + dummy_grad
                return SAM3_3DOut(
                    pred_logits=dummy_logits,
                    pred_boxes_2d=torch.zeros(1, 1, 4, device=device),
                    pred_boxes_3d=None, aux_outputs=None, geom_losses=None,
                    presence_logits=None, queries=None,
                    encoder_hidden_states=None, indices=None,
                )
            else:
                return Det3DOut(
                    boxes=[torch.zeros(0, 4, device=device) for _ in range(B_images)],
                    boxes3d=[torch.zeros(0, 10, device=device) for _ in range(B_images)],
                    scores=[torch.zeros(0, device=device) for _ in range(B_images)],
                    class_ids=[torch.zeros(0, dtype=torch.long, device=device) for _ in range(B_images)],
                    depth_maps=None, categories=None,
                )

        # =================================================================
        # Step 1: Prepare 4ch input (RGB + conditioned depth)
        # =================================================================
        depth_gt = getattr(batch, "depth_gt", None)
        depth_mask = getattr(batch, "depth_mask", None)

        depth_input, depth_valid = self._prepare_depth_input(
            depth_gt, depth_mask, B_images, H, W, device,
        )

        # Convert RGB: ImageNet norm -> SAM3 norm
        images_sam3 = self._convert_imagenet_to_sam3_norm(batch.images)

        # Convert depth: metric -> SAM3 norm
        depth_sam3 = normalize_depth_to_sam3(depth_input, depth_valid)

        # Concat to 4ch
        images_4ch = torch.cat([images_sam3, depth_sam3], dim=1)  # (B, 4, H, W)

        # =================================================================
        # Step 2: Run backbone (hooks capture intermediate features)
        # =================================================================
        # Reset hook storage
        self._stage2_features = None
        self._stage3_features = None

        # Run through SAM3 backbone (modified TinyViT accepts 4ch)
        backbone_out = {"img_batch_all_stages": batch.images}
        backbone_out.update(self.sam3.backbone.forward_image(images_4ch))
        text_out = self.sam3.backbone.forward_text(
            batch.unique_texts, device=device,
        )
        backbone_out.update(text_out)

        # =================================================================
        # Step 3: Run depth pipeline (from captured Stage 3 features)
        # =================================================================
        geom_losses = None
        depth_out = None

        if self._stage3_features is not None:
            depth_out = self._run_depth_pipeline(
                self._stage3_features, B_images, H, W,
            )

            # Compute depth losses during training
            if self.training:
                geom_losses = self._compute_depth_losses(
                    depth_pred=depth_out["depth_map"],
                    depth_gt=depth_gt,
                    depth_mask=depth_mask,
                    confidence_map=depth_out["confidence_map"],
                    K_pred=depth_out["K_pred"],
                    intrinsics_gt=batch.intrinsics,
                    image_hw=(H, W),
                )

        # =================================================================
        # Step 4: Extract depth latents (from captured Stage 2 features)
        # =================================================================
        depth_latents = None
        if self._stage2_features is not None:
            depth_latents = self._extract_depth_latents(self._stage2_features)

        # =================================================================
        # Step 5: SAM3 forward_grounding (encoder + decoder + matching)
        # =================================================================
        find_input = self._build_find_stage(batch, device)
        geometric_prompt = self._build_geometric_prompt(batch, device)

        find_target = None
        if self.training:
            find_target = self._build_find_target(batch)

        sam3_out = self.sam3.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=find_target,
            geometric_prompt=geometric_prompt,
        )

        # =================================================================
        # Step 6: Extract SAM3 outputs
        # =================================================================
        pred_logits = sam3_out["pred_logits"]
        pred_boxes_xyxy = sam3_out["pred_boxes_xyxy"]
        pred_boxes_cxcywh = sam3_out["pred_boxes"]
        queries = sam3_out.get("queries")
        encoder_hidden_states = sam3_out.get("encoder_hidden_states")
        presence_logits = sam3_out.get("presence_logit_dec")

        # O2M outputs
        pred_logits_o2m = sam3_out.get("pred_logits_o2m")
        pred_boxes_xyxy_o2m = sam3_out.get("pred_boxes_xyxy_o2m")
        pred_boxes_cxcywh_o2m = sam3_out.get("pred_boxes_o2m")
        queries_o2m = sam3_out.get("queries_o2m")

        sam3_aux_outputs = sam3_out.get("aux_outputs", [])

        # =================================================================
        # Step 7: 3D Head
        # =================================================================
        pred_boxes_3d = None
        pred_conf_3d = None
        aux_outputs = None

        if self.bbox3d_head is not None and queries is not None:
            # Ray embeddings from intrinsics (use predicted or GT)
            ray_embeddings = None
            if self.bbox3d_head.use_camera_prompt:
                if depth_out is not None and self.training:
                    # During training, alternate between GT and predicted intrinsics
                    if torch.rand(1).item() < self.intrinsic_use_gt_prob:
                        ray_K = batch.intrinsics
                    else:
                        ray_K = depth_out["K_pred"]
                elif depth_out is not None:
                    ray_K = depth_out["K_pred"]
                else:
                    ray_K = batch.intrinsics

                # Use Stage 3 spatial dims for ray downsample
                stage3_side = int(self._stage3_features.shape[1] ** 0.5) if self._stage3_features is not None else 32
                ray_downsample = max(H // stage3_side, 1)

                ray_embeddings = self.bbox3d_head.get_camera_embeddings(
                    ray_K, (H, W), ray_downsample,
                )

            # Align spatial resolution of depth_latents and ray_embeddings
            if depth_latents is not None and ray_embeddings is not None:
                B_d, N_d, C_d = depth_latents.shape
                B_r, N_r, C_r = ray_embeddings.shape
                if N_d != N_r:
                    H_d = int(N_d ** 0.5)
                    H_r = int(N_r ** 0.5)
                    depth_latents_2d = depth_latents.permute(0, 2, 1).reshape(B_d, C_d, H_d, H_d)
                    depth_latents_resized = F.adaptive_avg_pool2d(depth_latents_2d, (H_r, H_r))
                    depth_latents = depth_latents_resized.reshape(B_d, C_d, -1).permute(0, 2, 1)

            # Index per-image -> per-prompt
            if ray_embeddings is not None:
                ray_embeddings = ray_embeddings[batch.img_ids]
            if depth_latents is not None:
                depth_latents = depth_latents[batch.img_ids]

            # Deep supervision: all decoder layers
            all_layers_queries = []
            aux_indices_with_queries = []
            for i, aux_out in enumerate(sam3_aux_outputs):
                aux_queries = aux_out.get("queries")
                if aux_queries is not None:
                    all_layers_queries.append(aux_queries)
                    aux_indices_with_queries.append(i)
            all_layers_queries.append(queries)

            if len(all_layers_queries) > 1:
                hidden_states = torch.stack(all_layers_queries, dim=0)
            else:
                hidden_states = queries.unsqueeze(0)

            all_layers_boxes_3d, all_layers_conf_3d = self.bbox3d_head(
                hidden_states=hidden_states,
                ray_embeddings=ray_embeddings,
                depth_latents=depth_latents,
            )

            if len(all_layers_queries) > 1:
                pred_boxes_3d = all_layers_boxes_3d[-1]
                pred_conf_3d = all_layers_conf_3d[-1]
            else:
                pred_boxes_3d = all_layers_boxes_3d.squeeze(0)
                pred_conf_3d = all_layers_conf_3d.squeeze(0)

            # Build aux outputs for deep supervision
            if len(aux_indices_with_queries) > 0 and self.training:
                aux_outputs = []
                for layer_idx, orig_idx in enumerate(aux_indices_with_queries):
                    aux_out = sam3_aux_outputs[orig_idx]
                    aux_dict = {
                        "pred_logits": aux_out["pred_logits"],
                        "pred_boxes_2d": aux_out["pred_boxes_xyxy"],
                        "pred_boxes_3d": all_layers_boxes_3d[layer_idx],
                    }
                    if "presence_logit_dec" in aux_out:
                        aux_dict["presence_logits"] = aux_out["presence_logit_dec"]
                    aux_outputs.append(aux_dict)

        # O2M 3D predictions
        pred_boxes_3d_o2m = None
        pred_conf_3d_o2m = None
        if self.bbox3d_head is not None and queries_o2m is not None and self.training:
            o2m_hidden = queries_o2m.unsqueeze(0)
            o2m_3d, o2m_conf = self.bbox3d_head(
                hidden_states=o2m_hidden,
                ray_embeddings=ray_embeddings,
                depth_latents=depth_latents,
            )
            pred_boxes_3d_o2m = o2m_3d.squeeze(0)
            pred_conf_3d_o2m = o2m_conf.squeeze(0)

        # =================================================================
        # Return
        # =================================================================
        if self.training:
            sam3_indices = sam3_out.get("indices", None)
            return SAM3_3DOut(
                pred_logits=pred_logits,
                pred_boxes_2d=pred_boxes_xyxy,
                pred_boxes_3d=pred_boxes_3d,
                aux_outputs=aux_outputs,
                geom_losses=geom_losses,
                presence_logits=presence_logits,
                queries=queries,
                encoder_hidden_states=encoder_hidden_states,
                indices=sam3_indices,
                pred_boxes_2d_cxcywh=pred_boxes_cxcywh,
                pred_logits_o2m=pred_logits_o2m,
                pred_boxes_2d_o2m=pred_boxes_xyxy_o2m,
                pred_boxes_2d_cxcywh_o2m=pred_boxes_cxcywh_o2m,
                pred_boxes_3d_o2m=pred_boxes_3d_o2m,
                pred_conf_3d=pred_conf_3d,
                pred_conf_3d_o2m=pred_conf_3d_o2m,
            )

        # Test mode
        return self._forward_test(
            pred_logits=pred_logits,
            pred_boxes_2d=pred_boxes_xyxy,
            pred_boxes_3d=pred_boxes_3d,
            pred_conf_3d=pred_conf_3d,
            presence_logits=presence_logits,
            batch=batch,
            geom_out={"depth_map": depth_out["depth_map"]} if depth_out else None,
        )

    # ========================================================================
    # Pretrained weight loading
    # ========================================================================

    def _load_depth_weights(
        self,
        lingbot_pretrained: str,
        sam3_3d_checkpoint: str | None,
    ):
        """Load pretrained weights for depth pipeline and 3D head.

        Priority:
        - If sam3_3d_checkpoint provided: load depth neck/head/mask/intrinsic + 3D head
        - Else: load from lingbot_pretrained for depth components
        """
        if sam3_3d_checkpoint is not None:
            self._load_from_sam3_3d_checkpoint(sam3_3d_checkpoint)
        elif lingbot_pretrained is not None:
            self._load_from_lingbot_pretrained(lingbot_pretrained)

    def _load_from_sam3_3d_checkpoint(self, ckpt_path: str):
        """Load depth + 3D head weights from a trained SAM3_3D checkpoint."""
        print(f"[EfficientSAM3_3D] Loading from SAM3_3D checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        # Map SAM3_3D keys -> EfficientSAM3_3D keys
        # SAM3_3D: model.geometry_backend.neck.* -> self.depth_neck.*
        # SAM3_3D: model.geometry_backend.depth_head.* -> self.depth_head.*
        # SAM3_3D: model.geometry_backend.mask_head.* -> self.mask_head.*
        # SAM3_3D: model.geometry_backend.intrinsic_head.* -> self.intrinsic_head.*
        # SAM3_3D: model.bbox3d_head.* -> self.bbox3d_head.*
        # SAM3_3D: model.roi2det3d.* -> self.roi2det3d.*
        key_mapping = {
            "model.geometry_backend.neck.": "depth_neck.",
            "model.geometry_backend.depth_head.": "depth_head.",
            "model.geometry_backend.mask_head.": "mask_head.",
            "model.bbox3d_head.": "bbox3d_head.",
        }

        loaded_modules = set()
        my_state = self.state_dict()
        for old_key, value in state_dict.items():
            for prefix_old, prefix_new in key_mapping.items():
                if old_key.startswith(prefix_old):
                    new_key = old_key.replace(prefix_old, prefix_new)
                    if new_key in my_state:
                        if my_state[new_key].shape == value.shape:
                            my_state[new_key] = value
                            loaded_modules.add(prefix_new.rstrip("."))
                        else:
                            print(f"  Shape mismatch: {new_key} "
                                  f"({my_state[new_key].shape} vs {value.shape}), skipping")
                    break

        # Load intrinsic head (different architecture: LingBot uses cls_token, we use GAP)
        # The MLP layers might have different input dims, so we skip and train from scratch
        print(f"  Intrinsic head: training from scratch (different input: GAP vs cls_token)")

        # Load matched weights
        self.load_state_dict(my_state, strict=False)
        print(f"  Loaded modules: {sorted(loaded_modules)}")

        # ROI2Det3D (stateless, no weights needed)
        # depth_adapter: random init (new module)
        print(f"  depth_adapter: random init (new 448->1024 projection)")

    def _load_from_lingbot_pretrained(self, ckpt_path: str):
        """Load depth weights from LingBot pretrained model."""
        print(f"[EfficientSAM3_3D] Loading from LingBot pretrained: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)

        key_mapping = {
            "neck.": "depth_neck.",
            "depth_head.": "depth_head.",
            "mask_head.": "mask_head.",
        }

        loaded = 0
        my_state = self.state_dict()
        for old_key, value in state_dict.items():
            for prefix_old, prefix_new in key_mapping.items():
                if old_key.startswith(prefix_old):
                    new_key = old_key.replace(prefix_old, prefix_new)
                    if new_key in my_state and my_state[new_key].shape == value.shape:
                        my_state[new_key] = value
                        loaded += 1
                    break

        self.load_state_dict(my_state, strict=False)
        print(f"  Loaded {loaded} parameters from LingBot pretrained")

    def on_load_checkpoint(self, checkpoint):
        """Handle checkpoint loading for vis4d/Lightning."""
        state_dict = checkpoint.get("state_dict", {})
        has_sam3 = any("sam3." in k for k in state_dict)
        has_depth_neck = any("depth_neck." in k for k in state_dict)
        is_resume = has_sam3 and has_depth_neck

        if is_resume:
            print("[EfficientSAM3_3D] Resume training from checkpoint")
        else:
            print("[EfficientSAM3_3D] First training (fine-tuning)")
            # Reset training state
            for key in ["epoch", "global_step"]:
                if key in checkpoint:
                    checkpoint[key] = 0
            for key in ["optimizer_states", "lr_schedulers"]:
                if key in checkpoint:
                    del checkpoint[key]

    # ========================================================================
    # Utilities
    # ========================================================================

    def _print_param_summary(self):
        """Print parameter count summary."""
        def count_params(module):
            return sum(p.numel() for p in module.parameters())

        def count_trainable(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        total = count_params(self)
        trainable = count_trainable(self)

        print(f"\n{'='*60}")
        print(f"EfficientSAM3_3D Parameter Summary")
        print(f"{'='*60}")
        print(f"  SAM3 (backbone+encoder+decoder): {count_params(self.sam3)/1e6:.1f}M")
        print(f"  depth_adapter (448->1024):        {count_params(self.depth_adapter)/1e6:.2f}M")
        print(f"  depth_neck (ConvStack):           {count_params(self.depth_neck)/1e6:.1f}M")
        print(f"  depth_head (ConvStack):           {count_params(self.depth_head)/1e6:.1f}M")
        print(f"  mask_head (ConvStack):            {count_params(self.mask_head)/1e6:.1f}M")
        print(f"  intrinsic_head (GAP+MLP):         {count_params(self.intrinsic_head)/1e6:.2f}M")
        print(f"  bbox3d_head (3D detection):       {count_params(self.bbox3d_head)/1e6:.1f}M")
        print(f"  ---")
        print(f"  Total:     {total/1e6:.1f}M")
        print(f"  Trainable: {trainable/1e6:.1f}M")
        print(f"{'='*60}\n")
