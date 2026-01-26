"""SAM3_3D Loss Module.

This module implements the loss function for SAM3_3D, combining:
1. SAM3-style 2D losses (IABCEMdetr for classification, L1+GIoU for boxes)
2. 3D-MOOD-style 3D losses (delta_center, depth, dimensions, rotation)

Key Design Decisions:
- Uses SAM3's Hungarian matcher for assignment (already computed in model)
- Follows SAM3's loss normalization (global/local/none)
- Adds 3D regression losses on top of 2D losses
- Supports deep supervision on auxiliary decoder outputs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from vis4d.common.distributed import reduce_mean
from vis4d.op.loss.common import l1_loss
from vis4d.op.loss.reducer import SumWeightedLoss

from opendet3d.op.detect3d.grounding_dino_3d.coder import GroundingDINO3DCoder
from sam3.model.box_ops import fast_diag_box_iou, fast_diag_generalized_box_iou
from sam3.train.matcher import BinaryOneToManyMatcher
from sam3.train.loss.loss_fns import sigmoid_focal_loss


def _packed_to_padded(boxes_packed: Tensor, num_boxes: Tensor, fill_value: float = 0.0) -> Tensor:
    """Convert packed tensor to padded tensor.

    This function converts a packed (concatenated) tensor of bounding boxes
    to a batch-wise padded tensor, following SAM3's collator implementation.

    Args:
        boxes_packed: Packed boxes tensor of shape (N_total, 4) where
                     N_total = N_1 + N_2 + ... + N_B
        num_boxes: Number of boxes per image, shape (B,)
        fill_value: Value to use for padding (default: 0.0)

    Returns:
        Padded boxes tensor of shape (B, max_N, 4) where max_N = max(num_boxes)

    Example:
        >>> boxes = torch.tensor([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
        >>> num_boxes = torch.tensor([1, 2])
        >>> padded = _packed_to_padded(boxes, num_boxes)
        >>> padded.shape
        torch.Size([2, 2, 4])
    """
    B = num_boxes.shape[0]
    Ns = num_boxes.tolist()
    max_N = max(Ns)

    # Create padded tensor
    boxes_padded = boxes_packed.new_full((B, max_N, *boxes_packed.shape[1:]), fill_value)

    # Fill in actual boxes
    prev_idx = 0
    for i in range(B):
        next_idx = prev_idx + Ns[i]
        boxes_padded[i, :Ns[i]] = boxes_packed[prev_idx:next_idx]
        prev_idx = next_idx

    return boxes_padded


@dataclass
class SAM3_3DLossConfig:
    """Configuration for SAM3_3D loss.

    Follows SAM3's loss configuration style with additional 3D loss weights.
    """
    # ========== Global Scale Factors ==========
    # These allow adjusting the balance between 2D, 3D, and geometry losses
    # Default 1.0, can be adjusted in training config to tune 2D:3D:Geom ratio
    loss_2d_scale: float = 1.0  # Scale for 2D losses (cls, bbox, giou)
    loss_3d_scale: float = 1.0  # Scale for 3D losses (delta, depth, dim, rot)
    loss_geom_scale: float = 10.0  # Scale for geometry backend losses (SILog, SSI, camera angles)

    # ========== O2M (One-to-Many) Matcher Configuration ==========
    # Note: O2O matcher is configured in sam3_3d.py (self.sam3.matcher)
    use_o2m: bool = True  # Enable O2M matching
    o2m_alpha: float = 0.3  # Alpha for O2M cost computation
    o2m_threshold: float = 0.4  # IoU threshold for O2M matching
    o2m_topk: int = 4  # Top-k predictions per GT (SAM3 original: topk: 4)
    o2m_loss_weight: float = 2.0  # Weight for O2M loss (SAM3 original: o2m_weight: 2.0)

    # ========== 2D Loss Weights (SAM3 style) ==========
    # Classification loss (IABCEMdetr style)
    loss_cls_weight: float = 20.0  # SAM3 original
    pos_weight: float = 5.0  # SAM3 original (was incorrectly 10.0)
    gamma: float = 2.0  # SAM3 original focal (was incorrectly 0.0)
    alpha: float = 0.25  # IoU-aware alpha

    # IABCEMdetr advanced features
    use_weak_loss: bool = False  # Enable weak supervision (SAM3 original: weak_loss: False)
    weak_loss_weight: float = 1.0  # Weight for weak loss (only used if use_weak_loss=True)
    use_presence: bool = True  # Enable presence loss (per-category presence detection)
    presence_loss_weight: float = 20.0  # Weight for presence loss (SAM3 original: presence_weight: 20.0)
    presence_alpha: float = 0.5  # SAM3 original presence focal loss alpha
    presence_gamma: float = 0.0  # SAM3 original presence focal loss gamma

    # Box regression loss
    loss_bbox_weight: float = 5.0  # L1 loss weight
    loss_giou_weight: float = 2.0  # GIoU loss weight
    
    # ========== 3D Loss Weights (3D-MOOD style) ==========
    loss_delta_2d_weight: float = 1.0  # Delta 2D center
    loss_depth_weight: float = 1.0  # Log depth
    loss_dim_weight: float = 1.0  # Log dimensions
    loss_rot_weight: float = 1.0  # 6D rotation
    
    # ========== Geometry Backend Loss Weights ==========
    loss_silog_weight: float = 1.0  # SILog depth loss
    loss_phi_weight: float = 0.1  # Phi angle loss
    loss_theta_weight: float = 0.1  # Theta angle loss
    loss_opt_ssi_weight: float = 0.5  # SSI loss weight (UniDepthV2)
    
    # ========== Normalization ==========
    normalization: Literal["global", "local", "none"] = "global"
    
    # ========== Auxiliary Loss ==========
    aux_loss_weight: float = 1.0  # Weight for auxiliary decoder outputs
    
    # ========== Mask Loss (optional) ==========
    loss_mask_weight: float = 0.0  # Set > 0 to enable mask loss
    loss_dice_weight: float = 0.0  # Set > 0 to enable dice loss


class SAM3_3DLoss(nn.Module):
    """Loss function for SAM3_3D.
    
    Combines SAM3-style 2D losses with 3D-MOOD-style 3D losses.
    
    Loss Components:
    1. Classification: IABCEMdetr (IoU-aware BCE with soft targets)
    2. 2D Box: L1 + GIoU
    3. 3D Box: L1 for (delta_center, log_depth, log_dims, rot_6d)
    4. Geometry: SILog depth + phi/theta angles (from geometry backend)
    """
    
    def __init__(
        self,
        config: SAM3_3DLossConfig | None = None,
        box_coder: GroundingDINO3DCoder | None = None,
    ) -> None:
        """Initialize SAM3_3D loss.

        Args:
            config: Loss configuration
            box_coder: 3D box encoder/decoder for target encoding
        """
        super().__init__()
        self.config = config or SAM3_3DLossConfig()
        self.box_coder = box_coder or GroundingDINO3DCoder()
        self.reg_dims = self.box_coder.reg_dims

        # Note: O2O matcher is configured in sam3_3d.py (self.sam3.matcher)
        # We only need O2M matcher here for O2M loss computation

        # Initialize O2M matcher if enabled
        if self.config.use_o2m:
            self.o2m_matcher = BinaryOneToManyMatcher(
                alpha=self.config.o2m_alpha,
                threshold=self.config.o2m_threshold,
                topk=self.config.o2m_topk,
            )
        else:
            self.o2m_matcher = None

    def _build_targets_from_batch(
        self, batch: "SAM3_3DBatchedInputs"
    ) -> dict[str, Tensor]:
        """Build targets dict from SAM3_3DBatchedInputs.

        SAM3_3D uses per-category queries with multi-instance targets.
        The collator produces:
        - gt_boxes2d: (N_prompts, max_gt, 4) - multiple GTs per query
        - gt_boxes3d: (N_prompts, max_gt, 12) - multiple GTs per query (if available)
        - num_gts: (N_prompts,) - number of valid GTs per query (can be > 1)

        We convert this to the packed format expected by loss computation.

        Args:
            batch: SAM3_3DBatchedInputs containing GT boxes

        Returns:
            targets dict with:
            - boxes_xyxy: (N_total, 4) GT boxes in xyxy format (packed)
            - boxes_3d: (N_total, 12) 3D GT boxes (packed)
            - num_boxes: (N_prompts,) number of GTs per query
            - intrinsics: (N_prompts, 3, 3) camera intrinsics per prompt
        """
        device = batch.images.device
        N_prompts = batch.img_ids.shape[0]

        # Extract GT from batch
        gt_boxes2d = batch.gt_boxes2d  # (N_prompts, max_gt, 4) or (N_prompts, 4)
        gt_boxes3d = batch.gt_boxes3d  # (N_prompts, max_gt, 12) or None
        num_gts = batch.num_gts  # (N_prompts,) number of valid GTs per query

        if gt_boxes2d is None:
            # No GT available
            return {
                "boxes_xyxy": torch.zeros(0, 4, device=device),
                "boxes_3d": torch.zeros(0, 12, device=device),
                "classes": torch.zeros(0, dtype=torch.long, device=device),
                "num_boxes": torch.zeros(N_prompts, dtype=torch.long, device=device),
                "intrinsics": batch.intrinsics[batch.img_ids],
            }

        # Handle both old (N_prompts, 4) and new (N_prompts, max_gt, 4) formats
        if gt_boxes2d.dim() == 2:
            # Old format: (N_prompts, 4) - single GT per prompt
            boxes_xyxy = gt_boxes2d
            if num_gts is None:
                num_gts = torch.ones(N_prompts, dtype=torch.long, device=device)

            if gt_boxes3d is not None and gt_boxes3d.dim() == 2:
                boxes_3d = gt_boxes3d
            else:
                boxes_3d = torch.zeros(N_prompts, 12, device=device)
        else:
            # New format: (N_prompts, max_gt, 4) - multi-instance targets
            # Pack valid boxes into a flat tensor
            if num_gts is None:
                # Fallback: assume all boxes are valid
                num_gts = torch.tensor([gt_boxes2d.shape[1]] * N_prompts, dtype=torch.long, device=device)

            # Pack boxes into (N_total, 4)
            boxes_list = []
            boxes_3d_list = []
            for i in range(N_prompts):
                n_gt = num_gts[i].item()
                boxes_list.append(gt_boxes2d[i, :n_gt])  # (n_gt, 4)
                if gt_boxes3d is not None:
                    boxes_3d_list.append(gt_boxes3d[i, :n_gt])  # (n_gt, 12)

            if boxes_list:
                boxes_xyxy = torch.cat(boxes_list, dim=0)  # (N_total, 4)
            else:
                boxes_xyxy = torch.zeros(0, 4, device=device)

            if boxes_3d_list:
                boxes_3d = torch.cat(boxes_3d_list, dim=0)  # (N_total, 12)
            else:
                box3d_dim = gt_boxes3d.shape[-1] if gt_boxes3d is not None else 12
                boxes_3d = torch.zeros(boxes_xyxy.shape[0], box3d_dim, device=device)

        # SAM3 uses binary detection (all targets are class 1)
        N_total = boxes_xyxy.shape[0]
        classes = torch.ones(N_total, dtype=torch.long, device=device)

        # Get per-prompt intrinsics
        intrinsics = batch.intrinsics[batch.img_ids]  # (N_prompts, 3, 3)

        return {
            "boxes_xyxy": boxes_xyxy,
            "boxes_3d": boxes_3d,
            "classes": classes,
            "num_boxes": num_gts,  # (N_prompts,) - can be > 1 for multi-instance
            "intrinsics": intrinsics,
        }

    def forward(
        self,
        out: "SAM3_3DOutput",
        batch: "SAM3_3DBatchedInputs",
    ) -> dict[str, Tensor]:
        """Compute all losses.

        vis4d LossModule interface: expects either Tensor, dict, or namedtuple.
        We return a dict of tensors, and LossModule will sum them automatically.

        Following SAM3 and GDino3D's design, we compute 2D box L1 loss in normalized
        cxcywh space and GIoU loss in pixel xyxy space for consistent loss weights.

        Args:
            out: Model output (SAM3_3DOutput dataclass)
            batch: Input batch (SAM3_3DBatchedInputs dataclass)

        Returns:
            Dict of loss tensors (vis4d LossModule will sum them)
        """
        import time
        import os
        import torch
        _PROFILE_LOSS = os.environ.get("PROFILE_SAM3_3D", "0") == "1"
        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _loss_start = time.perf_counter()
        # Unpack model outputs
        pred_logits = out.pred_logits
        pred_boxes_2d = out.pred_boxes_2d
        pred_boxes_3d = out.pred_boxes_3d
        aux_outputs = out.aux_outputs
        geom_losses = out.geom_losses

        # Build targets from batch
        # Get per-prompt intrinsics by indexing into batch intrinsics
        B_images = batch.images.shape[0]
        N_prompts = batch.img_ids.shape[0]
        intrinsics = batch.intrinsics[batch.img_ids]  # (N_prompts, 3, 3)

        # Image size from batch
        image_size = (batch.images.shape[2], batch.images.shape[3])  # (H, W)

        targets = self._build_targets_from_batch(batch)
        losses = {}

        # Normalize targets to [0, 1] range (for matching and computation)
        normalized_targets = self._normalize_targets(targets)

        # Store image_size for pixel coordinate conversion
        if image_size is None and "image_size" in targets:
            image_size = targets["image_size"]

        # Get matching indices from SAM3's internal matching
        # SAM3's forward_grounding computes indices via _compute_matching when find_target is provided
        # Handle empty batch (N_prompts=0) case - return zero loss with grad
        if out.indices is None:
            device = pred_logits.device
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(f"[SAM3_3D Loss] Empty batch detected on rank {rank}, returning zero loss")

            # CRITICAL: Must still participate in all_reduce to prevent DDP deadlock
            # Other ranks may have non-empty batches and will call all_reduce
            if self.config.normalization == "global" and torch.distributed.is_initialized():
                dummy_num_boxes = torch.tensor(0.0, device=device)
                torch.distributed.all_reduce(dummy_num_boxes)

            # Use pred_logits.sum() * 0 to maintain computation graph for DDP
            zero_loss = pred_logits.sum() * 0
            return {
                "loss_cls": zero_loss,  # Keep grad for DDP
                "loss_bbox": zero_loss.clone(),
                "loss_giou": zero_loss.clone(),
            }

        batch_idx, src_idx, tgt_idx = out.indices

        # Move indices to the same device as predictions
        batch_idx = batch_idx.to(pred_logits.device)
        src_idx = src_idx.to(pred_logits.device)
        tgt_idx = tgt_idx.to(pred_logits.device) if tgt_idx is not None else None

        indices = (batch_idx, src_idx, tgt_idx)

        # Get number of boxes for normalization
        num_boxes = self._get_num_boxes(normalized_targets)
        
        # ========== 2D Losses (scaled by loss_2d_scale) ==========
        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()

        loss_cls = self._loss_classification(
            pred_logits, pred_boxes_2d, indices, normalized_targets, num_boxes
        )
        losses["loss_cls"] = (
            self.config.loss_2d_scale * loss_cls * self.config.loss_cls_weight
        )

        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _loss_cls_time = (time.perf_counter() - _t0) * 1000

        # Presence loss (per-category presence detection, following SAM3 original)
        # Uses presence_logit_dec to predict whether each category exists in the image
        if self.config.use_presence and out.presence_logits is not None:
            loss_presence = self._loss_presence(out.presence_logits, normalized_targets)
            losses["loss_presence"] = (
                self.config.loss_2d_scale * loss_presence * self.config.presence_loss_weight
            )

        # 2D box losses (L1 in normalized cxcywh, GIoU in pixel xyxy - following SAM3/GDino3D)
        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _t1 = time.perf_counter()

        loss_bbox, loss_giou = self._loss_boxes_2d(
            pred_boxes_2d, indices, normalized_targets, num_boxes, image_size
        )
        losses["loss_bbox"] = (
            self.config.loss_2d_scale * loss_bbox * self.config.loss_bbox_weight
        )
        losses["loss_giou"] = (
            self.config.loss_2d_scale * loss_giou * self.config.loss_giou_weight
        )

        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _loss_2d_box_time = (time.perf_counter() - _t1) * 1000

        # ========== O2M Loss (2D scaled by loss_2d_scale, 3D scaled by loss_3d_scale) ==========
        # Use real O2M outputs from SAM3 DAC mechanism (not O2O outputs)
        if self.config.use_o2m and self.o2m_matcher is not None and out.pred_logits_o2m is not None:
            o2m_losses = self._loss_o2m(
                pred_logits=out.pred_logits_o2m,
                pred_boxes_2d=out.pred_boxes_2d_o2m,
                pred_boxes_3d=out.pred_boxes_3d_o2m,
                targets=normalized_targets,
                num_boxes=num_boxes,
                intrinsics=intrinsics,
                image_size=image_size,
            )
            # Apply appropriate scale and loss weights (following SAM3 original)
            # SAM3 original: loss = loss_value * o2m_weight * loss_weight
            # We need to apply the individual loss weights, not just o2m_loss_weight
            o2m_weight_map = {
                "loss_cls": self.config.loss_cls_weight,
                "loss_bbox": self.config.loss_bbox_weight,
                "loss_giou": self.config.loss_giou_weight,
                "loss_delta_2d": self.config.loss_delta_2d_weight,
                "loss_depth": self.config.loss_depth_weight,
                "loss_dim": self.config.loss_dim_weight,
                "loss_rot": self.config.loss_rot_weight,
            }
            for key, value in o2m_losses.items():
                loss_weight = o2m_weight_map.get(key, 1.0)
                if key in ("loss_delta_2d", "loss_depth", "loss_dim", "loss_rot"):
                    # 3D losses use loss_3d_scale
                    losses[f"o2m_{key}"] = (
                        self.config.loss_3d_scale * value * loss_weight * self.config.o2m_loss_weight
                    )
                else:
                    # 2D losses (loss_cls, loss_bbox, loss_giou) use loss_2d_scale
                    losses[f"o2m_{key}"] = (
                        self.config.loss_2d_scale * value * loss_weight * self.config.o2m_loss_weight
                    )

        # ========== 3D Losses (scaled by loss_3d_scale) ==========
        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _t2 = time.perf_counter()
            _loss_3d_time = 0

        if pred_boxes_3d is not None and intrinsics is not None:
            loss_3d = self._loss_boxes_3d(
                pred_boxes_2d, pred_boxes_3d, indices, normalized_targets,
                intrinsics, num_boxes, image_size=image_size
            )
            # Apply loss_3d_scale to all 3D losses
            for key, value in loss_3d.items():
                losses[key] = self.config.loss_3d_scale * value

        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _loss_3d_time = (time.perf_counter() - _t2) * 1000

        # ========== Geometry Backend Losses (scaled by loss_geom_scale) ==========
        if geom_losses is not None:
            for key, value in geom_losses.items():
                weight = getattr(self.config, f"loss_{key}_weight", 1.0)
                losses[f"loss_{key}"] = (
                    self.config.loss_geom_scale * value * weight
                )

        # ========== Auxiliary Losses (Deep Supervision) ==========
        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _t3 = time.perf_counter()
            _loss_aux_time = 0

        _num_aux_layers = 0
        if aux_outputs is not None:
            _num_aux_layers = len(aux_outputs)
            for i, aux_out in enumerate(aux_outputs):
                aux_losses = self._compute_aux_loss(
                    aux_out, indices, normalized_targets, num_boxes, intrinsics, image_size
                )
                for key, value in aux_losses.items():
                    losses[f"d{i}.{key}"] = value * self.config.aux_loss_weight

        if _PROFILE_LOSS:
            torch.cuda.synchronize()
            _loss_aux_time = (time.perf_counter() - _t3) * 1000
            _loss_total_time = (time.perf_counter() - _loss_start) * 1000

            # Print loss timing summary (every N steps via profiler)
            from opendet3d.utils.profiler import profiler
            p = profiler()
            p.current_step_timings["loss_total"] = _loss_total_time / 1000
            p.current_step_timings["  loss_cls"] = _loss_cls_time / 1000
            p.current_step_timings["  loss_2d_box"] = _loss_2d_box_time / 1000
            p.current_step_timings["  loss_3d"] = _loss_3d_time / 1000
            p.current_step_timings["  loss_aux"] = _loss_aux_time / 1000
            p.current_step_timings["  loss_aux_layers"] = _num_aux_layers

        # ========== Ensure all losses are tensors ==========
        # vis4d LossModule expects dict of tensors
        for k, v in list(losses.items()):
            if not isinstance(v, Tensor):
                losses[k] = torch.tensor(v, device=pred_logits.device)

        # vis4d LossModule will sum all losses in the dict automatically
        return losses

    def _get_num_boxes(self, targets: dict) -> Tensor:
        """Get number of boxes for loss normalization."""
        num_boxes = targets["num_boxes"].sum().float()

        if self.config.normalization == "global":
            # Handle non-distributed case
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(num_boxes)
                world_size = torch.distributed.get_world_size()
                num_boxes = torch.clamp(num_boxes / world_size, min=1)
            else:
                # Non-distributed: just clamp
                num_boxes = torch.clamp(num_boxes, min=1)
        elif self.config.normalization == "local":
            num_boxes = torch.clamp(num_boxes, min=1)
        else:  # "none"
            num_boxes = torch.ones_like(num_boxes)

        return num_boxes

    def _loss_classification(
        self,
        pred_logits: Tensor,  # (B, S, 1)
        pred_boxes_2d: Tensor,  # (B, S, 4)
        indices: tuple[Tensor, Tensor, Tensor | None],
        targets: dict,
        num_boxes: Tensor,
    ) -> Tensor:
        """Compute IABCEMdetr-style classification loss.

        This follows SAM3's IoU-aware BCE loss with soft targets.
        Includes weak supervision for non-exhaustive annotations.
        """
        batch_idx, src_idx, tgt_idx = indices
        device = pred_logits.device
        B = pred_logits.shape[0]

        src_logits = pred_logits.squeeze(-1)  # (B, S)
        prob = src_logits.sigmoid()

        # Create target classes (0 for background, 1 for foreground)
        target_classes = torch.zeros_like(src_logits)
        target_classes[(batch_idx, src_idx)] = 1.0

        # Get matched boxes for IoU computation
        src_boxes_xyxy = pred_boxes_2d[(batch_idx, src_idx)]
        target_boxes_xyxy = (
            targets["boxes_xyxy"][tgt_idx] if tgt_idx is not None
            else targets["boxes_xyxy"]
        )

        # Compute IoU for soft targets
        with torch.no_grad():
            iou = fast_diag_box_iou(src_boxes_xyxy, target_boxes_xyxy)
            # Soft target: prob^alpha * iou^(1-alpha)
            t = prob[(batch_idx, src_idx)] ** self.config.alpha * iou ** (1 - self.config.alpha)
            t = torch.clamp(t, 0.01).detach()

            positive_target_classes = target_classes.clone()
            positive_target_classes[(batch_idx, src_idx)] = t.to(positive_target_classes.dtype)

        # Compute weak loss mask if enabled
        weak_mask = torch.ones_like(src_logits)
        if self.config.use_weak_loss:
            weak_mask = self._compute_weak_loss_mask(
                pred_boxes_2d, targets, batch_idx, src_idx
            )

        # BCE loss on positives with soft targets
        loss_pos = F.binary_cross_entropy_with_logits(
            src_logits, positive_target_classes, reduction="none"
        )
        loss_pos = loss_pos * target_classes * self.config.pos_weight

        # BCE loss on negatives with focal weighting and weak mask
        loss_neg = F.binary_cross_entropy_with_logits(
            src_logits, target_classes, reduction="none"
        )
        loss_neg = loss_neg * (1 - target_classes) * (prob ** self.config.gamma) * weak_mask

        loss_bce = loss_pos + loss_neg

        # Apply keep_loss mask if use_presence is enabled (following SAM3 original)
        # For prompts with no GT boxes, set classification loss to 0
        # Only presence loss supervises these prompts (to predict presence = 0)
        if self.config.use_presence:
            num_gts = targets.get("num_boxes", torch.zeros(B, dtype=torch.long, device=device))
            keep_loss = (num_gts > 0).float().unsqueeze(-1)  # (B, 1)
            loss_bce = loss_bce * keep_loss

        # SAM3 original: use mean() over all B×S logits (not sum()/num_boxes)
        # This is different from bbox/giou which use sum()/num_boxes
        # That's why loss_cls_weight=20 is needed to compensate
        loss_bce = loss_bce.mean()

        return loss_bce

    def _compute_weak_loss_mask(
        self,
        pred_boxes_2d: Tensor,  # (B, S, 4)
        targets: dict,
        batch_idx: Tensor,
        src_idx: Tensor,
    ) -> Tensor:
        """Compute weak loss mask for non-exhaustive annotations.

        For unmatched predictions, compute max IoU with all GT boxes.
        If max IoU > threshold, mask out the negative loss (might be unlabeled positive).

        Args:
            pred_boxes_2d: Predicted boxes (B, S, 4) normalized xyxy
            targets: Ground truth targets
            batch_idx: Matched prediction batch indices
            src_idx: Matched prediction query indices

        Returns:
            Weak mask (B, S): 1.0 for normal negatives, 0.0 for weak negatives
        """
        device = pred_boxes_2d.device
        B, S = pred_boxes_2d.shape[:2]

        # Create mask: 1 for unmatched, 0 for matched
        is_unmatched = torch.ones(B, S, dtype=torch.bool, device=device)
        is_unmatched[(batch_idx, src_idx)] = False

        weak_mask = torch.ones(B, S, dtype=torch.float32, device=device)

        if not is_unmatched.any():
            return weak_mask

        # For each unmatched prediction, compute max IoU with GT
        gt_boxes = targets["boxes_xyxy"]  # (N_total_gt, 4) normalized xyxy

        if gt_boxes.numel() == 0:
            return weak_mask

        # Reshape for batch-wise processing
        unmatched_boxes = pred_boxes_2d[is_unmatched]  # (N_unmatched, 4)

        # Compute IoU between all unmatched boxes and all GT boxes
        # This is expensive but necessary for weak supervision
        from sam3.model.box_ops import box_iou

        iou_matrix, _ = box_iou(unmatched_boxes, gt_boxes)  # (N_unmatched, N_total_gt)
        max_iou = iou_matrix.max(dim=1)[0]  # (N_unmatched,)

        # Threshold: if max IoU > 0.5, consider it a potential unlabeled positive
        weak_threshold = 0.5
        is_weak_negative = max_iou > weak_threshold

        # Update mask: set to 0 for weak negatives
        weak_mask_flat = torch.ones(B * S, dtype=torch.float32, device=device)
        unmatched_indices = is_unmatched.view(-1).nonzero(as_tuple=True)[0]
        weak_mask_flat[unmatched_indices[is_weak_negative]] = 0.0
        weak_mask = weak_mask_flat.view(B, S)

        return weak_mask

    def _loss_presence(
        self,
        presence_logits: Tensor,  # (N_prompts, 1) from presence_logit_dec
        targets: dict,
    ) -> Tensor:
        """Compute presence loss for per-category presence detection.

        Following SAM3 original: presence_logit_dec predicts whether each
        category/prompt has any visible objects in the image.

        This is different from empty image detection - it's per-category:
        - If category "car" has GT boxes -> target = 1
        - If category "bicycle" has no GT boxes -> target = 0

        Args:
            presence_logits: Predicted presence logits (N_prompts, 1) from presence_logit_dec
            targets: Ground truth targets with num_boxes field

        Returns:
            Presence loss scalar (sigmoid focal loss)
        """
        device = presence_logits.device
        N_prompts = presence_logits.shape[0]

        # Get number of GT boxes per prompt
        num_gts = targets.get("num_boxes", torch.zeros(N_prompts, dtype=torch.long, device=device))

        # Target: 1 if prompt has GT boxes, 0 otherwise (following SAM3 original)
        # SAM3 uses: keep_loss = (gt_padded_is_visible.sum(dim=-1)[..., None] != 0).float()
        presence_target = (num_gts > 0).float().unsqueeze(-1)  # (N_prompts, 1)

        # Sigmoid focal loss (following SAM3 original)
        # SAM3 uses: sigmoid_focal_loss(presence_logits, keep_loss, num_boxes=bs, alpha=0.5, gamma=0.0)
        loss_presence = sigmoid_focal_loss(
            presence_logits,
            presence_target,
            num_boxes=N_prompts,
            alpha=self.config.presence_alpha,
            gamma=self.config.presence_gamma,
        )

        return loss_presence

    def _loss_o2m(
        self,
        pred_logits: Tensor,  # (B, S, 1)
        pred_boxes_2d: Tensor,  # (B, S, 4)
        pred_boxes_3d: Tensor | None,  # (B, S, reg_dims)
        targets: dict,
        num_boxes: Tensor,
        intrinsics: Tensor | None = None,  # (B, 3, 3)
        image_size: tuple[int, int] | None = None,
    ) -> dict[str, Tensor]:
        """Compute O2M (One-to-Many) auxiliary loss.

        O2M matching allows multiple predictions to match the same GT,
        which helps improve recall and stabilize training.

        Following SAM3 original design, O2M computes ALL loss components
        (classification, 2D boxes, AND 3D boxes) for matched predictions.
        Normalization uses num_boxes (GT count), same as O2O.

        Args:
            pred_logits: Predicted objectness logits
            pred_boxes_2d: Predicted 2D boxes (normalized xyxy)
            pred_boxes_3d: Predicted 3D box parameters (optional)
            targets: Ground truth targets
            num_boxes: Number of boxes for normalization (GT count)
            intrinsics: Camera intrinsics for 3D loss (optional)
            image_size: (H, W) for converting to pixel coordinates

        Returns:
            Dictionary with O2M losses (loss_cls, loss_bbox, loss_giou,
            and optionally loss_delta_2d, loss_depth, loss_dim, loss_rot)
        """
        losses = {}
        device = pred_logits.device
        B, S = pred_logits.shape[:2]

        # Prepare targets in padded format for O2M matcher
        boxes_cxcywh = self._xyxy_to_cxcywh(targets["boxes_xyxy"])
        num_boxes_per_image = targets.get("num_boxes", torch.tensor([len(targets["boxes_xyxy"])], device=device))

        # Convert packed boxes to padded format (B, max_N, 4)
        boxes_padded = _packed_to_padded(boxes_cxcywh, num_boxes_per_image)

        # Create validity mask (B, max_N) - True for valid boxes, False for padding
        max_N = boxes_padded.shape[1]
        target_is_valid_padded = torch.zeros(B, max_N, dtype=torch.bool, device=device)
        for i in range(B):
            target_is_valid_padded[i, :num_boxes_per_image[i]] = True

        # Compute O2M matching
        outputs_dict = {
            "pred_logits": pred_logits,
            "pred_boxes": self._xyxy_to_cxcywh(pred_boxes_2d),
        }

        targets_dict = {
            "boxes_padded": boxes_padded,  # Use padded format for O2M
            "labels": targets["classes"],
            "num_boxes": num_boxes_per_image,
        }

        # Call O2M matcher with validity mask
        batch_idx, src_idx, tgt_idx = self.o2m_matcher(
            outputs_dict,
            targets_dict,
            target_is_valid_padded=target_is_valid_padded,
        )

        # Handle empty matching case
        if batch_idx.numel() == 0:
            # No matches, return zero losses (2D + 3D if applicable)
            zero_losses = {
                "loss_cls": torch.tensor(0.0, device=device),
                "loss_bbox": torch.tensor(0.0, device=device),
                "loss_giou": torch.tensor(0.0, device=device),
            }
            # Add 3D zero losses if 3D data is available
            if pred_boxes_3d is not None and intrinsics is not None and "boxes_3d" in targets:
                zero_losses.update({
                    "loss_delta_2d": torch.tensor(0.0, device=device),
                    "loss_depth": torch.tensor(0.0, device=device),
                    "loss_dim": torch.tensor(0.0, device=device),
                    "loss_rot": torch.tensor(0.0, device=device),
                })
            return zero_losses

        # O2M Classification Loss (following SAM3 original IABCEMdetr)
        src_logits = pred_logits.squeeze(-1)  # (B, S)
        prob = src_logits.sigmoid()

        # Create target classes
        target_classes = torch.zeros_like(src_logits)
        target_classes[(batch_idx, src_idx)] = 1.0

        # Get matched boxes for IoU computation (use tgt_idx to index into packed targets)
        src_boxes_xyxy = pred_boxes_2d[(batch_idx, src_idx)]
        target_boxes_xyxy = targets["boxes_xyxy"][tgt_idx]

        # Compute IoU for soft targets
        with torch.no_grad():
            iou = fast_diag_box_iou(src_boxes_xyxy, target_boxes_xyxy)
            t = prob[(batch_idx, src_idx)] ** self.config.alpha * iou ** (1 - self.config.alpha)
            t = torch.clamp(t, 0.01).detach()

            positive_target_classes = target_classes.clone()
            positive_target_classes[(batch_idx, src_idx)] = t.to(positive_target_classes.dtype)

        # Positive loss: BCE with soft target (same as main IABCEMdetr)
        loss_pos = F.binary_cross_entropy_with_logits(
            src_logits, positive_target_classes, reduction="none"
        )
        loss_pos = loss_pos * target_classes * self.config.pos_weight

        # Negative loss: focal BCE (same as main IABCEMdetr)
        loss_neg = F.binary_cross_entropy_with_logits(
            src_logits, target_classes, reduction="none"
        )
        loss_neg = loss_neg * (1 - target_classes) * (prob ** self.config.gamma)

        # Total cls loss: positive + negative (following SAM3 original)
        loss_bce = loss_pos + loss_neg

        # Apply keep_loss mask if use_presence is enabled (following SAM3 original)
        # For prompts with no GT boxes, set classification loss to 0
        # This is applied to O2M as well (SAM3 original: loss_bce = loss_bce * keep_loss)
        if self.config.use_presence:
            num_gts = targets.get("num_boxes", torch.zeros(B, dtype=torch.long, device=device))
            keep_loss = (num_gts > 0).float().unsqueeze(-1)  # (B, 1)
            loss_bce = loss_bce * keep_loss

        # SAM3 original: use mean() for cls (same as main loss)
        loss_cls = loss_bce.mean()

        losses["loss_cls"] = loss_cls

        # O2M Box Losses
        # Following SAM3 and GDino3D's design (consistent with _loss_boxes_2d):
        # - L1 loss: computed in normalized cxcywh space [0, 1]
        # - GIoU loss: computed in pixel xyxy space

        # L1 loss in normalized cxcywh space
        src_boxes_cxcywh = self._xyxy_to_cxcywh(src_boxes_xyxy)
        target_boxes_cxcywh = self._xyxy_to_cxcywh(target_boxes_xyxy)
        loss_bbox = F.l1_loss(src_boxes_cxcywh, target_boxes_cxcywh, reduction="none")
        loss_bbox = loss_bbox.sum() / num_boxes
        losses["loss_bbox"] = loss_bbox

        # GIoU loss in pixel xyxy space
        if image_size is not None:
            H, W = image_size
            factors = src_boxes_xyxy.new_tensor([W, H, W, H])
            src_boxes_pixel = src_boxes_xyxy * factors
            target_boxes_pixel = target_boxes_xyxy * factors
        else:
            # Fallback to normalized coordinates
            src_boxes_pixel = src_boxes_xyxy
            target_boxes_pixel = target_boxes_xyxy

        loss_giou = 1 - fast_diag_generalized_box_iou(src_boxes_pixel, target_boxes_pixel)
        loss_giou = loss_giou.sum() / num_boxes
        losses["loss_giou"] = loss_giou

        # O2M 3D Loss - same logic as O2O, just more matched pairs
        # Following SAM3 original design: O2M computes ALL loss components
        if pred_boxes_3d is not None and intrinsics is not None and "boxes_3d" in targets:
            o2m_indices = (batch_idx, src_idx, tgt_idx)
            loss_3d = self._loss_boxes_3d(
                pred_boxes_2d=pred_boxes_2d,
                pred_boxes_3d=pred_boxes_3d,
                indices=o2m_indices,
                targets=targets,
                intrinsics=intrinsics,
                num_boxes=num_boxes,  # Same normalization as 2D (GT count)
                image_size=image_size,
            )
            losses.update(loss_3d)

        return losses

    def _loss_boxes_2d(
        self,
        pred_boxes_2d: Tensor,  # (B, S, 4) normalized xyxy
        indices: tuple[Tensor, Tensor, Tensor | None],
        targets: dict,
        num_boxes: Tensor,
        image_size: tuple[int, int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute 2D box regression losses (L1 + GIoU).

        Following SAM3 and GDino3D's design:
        - L1 loss: computed in normalized cxcywh space [0, 1]
        - GIoU loss: computed in pixel xyxy space

        This allows direct use of GDino3D loss weights (bbox_loss_weight=5.0,
        iou_loss_weight=2.0).

        Args:
            pred_boxes_2d: Predicted boxes (normalized xyxy [0, 1])
            indices: Matching indices
            targets: Ground truth targets (normalized xyxy [0, 1])
            num_boxes: Number of boxes for normalization
            image_size: (H, W) for converting to pixel coordinates (for GIoU)

        Returns:
            (loss_bbox, loss_giou) tuple
        """
        batch_idx, src_idx, tgt_idx = indices

        src_boxes = pred_boxes_2d[(batch_idx, src_idx)]
        target_boxes = (
            targets["boxes_xyxy"][tgt_idx] if tgt_idx is not None
            else targets["boxes_xyxy"]
        )

        # L1 loss in normalized cxcywh space (following SAM3 and GDino3D)
        # Convert xyxy -> cxcywh for L1 loss computation
        src_boxes_cxcywh = self._xyxy_to_cxcywh(src_boxes)
        target_boxes_cxcywh = self._xyxy_to_cxcywh(target_boxes)
        loss_bbox = F.l1_loss(src_boxes_cxcywh, target_boxes_cxcywh, reduction="none")
        loss_bbox = loss_bbox.sum() / num_boxes

        # GIoU loss in pixel xyxy space (following GDino3D)
        if image_size is not None:
            H, W = image_size
            factors = src_boxes.new_tensor([W, H, W, H])
            src_boxes_pixel = src_boxes * factors
            target_boxes_pixel = target_boxes * factors
        else:
            # Fallback to normalized coordinates
            src_boxes_pixel = src_boxes
            target_boxes_pixel = target_boxes

        loss_giou = 1 - fast_diag_generalized_box_iou(src_boxes_pixel, target_boxes_pixel)
        loss_giou = loss_giou.sum() / num_boxes

        return loss_bbox, loss_giou

    def _loss_boxes_3d(
        self,
        pred_boxes_2d: Tensor,  # (B, S, 4)
        pred_boxes_3d: Tensor,  # (B, S, reg_dims)
        indices: tuple[Tensor, Tensor, Tensor | None],
        targets: dict,
        intrinsics: Tensor,
        num_boxes: Tensor,
        image_size: tuple[int, int] | None = None,
    ) -> dict[str, Tensor]:
        """Compute 3D box regression losses.

        Args:
            pred_boxes_2d: Predicted 2D boxes in normalized xyxy [0,1]. Shape (B, S, 4).
            pred_boxes_3d: Predicted 3D box parameters. Shape (B, S, reg_dims).
            indices: Matching indices (batch_idx, src_idx, tgt_idx).
            targets: Target dict containing boxes_3d.
            intrinsics: Camera intrinsics. Shape (B, 3, 3).
            num_boxes: Number of matched boxes for normalization.
            image_size: (H, W) tuple for converting normalized to pixel coords.
                Required for correct box_coder.encode() which expects pixel coords.
        """
        batch_idx, src_idx, tgt_idx = indices

        # Get matched predictions (for loss computation)
        src_boxes_3d = pred_boxes_3d[(batch_idx, src_idx)]

        # Get matched GT 2D boxes (for box_coder.encode target computation)
        # IMPORTANT: Use GT 2D boxes, NOT predicted boxes!
        # This matches GDino3D's design where encode() uses GT 2D boxes to compute
        # stable targets, while decode() at inference uses predicted 2D boxes.
        target_boxes_2d = (
            targets["boxes_xyxy"][tgt_idx] if tgt_idx is not None
            else targets["boxes_xyxy"]
        )

        # Get matched GT 3D boxes
        target_boxes_3d = (
            targets["boxes_3d"][tgt_idx] if tgt_idx is not None
            else targets["boxes_3d"]
        )

        # Get intrinsics for matched samples
        # Note: intrinsics is (B, 3, 3), need to index by batch_idx
        # Since box_coder.encode() expects single intrinsics (3, 3),
        # we need to process each matched box individually
        if len(batch_idx) == 0:
            # No matches, return zero losses
            return {
                "loss_delta_2d": torch.tensor(0.0, device=pred_boxes_2d.device),
                "loss_depth": torch.tensor(0.0, device=pred_boxes_2d.device),
                "loss_dim": torch.tensor(0.0, device=pred_boxes_2d.device),
                "loss_rot": torch.tensor(0.0, device=pred_boxes_2d.device),
            }

        target_boxes_3d_encoded_list = []
        weights_3d_list = []

        # Validate image_size is provided - required for correct box_coder.encode()
        if image_size is None:
            raise ValueError(
                "image_size is required for _loss_boxes_3d. "
                "box_coder.encode() expects pixel coordinates because "
                "project_points() returns pixel coords and "
                "delta_center = projected_3d_center - 2d_box_center (both in pixels)."
            )

        H, W = image_size
        factors = target_boxes_2d.new_tensor([W, H, W, H])

        for i in range(len(batch_idx)):
            # Use GT 2D box (normalized xyxy) and convert to pixel
            single_gt_box_2d = target_boxes_2d[i:i+1]  # GT 2D, normalized xyxy [0,1]
            single_gt_box_2d_pixel = single_gt_box_2d * factors  # Convert to pixel

            single_box_3d = target_boxes_3d[i:i+1]
            single_intrinsic = intrinsics[batch_idx[i]]  # (3, 3)

            encoded, weights = self.box_coder.encode(
                single_gt_box_2d_pixel, single_box_3d, single_intrinsic,
            )
            target_boxes_3d_encoded_list.append(encoded)
            weights_3d_list.append(weights)

        target_boxes_3d_encoded = torch.cat(target_boxes_3d_encoded_list, dim=0)
        weights_3d = torch.cat(weights_3d_list, dim=0)

        losses = {}

        # Delta 2D center loss
        loss_delta_2d = l1_loss(
            src_boxes_3d[:, :2],
            target_boxes_3d_encoded[:, :2],
            reducer=SumWeightedLoss(
                weight=weights_3d[:, :2], avg_factor=num_boxes.item()
            ),
        )
        losses["loss_delta_2d"] = loss_delta_2d * self.config.loss_delta_2d_weight

        # Depth loss
        loss_depth = l1_loss(
            src_boxes_3d[:, 2],
            target_boxes_3d_encoded[:, 2],
            reducer=SumWeightedLoss(
                weight=weights_3d[:, 2], avg_factor=num_boxes.item()
            ),
        )
        losses["loss_depth"] = loss_depth * self.config.loss_depth_weight

        # Dimension loss
        loss_dim = l1_loss(
            src_boxes_3d[:, 3:6],
            target_boxes_3d_encoded[:, 3:6],
            reducer=SumWeightedLoss(
                weight=weights_3d[:, 3:6], avg_factor=num_boxes.item()
            ),
        )
        losses["loss_dim"] = loss_dim * self.config.loss_dim_weight

        # Rotation loss
        loss_rot = l1_loss(
            src_boxes_3d[:, 6:],
            target_boxes_3d_encoded[:, 6:],
            reducer=SumWeightedLoss(
                weight=weights_3d[:, 6:], avg_factor=num_boxes.item()
            ),
        )
        losses["loss_rot"] = loss_rot * self.config.loss_rot_weight

        return losses

    def _compute_aux_loss(
        self,
        aux_out: dict,
        indices: tuple[Tensor, Tensor, Tensor | None],
        targets: dict,
        num_boxes: Tensor,
        intrinsics: Tensor | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> dict[str, Tensor]:
        """Compute losses for auxiliary decoder outputs.

        Following GDino3D's design, we compute all losses (2D + 3D) for auxiliary outputs
        to enable full deep supervision across all decoder layers.

        Args:
            aux_out: Auxiliary output dictionary containing pred_logits, pred_boxes_2d, pred_boxes_3d
            indices: Matching indices from matcher
            targets: Ground truth targets
            num_boxes: Number of boxes for normalization
            intrinsics: Camera intrinsics for 3D loss computation
            image_size: (H, W) for pixel coordinate conversion

        Returns:
            Dictionary of auxiliary losses
        """
        losses = {}

        # Classification loss (scaled by loss_2d_scale)
        if "pred_logits" in aux_out:
            loss_cls = self._loss_classification(
                aux_out["pred_logits"],
                aux_out.get("pred_boxes_2d", aux_out.get("pred_boxes")),
                indices,
                targets,
                num_boxes,
            )
            losses["loss_cls"] = (
                self.config.loss_2d_scale * loss_cls * self.config.loss_cls_weight
            )

        # 2D box loss (L1 in normalized cxcywh, GIoU in pixel xyxy, scaled by loss_2d_scale)
        if "pred_boxes_2d" in aux_out or "pred_boxes" in aux_out:
            pred_boxes = aux_out.get("pred_boxes_2d", aux_out.get("pred_boxes"))
            loss_bbox, loss_giou = self._loss_boxes_2d(
                pred_boxes, indices, targets, num_boxes, image_size
            )
            losses["loss_bbox"] = (
                self.config.loss_2d_scale * loss_bbox * self.config.loss_bbox_weight
            )
            losses["loss_giou"] = (
                self.config.loss_2d_scale * loss_giou * self.config.loss_giou_weight
            )

        # 3D box loss (scaled by loss_3d_scale, following GDino3D's deep supervision design)
        if "pred_boxes_3d" in aux_out and intrinsics is not None:
            pred_boxes_2d = aux_out.get("pred_boxes_2d", aux_out.get("pred_boxes"))
            loss_3d = self._loss_boxes_3d(
                pred_boxes_2d,
                aux_out["pred_boxes_3d"],
                indices,
                targets,
                intrinsics,
                num_boxes,
                image_size=image_size,
            )
            # Apply loss_3d_scale to all 3D losses
            for key, value in loss_3d.items():
                losses[key] = self.config.loss_3d_scale * value

        return losses

    def _normalize_targets(self, targets: dict) -> dict:
        """Ensure targets are in expected format for loss computation.

        Note: SAM3_3D collator always outputs GT boxes in normalized [0, 1] xyxy format.
        This function simply ensures the classes tensor exists (for binary classification).

        Args:
            targets: Dictionary containing ground truth data
                - boxes_xyxy: (N, 4) boxes in normalized xyxy [0, 1] format
                - classes: (N,) class labels (all ones for SAM3)
                - num_boxes: (N,) number of boxes per prompt (always 1)
                - boxes_3d: (N, 12) 3D boxes (optional)

        Returns:
            Targets dict with classes tensor guaranteed to exist
        """
        normalized = targets.copy()
        boxes_xyxy = targets["boxes_xyxy"]

        # Ensure classes tensor exists (all ones for binary classification)
        if "classes" not in normalized:
            num_boxes = boxes_xyxy.shape[0]
            normalized["classes"] = torch.ones(
                num_boxes, dtype=torch.long, device=boxes_xyxy.device
            )

        return normalized

    def _xyxy_to_cxcywh(self, boxes_xyxy: Tensor) -> Tensor:
        """Convert boxes from xyxy to cxcywh format.

        Args:
            boxes_xyxy: (N, 4) boxes in xyxy format

        Returns:
            boxes_cxcywh: (N, 4) boxes in cxcywh format
        """
        x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return torch.stack([cx, cy, w, h], dim=-1)

