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
from sam3.train.matcher import BinaryHungarianMatcher, BinaryOneToManyMatcher


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
    # ========== Matcher Configuration ==========
    matcher_cost_class: float = 2.0
    matcher_cost_bbox: float = 5.0
    matcher_cost_giou: float = 2.0

    # O2M (One-to-Many) Matcher Configuration
    use_o2m: bool = True  # Enable O2M matching
    o2m_alpha: float = 0.3  # Alpha for O2M cost computation
    o2m_threshold: float = 0.4  # IoU threshold for O2M matching
    o2m_topk: int = 6  # Top-k predictions per GT
    o2m_loss_weight: float = 1.0  # Weight for O2M loss

    # ========== 2D Loss Weights (SAM3 style) ==========
    # Classification loss (IABCEMdetr style)
    loss_cls_weight: float = 20.0  # Fixed: was 2.0
    pos_weight: float = 10.0  # Fixed: was 1.0
    gamma: float = 0.0  # Focal loss gamma (0 = no focal)
    alpha: float = 0.25  # IoU-aware alpha

    # IABCEMdetr advanced features
    use_weak_loss: bool = True  # Enable weak supervision (non-exhaustive annotations)
    weak_loss_weight: float = 1.0  # Weight for weak loss
    use_presence: bool = True  # Enable presence loss (empty image detection)
    presence_loss_weight: float = 1.0  # Weight for presence loss

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

        # Initialize matcher
        self.matcher = BinaryHungarianMatcher(
            cost_class=self.config.matcher_cost_class,
            cost_bbox=self.config.matcher_cost_bbox,
            cost_giou=self.config.matcher_cost_giou,
        )

        # Initialize O2M matcher if enabled
        if self.config.use_o2m:
            self.o2m_matcher = BinaryOneToManyMatcher(
                alpha=self.config.o2m_alpha,
                threshold=self.config.o2m_threshold,
                topk=self.config.o2m_topk,
            )
        else:
            self.o2m_matcher = None
    
    def forward(
        self,
        # Model outputs
        pred_logits: Tensor,  # (B, S, 1)
        pred_boxes_2d: Tensor,  # (B, S, 4) normalized xyxy
        pred_boxes_3d: Tensor | None,  # (B, S, reg_dims)
        aux_outputs: list[dict] | None,
        geom_losses: dict[str, Tensor] | None,

        # Targets
        targets: dict,

        # Intrinsics for 3D encoding
        intrinsics: Tensor | None = None,  # (B, 3, 3)
    ) -> dict[str, Tensor]:
        """Compute all losses.

        Args:
            pred_logits: Predicted objectness logits
            pred_boxes_2d: Predicted 2D boxes (normalized xyxy)
            pred_boxes_3d: Predicted 3D box parameters
            aux_outputs: Auxiliary outputs from decoder layers
            geom_losses: Geometry backend losses
            targets: Ground truth targets
            intrinsics: Camera intrinsics

        Returns:
            Dictionary of loss values
        """
        losses = {}

        # Normalize targets to [0, 1] range
        normalized_targets = self._normalize_targets(targets)

        # Compute matching indices using BinaryHungarianMatcher
        outputs_dict = {
            "pred_logits": pred_logits,  # (B, S, 1)
            "pred_boxes": self._xyxy_to_cxcywh(pred_boxes_2d),  # (B, S, 4) cxcywh
        }

        # Matcher expects concatenated boxes across all batches
        targets_dict = {
            "boxes": self._xyxy_to_cxcywh(normalized_targets["boxes_xyxy"]),  # (N_total, 4)
            "labels": normalized_targets["classes"],  # (N_total,)
            "num_boxes": normalized_targets.get("num_boxes", torch.tensor([normalized_targets["boxes_xyxy"].shape[0]], device=normalized_targets["boxes_xyxy"].device)),
        }

        # Run matcher
        indices_list = self.matcher(outputs_dict, targets_dict)

        # Convert list of tuples to flat tensors
        batch_idx_list = []
        src_idx_list = []
        for i, (src, _) in enumerate(indices_list):
            # Handle both 1-D and 0-D tensors
            if src.ndim == 0:
                # Single match, convert to list
                src_list = [src.item()]
            else:
                src_list = src.tolist()

            batch_idx_list.extend([i] * len(src_list))
            src_idx_list.extend(src_list)

        batch_idx = torch.tensor(batch_idx_list, dtype=torch.long, device=pred_logits.device) if batch_idx_list else torch.empty(0, dtype=torch.long, device=pred_logits.device)
        src_idx = torch.tensor(src_idx_list, dtype=torch.long, device=pred_logits.device) if src_idx_list else torch.empty(0, dtype=torch.long, device=pred_logits.device)
        indices = (batch_idx, src_idx, None)

        # Get number of boxes for normalization
        num_boxes = self._get_num_boxes(normalized_targets)
        
        # ========== 2D Losses ==========
        loss_cls = self._loss_classification(
            pred_logits, pred_boxes_2d, indices, normalized_targets, num_boxes
        )
        losses["loss_cls"] = loss_cls * self.config.loss_cls_weight

        # Presence loss (empty image detection)
        if self.config.use_presence:
            loss_presence = self._loss_presence(pred_logits, normalized_targets)
            losses["loss_presence"] = loss_presence * self.config.presence_loss_weight

        loss_bbox, loss_giou = self._loss_boxes_2d(
            pred_boxes_2d, indices, normalized_targets, num_boxes
        )
        losses["loss_bbox"] = loss_bbox * self.config.loss_bbox_weight
        losses["loss_giou"] = loss_giou * self.config.loss_giou_weight

        # ========== O2M Loss ==========
        if self.config.use_o2m and self.o2m_matcher is not None:
            o2m_losses = self._loss_o2m(
                pred_logits, pred_boxes_2d, normalized_targets, num_boxes
            )
            for key, value in o2m_losses.items():
                losses[f"o2m_{key}"] = value * self.config.o2m_loss_weight

        # ========== 3D Losses ==========
        if pred_boxes_3d is not None and intrinsics is not None:
            loss_3d = self._loss_boxes_3d(
                pred_boxes_2d, pred_boxes_3d, indices, normalized_targets, intrinsics, num_boxes
            )
            losses.update(loss_3d)

        # ========== Geometry Backend Losses ==========
        if geom_losses is not None:
            for key, value in geom_losses.items():
                weight = getattr(self.config, f"loss_{key}_weight", 1.0)
                losses[f"loss_{key}"] = value * weight

        # ========== Auxiliary Losses ==========
        if aux_outputs is not None:
            for i, aux_out in enumerate(aux_outputs):
                aux_losses = self._compute_aux_loss(aux_out, indices, normalized_targets, num_boxes)
                for key, value in aux_losses.items():
                    losses[f"d{i}.{key}"] = value * self.config.aux_loss_weight
        
        # ========== Total Loss ==========
        losses["loss_total"] = sum(v for k, v in losses.items() if k.startswith("loss_"))

        return losses

    def _get_num_boxes(self, targets: dict) -> Tensor:
        """Get number of boxes for loss normalization."""
        num_boxes = targets["num_boxes"].sum().float()

        if self.config.normalization == "global":
            torch.distributed.all_reduce(num_boxes)
            world_size = torch.distributed.get_world_size()
            num_boxes = torch.clamp(num_boxes / world_size, min=1)
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
            positive_target_classes[(batch_idx, src_idx)] = t

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

        loss_bce = (loss_pos + loss_neg).sum() / num_boxes

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
        pred_logits: Tensor,  # (B, S, 1)
        targets: dict,
    ) -> Tensor:
        """Compute presence loss for empty image detection.

        Presence loss encourages the model to predict all negatives for images
        without any objects (empty images). This helps with false positive reduction.

        Args:
            pred_logits: Predicted objectness logits (B, S, 1)
            targets: Ground truth targets with num_boxes field

        Returns:
            Presence loss scalar
        """
        device = pred_logits.device
        B, S = pred_logits.shape[:2]

        src_logits = pred_logits.squeeze(-1)  # (B, S)

        # Identify empty images (num_gts == 0)
        num_gts = targets.get("num_boxes", torch.zeros(B, dtype=torch.long, device=device))
        is_empty = (num_gts == 0)  # (B,)

        if not is_empty.any():
            # No empty images in batch, no presence loss
            return torch.tensor(0.0, device=device)

        # For empty images, all predictions should be negative
        empty_logits = src_logits[is_empty]  # (N_empty * S,)
        target_empty = torch.zeros_like(empty_logits)

        # BCE loss for empty image predictions
        loss_presence = F.binary_cross_entropy_with_logits(
            empty_logits, target_empty, reduction="mean"
        )

        return loss_presence

    def _loss_o2m(
        self,
        pred_logits: Tensor,  # (B, S, 1)
        pred_boxes_2d: Tensor,  # (B, S, 4)
        targets: dict,
        num_boxes: Tensor,
    ) -> dict[str, Tensor]:
        """Compute O2M (One-to-Many) auxiliary loss.

        O2M matching allows multiple predictions to match the same GT,
        which helps improve recall and stabilize training.

        Args:
            pred_logits: Predicted objectness logits
            pred_boxes_2d: Predicted 2D boxes (normalized xyxy)
            targets: Ground truth targets
            num_boxes: Number of boxes for normalization

        Returns:
            Dictionary with O2M losses (loss_cls, loss_bbox, loss_giou)
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
            # No matches, return zero losses
            return {
                "loss_cls": torch.tensor(0.0, device=device),
                "loss_bbox": torch.tensor(0.0, device=device),
                "loss_giou": torch.tensor(0.0, device=device),
            }

        # O2M Classification Loss
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
            positive_target_classes[(batch_idx, src_idx)] = t

        # BCE loss (only on positives for O2M)
        loss_pos = F.binary_cross_entropy_with_logits(
            src_logits, positive_target_classes, reduction="none"
        )
        loss_pos = loss_pos * target_classes * self.config.pos_weight
        loss_cls = loss_pos.sum() / num_boxes

        losses["loss_cls"] = loss_cls

        # O2M Box Losses
        src_boxes = src_boxes_xyxy  # Already extracted above
        target_boxes = target_boxes_xyxy  # Already extracted above

        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox.sum() / num_boxes
        losses["loss_bbox"] = loss_bbox

        # GIoU loss
        loss_giou = 1 - fast_diag_generalized_box_iou(src_boxes, target_boxes)
        loss_giou = loss_giou.sum() / num_boxes
        losses["loss_giou"] = loss_giou

        return losses

    def _loss_boxes_2d(
        self,
        pred_boxes_2d: Tensor,  # (B, S, 4) normalized xyxy
        indices: tuple[Tensor, Tensor, Tensor | None],
        targets: dict,
        num_boxes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute 2D box regression losses (L1 + GIoU)."""
        batch_idx, src_idx, tgt_idx = indices

        src_boxes = pred_boxes_2d[(batch_idx, src_idx)]
        target_boxes = (
            targets["boxes_xyxy"][tgt_idx] if tgt_idx is not None
            else targets["boxes_xyxy"]
        )

        # L1 loss
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox.sum() / num_boxes

        # GIoU loss
        loss_giou = 1 - fast_diag_generalized_box_iou(src_boxes, target_boxes)
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
    ) -> dict[str, Tensor]:
        """Compute 3D box regression losses."""
        batch_idx, src_idx, tgt_idx = indices

        # Get matched predictions
        src_boxes_2d = pred_boxes_2d[(batch_idx, src_idx)]
        src_boxes_3d = pred_boxes_3d[(batch_idx, src_idx)]

        # Get matched targets
        target_boxes_3d = (
            targets["boxes_3d"][tgt_idx] if tgt_idx is not None
            else targets["boxes_3d"]
        )

        # Get intrinsics for matched samples
        # Note: intrinsics is (B, 3, 3), need to index by batch_idx
        matched_intrinsics = intrinsics[batch_idx]

        # Encode 3D targets
        target_boxes_3d_encoded, weights_3d = self.box_coder.encode(
            src_boxes_2d, target_boxes_3d, matched_intrinsics
        )

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
    ) -> dict[str, Tensor]:
        """Compute losses for auxiliary decoder outputs."""
        losses = {}

        # Classification loss
        if "pred_logits" in aux_out:
            loss_cls = self._loss_classification(
                aux_out["pred_logits"],
                aux_out.get("pred_boxes_2d", aux_out.get("pred_boxes")),
                indices,
                targets,
                num_boxes,
            )
            losses["loss_cls"] = loss_cls * self.config.loss_cls_weight

        # 2D box loss
        if "pred_boxes_2d" in aux_out or "pred_boxes" in aux_out:
            pred_boxes = aux_out.get("pred_boxes_2d", aux_out.get("pred_boxes"))
            loss_bbox, loss_giou = self._loss_boxes_2d(
                pred_boxes, indices, targets, num_boxes
            )
            losses["loss_bbox"] = loss_bbox * self.config.loss_bbox_weight
            losses["loss_giou"] = loss_giou * self.config.loss_giou_weight

        return losses

    def _normalize_targets(self, targets: dict) -> dict:
        """Normalize target boxes to [0, 1] range if needed.

        Args:
            targets: Dictionary containing ground truth data
                - boxes_xyxy: (N, 4) boxes in xyxy format
                - classes: (N,) class labels (all ones for SAM3)
                - num_boxes: scalar or (B,) number of boxes per sample
                - boxes_3d: (N, 7) 3D boxes (optional)
                - image_size: (2,) or (B, 2) image dimensions (H, W)

        Returns:
            Normalized targets with boxes_xyxy in [0, 1] range
        """
        normalized = targets.copy()

        boxes_xyxy = targets["boxes_xyxy"]

        # Check if boxes are already normalized (max value <= 1.0)
        if boxes_xyxy.numel() > 0 and boxes_xyxy.max() > 1.0:
            # Boxes are in pixel coordinates, need to normalize
            if "image_size" in targets:
                img_h, img_w = targets["image_size"]
                # Normalize: [x1, y1, x2, y2] / [W, H, W, H]
                scale = torch.tensor(
                    [img_w, img_h, img_w, img_h],
                    dtype=boxes_xyxy.dtype,
                    device=boxes_xyxy.device
                )
                normalized["boxes_xyxy"] = boxes_xyxy / scale
            else:
                # If no image_size provided, assume boxes are already normalized
                # This should not happen, but we handle it gracefully
                pass

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

