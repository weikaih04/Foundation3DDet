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
from opendet3d.op.box import box_ops


@dataclass
class SAM3_3DLossConfig:
    """Configuration for SAM3_3D loss.
    
    Follows SAM3's loss configuration style with additional 3D loss weights.
    """
    # ========== 2D Loss Weights (SAM3 style) ==========
    # Classification loss (IABCEMdetr style)
    loss_cls_weight: float = 2.0
    pos_weight: float = 1.0  # Weight for positive samples
    gamma: float = 0.0  # Focal loss gamma (0 = no focal)
    alpha: float = 0.25  # IoU-aware alpha
    
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
    
    def forward(
        self,
        # Model outputs
        pred_logits: Tensor,  # (B, S, 1)
        pred_boxes_2d: Tensor,  # (B, S, 4) normalized xyxy
        pred_boxes_3d: Tensor | None,  # (B, S, reg_dims)
        aux_outputs: list[dict] | None,
        geom_losses: dict[str, Tensor] | None,
        
        # Matching indices (from SAM3 matcher)
        indices: tuple[Tensor, Tensor, Tensor | None],  # (batch_idx, src_idx, tgt_idx)
        
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
            indices: Matching indices (batch_idx, src_idx, tgt_idx)
            targets: Ground truth targets
            intrinsics: Camera intrinsics
            
        Returns:
            Dictionary of loss values
        """
        losses = {}
        
        # Get number of boxes for normalization
        num_boxes = self._get_num_boxes(targets)
        
        # ========== 2D Losses ==========
        loss_cls = self._loss_classification(
            pred_logits, pred_boxes_2d, indices, targets, num_boxes
        )
        losses["loss_cls"] = loss_cls * self.config.loss_cls_weight
        
        loss_bbox, loss_giou = self._loss_boxes_2d(
            pred_boxes_2d, indices, targets, num_boxes
        )
        losses["loss_bbox"] = loss_bbox * self.config.loss_bbox_weight
        losses["loss_giou"] = loss_giou * self.config.loss_giou_weight
        
        # ========== 3D Losses ==========
        if pred_boxes_3d is not None and intrinsics is not None:
            loss_3d = self._loss_boxes_3d(
                pred_boxes_2d, pred_boxes_3d, indices, targets, intrinsics, num_boxes
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
                aux_losses = self._compute_aux_loss(aux_out, indices, targets, num_boxes)
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
            iou = box_ops.fast_diag_box_iou(src_boxes_xyxy, target_boxes_xyxy)
            # Soft target: prob^alpha * iou^(1-alpha)
            t = prob[(batch_idx, src_idx)] ** self.config.alpha * iou ** (1 - self.config.alpha)
            t = torch.clamp(t, 0.01).detach()

            positive_target_classes = target_classes.clone()
            positive_target_classes[(batch_idx, src_idx)] = t

        # BCE loss on positives with soft targets
        loss_pos = F.binary_cross_entropy_with_logits(
            src_logits, positive_target_classes, reduction="none"
        )
        loss_pos = loss_pos * target_classes * self.config.pos_weight

        # BCE loss on negatives with focal weighting
        loss_neg = F.binary_cross_entropy_with_logits(
            src_logits, target_classes, reduction="none"
        )
        loss_neg = loss_neg * (1 - target_classes) * (prob ** self.config.gamma)

        loss_bce = (loss_pos + loss_neg).sum() / num_boxes

        return loss_bce

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
        loss_giou = 1 - box_ops.fast_diag_generalized_box_iou(src_boxes, target_boxes)
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

