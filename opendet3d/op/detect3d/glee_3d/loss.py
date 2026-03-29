"""GLEE_3D Loss Module.

Combines:
1. GLEE's 2D losses via SetCriterion (focal cls, L1+GIoU boxes, DN, deep supervision)
2. 3D regression losses (delta_center, depth, dimensions, rotation) from SAM3_3D pattern
3. Geometry backend losses (SILog, phi, theta) from LingbotDepth
4. DN 3D losses (DN queries also get 3D supervision)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from vis4d.op.loss.common import l1_loss
from vis4d.op.loss.reducer import SumWeightedLoss

from opendet3d.op.detect3d.grounding_dino_3d.coder import GroundingDINO3DCoder

# GLEE imports
from glee.models.criterion import SetCriterion
from glee.models.matcher import HungarianMatcher


@dataclass
class GLEE_3DLossConfig:
    """Loss configuration for GLEE_3D."""

    # 2D loss weights (used by GLEE's SetCriterion weight_dict)
    loss_ce_weight: float = 4.0
    loss_bbox_weight: float = 5.0
    loss_giou_weight: float = 2.0
    loss_conf_weight: float = 4.0

    # 3D loss weights
    loss_delta_2d_weight: float = 1.0
    loss_depth_weight: float = 1.0
    loss_dim_weight: float = 1.0
    loss_rot_weight: float = 1.0

    # Global scale factors
    loss_2d_scale: float = 1.0
    loss_3d_scale: float = 1.0
    loss_geom_scale: float = 5.0

    # 3D Confidence Head
    use_3d_conf: bool = True
    loss_3d_conf_weight: float = 20.0
    conf_depth_weight: float = 0.7
    conf_iou_3d_weight: float = 0.3
    conf_alpha: float = 0.25  # soft target mixing: prob^alpha * quality^(1-alpha)
    conf_gamma: float = 2.0   # focal loss gamma for negatives
    conf_pos_weight: float = 5.0  # positive loss weight

    # Ignore suppress
    use_ignore_suppress: bool = True
    ignore_iou_threshold: float = 0.5

    # Geometry loss weights
    loss_silog_weight: float = 1.0
    loss_phi_weight: float = 0.1
    loss_theta_weight: float = 0.1

    # Deep supervision
    aux_loss_weight: float = 1.0

    # GLEE SetCriterion params
    dn: str = "standard"  # "no", "standard", or "seg"
    num_decoder_layers: int = 9


def build_glee_criterion(config: GLEE_3DLossConfig) -> SetCriterion:
    """Build GLEE's SetCriterion for 2D losses."""
    matcher = HungarianMatcher(
        cost_class=config.loss_ce_weight,
        cost_mask=0.0,  # no mask
        cost_dice=0.0,  # no mask
        cost_box=config.loss_bbox_weight,
        cost_giou=config.loss_giou_weight,
        num_points=0,
    )

    # Weight dict for all loss components
    weight_dict = {
        "loss_ce": config.loss_ce_weight,
        "loss_bbox": config.loss_bbox_weight,
        "loss_giou": config.loss_giou_weight,
        "loss_conf": config.loss_conf_weight,
        "loss_mask": 0.0,  # disabled
        "loss_dice": 0.0,  # disabled
    }

    # DN losses (same weights)
    if config.dn != "no":
        dn_weight_dict = {}
        for k, v in weight_dict.items():
            if "mask" not in k and "dice" not in k:
                dn_weight_dict[k + "_dn"] = v
        weight_dict.update(dn_weight_dict)

    # Deep supervision losses (per decoder layer)
    for i in range(config.num_decoder_layers):
        for k, v in list(weight_dict.items()):
            if "_dn" not in k and not any(
                k.endswith(f"_{j}") for j in range(config.num_decoder_layers)
            ):
                weight_dict[k + f"_{i}"] = v

    # num_classes must match what GLEE criterion's self.num_classes[task] expects.
    # Use 'grounding' task (num_classes=1) since GLEE_3D uses contrastive
    # classification via class_embeddings, not fixed category count.
    criterion = SetCriterion(
        num_classes=1,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1,
        losses=["labels", "boxes", "conf"],  # no "masks"; conf ensures pred_scores in graph
        num_points=0,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        dn=config.dn,
        dn_losses=["labels", "boxes"],
        panoptic_on=False,
        semantic_ce_loss=False,
    )

    return criterion


class GLEE_3DLoss(nn.Module):
    """Loss for GLEE_3D combining 2D (GLEE) + 3D (from SAM3_3D pattern)."""

    def __init__(
        self,
        config: GLEE_3DLossConfig,
        box_coder: GroundingDINO3DCoder,
        criterion_2d: SetCriterion | None = None,
        task: str = "glee3d",
    ):
        super().__init__()
        self.config = config
        self.box_coder = box_coder
        self.task = task

        if criterion_2d is not None:
            self.criterion_2d = criterion_2d
        else:
            self.criterion_2d = build_glee_criterion(config)

    def forward(
        self,
        pred_logits: Tensor,  # (B, Q, C)
        pred_scores: Tensor,  # (B, Q, 1)
        pred_boxes_2d: Tensor,  # (B, Q, 4) cxcywh normalized
        pred_boxes_3d: Tensor | None,  # (B, Q, 12)
        pred_conf_3d: Tensor | None,  # (B, Q, 1) 3D confidence
        pred_boxes_3d_dn: Tensor | None,  # (B, DN_Q, 12) DN 3D preds
        aux_outputs: list[dict] | None,
        geom_losses: dict[str, Tensor] | None,
        mask_dict: dict | None,
        targets: list[dict],  # GLEE format per-image
        gt_boxes_3d: list[Tensor],  # per-image 3D GT
        gt_boxes_2d_pixel: list[Tensor],  # per-image 2D GT xyxy pixel
        intrinsics: Tensor,  # (B, 3, 3)
        image_size: tuple[int, int],  # (H, W)
        ignore_boxes_2d: list[Tensor] | None = None,  # per-image ignore boxes xyxy pixel
    ) -> dict[str, Tensor]:
        """Compute all losses.

        Returns dict of named loss tensors (already weighted).
        """
        losses = {}

        # ==============================================================
        # 1. GLEE 2D losses (includes DN + deep supervision)
        # ==============================================================
        outputs_2d = {
            "pred_logits": pred_logits,
            "pred_scores": pred_scores,
            "pred_boxes": pred_boxes_2d,
            "pred_masks": None,
        }

        # Ignore suppress: compute mask for predictions overlapping ignore boxes
        if (
            self.config.use_ignore_suppress
            and ignore_boxes_2d is not None
        ):
            H, W = image_size
            outputs_2d["ignore_neg_mask"] = self._compute_ignore_neg_mask(
                pred_boxes_2d, ignore_boxes_2d, image_size,
                threshold=self.config.ignore_iou_threshold,
            )

        if aux_outputs is not None:
            outputs_2d["aux_outputs"] = aux_outputs

        losses_2d = self.criterion_2d(
            outputs_2d, targets, mask_dict=mask_dict, task=self.task
        )

        # Apply 2D scale and add to losses (skip mask/dice)
        for k, v in losses_2d.items():
            if "mask" in k or "dice" in k:
                continue
            if k in self.criterion_2d.weight_dict:
                losses[k] = v * self.criterion_2d.weight_dict[k] * self.config.loss_2d_scale
            else:
                losses[k] = v * self.config.loss_2d_scale

        # ==============================================================
        # 2. 3D losses on matched pairs
        # ==============================================================
        if pred_boxes_3d is not None:
            indices = self.criterion_2d.last_indices
            losses_3d = self._loss_boxes_3d(
                pred_boxes_3d=pred_boxes_3d,
                indices=indices,
                gt_boxes_3d=gt_boxes_3d,
                gt_boxes_2d_pixel=gt_boxes_2d_pixel,
                intrinsics=intrinsics,
                image_size=image_size,
            )
            for k, v in losses_3d.items():
                losses[k] = v * self.config.loss_3d_scale

        # ==============================================================
        # 3. 3D deep supervision on auxiliary layers
        # ==============================================================
        if pred_boxes_3d is not None and aux_outputs is not None:
            for i, aux in enumerate(aux_outputs):
                aux_pred_3d = aux.get("pred_boxes_3d")
                if aux_pred_3d is None:
                    continue

                # Re-run matcher for this aux layer
                aux_outputs_2d_i = {
                    "pred_logits": aux["pred_logits"],
                    "pred_scores": aux.get("pred_scores", pred_scores),
                    "pred_boxes": aux["pred_boxes"],
                }
                if self.task in self.criterion_2d.no_mask_tasks:
                    aux_indices = self.criterion_2d.matcher(
                        aux_outputs_2d_i, targets, self.task, ["cls", "box"]
                    )
                else:
                    aux_indices = self.criterion_2d.matcher(
                        aux_outputs_2d_i, targets, self.task
                    )

                aux_losses_3d = self._loss_boxes_3d(
                    pred_boxes_3d=aux_pred_3d,
                    indices=aux_indices,
                    gt_boxes_3d=gt_boxes_3d,
                    gt_boxes_2d_pixel=gt_boxes_2d_pixel,
                    intrinsics=intrinsics,
                    image_size=image_size,
                )
                for k, v in aux_losses_3d.items():
                    losses[f"{k}_{i}"] = (
                        v * self.config.loss_3d_scale * self.config.aux_loss_weight
                    )

        # ==============================================================
        # 3.5 3D Confidence Loss (same as SAM3_3D's _loss_3d_classification)
        # ==============================================================
        if (
            self.config.use_3d_conf
            and pred_conf_3d is not None
            and pred_boxes_3d is not None
        ):
            indices = self.criterion_2d.last_indices
            loss_3d_conf = self._loss_3d_confidence(
                pred_conf_3d=pred_conf_3d,
                pred_boxes_2d=pred_boxes_2d,
                pred_boxes_3d=pred_boxes_3d,
                indices=indices,
                gt_boxes_3d=gt_boxes_3d,
                intrinsics=intrinsics,
                image_size=image_size,
            )
            losses["loss_3d_conf"] = self.config.loss_3d_conf_weight * loss_3d_conf

        # ==============================================================
        # 4. DN 3D losses (DN queries have known GT mapping)
        # ==============================================================
        if pred_boxes_3d_dn is not None and mask_dict is not None:
            dn_3d_losses = self._loss_boxes_3d_dn(
                pred_boxes_3d_dn=pred_boxes_3d_dn,
                mask_dict=mask_dict,
                targets=targets,
                gt_boxes_3d=gt_boxes_3d,
                gt_boxes_2d_pixel=gt_boxes_2d_pixel,
                intrinsics=intrinsics,
                image_size=image_size,
            )
            for k, v in dn_3d_losses.items():
                losses[f"{k}_dn"] = v * self.config.loss_3d_scale

        # ==============================================================
        # 5. Geometry backend losses (independent)
        # ==============================================================
        if geom_losses is not None:
            for k, v in geom_losses.items():
                weight = getattr(self.config, f"loss_{k}_weight", 1.0)
                losses[f"loss_{k}"] = v * weight * self.config.loss_geom_scale

        return losses

    # ------------------------------------------------------------------
    # 3D loss computation (reuses SAM3_3D pattern)
    # ------------------------------------------------------------------

    def _loss_3d_confidence(
        self,
        pred_conf_3d: Tensor,  # (B, Q, 1)
        pred_boxes_2d: Tensor,  # (B, Q, 4) cxcywh normalized
        pred_boxes_3d: Tensor,  # (B, Q, 12) encoded
        indices: list[tuple[Tensor, Tensor]],  # per-image (src_idx, tgt_idx)
        gt_boxes_3d: list[Tensor],  # per-image [N_i, 10]
        intrinsics: Tensor,  # (B, 3, 3)
        image_size: tuple[int, int],
    ) -> Tensor:
        """3D confidence loss (copied from SAM3_3D's _loss_3d_classification).

        Positive: soft target = conf_depth_weight * depth_quality + conf_iou_3d_weight * iou_3d
        Negative: target=0, focal weighting
        """
        B, Q, _ = pred_conf_3d.shape
        device = pred_conf_3d.device
        H, W = image_size

        prob = pred_conf_3d.sigmoid()
        target_classes = torch.zeros(B, Q, 1, device=device)

        # Mark matched queries as positive
        total_matched = 0
        for b in range(B):
            src_idx, tgt_idx = indices[b]
            if len(src_idx) == 0:
                continue
            target_classes[b, src_idx] = 1.0
            total_matched += len(src_idx)

        if total_matched == 0:
            return pred_conf_3d.sum() * 0.0

        # Compute quality for positive targets (no_grad)
        with torch.no_grad():
            positive_target = target_classes.clone()

            for b in range(B):
                src_idx, tgt_idx = indices[b]
                if len(src_idx) == 0:
                    continue

                src_3d = pred_boxes_3d[b, src_idx]  # (M, 12)
                gt_3d_b = gt_boxes_3d[b][tgt_idx]   # (M, 10)

                # 1. Depth quality
                depth_scale = self.box_coder.depth_scale
                pred_log_z = src_3d[:, 2] / depth_scale
                gt_z = gt_3d_b[:, 2].clamp(min=0.1)
                gt_log_z = torch.log(gt_z)
                depth_quality = torch.exp(-torch.abs(pred_log_z - gt_log_z))
                depth_quality = torch.nan_to_num(depth_quality, nan=0.0)

                # 2. 3D IoU (use safe implementation)
                from opendet3d.op.box.iou_3d_safe import batch_box3d_iou

                # Decode predictions for IoU
                boxes_2d_b = pred_boxes_2d[b, src_idx]  # (M, 4) cxcywh norm
                cx, cy, w, h = boxes_2d_b.unbind(-1)
                boxes_xyxy = torch.stack([
                    (cx - w/2) * W, (cy - h/2) * H,
                    (cx + w/2) * W, (cy + h/2) * H], dim=-1)

                pred_decoded_list = []
                for j in range(len(src_idx)):
                    dec = self.box_coder.decode(
                        boxes_xyxy[j:j+1], src_3d[j:j+1], intrinsics[b])
                    pred_decoded_list.append(dec)
                pred_decoded = torch.cat(pred_decoded_list, dim=0)

                iou_3d = batch_box3d_iou(pred_decoded, gt_3d_b[:, :10])

                # 3. Combined quality
                quality = (
                    self.config.conf_depth_weight * depth_quality
                    + self.config.conf_iou_3d_weight * iou_3d
                ).clamp(0.0, 1.0)
                quality = torch.nan_to_num(quality, nan=0.0)

                # 4. Soft target
                t = (
                    prob[b, src_idx].squeeze(-1) ** self.config.conf_alpha
                    * quality ** (1 - self.config.conf_alpha)
                ).clamp(min=0.01).detach()

                positive_target[b, src_idx] = t.unsqueeze(-1)

        # Positive loss
        loss_pos = F.binary_cross_entropy_with_logits(
            pred_conf_3d, positive_target, reduction="none"
        )
        loss_pos = loss_pos * target_classes * self.config.conf_pos_weight

        # Negative loss with focal weighting
        loss_neg = F.binary_cross_entropy_with_logits(
            pred_conf_3d, target_classes, reduction="none"
        )
        loss_neg = loss_neg * (1 - target_classes) * (prob ** self.config.conf_gamma)

        return (loss_pos + loss_neg).mean()

    def _compute_ignore_neg_mask(
        self,
        pred_boxes_cxcywh: Tensor,  # (B, Q, 4) normalized cxcywh
        ignore_boxes_2d: list[Tensor],  # per-image ignore boxes xyxy pixel
        image_size: tuple[int, int],
        threshold: float = 0.5,
    ) -> Tensor:
        """Compute mask for predictions overlapping ignore boxes.

        Same logic as SAM3_3D._compute_ignore_neg_mask.

        Returns:
            mask: (B, Q) float. 1.0 = suppress negative loss, 0.0 = normal.
        """
        import torchvision.ops

        B, Q, _ = pred_boxes_cxcywh.shape
        H, W = image_size
        device = pred_boxes_cxcywh.device
        mask = torch.zeros(B, Q, device=device)

        # Convert pred cxcywh normalized → xyxy normalized
        cx, cy, w, h = pred_boxes_cxcywh.unbind(-1)
        pred_xyxy = torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)

        for b in range(B):
            ign = ignore_boxes_2d[b]
            if len(ign) == 0:
                continue
            # Convert ignore boxes from pixel to normalized
            if not isinstance(ign, Tensor):
                ign = torch.as_tensor(ign, dtype=torch.float32, device=device)
            ign = ign.to(device)
            ign_norm = ign / torch.tensor([W, H, W, H], dtype=torch.float32, device=device)

            iou = torchvision.ops.box_iou(pred_xyxy[b], ign_norm)  # (Q, N_ign)
            mask[b] = (iou.max(dim=1).values > threshold).float()

        return mask

    def _loss_boxes_3d(
        self,
        pred_boxes_3d: Tensor,  # (B, Q, 12)
        indices: list[tuple[Tensor, Tensor]],  # per-image (src_idx, tgt_idx)
        gt_boxes_3d: list[Tensor],  # per-image [N_i, 10]
        gt_boxes_2d_pixel: list[Tensor],  # per-image [N_i, 4] xyxy pixel
        intrinsics: Tensor,  # (B, 3, 3)
        image_size: tuple[int, int],
    ) -> dict[str, Tensor]:
        """Compute 3D box regression losses on matched pairs.

        GLEE uses per-image matching: indices[i] = (src_idx, tgt_idx) for image i.
        This differs from SAM3_3D which packs everything into (batch_idx, src_idx, tgt_idx).
        """
        device = pred_boxes_3d.device
        B = pred_boxes_3d.shape[0]

        all_src_3d = []
        all_target_encoded = []
        all_weights = []

        for b in range(B):
            src_idx, tgt_idx = indices[b]
            if len(src_idx) == 0:
                continue

            # Matched predictions
            src_3d = pred_boxes_3d[b, src_idx]  # (M, 12)

            # Matched GT
            gt_3d_b = gt_boxes_3d[b][tgt_idx]  # (M, 10)
            gt_2d_pixel_b = gt_boxes_2d_pixel[b][tgt_idx]  # (M, 4) xyxy pixel
            intrinsics_b = intrinsics[b]  # (3, 3)

            # Encode each GT 3D box
            for j in range(len(src_idx)):
                single_gt_3d = gt_3d_b[j : j + 1]

                # Skip invalid 3D boxes
                if single_gt_3d.abs().sum() < 1e-6:
                    reg_dims = pred_boxes_3d.shape[-1]
                    all_src_3d.append(src_3d[j : j + 1])
                    all_target_encoded.append(
                        torch.zeros(1, reg_dims, device=device)
                    )
                    all_weights.append(
                        torch.zeros(1, reg_dims, device=device)
                    )
                    continue

                single_gt_2d_pixel = gt_2d_pixel_b[j : j + 1]
                encoded, weights = self.box_coder.encode(
                    single_gt_2d_pixel, single_gt_3d, intrinsics_b
                )
                all_src_3d.append(src_3d[j : j + 1])
                all_target_encoded.append(encoded)
                all_weights.append(weights)

        # Handle no matches
        if len(all_src_3d) == 0:
            zero = torch.tensor(0.0, device=device)
            return {
                "loss_delta_2d": zero,
                "loss_depth": zero,
                "loss_dim": zero,
                "loss_rot": zero,
            }

        src_3d_cat = torch.cat(all_src_3d, dim=0)
        target_cat = torch.cat(all_target_encoded, dim=0)
        weights_cat = torch.cat(all_weights, dim=0)
        num_boxes = max(float(src_3d_cat.shape[0]), 1.0)

        return self._compute_3d_l1_losses(
            src_3d_cat, target_cat, weights_cat, num_boxes
        )

    def _loss_boxes_3d_dn(
        self,
        pred_boxes_3d_dn: Tensor,  # (B, pad_size, 12) DN 3D predictions
        mask_dict: dict,
        targets: list[dict],
        gt_boxes_3d: list[Tensor],
        gt_boxes_2d_pixel: list[Tensor],
        intrinsics: Tensor,
        image_size: tuple[int, int],
    ) -> dict[str, Tensor]:
        """Compute 3D losses on DN queries.

        Each DN query has a known GT correspondence. We reconstruct the
        DN-to-GT mapping (same as GLEE criterion's exc_idx) and compute
        3D L1 losses using clean GT 2D boxes for box_coder.encode().

        mask_dict keys:
            'pad_size': total DN query positions
            'scalar': number of repetitions per GT
            'batch_idx': which image each DN entry belongs to
            'map_known_indice': position in DN query space for each entry
        """
        device = pred_boxes_3d_dn.device
        B = pred_boxes_3d_dn.shape[0]
        scalar = mask_dict["scalar"]
        pad_size = mask_dict["pad_size"]
        single_pad = pad_size // scalar

        # Reconstruct exc_idx: DN query position → per-image GT index
        # Same logic as GLEE criterion.py lines 470-487
        all_src_3d = []
        all_target_encoded = []
        all_weights = []

        for b in range(B):
            num_gts = len(targets[b]["labels"])
            if num_gts == 0:
                continue

            # For each scalar group, map DN positions to GT indices
            t = torch.arange(0, num_gts, device=device).long()
            t_rep = t.unsqueeze(0).repeat(scalar, 1)  # (scalar, num_gts)
            tgt_idx = t_rep.flatten()  # per-image GT indices

            output_idx = (
                torch.arange(scalar, device=device).long().unsqueeze(1)
                * single_pad
                + t.unsqueeze(0)
            ).flatten()  # DN query positions

            # Get DN 3D predictions for this image
            src_3d = pred_boxes_3d_dn[b, output_idx]  # (M, 12)

            # Get clean GT 2D and 3D boxes
            gt_3d_b = gt_boxes_3d[b][tgt_idx]  # (M, 10)
            gt_2d_pixel_b = gt_boxes_2d_pixel[b][tgt_idx]  # (M, 4)
            intrinsics_b = intrinsics[b]

            # Encode each GT 3D box
            for j in range(len(tgt_idx)):
                single_gt_3d = gt_3d_b[j : j + 1]
                if single_gt_3d.abs().sum() < 1e-6:
                    reg_dims = pred_boxes_3d_dn.shape[-1]
                    all_src_3d.append(src_3d[j : j + 1])
                    all_target_encoded.append(
                        torch.zeros(1, reg_dims, device=device)
                    )
                    all_weights.append(
                        torch.zeros(1, reg_dims, device=device)
                    )
                    continue

                single_gt_2d_pixel = gt_2d_pixel_b[j : j + 1]
                encoded, weights = self.box_coder.encode(
                    single_gt_2d_pixel, single_gt_3d, intrinsics_b
                )
                all_src_3d.append(src_3d[j : j + 1])
                all_target_encoded.append(encoded)
                all_weights.append(weights)

        if len(all_src_3d) == 0:
            zero = torch.tensor(0.0, device=device)
            return {
                "loss_delta_2d": zero,
                "loss_depth": zero,
                "loss_dim": zero,
                "loss_rot": zero,
            }

        src_3d_cat = torch.cat(all_src_3d, dim=0)
        target_cat = torch.cat(all_target_encoded, dim=0)
        weights_cat = torch.cat(all_weights, dim=0)
        num_boxes = max(float(src_3d_cat.shape[0]), 1.0)

        return self._compute_3d_l1_losses(
            src_3d_cat, target_cat, weights_cat, num_boxes
        )

    def _compute_3d_l1_losses(
        self,
        src_3d: Tensor,  # (N, 12)
        target_3d: Tensor,  # (N, 12)
        weights: Tensor,  # (N, 12)
        num_boxes: float,
    ) -> dict[str, Tensor]:
        """Compute per-component L1 losses on encoded 3D params."""
        losses = {}

        # Delta 2D center
        losses["loss_delta_2d"] = l1_loss(
            src_3d[:, :2],
            target_3d[:, :2],
            reducer=SumWeightedLoss(
                weight=weights[:, :2], avg_factor=num_boxes
            ),
        ) * self.config.loss_delta_2d_weight

        # Log depth
        losses["loss_depth"] = l1_loss(
            src_3d[:, 2],
            target_3d[:, 2],
            reducer=SumWeightedLoss(
                weight=weights[:, 2], avg_factor=num_boxes
            ),
        ) * self.config.loss_depth_weight

        # Log dimensions
        losses["loss_dim"] = l1_loss(
            src_3d[:, 3:6],
            target_3d[:, 3:6],
            reducer=SumWeightedLoss(
                weight=weights[:, 3:6], avg_factor=num_boxes
            ),
        ) * self.config.loss_dim_weight

        # 6D rotation
        losses["loss_rot"] = l1_loss(
            src_3d[:, 6:],
            target_3d[:, 6:],
            reducer=SumWeightedLoss(
                weight=weights[:, 6:], avg_factor=num_boxes
            ),
        ) * self.config.loss_rot_weight

        return losses
