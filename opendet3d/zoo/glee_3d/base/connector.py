"""GLEE_3D data connector and collator.

Reuses the same vis4d dataset pipeline as SAM3_3D (same transforms,
same per-sample dict format), but with simpler per-image batching
instead of SAM3_3D's per-category batching.
"""

from __future__ import annotations

import torch
from torch import Tensor

from vis4d.data.const import CommonKeys as K

from opendet3d.model.detect3d.glee_3d import GLEE_3DBatchedInputs


class GLEE_3DCollator:
    """Collates per-image vis4d samples into GLEE_3DBatchedInputs.

    Per-image batching (simpler than SAM3_3DCollator's per-category batching).
    All categories detected simultaneously per image by GLEE.

    Input format per sample (from vis4d dataset + transforms):
        K.images:           Tensor (3, H, W) float32, ImageNet normalized
        K.intrinsics:       Tensor (3, 3)
        K.boxes2d:          Tensor (N, 4) pixel xyxy in padded space
        K.boxes2d_classes:  Tensor (N,) int64
        K.boxes2d_names:    list[str] (per-box category names)
        K.boxes3d:          Tensor (N, 11) [cx,cy,cz,w,l,h,qw,qx,qy,qz,conf]
        K.depth_maps:       Tensor (H, W) or None
        "padding":          [pad_left, pad_right, pad_top, pad_bottom]
        "sample_names":     str
        "dataset_name":     str
        K.original_hw:      tuple (H_orig, W_orig)
    """

    def __init__(
        self,
        filter_empty_boxes: bool = True,
        visual_prompt_prob: float = 0.3,
        box_noise_std: float = 0.1,
        object_text_prob: float = 0.5,
    ):
        self.filter_empty_boxes = filter_empty_boxes
        self.visual_prompt_prob = visual_prompt_prob
        self.box_noise_std = box_noise_std
        # When visual prompt is active, probability of replacing class names
        # with ["object"] (class-agnostic mode, like GLEE's SA1B training)
        self.object_text_prob = object_text_prob

    def __call__(self, batch: list[dict]) -> GLEE_3DBatchedInputs:
        images_list = []
        intrinsics_list = []
        targets_list = []
        gt_boxes_3d_list = []
        gt_boxes_2d_pixel_list = []
        ignore_boxes_2d_list = []
        depth_gt_list = []
        depth_mask_list = []
        sample_names = []
        dataset_names = []
        original_hw_list = []
        padding_list = []

        # First pass: collect all unique class names across batch
        # boxes2d_names is a GLOBAL class name list per image (not per-box)
        all_class_names = set()
        for sample in batch:
            names = sample.get(K.boxes2d_names, sample.get("boxes2d_names", []))
            if isinstance(names, (list, tuple)):
                all_class_names.update(names)

        class_names = sorted(all_class_names)
        name_to_idx = {name: i for i, name in enumerate(class_names)}

        # Second pass: build batched data
        for sample in batch:
            # Image: (3, H, W)
            img = sample[K.images]
            if isinstance(img, torch.Tensor):
                if img.dim() == 4 and img.shape[0] == 1:
                    img = img.squeeze(0)
            images_list.append(img)

            # Intrinsics: (3, 3)
            intrinsics_list.append(
                torch.as_tensor(sample[K.intrinsics], dtype=torch.float32)
            )

            # 2D boxes: (N, 4) pixel xyxy
            boxes_2d = sample[K.boxes2d]
            if not isinstance(boxes_2d, torch.Tensor):
                boxes_2d = torch.as_tensor(boxes_2d, dtype=torch.float32)

            # boxes2d_names is a GLOBAL class name list (not per-box!)
            # boxes2d_classes is per-box class indices into boxes2d_names
            global_names = sample.get(
                K.boxes2d_names, sample.get("boxes2d_names", [])
            )
            box_class_ids = sample.get(
                K.boxes2d_classes, sample.get("boxes2d_classes")
            )
            if not isinstance(box_class_ids, torch.Tensor):
                box_class_ids = torch.as_tensor(box_class_ids, dtype=torch.long)

            H, W = img.shape[-2:]
            n_boxes = len(boxes_2d)

            if n_boxes > 0:
                # Convert pixel xyxy → normalized cxcywh for GLEE targets
                x1 = boxes_2d[:, 0] / W
                y1 = boxes_2d[:, 1] / H
                x2 = boxes_2d[:, 2] / W
                y2 = boxes_2d[:, 3] / H
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w = (x2 - x1).clamp(min=1e-6)
                h = (y2 - y1).clamp(min=1e-6)
                boxes_cxcywh = torch.stack([cx, cy, w, h], dim=-1)

                # Map per-box class IDs → batch-global label indices
                per_box_names = [global_names[int(cid)] for cid in box_class_ids]
                labels = torch.tensor(
                    [name_to_idx[n] for n in per_box_names],
                    dtype=torch.long,
                )

                target = {"labels": labels, "boxes": boxes_cxcywh}
            else:
                target = {
                    "labels": torch.zeros(0, dtype=torch.long),
                    "boxes": torch.zeros(0, 4),
                }

            targets_list.append(target)

            # GT 2D boxes in pixel xyxy (for 3D loss encoding)
            gt_boxes_2d_pixel_list.append(
                boxes_2d.float() if n_boxes > 0
                else torch.zeros(0, 4)
            )

            # 3D boxes
            boxes_3d = sample.get(K.boxes3d)
            if boxes_3d is not None and not isinstance(boxes_3d, torch.Tensor):
                boxes_3d = torch.as_tensor(boxes_3d, dtype=torch.float32)

            if boxes_3d is not None and len(boxes_3d) > 0:
                # boxes3d format: [cx,cy,cz,w,l,h,qw,qx,qy,qz] (10 or 11 dims)
                gt_boxes_3d_list.append(boxes_3d[:, :10].float())
            else:
                gt_boxes_3d_list.append(
                    torch.zeros(max(n_boxes, 0), 10)
                )

            # Depth maps
            depth = sample.get(K.depth_maps)
            if depth is not None:
                if not isinstance(depth, torch.Tensor):
                    depth = torch.as_tensor(depth, dtype=torch.float32)
                if depth.dim() == 2:
                    depth = depth.unsqueeze(0)  # (1, H, W)
                depth_gt_list.append(depth)
                # Depth mask: valid where depth > 0
                depth_mask_list.append((depth > 0).float())
            else:
                depth_gt_list.append(None)
                depth_mask_list.append(None)

            # Metadata
            sample_names.append(
                sample.get(K.sample_names, sample.get("sample_names", ""))
            )
            dataset_names.append(
                sample.get("dataset_name", "")
            )
            original_hw_list.append(
                sample.get(K.original_hw, sample.get("original_hw"))
            )
            padding_list.append(sample.get("padding"))

            # Ignore boxes
            ign = sample.get("ignore_boxes2d")
            if ign is not None and not isinstance(ign, Tensor):
                ign = torch.as_tensor(ign, dtype=torch.float32)
            ignore_boxes_2d_list.append(
                ign if ign is not None else torch.zeros(0, 4)
            )

        # Stack tensors
        images = torch.stack(images_list, dim=0)
        intrinsics = torch.stack(intrinsics_list, dim=0)

        # Stack depth if all available
        depth_gt = None
        depth_mask = None
        if all(d is not None for d in depth_gt_list):
            depth_gt = torch.stack(depth_gt_list, dim=0)
            depth_mask = torch.stack(depth_mask_list, dim=0)

        # Generate visual prompts (box/point) for training
        # May also replace class_names with ["object"] for class-agnostic mode
        spatial_masks, visual_prompt_type, class_names = self._generate_visual_prompts(
            batch, images, gt_boxes_2d_pixel_list, class_names
        )

        # If class_names changed to ["object"], remap all target labels to 0
        if len(class_names) == 1 and class_names[0] == "object":
            for target in targets_list:
                target["labels"] = torch.zeros_like(target["labels"])

        return GLEE_3DBatchedInputs(
            images=images,
            intrinsics=intrinsics,
            class_names=class_names,
            spatial_masks=spatial_masks,
            visual_prompt_type=visual_prompt_type,
            targets=targets_list,
            gt_boxes_3d=gt_boxes_3d_list,
            gt_boxes_2d_pixel=gt_boxes_2d_pixel_list,
            ignore_boxes_2d=ignore_boxes_2d_list,
            depth_gt=depth_gt,
            depth_mask=depth_mask,
            sample_names=sample_names,
            dataset_name=dataset_names,
            original_hw=original_hw_list,
            padding=padding_list,
        )


    def _generate_visual_prompts(
        self,
        batch: list[dict],
        images: Tensor,
        gt_boxes_2d_pixel_list: list[Tensor],
        class_names: list[str],
    ) -> tuple[list[Tensor] | None, str, list[str]]:
        """Generate visual prompts (box/point) for training.

        Per-batch decision: visual_prompt_prob chance of adding visual prompts.
        If yes, every image in batch gets a prompt:
          - Box prompt: random GT box + jitter → filled rect mask
          - Point prompt: only if mask data available, sample from GT mask

        Also handles class-agnostic mode: with object_text_prob, replaces
        class_names with ["object"] (like GLEE's SA1B training).

        Args:
            batch: original per-sample dicts (for mask data access)
            images: (B, 3, H, W) stacked images
            gt_boxes_2d_pixel_list: per-image GT boxes in pixel xyxy

        Returns:
            (spatial_masks, visual_prompt_type, class_names)
            class_names may be replaced with ["object"] for class-agnostic training.
        """
        import numpy as np

        # Per-batch decision
        if np.random.rand() > self.visual_prompt_prob:
            return None, "box", class_names

        B, _, H, W = images.shape
        spatial_masks = []
        prompt_type = "box"  # default

        for i in range(B):
            boxes = gt_boxes_2d_pixel_list[i]
            if len(boxes) == 0:
                # No GT boxes → empty mask
                spatial_masks.append(torch.zeros(1, H, W))
                continue

            # Check if mask data available (for point prompt)
            masks_2d = batch[i].get(K.instance_masks, batch[i].get("masks2d")) if i < len(batch) else None
            has_mask = masks_2d is not None and len(masks_2d) > 0

            if has_mask and np.random.rand() < 0.5:
                # Point prompt: sample from GT mask
                prompt_type = "point"
                idx = np.random.randint(len(masks_2d))
                mask = masks_2d[idx]
                if not isinstance(mask, torch.Tensor):
                    mask = torch.as_tensor(mask)
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)  # (1, H, W)
                # Sample a random point from mask
                nonzero = mask[0].nonzero()  # (N, 2) = (y, x)
                if len(nonzero) > 0:
                    pt_idx = np.random.randint(len(nonzero))
                    py, px = nonzero[pt_idx].tolist()
                    # Create small rectangle around point
                    ph, pw_ = H // 40, W // 40
                    point_mask = torch.zeros(1, H, W)
                    point_mask[0,
                               max(0, py - ph):min(H, py + ph),
                               max(0, px - pw_):min(W, px + pw_)] = 1.0
                    spatial_masks.append(point_mask)
                else:
                    spatial_masks.append(torch.zeros(1, H, W))
            else:
                # Box prompt: random GT box + jitter
                prompt_type = "box"
                idx = np.random.randint(len(boxes))
                box = boxes[idx].numpy() if isinstance(boxes[idx], torch.Tensor) else boxes[idx]
                if boxes.dim() == 2:
                    box = boxes[idx].numpy()

                # Add noise (same as SAM3_3D's noise_box)
                box_np = np.array(box, dtype=np.float32)
                if self.box_noise_std > 0:
                    noise = self.box_noise_std * np.random.randn(4)
                    bw, bh = box_np[2] - box_np[0], box_np[3] - box_np[1]
                    noise *= np.array([bw, bh, bw, bh])
                    box_np = box_np + noise
                    box_np = np.clip(box_np, 0, [W, H, W, H])

                x1, y1, x2, y2 = box_np.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                box_mask = torch.zeros(1, H, W)
                if y2 > y1 and x2 > x1:
                    box_mask[0, y1:y2, x1:x2] = 1.0
                spatial_masks.append(box_mask)

        # Class-agnostic mode: replace class names with ["object"]
        # Like GLEE's SA1B training where all objects are class "object"
        use_object_text = np.random.rand() < self.object_text_prob

        if use_object_text:
            class_names = ["object"]

        return spatial_masks, prompt_type, class_names


class GLEE_3DPassthroughConnector:
    """Train connector: passes GLEE_3DBatchedInputs directly to model.

    vis4d calls: self.model(**self.train_data_connector(batch))
    So we return {"batch": batch} which maps to model.forward(batch=batch).
    """

    def __call__(self, data) -> dict:
        return {"batch": data}


class GLEE_3DLossConnector:
    """Connects model output + batch data to loss function inputs."""

    def __call__(
        self,
        prediction: dict,
        batch: GLEE_3DBatchedInputs,
    ) -> dict:
        out = prediction  # GLEE_3DOut (NamedTuple, dict-like)
        H, W = batch.images.shape[-2:]

        return {
            "pred_logits": out["pred_logits"],
            "pred_scores": out["pred_scores"],
            "pred_boxes_2d": out["pred_boxes_2d"],
            "pred_boxes_3d": out["pred_boxes_3d"],
            "pred_conf_3d": out["pred_conf_3d"],
            "pred_boxes_3d_dn": out["pred_boxes_3d_dn"],
            "aux_outputs": out["aux_outputs"],
            "geom_losses": out["geom_losses"],
            "mask_dict": out["mask_dict"],
            "targets": batch.targets,
            "gt_boxes_3d": batch.gt_boxes_3d,
            "gt_boxes_2d_pixel": batch.gt_boxes_2d_pixel,
            "intrinsics": batch.intrinsics,
            "ignore_boxes_2d": getattr(batch, "ignore_boxes_2d", None),
            "image_size": (H, W),
        }
