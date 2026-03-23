"""Spatial transforms for per-box binary masks (masks2d).

masks2d is a list (per image in batch) of (N, H, W) uint8 arrays,
where N is the number of boxes in that image. Each mask slice is a
binary mask for one box. These transforms keep masks aligned with
images, boxes2d, and depth_maps through the spatial augmentation
pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from vis4d.data.transforms.base import Transform

MASKS2D_KEY = "masks2d"


@Transform(
    [MASKS2D_KEY, "transforms.resize.target_shape"],
    MASKS2D_KEY,
)
class ResizeMasks2D:
    """Resize per-box masks using nearest interpolation."""

    def __call__(
        self,
        masks_list,
        target_shapes,
    ):
        """Resize masks."""
        if masks_list is None:
            return masks_list
        for i, (masks, target_shape) in enumerate(
            zip(masks_list, target_shapes)
        ):
            if masks is None or len(masks) == 0:
                continue
            # masks: (N, H, W) uint8
            t = torch.from_numpy(masks).float().unsqueeze(1)  # (N,1,H,W)
            t = F.interpolate(
                t, size=target_shape, mode="nearest"
            )
            masks_list[i] = (
                t.squeeze(1).to(torch.uint8).numpy()
            )  # (N, H', W')
        return masks_list


@Transform([MASKS2D_KEY, "transforms.crop.crop_box"], MASKS2D_KEY)
class CropMasks2D:
    """Crop per-box masks."""

    def __call__(
        self,
        masks_list,
        crop_box_list,
    ):
        """Crop masks."""
        if masks_list is None:
            return masks_list
        for i, (masks, crop_box) in enumerate(
            zip(masks_list, crop_box_list)
        ):
            if masks is None or len(masks) == 0:
                continue
            x1, y1, x2, y2 = crop_box
            masks_list[i] = masks[:, y1:y2, x1:x2]
        return masks_list


@Transform(MASKS2D_KEY, MASKS2D_KEY)
class FlipMasks2D:
    """Flip per-box masks horizontally."""

    def __call__(
        self,
        masks_list,
    ):
        """Flip masks."""
        if masks_list is None:
            return masks_list
        for i, masks in enumerate(masks_list):
            if masks is None or len(masks) == 0:
                continue
            masks_list[i] = np.ascontiguousarray(
                masks[:, :, ::-1]
            )
        return masks_list


@Transform([MASKS2D_KEY, "transforms.pad"], MASKS2D_KEY)
class CenterPadMasks2D:
    """Center-pad per-box masks."""

    def __call__(
        self,
        masks_list,
        pad_params,
    ):
        """Pad masks."""
        if masks_list is None:
            return masks_list
        for i, (masks, pad_param) in enumerate(
            zip(masks_list, pad_params)
        ):
            if masks is None or len(masks) == 0:
                continue
            pad = (
                pad_param["pad_left"],
                pad_param["pad_right"],
                pad_param["pad_top"],
                pad_param["pad_bottom"],
            )
            t = torch.from_numpy(masks).unsqueeze(1)  # (N,1,H,W)
            t = F.pad(t, pad, mode="constant", value=0)
            masks_list[i] = t.squeeze(1).numpy()  # (N, H', W')
        return masks_list
