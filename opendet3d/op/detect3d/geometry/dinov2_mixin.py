"""Mixin for geometry backends that use DINOv2 encoder."""

import torch
import torch.nn.functional as F
from torch import Tensor


class DINOv2Mixin:
    """
    Mixin for geometry backends that use DINOv2 encoder.

    Provides common functionality:
    - Image normalization (ImageNet stats)
    - Patch size handling (resize to 14x14 multiples)
    - Intrinsic adjustment when resizing
    - Depth map resizing back to original dimensions
    """

    DINOV2_PATCH_SIZE = 14
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def _normalize_image_for_dinov2(self, images: Tensor) -> Tensor:
        """
        Normalize images for DINOv2 using ImageNet statistics.

        Args:
            images: [B, 3, H, W] in [0, 255] range

        Returns:
            Normalized images [B, 3, H, W] in approximately [-2.5, 2.5] range
        """
        # Convert to [0, 1]
        images = images / 255.0

        # Normalize with ImageNet stats
        mean = torch.tensor(
            self.IMAGENET_MEAN, device=images.device, dtype=images.dtype
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            self.IMAGENET_STD, device=images.device, dtype=images.dtype
        ).view(1, 3, 1, 1)

        return (images - mean) / std

    def _make_divisible_by_dinov2_patch(
        self,
        images: Tensor,
        intrinsics: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Resize images to make dimensions divisible by DINOv2 patch size (14).
        Also adjusts camera intrinsics to match the new image dimensions.

        DINOv2 uses 14x14 patches, so input images must have dimensions that
        are multiples of 14. This method resizes images to the nearest multiple
        and adjusts the camera intrinsics accordingly.

        Args:
            images: [B, 3, H, W] input images
            intrinsics: [B, 3, 3] camera intrinsics matrix
                [[fx,  0, cx],
                 [ 0, fy, cy],
                 [ 0,  0,  1]]

        Returns:
            Tuple of (resized_images, adjusted_intrinsics)

        Example:
            Input image: 800x600
            Nearest multiples of 14: 798x602
            Scale factors: sx = 602/600, sy = 798/800

            Adjusted intrinsics:
            - fx' = fx * sx  (focal length x scaled by width ratio)
            - fy' = fy * sy  (focal length y scaled by height ratio)
            - cx' = cx * sx  (principal point x scaled by width ratio)
            - cy' = cy * sy  (principal point y scaled by height ratio)

        Mathematical Justification:
            Camera projection: u = fx * (X/Z) + cx, v = fy * (Y/Z) + cy
            When image is resized: u' = u * (W'/W), v' = v * (H'/H)
            To maintain projection: fx' = fx * (W'/W), cx' = cx * (W'/W)
                                    fy' = fy * (H'/H), cy' = cy * (H'/H)
        """
        B, C, H, W = images.shape

        # Calculate new dimensions (round to nearest multiple of 14)
        def nearest_multiple(x: int, p: int) -> int:
            """Round x to nearest multiple of p."""
            down = (x // p) * p
            up = down + p
            return up if abs(up - x) <= abs(x - down) else down

        new_H = max(
            self.DINOV2_PATCH_SIZE, nearest_multiple(H, self.DINOV2_PATCH_SIZE)
        )
        new_W = max(
            self.DINOV2_PATCH_SIZE, nearest_multiple(W, self.DINOV2_PATCH_SIZE)
        )

        # If already divisible, return as-is
        if new_H == H and new_W == W:
            return images, intrinsics

        # Resize images using bilinear interpolation
        images_resized = F.interpolate(
            images,
            size=(new_H, new_W),
            mode="bilinear",
            align_corners=False,
        )

        # Adjust camera intrinsics
        # When image is resized from (H, W) to (new_H, new_W):
        # - All x-coordinates scale by (new_W / W)
        # - All y-coordinates scale by (new_H / H)
        #
        # Intrinsic matrix transformation:
        # [[fx,  0, cx],      [[fx * sx,      0, cx * sx],
        #  [ 0, fy, cy],  =>   [      0, fy * sy, cy * sy],
        #  [ 0,  0,  1]]       [      0,      0,       1]]
        #
        # where sx = new_W / W, sy = new_H / H

        intrinsics_adjusted = intrinsics.clone()

        # Scale x-related parameters (fx, cx) by width ratio
        scale_x = new_W / W
        intrinsics_adjusted[:, 0, 0] *= scale_x  # fx
        intrinsics_adjusted[:, 0, 2] *= scale_x  # cx

        # Scale y-related parameters (fy, cy) by height ratio
        scale_y = new_H / H
        intrinsics_adjusted[:, 1, 1] *= scale_y  # fy
        intrinsics_adjusted[:, 1, 2] *= scale_y  # cy

        # Note: intrinsics_adjusted[:, 2, 2] = 1 (unchanged)

        return images_resized, intrinsics_adjusted

    def _resize_depth_to_original(
        self,
        depth_map: Tensor,
        original_hw: tuple[int, int],
    ) -> Tensor:
        """
        Resize depth map back to original image dimensions.

        After processing with resized images, we need to resize the depth
        prediction back to the original image dimensions for loss computation
        and downstream tasks.

        Args:
            depth_map: [B, 1, H', W'] depth prediction
            original_hw: (H, W) original image size

        Returns:
            Resized depth map [B, 1, H, W]
        """
        _, _, H_curr, W_curr = depth_map.shape
        H_orig, W_orig = original_hw

        if H_curr == H_orig and W_curr == W_orig:
            return depth_map

        return F.interpolate(
            depth_map,
            size=(H_orig, W_orig),
            mode="bilinear",
            align_corners=False,
        )

