"""3D bounding box encoder."""

from __future__ import annotations

import torch
from torch import Tensor
from vis4d.data.const import AxisMode
from vis4d.op.geometry.projection import project_points, unproject_points
from vis4d.op.geometry.rotation import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotation_matrix_yaw,
)

from opendet3d.op.geometric.rotation import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


def _normalize_rotation_half(poses: Tensor) -> Tensor:
    """Normalize rotation matrices to [0, pi) yaw range.

    For objects with 180-degree rotational ambiguity (e.g. tables, chairs),
    this folds yaw into [0, pi) so that 90 and 270 map to the same target.
    Also handles boundary: 180 and 0 map to the same target.

    Uses Y-axis rotation (OPENCV convention) to detect and flip.
    """
    import math

    yaw = rotation_matrix_yaw(
        poses, axis_mode=AxisMode.OPENCV
    )[:, 1]  # [N]
    # Flip by 180 around Y-axis: Ry(pi) = diag(-1, 1, -1)
    # yaw in [-pi, 0) or yaw ~= pi -> flip to [0, pi)
    flip_mask = (yaw < 0) | (yaw > math.pi - 1e-4)
    poses_out = poses.clone()
    # R_new = R @ Ry(pi), Ry(pi) negates columns 0 and 2
    poses_out[flip_mask, :, 0] = -poses[flip_mask, :, 0]
    poses_out[flip_mask, :, 2] = -poses[flip_mask, :, 2]
    return poses_out


class GroundingDINO3DCoder:
    """Grounding DINO 3D box Coder."""

    def __init__(
        self,
        center_scale: float = 10.0,
        depth_scale: float = 2.0,
        dim_scale: float = 2.0,
        orientation: str = "rotation_6d",
        ambiguous_rotation: bool = False,
    ) -> None:
        """Initialize the Grounding DINO 3D encoder."""
        self.center_scale = center_scale
        self.depth_scale = depth_scale
        self.dim_scale = dim_scale
        self.ambiguous_rotation = ambiguous_rotation
        if ambiguous_rotation:
            print(
                "[GroundingDINO3DCoder] ambiguous_rotation=True: "
                "GT rotation normalized to [0, 180) yaw range"
            )

        assert orientation in {
            "yaw",
            "rotation_6d",
        }, f"Invalid orientation {orientation}."
        self.orientation = orientation

        if orientation == "yaw":
            reg_dims = 8
        elif orientation == "rotation_6d":
            reg_dims = 12

        self.reg_dims = reg_dims

    def encode(
        self, boxes: Tensor, boxes3d: Tensor, intrinsics: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encode the 3D bounding boxes.

        Args:
            boxes: 2D boxes in PIXEL xyxy format. Shape (N, 4).
                   IMPORTANT: Should be GT 2D boxes during training (not predictions!)
                   This ensures stable targets. At inference, decode() uses pred boxes.
            boxes3d: GT 3D boxes [center_3d(3), dims(3), quat(4)]. Shape (N, 10).
            intrinsics: Camera intrinsics. Shape (3, 3) or (N, 3, 3).

        Returns:
            boxes3d_target: Encoded targets [delta_2d(2), log_depth(1), log_dims(3), rot_6d(6)].
            boxes3d_weights: Per-element weights (0 for invalid depth/dims).
        """
        projected_center_3d = project_points(boxes3d[:, :3], intrinsics)
        ctr_x = (boxes[:, 0] + boxes[:, 2]) / 2
        ctr_y = (boxes[:, 1] + boxes[:, 3]) / 2
        center_2d = torch.stack([ctr_x, ctr_y], -1)

        delta_center = projected_center_3d - center_2d

        delta_center /= self.center_scale

        valid_depth = boxes3d[:, 2] > 0

        depth = torch.where(
            valid_depth,
            torch.log(boxes3d[:, 2]) * self.depth_scale,
            boxes3d[:, 2].new_zeros(1),
        )
        depth = depth.unsqueeze(-1)

        valid_dims = boxes3d[:, 3:6] > 0
        dims = torch.where(
            valid_dims,
            torch.log(boxes3d[:, 3:6]) * self.dim_scale,
            boxes3d[:, 2].new_zeros(1),
        )

        poses = quaternion_to_matrix(boxes3d[:, 6:])

        if self.ambiguous_rotation:
            poses = _normalize_rotation_half(poses)

        if self.orientation == "yaw":
            yaw = rotation_matrix_yaw(
                poses,
                axis_mode=AxisMode.OPENCV,
            )[:, 1]

            sin_yaw = torch.sin(yaw).unsqueeze(-1)
            cos_yaw = torch.cos(yaw).unsqueeze(-1)

            boxes3d_target = torch.cat(
                [delta_center, depth, dims, sin_yaw, cos_yaw], -1
            )
        elif self.orientation == "rotation_6d":
            rot_6d = matrix_to_rotation_6d(poses)

            boxes3d_target = torch.cat([delta_center, depth, dims, rot_6d], -1)

        boxes3d_weights = torch.ones_like(boxes3d_target)
        boxes3d_weights[:, 2] = valid_depth.float()
        boxes3d_weights[:, 3:6] = valid_dims.float()

        return boxes3d_target, boxes3d_weights

    def decode(
        self, boxes: Tensor, boxes3d: Tensor, intrinsics: Tensor
    ) -> Tensor:
        """Decode the 3D bounding boxes."""
        delta_center = boxes3d[:, :2] * self.center_scale

        ctr_x = (boxes[:, 0] + boxes[:, 2]) / 2
        ctr_y = (boxes[:, 1] + boxes[:, 3]) / 2
        center_2d = torch.stack([ctr_x, ctr_y], -1)

        proj_center_3d = center_2d + delta_center

        depth = torch.exp(boxes3d[:, 2] / self.depth_scale)

        center_3d = unproject_points(proj_center_3d, depth, intrinsics)

        dims = torch.exp(boxes3d[:, 3:6] / self.dim_scale)

        if self.orientation == "yaw":
            yaw = torch.atan2(boxes3d[:, 6], boxes3d[:, 7])

            orientation = torch.stack(
                [torch.zeros_like(yaw), yaw, torch.zeros_like(yaw)], -1
            )

            orientation = matrix_to_quaternion(
                euler_angles_to_matrix(orientation)
            )
        elif self.orientation == "rotation_6d":
            poses = rotation_6d_to_matrix(boxes3d[:, 6:])

            if self.ambiguous_rotation:
                poses = _normalize_rotation_half(poses)

            orientation = matrix_to_quaternion(poses)

        return torch.cat([center_3d, dims, orientation], dim=1)
