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


def _normalize_canonical(
    poses: Tensor, dims: Tensor,
) -> tuple[Tensor, Tensor]:
    """Normalize rotation and dimensions to canonical form.

    Eliminates OBB rotation ambiguity via 2 steps:

    Step 1 - Force W <= L:
        If W > L, swap W and L, then apply Ry(90 deg) to rotation.
        boxes3d dims = [W, L, H]. Canonical: X=L, Z=W, so swapping
        W<->L requires rotating 90 deg around Y to keep the box
        geometry identical.
        Ry(90): new_col0 = old_col2, new_col2 = -old_col0

    Step 2 - Normalize yaw to [0, pi):
        Same as _normalize_rotation_half. Apply Ry(180 deg) if yaw < 0
        or yaw >= pi.

    Together these reduce 4-fold Ry ambiguity to 1-fold.
    (Rx(180) upside-down ambiguity is left to data preprocessing.)

    Args:
        poses: Rotation matrices [N, 3, 3].
        dims: Dimensions [N, 3] as [W, L, H].

    Returns:
        poses_out: Normalized rotation matrices [N, 3, 3].
        dims_out: Normalized dimensions [N, 3] with W <= L.
    """
    import math

    poses_out = poses.clone()
    dims_out = dims.clone()

    # Step 1: Force W <= L
    # dims = [W, L, H], indices 0, 1, 2
    swap_mask = dims_out[:, 0] > dims_out[:, 1]  # W > L
    if swap_mask.any():
        # Swap W and L
        w_old = dims_out[swap_mask, 0].clone()
        dims_out[swap_mask, 0] = dims_out[swap_mask, 1]
        dims_out[swap_mask, 1] = w_old

        # Apply Ry(90 deg): R_new = R @ Ry(90)
        # Ry(90) = [[0,0,1],[0,1,0],[-1,0,0]]
        # col0_new = R @ [0,0,-1]^T = -col2
        # col1_new = R @ [0,1,0]^T  = col1 (unchanged)
        # col2_new = R @ [1,0,0]^T  = col0
        col0 = poses_out[swap_mask, :, 0].clone()
        col2 = poses_out[swap_mask, :, 2].clone()
        poses_out[swap_mask, :, 0] = -col2
        poses_out[swap_mask, :, 2] = col0

    # Step 2: Normalize yaw to [0, pi)
    yaw = rotation_matrix_yaw(
        poses_out, axis_mode=AxisMode.OPENCV
    )[:, 1]  # [N]
    flip_mask = (yaw < 0) | (yaw > math.pi - 1e-4)
    if flip_mask.any():
        # R_new = R @ Ry(pi), negates columns 0 and 2
        poses_out[flip_mask, :, 0] = -poses_out[flip_mask, :, 0]
        poses_out[flip_mask, :, 2] = -poses_out[flip_mask, :, 2]

    return poses_out, dims_out


class GroundingDINO3DCoder:
    """Grounding DINO 3D box Coder."""

    def __init__(
        self,
        center_scale: float = 10.0,
        depth_scale: float = 2.0,
        dim_scale: float = 2.0,
        orientation: str = "rotation_6d",
        ambiguous_rotation: bool = False,
        canonical_rotation: bool = False,
    ) -> None:
        """Initialize the Grounding DINO 3D encoder."""
        self.center_scale = center_scale
        self.depth_scale = depth_scale
        self.dim_scale = dim_scale
        self.ambiguous_rotation = ambiguous_rotation
        self.canonical_rotation = canonical_rotation
        if canonical_rotation:
            print(
                "[GroundingDINO3DCoder] canonical_rotation=True: "
                "dims normalized to W<=L, yaw to [0, 180)"
            )
        elif ambiguous_rotation:
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

        raw_dims = boxes3d[:, 3:6]  # [W, L, H]

        poses = quaternion_to_matrix(boxes3d[:, 6:])

        if self.canonical_rotation:
            poses, raw_dims = _normalize_canonical(poses, raw_dims)
        elif self.ambiguous_rotation:
            poses = _normalize_rotation_half(poses)

        valid_dims = raw_dims > 0
        dims = torch.where(
            valid_dims,
            torch.log(raw_dims) * self.dim_scale,
            raw_dims.new_zeros(1),
        )

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

            poses = euler_angles_to_matrix(orientation)
        elif self.orientation == "rotation_6d":
            poses = rotation_6d_to_matrix(boxes3d[:, 6:])

        if self.canonical_rotation:
            poses, dims = _normalize_canonical(poses, dims)
        elif self.ambiguous_rotation:
            poses = _normalize_rotation_half(poses)

        orientation = matrix_to_quaternion(poses)

        return torch.cat([center_3d, dims, orientation], dim=1)
