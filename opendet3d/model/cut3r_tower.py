"""
CUT3R Tower for extracting 3D geometric features.
Adapted from VLM-3R's implementation - EXACTLY THE SAME.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import sys
import os

# Add CUT3R to path
cut3r_path = os.path.join(os.path.dirname(__file__), '../../CUT3R')
if cut3r_path not in sys.path:
    sys.path.insert(0, cut3r_path)

from src.dust3r.model import ARCroco3DStereo


def prepare_input(pixel_values):
    """
    Prepare input for CUT3R model.
    Fixed to match CUT3R's expected input size.

    Args:
        pixel_values: [B, C, H, W] tensor

    Returns:
        views: List of view dictionaries
    """
    # Resize to 512x512 (MATCH CUT3R MODEL'S TRAINING SIZE)
    # The CUT3R checkpoint was trained on 512x512 images, not 432x432
    pixel_values = F.interpolate(pixel_values, size=(512, 512), mode='bilinear')

    # Add frame dimension: [B, C, H, W] -> [1, B, C, H, W] (SAME AS VLM-3R)
    pixel_values = pixel_values.unsqueeze(1)  # Note: unsqueeze(1) not unsqueeze(0)

    # Check shape
    if not isinstance(pixel_values, torch.Tensor) or pixel_values.ndim != 5:
        raise ValueError(f"Expected pixel_values to be a 5D tensor (F, B, C, H, W), got {type(pixel_values)} with shape {getattr(pixel_values, 'shape', 'N/A')}")

    F_max, B, C, H, W = pixel_values.shape
    device = pixel_values.device

    views = []
    for i in range(F_max):
        current_frame_batch = pixel_values[i]  # Shape (B, C, H, W)
        view = {
            "img": current_frame_batch,
            "ray_map": torch.full(
                (B, 6, H, W),
                torch.nan,
            ).to(device),
            "true_shape": torch.tensor([H, W], device=device).expand(B, -1),  # Shape (B, 2)
            "idx": i,
            "instance": [str(j) for j in range(B)],  # List of B instances
            "camera_pose": torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1),  # Shape (B, 4, 4)
            "img_mask": torch.tensor(True, device=device).expand(B),  # Shape (B)
            "ray_mask": torch.tensor(False, device=device).expand(B),  # Shape (B)
            "update": torch.tensor(True, device=device).expand(B),  # Shape (B) - VLM-3R adds this
            "reset": torch.tensor(False, device=device).expand(B),  # Shape (B)
        }
        views.append(view)

    return views


class CUT3RTower(nn.Module):
    """
    CUT3R Tower for extracting 3D geometric features.
    EXACTLY copied from VLM-3R's Cut3rEncoder.

    This module wraps the pretrained CUT3R model (512x512 input) and extracts:
    - Camera tokens: Global 3D scene representation [B, 1, 1024]
    - Patch tokens: Local 3D geometric features [B, 1024, 1024] (32x32 grid)

    Args:
        cut3r_checkpoint: Path to CUT3R checkpoint
        freeze: Whether to freeze CUT3R weights (default: True)
    """

    def __init__(
        self,
        cut3r_checkpoint: str,
        freeze: bool = True
    ):
        super().__init__()

        # Load CUT3R model (SAME AS VLM-3R)
        self.cut3r = ARCroco3DStereo.from_pretrained(cut3r_checkpoint)

        # NOTE: If loading a unified checkpoint (3D-MOOD + CUT3R fusion trained together),
        # the weights here will be overwritten by load_state_dict() later.
        # This is intentional - we need to initialize the architecture first.

        # Freeze weights if specified (SAME AS VLM-3R)
        if freeze:
            self.cut3r.eval()
            for param in self.cut3r.parameters():
                param.requires_grad = False

        self.freeze = freeze

        # Output dimensions (for 432x432 input: 27x27 patches, SAME AS VLM-3R)
        # Note: CUT3R model actually outputs 768-dim features (dec_embed_dim=768)
        self.camera_token_dim = 768
        self.patch_token_dim = 768
        self.num_patch_tokens = 729  # 27x27 grid

    def forward(
        self,
        pixel_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract 3D features from images using CUT3R.
        EXACTLY from VLM-3R's Cut3rEncoder.forward

        Args:
            pixel_values: [B, 3, H, W] input images

        Returns:
            camera_tokens: [B, 1, 768] global 3D scene representation
            patch_tokens: [B, 729, 768] local 3D geometric features (27x27 grid)
        """
        # Prepare input (SAME AS VLM-3R)
        views = prepare_input(pixel_values=pixel_values)

        # Encode views (SAME AS VLM-3R)
        shape, feat_ls, pos = self.cut3r._encode_views(views)
        feat = feat_ls[-1]

        # Initialize state (SAME AS VLM-3R)
        state_feat, state_pos = self.cut3r._init_state(feat[0], pos[0])
        mem = self.cut3r.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()

        # Process each view (SAME AS VLM-3R)
        patch_features = []
        camera_tokens = []

        for i in range(len(views)):
            feat_i = feat[i].to(pixel_values.dtype)
            pos_i = pos[i]

            # Get pose features (SAME AS VLM-3R)
            if self.cut3r.pose_head_flag:
                global_img_feat_i = self.cut3r._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.cut3r.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.cut3r.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None

            # Recurrent rollout (SAME AS VLM-3R)
            new_state_feat, dec = self.cut3r._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=views[i]["img_mask"],
                reset_mask=views[i]["reset"],
                update=views[i].get("update", None),
            )

            # Update memory (SAME AS VLM-3R)
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.cut3r.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )

            # Extract features (SAME AS VLM-3R)
            # add camera token
            camera_tokens.append(dec[-1][:, :1].clone())
            # add patch features
            patch_features.append(dec[-1][:, 1:].clone())

            # Update state (SAME AS VLM-3R)
            img_mask = views[i]["img_mask"]
            update = views[i].get("update", None)
            if update is not None:
                update_mask = (img_mask & update)
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].to(pixel_values.dtype)

            state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)
            mem = new_mem * update_mask + mem * (1 - update_mask)

            reset_mask = views[i]["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].to(pixel_values.dtype)
                state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
                mem = init_mem * reset_mask + mem * (1 - reset_mask)

        # Stack and rearrange (SAME AS VLM-3R)
        patch_features = torch.stack(patch_features, dim=0)  # [frame, batch, token_num, token_dim]
        # VLM-3R uses: rearrange(patch_features, 'frame batch token_num token_dim -> (batch frame) token_num token_dim')
        # Since we only have 1 frame, this simplifies to: [1, B, 729, 768] -> [B, 729, 768]
        patch_features = patch_features.squeeze(0)  # Remove frame dimension

        camera_tokens = torch.stack(camera_tokens, dim=0)  # [frame, batch, token_num, token_dim]
        # VLM-3R uses: rearrange(camera_tokens, 'frame batch token_num token_dim -> (batch frame) token_num token_dim')
        # Since we only have 1 frame: [1, B, 1, 768] -> [B, 1, 768]
        camera_tokens = camera_tokens.squeeze(0)  # Remove frame dimension

        return camera_tokens, patch_features
