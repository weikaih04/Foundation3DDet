"""
CUT3R Tower for extracting 3D geometric features.
Adapted from VLM-3R's implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
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
    Adapted from VLM-3R's prepare_input function.
    
    Args:
        pixel_values: [B, C, H, W] tensor
    
    Returns:
        views: List of view dictionaries
    """
    # Resize to CUT3R input size (432x432)
    pixel_values = F.interpolate(pixel_values, size=(432, 432), mode='bilinear')
    
    # Add frame dimension: [B, C, H, W] -> [1, B, C, H, W]
    pixel_values = pixel_values.unsqueeze(0)
    
    views = []
    F_max, B, C, H, W = pixel_values.shape
    device = pixel_values.device
    
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
            "reset": torch.tensor(False, device=device).expand(B),  # Shape (B)
        }
        views.append(view)
    
    return views


class CUT3RTower(nn.Module):
    """
    CUT3R Tower for extracting 3D geometric features.
    
    This module wraps the pretrained CUT3R model and extracts:
    - Camera tokens: Global 3D scene representation [B, 1, 1024]
    - Patch tokens: Local 3D geometric features [B, 729, 1024] (27x27 grid)
    
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
        
        # Load CUT3R model
        print(f"Loading CUT3R from {cut3r_checkpoint}")
        self.cut3r = ARCroco3DStereo.from_pretrained(cut3r_checkpoint)
        
        # Freeze weights if specified
        if freeze:
            for param in self.cut3r.parameters():
                param.requires_grad = False
            self.cut3r.eval()
            print("CUT3R weights frozen")
        
        self.freeze = freeze
        
        # Output dimensions
        self.camera_token_dim = 1024
        self.patch_token_dim = 1024
        self.num_patch_tokens = 729  # 27x27 grid
    
    def forward(
        self,
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract 3D features from images using CUT3R.
        
        Args:
            images: [B, 3, H, W] input images
        
        Returns:
            camera_tokens: [B, 1, 1024] global 3D scene representation
            patch_tokens: [B, 729, 1024] local 3D geometric features (27x27 grid)
        """
        # Set to eval mode if frozen
        if self.freeze:
            self.cut3r.eval()
        
        # Prepare input
        views = prepare_input(images)
        
        # Forward through CUT3R (adapted from VLM-3R)
        with torch.set_grad_enabled(not self.freeze):
            # Encode views
            shape, feat_ls, pos = self.cut3r._encode_views(views)
            feat = feat_ls[-1]
            
            # Initialize state
            state_feat, state_pos = self.cut3r._init_state(feat[0], pos[0])
            mem = self.cut3r.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
            init_state_feat = state_feat.clone()
            init_mem = mem.clone()
            
            # Process each view (frame)
            camera_tokens_list = []
            patch_features_list = []
            
            for i in range(len(views)):
                feat_i = feat[i].to(images.dtype)
                pos_i = pos[i]
                
                # Get pose features
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
                
                # Recurrent rollout
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
                
                # Update memory
                out_pose_feat_i = dec[-1][:, 0:1]
                new_mem = self.cut3r.pose_retriever.update_mem(
                    mem, global_img_feat_i, out_pose_feat_i
                )
                
                # Extract camera token and patch features
                camera_tokens_list.append(dec[-1][:, :1].clone())  # [B, 1, 1024]
                patch_features_list.append(dec[-1][:, 1:].clone())  # [B, 729, 1024]
                
                # Update state
                img_mask = views[i]["img_mask"]
                update = views[i].get("update", None)
                if update is not None:
                    update_mask = (img_mask & update)
                else:
                    update_mask = img_mask
                update_mask = update_mask[:, None, None].to(images.dtype)
                
                state_feat = new_state_feat * update_mask + state_feat * (1 - update_mask)
                mem = new_mem * update_mask + mem * (1 - update_mask)
                
                reset_mask = views[i]["reset"]
                if reset_mask is not None:
                    reset_mask = reset_mask[:, None, None].to(images.dtype)
                    state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
                    mem = init_mem * reset_mask + mem * (1 - reset_mask)
        
        # Stack and rearrange
        # Since we only have 1 frame, just take the first one
        camera_tokens = camera_tokens_list[0]  # [B, 1, 1024]
        patch_tokens = patch_features_list[0]  # [B, 729, 1024]
        
        return camera_tokens, patch_tokens
    
    def train(self, mode: bool = True):
        """Override train to keep CUT3R in eval mode if frozen."""
        super().train(mode)
        if self.freeze:
            self.cut3r.eval()
        return self
