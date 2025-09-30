"""
Deformable Gated Fusion for CUT3R features.
Lightweight version using deformable attention for low/mid-level features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class DeformableGatedFusion(nn.Module):
    """
    Deformable Gated Fusion module.
    
    Uses deformable attention to efficiently sample from CUT3R features.
    Only samples n_points (e.g., 2 or 4) instead of attending to all 730 tokens.
    
    Args:
        d_model: Visual feature dimension (e.g., 96, 192, 384)
        d_cut3r: CUT3R feature dimension (1024)
        n_heads: Number of attention heads (default: 8)
        n_points: Number of sampling points per head (default: 4)
        dropout: Dropout probability (default: 0.1)
        use_relative_pos_bias: Whether to use relative position bias (default: False)
    """
    
    def __init__(
        self,
        d_model: int,
        d_cut3r: int = 1024,
        n_heads: int = 8,
        n_points: int = 4,
        dropout: float = 0.1,
        use_relative_pos_bias: bool = False
    ):
        super().__init__()
        
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.use_relative_pos_bias = use_relative_pos_bias
        
        # ========== 1. Layer Normalization ==========
        self.norm_visual = nn.LayerNorm(d_model)
        self.norm_cut3r = nn.LayerNorm(d_cut3r)
        
        # ========== 2. CUT3R Projection ==========
        self.cut3r_proj = nn.Sequential(
            nn.Linear(d_cut3r, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # ========== 3. Deformable Attention Components ==========
        # Sampling offsets: predict where to sample from CUT3R features
        # Output: [n_heads, n_points, 2] for each query
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        
        # Attention weights: predict importance of each sampled point
        # Output: [n_heads, n_points] for each query
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        
        # Value projection
        self.value_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
        # ========== 4. Feed-Forward Network ==========
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # ========== 5. Gate Network ==========
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate[-2].bias, -5.0)
        
        # ========== 6. Output Normalization ==========
        self.norm_out = nn.LayerNorm(d_model)
        
        # ========== 7. Initialize Parameters ==========
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters for deformable attention."""
        # Initialize sampling offsets to 0
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        
        # Initialize sampling offset biases to form a grid
        # This creates initial sampling points in a circular pattern
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.n_heads, 1, 2).repeat(1, self.n_points, 1)
        
        # Scale down the initial offsets
        for i in range(self.n_points):
            grid_init[:, i, :] *= (i + 1) / self.n_points
        
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        # Initialize attention weights
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        
        # Initialize projections
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)
    
    def forward(
        self,
        visual_feat: torch.Tensor,
        cut3r_features: torch.Tensor,
        pos_bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with deformable gated fusion.
        
        Args:
            visual_feat: [B, C, H, W] visual features
            cut3r_features: [B, 730, 1024] CUT3R features
            pos_bias: Optional position bias (not used in deformable version)
        
        Returns:
            fused_feat: [B, C, H, W] fused features
            gate_values: [B, H*W, 1] gate values
        """
        B, C, H, W = visual_feat.shape
        
        # ========== Step 1: Flatten Visual Features ==========
        visual_flat = visual_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
        visual_norm = self.norm_visual(visual_flat)
        
        # ========== Step 2: Project CUT3R Features ==========
        cut3r_norm = self.norm_cut3r(cut3r_features)  # [B, 730, 1024]
        cut3r_proj = self.cut3r_proj(cut3r_norm)      # [B, 730, C]
        
        # ========== Step 3: Deformable Attention ==========
        # Reshape CUT3R to 2D grid (27x27 for patch tokens + 1 camera token)
        # We'll treat it as a 27x27 grid and ignore the camera token for sampling
        cut3r_patches = cut3r_proj[:, 1:, :]  # [B, 729, C] (skip camera token)
        cut3r_grid = cut3r_patches.reshape(B, 27, 27, C).permute(0, 3, 1, 2)  # [B, C, 27, 27]
        
        # Predict sampling offsets and attention weights
        sampling_offsets = self.sampling_offsets(visual_norm)  # [B, H*W, n_heads*n_points*2]
        sampling_offsets = sampling_offsets.view(B, H * W, self.n_heads, self.n_points, 2)
        
        attention_weights = self.attention_weights(visual_norm)  # [B, H*W, n_heads*n_points]
        attention_weights = attention_weights.view(B, H * W, self.n_heads, self.n_points)
        attention_weights = F.softmax(attention_weights, dim=-1)  # Normalize across points
        
        # Perform deformable sampling
        attn_out = self._deformable_sampling(
            cut3r_grid,
            sampling_offsets,
            attention_weights,
            H, W
        )  # [B, H*W, C]
        
        # ========== Step 4: Feed-Forward Network ==========
        ffn_out = self.ffn(attn_out)  # [B, H*W, C]
        
        # ========== Step 5: Gated Residual Connection ==========
        gate_values = self.gate(visual_flat)  # [B, H*W, 1]
        fused = visual_flat + gate_values * ffn_out  # [B, H*W, C]
        
        # ========== Step 6: Output Normalization ==========
        fused = self.norm_out(fused)
        
        # ========== Step 7: Reshape Back ==========
        fused = fused.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        
        return fused, gate_values
    
    def _deformable_sampling(
        self,
        value: torch.Tensor,
        sampling_offsets: torch.Tensor,
        attention_weights: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        Perform deformable sampling from CUT3R features.
        
        Args:
            value: [B, C, 27, 27] CUT3R features as 2D grid
            sampling_offsets: [B, H*W, n_heads, n_points, 2] sampling offsets
            attention_weights: [B, H*W, n_heads, n_points] attention weights
            H, W: Spatial dimensions of query
        
        Returns:
            output: [B, H*W, C] sampled and aggregated features
        """
        B, C, H_kv, W_kv = value.shape
        N_q = H * W
        
        # Normalize offsets to [-1, 1] range for grid_sample
        # sampling_offsets are in normalized coordinates [0, 1]
        # We need to convert to [-1, 1] for grid_sample
        sampling_locations = sampling_offsets.sigmoid()  # [B, N_q, n_heads, n_points, 2]
        sampling_locations = 2.0 * sampling_locations - 1.0  # Convert to [-1, 1]
        
        # Reshape for grid_sample
        # [B, N_q, n_heads, n_points, 2] -> [B*n_heads, N_q, n_points, 2]
        sampling_locations = sampling_locations.permute(0, 2, 1, 3, 4)  # [B, n_heads, N_q, n_points, 2]
        sampling_locations = sampling_locations.reshape(B * self.n_heads, N_q, self.n_points, 2)
        
        # Expand value for each head
        # [B, C, H_kv, W_kv] -> [B, n_heads, C//n_heads, H_kv, W_kv] -> [B*n_heads, C//n_heads, H_kv, W_kv]
        value_per_head = value.view(B, self.n_heads, C // self.n_heads, H_kv, W_kv)
        value_per_head = value_per_head.reshape(B * self.n_heads, C // self.n_heads, H_kv, W_kv)
        
        # Sample features using grid_sample
        # [B*n_heads, C//n_heads, N_q, n_points]
        sampled_features = F.grid_sample(
            value_per_head,
            sampling_locations,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )
        
        # Reshape sampled features
        # [B*n_heads, C//n_heads, N_q, n_points] -> [B, n_heads, C//n_heads, N_q, n_points]
        sampled_features = sampled_features.view(B, self.n_heads, C // self.n_heads, N_q, self.n_points)

        # Apply attention weights
        # attention_weights: [B, N_q, n_heads, n_points] -> [B, n_heads, N_q, n_points]
        attention_weights = attention_weights.permute(0, 2, 1, 3)  # [B, n_heads, N_q, n_points]

        # Weighted sum over sampling points
        # sampled_features: [B, n_heads, C//n_heads, N_q, n_points]
        # attention_weights: [B, n_heads, N_q, n_points]
        # We need to broadcast attention_weights to match sampled_features
        attention_weights = attention_weights.unsqueeze(2)  # [B, n_heads, 1, N_q, n_points]

        # Weighted sum: [B, n_heads, C//n_heads, N_q, n_points] * [B, n_heads, 1, N_q, n_points]
        # -> [B, n_heads, C//n_heads, N_q]
        output = (sampled_features * attention_weights).sum(dim=-1)

        # Reshape to [B, N_q, C]
        # [B, n_heads, C//n_heads, N_q] -> [B, N_q, n_heads, C//n_heads] -> [B, N_q, C]
        output = output.permute(0, 3, 1, 2).reshape(B, N_q, C)
        
        # Apply output projection
        output = self.output_proj(output)
        
        return output

