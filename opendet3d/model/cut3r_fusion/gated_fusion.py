"""
Gated Cross-Attention Fusion for CUT3R features.
Full attention version for high-level features (Level 3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GatedCUT3RFusion(nn.Module):
    """
    Gated Cross-Attention Fusion module.
    
    Uses full cross-attention to fuse visual features with CUT3R 3D features.
    Includes a gated residual connection initialized to ~0 for smooth training.
    
    Args:
        d_model: Visual feature dimension (e.g., 768 for Level 3)
        d_cut3r: CUT3R feature dimension (1024)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
        use_relative_pos_bias: Whether to use relative position bias (default: False)
    """
    
    def __init__(
        self,
        d_model: int = 768,
        d_cut3r: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_relative_pos_bias: bool = False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_relative_pos_bias = use_relative_pos_bias
        
        # ========== 1. Layer Normalization ==========
        self.norm_visual = nn.LayerNorm(d_model)
        self.norm_cut3r = nn.LayerNorm(d_cut3r)
        
        # ========== 2. CUT3R Projection ==========
        # Project CUT3R features from 1024 to d_model
        self.cut3r_proj = nn.Sequential(
            nn.Linear(d_cut3r, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # ========== 3. Cross-Attention ==========
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # ========== 4. Feed-Forward Network ==========
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        
        # ========== 5. Gate Network ==========
        # Gate initialized to ~0 (sigmoid(-5) ≈ 0.0067)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        # Initialize gate bias to -5 for smooth training start
        nn.init.constant_(self.gate[-2].bias, -5.0)
        
        # ========== 6. Output Normalization ==========
        self.norm_out = nn.LayerNorm(d_model)
    
    def forward(
        self,
        visual_feat: torch.Tensor,
        cut3r_features: torch.Tensor,
        pos_bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with gated cross-attention fusion.
        
        Args:
            visual_feat: [B, C, H, W] visual features
            cut3r_features: [B, 730, 1024] CUT3R features (1 camera + 729 patch tokens)
            pos_bias: [num_heads, H*W, 730] optional relative position bias
        
        Returns:
            fused_feat: [B, C, H, W] fused features
            attn_weights: [B, H*W, 730] attention weights (for visualization)
            gate_values: [B, H*W, 1] gate values (for monitoring)
        """
        B, C, H, W = visual_feat.shape
        
        # ========== Step 1: Flatten Visual Features ==========
        visual_flat = visual_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]
        visual_norm = self.norm_visual(visual_flat)
        
        # ========== Step 2: Project CUT3R Features ==========
        cut3r_norm = self.norm_cut3r(cut3r_features)  # [B, 730, 1024]
        cut3r_proj = self.cut3r_proj(cut3r_norm)      # [B, 730, C]
        
        # ========== Step 3: Cross-Attention ==========
        if pos_bias is not None and self.use_relative_pos_bias:
            # Manual attention with position bias
            attn_out, attn_weights = self._cross_attn_with_bias(
                visual_norm, cut3r_proj, pos_bias
            )
        else:
            # Standard cross-attention
            attn_out, attn_weights = self.cross_attn(
                query=visual_norm,
                key=cut3r_proj,
                value=cut3r_proj,
                need_weights=True,
                average_attn_weights=True
            )
        # attn_out: [B, H*W, C]
        # attn_weights: [B, H*W, 730]
        
        # ========== Step 4: Feed-Forward Network ==========
        ffn_out = self.ffn(attn_out)  # [B, H*W, C]
        
        # ========== Step 5: Gated Residual Connection ==========
        gate_values = self.gate(visual_flat)  # [B, H*W, 1]
        # Initial: gate ≈ 0.007, so fused ≈ visual_flat (smooth start)
        # After training: gate ≈ 0.5, so fused = visual + 0.5 * 3D_info
        fused = visual_flat + gate_values * ffn_out  # [B, H*W, C]
        
        # ========== Step 6: Output Normalization ==========
        fused = self.norm_out(fused)
        
        # ========== Step 7: Reshape Back ==========
        fused = fused.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        
        return fused, attn_weights, gate_values
    
    def _cross_attn_with_bias(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        pos_bias: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention with relative position bias.
        
        Args:
            query: [B, N_q, C]
            key: [B, N_kv, C]
            pos_bias: [num_heads, N_q, N_kv]
        
        Returns:
            output: [B, N_q, C]
            attn_weights: [B, N_q, N_kv]
        """
        B, N_q, C = query.shape
        N_kv = key.shape[1]
        
        # Q, K, V projections (using cross_attn's weights)
        # Note: This is a simplified version. In practice, you'd need to access
        # the internal weights of self.cross_attn
        q = query.reshape(B, N_q, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = key.reshape(B, N_kv, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = k  # Value = Key
        # q, k, v: [B, num_heads, N, C//num_heads]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / ((C // self.num_heads) ** 0.5)
        # attn: [B, num_heads, N_q, N_kv]
        
        # Add position bias
        attn = attn + pos_bias.unsqueeze(0)  # [B, num_heads, N_q, N_kv]
        
        # Softmax
        attn_weights = F.softmax(attn, dim=-1)
        
        # Weighted sum
        output = attn_weights @ v  # [B, num_heads, N_q, C//num_heads]
        output = output.transpose(1, 2).reshape(B, N_q, C)
        
        # Average attention weights across heads
        attn_weights_avg = attn_weights.mean(1)  # [B, N_q, N_kv]
        
        return output, attn_weights_avg

