"""Depth-Memory Fusion Module for GroundingDINO3D.

This module fuses depth latents from geometry backends into the encoder memory
of the transformer. The fusion happens after the encoder and before the decoder.

Four fusion strategies are supported:
1. "add": Direct addition with normal initialization (simple but may disrupt pretrained weights)
2. "zero_add": Zero-initialized conv + addition (ControlNet style, stable training)
3. "concat": Concatenation + projection (preserves both signals, identity init for stability)
4. "gating": Gating fusion with sigmoid gate (adaptive fusion based on depth features)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


FusionType = Literal["add", "zero_add", "concat", "gating"]


class DepthMemoryFusion(nn.Module):
    """Fuse depth latents into encoder memory for multi-scale features.

    The depth latents are single-scale (e.g., 1/8 resolution), while memory is
    multi-scale (e.g., 1/8, 1/16, 1/32, 1/64). This module:
    1. Reshapes depth latents from [B, N, C] back to [B, C, H, W]
    2. Interpolates to each memory level's spatial size
    3. Projects depth dim to memory dim via conv
    4. Fuses with memory using the specified strategy

    Args:
        depth_dim: Channel dimension of depth latents (from geometry backend).
        memory_dim: Channel dimension of memory (from encoder, typically 256).
        num_levels: Number of feature levels (typically 4).
        fusion_type: Fusion strategy - "add", "zero_add", "concat", or "gating".
    """

    def __init__(
        self,
        depth_dim: int,
        memory_dim: int,
        num_levels: int = 4,
        fusion_type: FusionType = "zero_add",
    ) -> None:
        super().__init__()

        self.depth_dim = depth_dim
        self.memory_dim = memory_dim
        self.num_levels = num_levels
        self.fusion_type = fusion_type

        # Create per-level projection convs
        # Each level has its own conv to allow level-specific learning
        self.depth_projs = nn.ModuleList()

        # LayerNorm for add fusion to stabilize training
        # This normalizes depth_proj output to match memory feature distribution
        # NOTE: zero_add should NOT use LayerNorm because:
        #   - Zero-initialized weights produce near-zero outputs
        #   - LayerNorm normalizes near-zero inputs to normal scale (mean=0, std=1)
        #   - This creates noise that disrupts pretrained memory features
        self.depth_norms = nn.ModuleList() if fusion_type == "add" else None

        if fusion_type == "concat":
            # Concat: depth + memory -> memory_dim
            for _ in range(num_levels):
                proj = nn.Conv2d(depth_dim + memory_dim, memory_dim, kernel_size=1, bias=True)
                nn.init.zeros_(proj.weight)
                nn.init.zeros_(proj.bias)
                with torch.no_grad():
                    for i in range(memory_dim):
                        proj.weight[i, depth_dim + i, 0, 0] = 1.0
                self.depth_projs.append(proj)
        elif fusion_type == "gating":
            # Gating: gate = sigmoid(proj(depth)), fused = memory * gate + depth_proj
            self.gate_projs = nn.ModuleList()
            for _ in range(num_levels):
                # Gate projection: depth -> memory_dim -> sigmoid
                gate = nn.Sequential(
                    nn.Conv2d(depth_dim, memory_dim, kernel_size=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(memory_dim, memory_dim, kernel_size=1, bias=True),
                    nn.Sigmoid()
                )
                # Depth feature projection (zero-init for stable training)
                depth_proj = nn.Conv2d(depth_dim, memory_dim, kernel_size=1, bias=True)
                nn.init.zeros_(depth_proj.weight)
                nn.init.zeros_(depth_proj.bias)
                self.gate_projs.append(gate)
                self.depth_projs.append(depth_proj)
        else:
            # Add variants: depth_dim -> memory_dim
            for _ in range(num_levels):
                proj = nn.Conv2d(depth_dim, memory_dim, kernel_size=1, bias=True)
                if fusion_type == "zero_add":
                    nn.init.zeros_(proj.weight)
                    nn.init.zeros_(proj.bias)
                else:
                    nn.init.xavier_uniform_(proj.weight)
                    nn.init.zeros_(proj.bias)
                self.depth_projs.append(proj)

                # Add LayerNorm for each level to normalize depth_proj output
                # This ensures depth features have similar scale to memory features
                if self.depth_norms is not None:
                    self.depth_norms.append(nn.LayerNorm(memory_dim))
    
    # Class-level counter for debug logging
    _forward_count = 0

    def forward(
        self,
        depth_latents: Tensor,
        depth_latents_hw: tuple[int, int],
        memory: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
    ) -> Tensor:
        """Fuse depth latents into memory.

        Args:
            depth_latents: Depth features [B, N, depth_dim] where N = H_d * W_d.
            depth_latents_hw: Spatial size of depth latents (H_d, W_d).
            memory: Encoder output [B, sum(H_i * W_i), memory_dim].
            spatial_shapes: Per-level shapes [num_levels, 2], each row is (H_i, W_i).
            level_start_index: Start index for each level [num_levels].

        Returns:
            Fused memory [B, sum(H_i * W_i), memory_dim].
        """
        B = memory.shape[0]
        device = memory.device
        H_d, W_d = depth_latents_hw

        # Debug: Log fusion execution (only first 5 calls to avoid spam)
        DepthMemoryFusion._forward_count += 1
        if DepthMemoryFusion._forward_count <= 5:
            print(f"\n[DepthMemoryFusion] ✅ FUSION ACTIVE (call #{DepthMemoryFusion._forward_count})")
            print(f"  fusion_type: {self.fusion_type}")
            print(f"  depth_latents: {depth_latents.shape} (dim={self.depth_dim})")
            print(f"  depth_latents_hw: {depth_latents_hw}")
            print(f"  memory: {memory.shape} (dim={self.memory_dim})")
            print(f"  spatial_shapes: {spatial_shapes.tolist()}")
        
        # Reshape depth latents: [B, N, C] -> [B, C, H_d, W_d]
        depth_2d = depth_latents.permute(0, 2, 1).reshape(B, self.depth_dim, H_d, W_d)
        
        # Process each level
        fused_parts = []
        spatial_shapes_list = spatial_shapes.tolist()
        level_start_index_list = level_start_index.tolist()
        
        for lvl in range(self.num_levels):
            H_lvl, W_lvl = int(spatial_shapes_list[lvl][0]), int(spatial_shapes_list[lvl][1])
            start_idx = int(level_start_index_list[lvl])
            end_idx = start_idx + H_lvl * W_lvl
            
            # Extract this level's memory: [B, H_lvl * W_lvl, memory_dim]
            memory_lvl = memory[:, start_idx:end_idx, :]
            
            # Interpolate depth to this level's spatial size
            depth_lvl = F.interpolate(
                depth_2d, size=(H_lvl, W_lvl), mode="bilinear", align_corners=False
            )  # [B, depth_dim, H_lvl, W_lvl]
            
            if self.fusion_type == "concat":
                # Reshape memory to 2D for concat
                memory_2d = memory_lvl.permute(0, 2, 1).reshape(B, self.memory_dim, H_lvl, W_lvl)
                # Concat and project
                concat_feat = torch.cat([depth_lvl, memory_2d], dim=1)
                fused_2d = self.depth_projs[lvl](concat_feat)
                fused_lvl = fused_2d.flatten(2).permute(0, 2, 1)
            elif self.fusion_type == "gating":
                # Gating: fused = memory * gate + depth_proj
                memory_2d = memory_lvl.permute(0, 2, 1).reshape(B, self.memory_dim, H_lvl, W_lvl)
                gate = self.gate_projs[lvl](depth_lvl)  # [B, memory_dim, H, W], values in [0, 1]
                depth_proj = self.depth_projs[lvl](depth_lvl)  # [B, memory_dim, H, W]
                fused_2d = memory_2d * gate + depth_proj
                fused_lvl = fused_2d.flatten(2).permute(0, 2, 1)
            else:
                # Add variants: project depth and add to memory
                depth_proj = self.depth_projs[lvl](depth_lvl)
                depth_proj_flat = depth_proj.flatten(2).permute(0, 2, 1)

                # Apply LayerNorm to normalize depth_proj output
                # This prevents distribution mismatch with memory features
                if self.depth_norms is not None:
                    depth_proj_flat = self.depth_norms[lvl](depth_proj_flat)

                fused_lvl = memory_lvl + depth_proj_flat

            fused_parts.append(fused_lvl)
        
        # Concatenate all levels back
        fused_memory = torch.cat(fused_parts, dim=1)  # [B, total_tokens, memory_dim]
        
        return fused_memory

