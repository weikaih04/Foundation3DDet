"""
Multi-Scale CUT3R Fusion Module.
Flexible wrapper that supports configurable fusion for different levels.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple

from .gated_fusion import GatedCUT3RFusion

# Deformable fusion is optional - only import if needed
try:
    from .deformable_fusion import DeformableGatedFusion
    DEFORMABLE_AVAILABLE = True
except ImportError:
    DEFORMABLE_AVAILABLE = False
    DeformableGatedFusion = None


class MultiScaleCUT3RFusion(nn.Module):
    """
    Multi-Scale CUT3R Fusion Module.

    Supports flexible configuration of which levels to fuse and what strategy to use.
    Each level can use either:
    - 'full': Full cross-attention (recommended, simple and effective)
    - 'deformable': Lightweight deformable attention (optional, for efficiency)

    Args:
        d_models: List of feature dimensions for each level [96, 192, 384, 768]
        d_cut3r: CUT3R feature dimension (default: 1024)
        num_heads: Number of attention heads (default: 8)
        fusion_levels: List of levels to fuse, e.g., [2, 3] or [0, 1, 2, 3]
        fusion_strategies: Dict mapping level to strategy config
            Example (Simple - All Full Attention):
            {
                0: {'type': 'full'},
                1: {'type': 'full'},
                2: {'type': 'full'},
                3: {'type': 'full'}
            }

            Example (Advanced - Mixed):
            {
                0: {'type': 'deformable', 'n_points': 2},  # Optional
                1: {'type': 'deformable', 'n_points': 4},  # Optional
                2: {'type': 'full'},
                3: {'type': 'full'}
            }
        dropout: Dropout probability (default: 0.1)
        use_relative_pos_bias: Whether to use relative position bias (default: False)
    
    Example:
        >>> # Simple config: Fuse L2 + L3 with Full Attention (Recommended)
        >>> fusion = MultiScaleCUT3RFusion(
        ...     d_models=[96, 192, 384, 768],
        ...     fusion_levels=[2, 3],
        ...     fusion_strategies={
        ...         2: {'type': 'full'},
        ...         3: {'type': 'full'}
        ...     }
        ... )
        >>>
        >>> # Advanced config: Mixed strategies (requires deformable_fusion.py)
        >>> fusion = MultiScaleCUT3RFusion(
        ...     d_models=[96, 192, 384, 768],
        ...     fusion_levels=[0, 1, 2, 3],
        ...     fusion_strategies={
        ...         0: {'type': 'deformable', 'n_points': 2},
        ...         1: {'type': 'deformable', 'n_points': 4},
        ...         2: {'type': 'full'},
        ...         3: {'type': 'full'}
        ...     }
        ... )
    """
    
    def __init__(
        self,
        d_models: List[int] = [96, 192, 384, 768],
        d_cut3r: int = 1024,
        num_heads: int = 8,
        fusion_levels: List[int] = [2, 3],
        fusion_strategies: Optional[Dict[int, Dict]] = None,
        dropout: float = 0.1,
        use_relative_pos_bias: bool = False
    ):
        super().__init__()
        
        self.fusion_levels = fusion_levels
        self.use_relative_pos_bias = use_relative_pos_bias
        
        # ========== Default Fusion Strategies ==========
        if fusion_strategies is None:
            # Default: Simple Full Attention for all levels
            fusion_strategies = {
                0: {'type': 'full'},
                1: {'type': 'full'},
                2: {'type': 'full'},
                3: {'type': 'full'}
            }
        self.fusion_strategies = fusion_strategies
        
        # ========== Create Fusion Modules for Each Level ==========
        self.fusion_modules = nn.ModuleDict()
        
        for lvl in fusion_levels:
            if lvl >= len(d_models):
                raise ValueError(
                    f"Level {lvl} is out of range. d_models has {len(d_models)} levels."
                )
            
            d_model = d_models[lvl]
            strategy = fusion_strategies[lvl]
            
            if strategy['type'] == 'full':
                # Full Cross-Attention Fusion (Recommended)
                self.fusion_modules[f'level_{lvl}'] = GatedCUT3RFusion(
                    d_model=d_model,
                    d_cut3r=d_cut3r,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_relative_pos_bias=use_relative_pos_bias
                )
            elif strategy['type'] == 'deformable':
                # Deformable Attention Fusion (Optional - for efficiency)
                if not DEFORMABLE_AVAILABLE:
                    raise ImportError(
                        f"Deformable fusion requested for level {lvl}, but "
                        f"deformable_fusion.py is not available. "
                        f"Either use 'full' attention or ensure deformable_fusion.py exists."
                    )
                self.fusion_modules[f'level_{lvl}'] = DeformableGatedFusion(
                    d_model=d_model,
                    d_cut3r=d_cut3r,
                    n_heads=num_heads,
                    n_points=strategy.get('n_points', 4),  # Default to 4 points
                    dropout=dropout,
                    use_relative_pos_bias=use_relative_pos_bias
                )
            else:
                raise ValueError(
                    f"Unknown fusion strategy type: {strategy['type']}. "
                    f"Must be 'full' or 'deformable'."
                )
        
        print(f"[MultiScaleCUT3RFusion] Initialized with:")
        print(f"  - Fusion levels: {fusion_levels}")
        print(f"  - Strategies: {fusion_strategies}")
        print(f"  - Use relative pos bias: {use_relative_pos_bias}")
    
    def forward(
        self,
        multi_scale_feats: List[torch.Tensor],
        cut3r_features: torch.Tensor
    ) -> Tuple[List[torch.Tensor], Dict]:
        """
        Forward pass with multi-scale fusion.
        
        Args:
            multi_scale_feats: List of 4 tensors
                - Level 0: [B, 96, 200, 333]
                - Level 1: [B, 192, 100, 166]
                - Level 2: [B, 384, 50, 83]
                - Level 3: [B, 768, 25, 41]
            cut3r_features: [B, 730, 1024]
                - Camera token: [B, 1, 1024]
                - Patch tokens: [B, 729, 1024] (27x27 grid)
        
        Returns:
            fused_feats: List of 4 tensors (same shapes as input)
            fusion_info: Dict containing:
                - 'attention_weights': Dict[str, Tensor] (only for full attention levels)
                - 'gate_values': Dict[str, Tensor]
                - 'fusion_strength': Dict[str, float] (mean gate value per level)
        """
        fused_feats = []
        fusion_info = {
            'attention_weights': {},
            'gate_values': {},
            'fusion_strength': {}
        }
        
        for i, feat in enumerate(multi_scale_feats):
            if i in self.fusion_levels:
                # ========== Fuse this level ==========
                fusion_module = self.fusion_modules[f'level_{i}']
                strategy = self.fusion_strategies[i]
                
                # Forward through fusion module
                if strategy['type'] == 'full':
                    # Full attention returns: fused, attn_weights, gate_values
                    fused, attn_weights, gate_vals = fusion_module(
                        feat, cut3r_features, pos_bias=None
                    )
                    fusion_info['attention_weights'][f'level_{i}'] = attn_weights
                else:
                    # Deformable attention returns: fused, gate_values
                    fused, gate_vals = fusion_module(
                        feat, cut3r_features, pos_bias=None
                    )
                
                # Store gate values and fusion strength
                fusion_info['gate_values'][f'level_{i}'] = gate_vals
                fusion_info['fusion_strength'][f'level_{i}'] = gate_vals.mean().item()
                
                fused_feats.append(fused)
            else:
                # ========== Skip this level ==========
                fused_feats.append(feat)
                fusion_info['fusion_strength'][f'level_{i}'] = 0.0
        
        return fused_feats, fusion_info
    
    def get_fusion_summary(self) -> str:
        """Get a summary of the fusion configuration."""
        summary = []
        summary.append("=" * 60)
        summary.append("Multi-Scale CUT3R Fusion Configuration")
        summary.append("=" * 60)
        
        for lvl in range(4):
            if lvl in self.fusion_levels:
                strategy = self.fusion_strategies[lvl]
                module = self.fusion_modules[f'level_{lvl}']
                
                # Count parameters
                n_params = sum(p.numel() for p in module.parameters())

                # Format strategy info
                if strategy['type'] == 'deformable':
                    n_points = strategy.get('n_points', 4)
                    strategy_info = f"DEFORMABLE (n_points={n_points})"
                else:
                    strategy_info = "FULL ATTENTION"

                summary.append(
                    f"Level {lvl}: {strategy_info}, "
                    f"params={n_params/1e6:.2f}M"
                )
            else:
                summary.append(f"Level {lvl}: SKIP")
        
        summary.append("=" * 60)
        
        # Total parameters
        total_params = sum(p.numel() for p in self.parameters())
        summary.append(f"Total parameters: {total_params/1e6:.2f}M")
        summary.append("=" * 60)
        
        return "\n".join(summary)

