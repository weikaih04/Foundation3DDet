"""
CUT3R Fusion Configuration Examples for 3D-MOOD.

This file provides different configuration presets for CUT3R fusion.
"""

# ============================================================================
# Configuration 1: Minimal (Only Level 3) - For Quick Validation
# ============================================================================
# Speed impact: 1.22x
# Parameters: +10M
# Use case: Quick validation that fusion works
# Strategy: Simple Full Attention (Recommended)
# ============================================================================

cut3r_fusion_minimal = dict(
    # Path to CUT3R checkpoint
    cut3r_checkpoint='path/to/cut3r_checkpoint.pth',

    # Freeze CUT3R weights (recommended)
    cut3r_freeze=True,

    # Fusion configuration - Simple Full Attention
    fusion_levels=[3],  # Only fuse highest level
    fusion_strategies={
        3: {'type': 'full'}  # Simple and effective
    },

    # Attention parameters
    fusion_num_heads=8,
    fusion_dropout=0.1,
    use_relative_pos_bias=False,
)


# ============================================================================
# Configuration 2: Recommended (Level 2 + 3) - Best Balance
# ============================================================================
# Speed impact: 1.3x (with Full Attention on both levels)
# Parameters: +15M
# Use case: Recommended for production use
# Expected improvement: +3-4% AP3D
# Strategy: Simple Full Attention (Recommended)
# ============================================================================

cut3r_fusion_recommended = dict(
    # Path to CUT3R checkpoint
    cut3r_checkpoint='path/to/cut3r_checkpoint.pth',

    # Freeze CUT3R weights (recommended)
    cut3r_freeze=True,

    # Fusion configuration - Simple Full Attention
    fusion_levels=[2, 3],  # Fuse mid and high levels
    fusion_strategies={
        2: {'type': 'full'},  # Simple and effective
        3: {'type': 'full'}   # Simple and effective
    },

    # Attention parameters
    fusion_num_heads=8,
    fusion_dropout=0.1,
    use_relative_pos_bias=False,
)


# ============================================================================
# Configuration 3: Full (All 4 Levels) - Maximum Performance
# ============================================================================
# Speed impact: 2.0x (with Full Attention on all levels)
# Parameters: +40M
# Use case: When you want maximum performance and have compute budget
# Expected improvement: +4-5% AP3D
# Strategy: Simple Full Attention (Recommended)
# Note: This is computationally expensive but simple and effective
# ============================================================================

cut3r_fusion_full = dict(
    # Path to CUT3R checkpoint
    cut3r_checkpoint='path/to/cut3r_checkpoint.pth',

    # Freeze CUT3R weights (recommended)
    cut3r_freeze=True,

    # Fusion configuration - Simple Full Attention on all levels
    fusion_levels=[0, 1, 2, 3],  # Fuse all levels
    fusion_strategies={
        0: {'type': 'full'},  # Simple but expensive
        1: {'type': 'full'},  # Simple but expensive
        2: {'type': 'full'},  # Simple and effective
        3: {'type': 'full'}   # Simple and effective
    },

    # Attention parameters
    fusion_num_heads=8,
    fusion_dropout=0.1,
    use_relative_pos_bias=False,
)


# ============================================================================
# Configuration 4: Efficient (With Deformable Attention) - OPTIONAL
# ============================================================================
# Speed impact: 1.22x
# Parameters: +12.6M
# Use case: When you want efficiency (requires deformable_fusion.py)
# Expected improvement: +3% AP3D
# Strategy: Mixed (Deformable for L2, Full for L3)
# Note: This is optional - only use if you need efficiency
# ============================================================================

cut3r_fusion_efficient = dict(
    # Path to CUT3R checkpoint
    cut3r_checkpoint='path/to/cut3r_checkpoint.pth',

    # Freeze CUT3R weights (recommended)
    cut3r_freeze=True,

    # Fusion configuration - Mixed strategy (OPTIONAL)
    fusion_levels=[2, 3],
    fusion_strategies={
        2: {'type': 'deformable', 'n_points': 4},  # Optional: for efficiency
        3: {'type': 'full'}                         # Recommended
    },

    # Attention parameters
    fusion_num_heads=8,
    fusion_dropout=0.1,
    use_relative_pos_bias=False,
)


# ============================================================================
# Configuration 5: Experimental (With Relative Position Bias)
# ============================================================================
# Speed impact: 1.35x
# Parameters: +15.8M
# Use case: Experimental - test if position bias helps
# Expected improvement: +0.5-1% over recommended config
# Strategy: Simple Full Attention + Position Bias
# ============================================================================

cut3r_fusion_with_pos_bias = dict(
    # Path to CUT3R checkpoint
    cut3r_checkpoint='path/to/cut3r_checkpoint.pth',

    # Freeze CUT3R weights (recommended)
    cut3r_freeze=True,

    # Fusion configuration
    fusion_levels=[2, 3],
    fusion_strategies={
        2: {'type': 'full'},
        3: {'type': 'full'}
    },

    # Attention parameters
    fusion_num_heads=8,
    fusion_dropout=0.1,
    use_relative_pos_bias=True,  # Enable position bias
)


# ============================================================================
# Configuration 6: Ablation - Only Low Levels (L0 + L1)
# ============================================================================
# Speed impact: 1.5x (with Full Attention)
# Parameters: +20M
# Use case: Ablation study - does low-level fusion help?
# Strategy: Simple Full Attention
# Note: Computationally expensive, likely not worth it
# ============================================================================

cut3r_fusion_low_levels = dict(
    cut3r_checkpoint='path/to/cut3r_checkpoint.pth',
    cut3r_freeze=True,
    fusion_levels=[0, 1],
    fusion_strategies={
        0: {'type': 'full'},  # Expensive
        1: {'type': 'full'},  # Expensive
    },
    fusion_num_heads=8,
    fusion_dropout=0.1,
    use_relative_pos_bias=False,
)


# ============================================================================
# Configuration 7: Fine-tuning CUT3R
# ============================================================================
# Speed impact: 1.3x
# Parameters: +15M (all trainable)
# Use case: When you want to fine-tune CUT3R on your dataset
# Warning: Requires more GPU memory and careful learning rate tuning
# Strategy: Simple Full Attention
# ============================================================================

cut3r_fusion_finetune = dict(
    cut3r_checkpoint='path/to/cut3r_checkpoint.pth',
    cut3r_freeze=False,  # Allow CUT3R to be fine-tuned
    fusion_levels=[2, 3],
    fusion_strategies={
        2: {'type': 'full'},
        3: {'type': 'full'}
    },
    fusion_num_heads=8,
    fusion_dropout=0.1,
    use_relative_pos_bias=False,
)


# ============================================================================
# Helper Function: Get Config by Name
# ============================================================================

def get_cut3r_fusion_config(config_name='recommended'):
    """
    Get CUT3R fusion configuration by name.
    
    Args:
        config_name: One of ['minimal', 'recommended', 'full', 'with_pos_bias',
                             'low_levels', 'finetune']
    
    Returns:
        dict: Configuration dictionary
    
    Example:
        >>> config = get_cut3r_fusion_config('recommended')
        >>> model = GroundingDINO3D(..., **config)
    """
    configs = {
        'minimal': cut3r_fusion_minimal,
        'recommended': cut3r_fusion_recommended,
        'full': cut3r_fusion_full,
        'efficient': cut3r_fusion_efficient,  # Optional: uses deformable
        'with_pos_bias': cut3r_fusion_with_pos_bias,
        'low_levels': cut3r_fusion_low_levels,
        'finetune': cut3r_fusion_finetune,
    }
    
    if config_name not in configs:
        raise ValueError(
            f"Unknown config name: {config_name}. "
            f"Available: {list(configs.keys())}"
        )
    
    return configs[config_name]


# ============================================================================
# Training Configuration Example
# ============================================================================

# Example: How to use in your training config
"""
from configs.cut3r_fusion_configs import get_cut3r_fusion_config

# Get fusion config
fusion_config = get_cut3r_fusion_config('recommended')

# Update CUT3R checkpoint path
fusion_config['cut3r_checkpoint'] = '/path/to/your/cut3r_checkpoint.pth'

# Create model
model = dict(
    type='GroundingDINO3D',
    
    # ... other model configs ...
    
    basemodel=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        out_indices=(0, 1, 2, 3),
    ),
    
    neck=dict(
        type='ChannelMapper',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=4,
    ),
    
    # ... other configs ...
    
    # Add CUT3R Fusion
    **fusion_config,
)

# Training strategy
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            # Lower learning rate for pretrained components
            'backbone': dict(lr_mult=0.1),
            'neck': dict(lr_mult=0.1),
            
            # Normal learning rate for fusion modules
            'cut3r_fusion': dict(lr_mult=1.0),
            
            # CUT3R tower frozen by default
            # If fine-tuning: 'cut3r_tower': dict(lr_mult=0.01),
        }
    )
)

# Warm-up schedule
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[6, 9]  # Decay at epoch 6 and 9
)
"""

