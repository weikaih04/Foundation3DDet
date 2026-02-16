"""SAM3_3D Optimizer config.

This module provides SAM3_3D-specific optimizer configuration with param_groups
that properly handle SAM3's module naming convention (e.g., sam3.backbone).

Two main configurations:
1. Default (0.1x lr): All pretrained components use 0.1x lr (like GDino3D)
2. Freeze backbone: Freeze vision + depth backbones, finetune the rest

SAM3_3D Components:
- sam3.backbone: ViT vision backbone (pretrained)
- sam3.geometry_encoder: Geometric prompt encoder (pretrained)
- sam3.transformer: Transformer encoder + decoder (pretrained)
- geometry_backend.encoder: DINOv2 depth encoder (pretrained)
- geometry_backend.decoder: Depth decoder (finetune)
- bbox3d_head: 3D detection head (train from scratch)
- early_depth_fusion: Depth fusion module (train from scratch)
"""

from __future__ import annotations

from ml_collections import ConfigDict
from vis4d.config.typing import ExperimentParameters

from opendet3d.zoo.gdino3d.base.optim import get_optim_cfg


def get_sam3_3d_param_groups(
    backbone_lr_mult: float = 0.1,
    depth_encoder_lr_mult: float = 0.1,
) -> list[dict]:
    """Get SAM3_3D default param_groups.

    When backbone_freeze_blocks / encoder_freeze_blocks are used,
    frozen params have requires_grad=False and won't update regardless
    of lr_mult. The unfrozen blocks get full lr (1.0x) for effective
    adaptation.

    Args:
        backbone_lr_mult: LR multiplier for SAM3 vision backbone.
        depth_encoder_lr_mult: LR multiplier for depth encoder.

    Returns:
        List of param_group dicts.

    Default lr_mult:
        - sam3.backbone: 1.0x (frozen blocks won't update; unfrozen get full lr)
        - geometry_backend.encoder: 1.0x (same logic)
        - Others: 1.0x (sam3.transformer, bbox3d_head, etc.)
    """
    return [
        # SAM3 vision backbone - frozen blocks have requires_grad=False,
        # unfrozen blocks (last N) get full lr for effective adaptation
        {"custom_keys": ["sam3.backbone"], "lr_mult": backbone_lr_mult},
        # Depth encoder - same: frozen blocks won't update, unfrozen get full lr
        {"custom_keys": ["geometry_backend.encoder"], "lr_mult": depth_encoder_lr_mult},
        # These train with full lr (1.0x) by default:
        # - sam3.geometry_encoder (prompt encoder)
        # - sam3.transformer (encoder + decoder)
        # - bbox3d_head
        # - early_depth_fusion
        # - geometry_backend.decoder
    ]


def get_sam3_3d_freeze_backbone_param_groups() -> list[dict]:
    """Get param_groups with frozen backbones.

    Freezes (0.0x lr):
        - sam3.backbone (vision backbone)
        - geometry_backend.encoder (depth backbone)

    Trains (1.0x lr):
        - sam3.geometry_encoder
        - sam3.transformer
        - bbox3d_head
        - early_depth_fusion
        - geometry_backend.decoder

    Returns:
        List of param_group dicts.
    """
    return [
        # Freeze vision backbone
        {"custom_keys": ["sam3.backbone"], "lr_mult": 0.0},
        # Freeze depth backbone
        {"custom_keys": ["geometry_backend.encoder"], "lr_mult": 0.0},
        # These train with full lr (1.0x) by default:
        # - sam3.geometry_encoder
        # - sam3.transformer
        # - bbox3d_head
        # - early_depth_fusion
        # - geometry_backend.decoder
    ]


def get_sam3_3d_freeze_all_pretrained_param_groups() -> list[dict]:
    """Get param_groups that freeze ALL pretrained components.

    Only trains:
        - bbox3d_head (1.0x lr)
        - early_depth_fusion (1.0x lr)

    Freezes everything else (0.0x lr):
        - sam3.backbone
        - sam3.geometry_encoder
        - sam3.transformer
        - geometry_backend (encoder + decoder)

    Returns:
        List of param_group dicts.
    """
    return [
        # Freeze all SAM3 components
        {"custom_keys": ["sam3.backbone"], "lr_mult": 0.0},
        {"custom_keys": ["sam3.geometry_encoder"], "lr_mult": 0.0},
        {"custom_keys": ["sam3.transformer"], "lr_mult": 0.0},
        # Freeze entire geometry backend
        {"custom_keys": ["geometry_backend"], "lr_mult": 0.0},
        # Train only 3D head and fusion
        {"custom_keys": ["bbox3d_head"], "lr_mult": 1.0},
        {"custom_keys": ["early_depth_fusion"], "lr_mult": 1.0},
    ]


def get_sam3_3d_optim_cfg(
    params: ExperimentParameters,
    param_groups: list[dict] | None = None,
    freeze_backbone: bool = False,
    freeze_all_pretrained: bool = False,
    epoch_based: bool = True,
    warmup: bool = True,
) -> list[ConfigDict]:
    """Get SAM3_3D optimizer configuration.

    Three modes:
    1. Default (freeze_backbone=False, freeze_all_pretrained=False):
       All pretrained components use 0.1x lr, like GDino3D.

    2. Freeze backbone (freeze_backbone=True):
       Freeze vision + depth backbones, finetune transformer with 0.1x lr.

    3. Freeze all pretrained (freeze_all_pretrained=True):
       Only train bbox3d_head and early_depth_fusion.

    Args:
        params: Experiment parameters (lr, epochs, etc.).
        param_groups: Custom param_groups. Overrides freeze_* flags if provided.
        freeze_backbone: If True, freeze vision and depth backbones.
        freeze_all_pretrained: If True, freeze all pretrained components.
        epoch_based: Whether to use epoch-based lr scheduling.
        warmup: Whether to use lr warmup.

    Returns:
        List of optimizer configs.
    """
    # Convert FieldReference to actual values for control flow
    _freeze_backbone = freeze_backbone
    _freeze_all_pretrained = freeze_all_pretrained
    if hasattr(freeze_backbone, 'get'):
        _freeze_backbone = freeze_backbone.get()
    if hasattr(freeze_all_pretrained, 'get'):
        _freeze_all_pretrained = freeze_all_pretrained.get()

    # Determine mode and param_groups
    if param_groups is None:
        if _freeze_all_pretrained:
            mode = "Freeze All Pretrained (only train 3D head + fusion)"
            param_groups = get_sam3_3d_freeze_all_pretrained_param_groups()
        elif _freeze_backbone:
            mode = "Freeze Backbone (freeze vision + depth backbones)"
            param_groups = get_sam3_3d_freeze_backbone_param_groups()
        else:
            mode = "Default (0.1x lr for pretrained, like GDino3D)"
            param_groups = get_sam3_3d_param_groups()
    else:
        mode = "Custom"

    # Print config summary
    # Get actual lr value (handle FieldReference from ml_collections)
    base_lr = params.lr
    if hasattr(base_lr, 'get'):
        base_lr = base_lr.get()

    print("\n" + "=" * 80)
    print("[SAM3_3D Optimizer Config]")
    print(f"  Mode: {mode}")
    print(f"  Base LR: {base_lr}")
    print()
    print("  Param Groups:")
    for group in param_groups:
        keys = ", ".join(group["custom_keys"])
        lr_mult = group.get("lr_mult", 1.0)
        if lr_mult == 0.0:
            status = "FROZEN (0.0x)"
        else:
            status = f"{lr_mult}x lr"
        print(f"    {keys:32s} -> {status}")
    print(f"    {'Others (default)':32s} -> 1.0x lr")
    print("=" * 80 + "\n")

    return get_optim_cfg(
        params,
        param_groups=param_groups,
        epoch_based=epoch_based,
        warmup=warmup,
    )
