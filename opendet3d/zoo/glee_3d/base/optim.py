"""GLEE_3D optimizer configuration."""

from __future__ import annotations

from vis4d.config.typing import ExperimentParameters

from opendet3d.zoo.gdino3d.base.optim import get_optim_cfg


def get_glee_3d_default_param_groups() -> list[dict]:
    """Default param groups: 0.1x lr for pretrained, 1x for new."""
    return [
        {"custom_keys": ["glee.backbone"], "lr_mult": 0.1},
        {"custom_keys": ["glee.text_encoder"], "lr_mult": 0.0},  # frozen
        {"custom_keys": ["glee.pixel_decoder"], "lr_mult": 1.0},
        {"custom_keys": ["glee.predictor"], "lr_mult": 1.0},
        {"custom_keys": ["geometry_backend.encoder"], "lr_mult": 0.1},
        {"custom_keys": ["geometry_backend.decoder"], "lr_mult": 1.0},
        # bbox3d_head, early_depth_fusion: default 1.0x
    ]


def get_glee_3d_freeze_backbone_param_groups() -> list[dict]:
    """Freeze vision + depth backbones, train decoders + 3D head."""
    return [
        {"custom_keys": ["glee.backbone"], "lr_mult": 0.0},
        {"custom_keys": ["glee.text_encoder"], "lr_mult": 0.0},
        {"custom_keys": ["glee.pixel_decoder"], "lr_mult": 1.0},
        {"custom_keys": ["glee.predictor"], "lr_mult": 1.0},
        {"custom_keys": ["geometry_backend.encoder"], "lr_mult": 0.0},
        {"custom_keys": ["geometry_backend.decoder"], "lr_mult": 1.0},
    ]


def get_glee_3d_freeze_all_pretrained_param_groups() -> list[dict]:
    """Freeze all GLEE + geometry, only train 3D head + fusion."""
    return [
        {"custom_keys": ["glee"], "lr_mult": 0.0},
        {"custom_keys": ["geometry_backend"], "lr_mult": 0.0},
        {"custom_keys": ["bbox3d_head"], "lr_mult": 1.0},
        {"custom_keys": ["early_depth_fusion"], "lr_mult": 1.0},
    ]


def get_glee_3d_optim_cfg(
    params: ExperimentParameters,
    param_groups: list[dict] | None = None,
    freeze_backbone: bool = False,
    freeze_all_pretrained: bool = False,
):
    """Get optimizer config for GLEE_3D."""
    if param_groups is None:
        # Resolve FieldReference: .get() returns the underlying value
        _freeze_all = getattr(freeze_all_pretrained, 'get', lambda: freeze_all_pretrained)()
        _freeze_bb = getattr(freeze_backbone, 'get', lambda: freeze_backbone)()
        if _freeze_all:
            param_groups = get_glee_3d_freeze_all_pretrained_param_groups()
            mode = "freeze_all_pretrained"
        elif _freeze_bb:
            param_groups = get_glee_3d_freeze_backbone_param_groups()
            mode = "freeze_backbone"
        else:
            param_groups = get_glee_3d_default_param_groups()
            mode = "default"
    else:
        mode = "custom"

    print(f"[GLEE_3D Optim] mode={mode}")
    for pg in param_groups:
        print(f"  {pg['custom_keys']}: lr_mult={pg['lr_mult']}")

    return get_optim_cfg(params, param_groups=param_groups)
