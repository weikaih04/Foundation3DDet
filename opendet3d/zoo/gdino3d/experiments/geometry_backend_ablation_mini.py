"""Geometry Backend Ablation Experiments - Mini Dataset Version.

This config file is identical to geometry_backend_ablation.py but uses mini datasets
(100 samples per dataset) for fast testing and debugging.

Total samples: ~1,200 (vs ~175,000 for full dataset)
Loading time: ~10x faster

Usage:
    vis4d fit --config opendet3d/zoo/gdino3d/experiments/geometry_backend_ablation_mini.py ...
"""

from __future__ import annotations

from opendet3d.zoo.gdino3d.experiments.geometry_backend_ablation import (
    get_detany3d_config as _get_detany3d_config_full,
    get_unidepth_head_config as _get_unidepth_head_config_full,
    get_unidepth_head_dino_config as _get_unidepth_head_dino_config_full,
    get_unidepth_v2_config as _get_unidepth_v2_config_full,
    _build_geometry_backend_experiment,
)
from vis4d.config.typing import ExperimentConfig


# ============================================================================
# Mini Dataset Configurations (100 samples per dataset)
# ============================================================================


def get_unidepth_head_config(*args, **kwargs) -> ExperimentConfig:
    """UniDepthHead (original 3D-MOOD) with mini dataset."""
    return _build_geometry_backend_experiment(
        exp_name="gdino3d_geom-ablation_unidepth-head_mini100",
        geometry_backend_type="unidepth_head",
        use_geom_backend_loss=False,
        use_mini_dataset=True,
        mini_dataset_size=100,
    )


def get_unidepth_head_dino_config(*args, **kwargs) -> ExperimentConfig:
    """UniDepthHead + DINOv2 with mini dataset."""
    return _build_geometry_backend_experiment(
        exp_name="gdino3d_geom-ablation_unidepth-head-dino-vits_mini100",
        geometry_backend_type="unidepth_head_dino",
        use_geom_backend_loss=True,  # Use geometry backend loss (returns only depth_loss)
        dino_model="vit_small",
        dino_pretrained="checkpoints/dinov2_backbones/da3_s_dinov2_backbone.pth",
        use_mini_dataset=True,
        mini_dataset_size=100,
    )


def get_detany3d_config(*args, **kwargs) -> ExperimentConfig:
    """DetAny3D (DINOv2 + UnidepthDecoderDinoOnly) with mini dataset."""
    return _build_geometry_backend_experiment(
        exp_name="gdino3d_geom-ablation_detany3d-vits_mini100",
        geometry_backend_type="detany3d",
        use_geom_backend_loss=True,
        dino_model="vit_small",
        dino_pretrained="checkpoints/dinov2_backbones/da3_s_dinov2_backbone.pth",
        detany3d_decoder_pretrained="checkpoints/depth_heads/detany3d_decoder.pth",
        detany3d_decoder_dino_model="vit_large",
        use_mini_dataset=True,
        mini_dataset_size=100,
    )


def get_unidepth_v2_config(*args, **kwargs) -> ExperimentConfig:
    """UniDepthV2 (DINOv2 + UniDepthV2 Decoder) with mini dataset."""
    return _build_geometry_backend_experiment(
        exp_name="gdino3d_geom-ablation_unidepth-v2-vits_mini100",
        geometry_backend_type="unidepth_v2",
        use_geom_backend_loss=True,
        unidepth_v2_version="v2-vits14",
        # Don't use separate decoder weights - use HuggingFace model directly
        # unidepth_v2_decoder_pretrained="checkpoints/depth_heads/unidepth_v2_decoder_vits.pth",
        use_mini_dataset=True,
        mini_dataset_size=100,
    )


# Default config function for vis4d CLI
def get_config(config_name: str = "unidepth_head_dino", *args, **kwargs) -> ExperimentConfig:
    """Get config by name.

    Args:
        config_name: Name of the config to load. Options:
            - "unidepth_head": UniDepthHead (original 3D-MOOD)
            - "unidepth_head_dino": UniDepthHead + DINOv2 (default)
            - "detany3d": DetAny3D (DINOv2 + UnidepthDecoderDinoOnly)
            - "unidepth_v2": UniDepthV2 (DINOv2 + UniDepthV2 Decoder)

    Returns:
        ExperimentConfig for the specified geometry backend.
    """
    # Strip leading colon if present (vis4d adds it when using :: syntax)
    if config_name.startswith(":"):
        config_name = config_name[1:]

    config_map = {
        "unidepth_head": get_unidepth_head_config,
        "unidepth_head_dino": get_unidepth_head_dino_config,
        "detany3d": get_detany3d_config,
        "unidepth_v2": get_unidepth_v2_config,
    }

    if config_name not in config_map:
        raise ValueError(
            f"Unknown config name: {config_name}. "
            f"Available options: {list(config_map.keys())}"
        )

    return config_map[config_name](*args, **kwargs)

