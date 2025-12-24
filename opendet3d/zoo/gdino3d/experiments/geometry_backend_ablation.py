"""Geometry Backend Ablation Experiments.

This module provides experiment configurations for comparing different geometry
backends in 3D-MOOD:

1. UniDepthHead (baseline): Original 3D-MOOD with Swin + FPN + UniDepthHead
2. UniDepthHead + DINOv2: DINOv2 (all blocks, grouped) + UniDepthHead
3. DetAny3D: DINOv2 (last 4 blocks) + Unidepth_Decoder
4. UniDepthV2: DINOv2 (all blocks, grouped) + UniDepthV2 Decoder

Each backend has its own:
- Feature extraction (Swin/FPN vs DINOv2)
- Depth decoder architecture
- Loss functions

Usage:
    # Run UniDepthHead baseline (Swin + FPN)
    python -m vis4d.engine.run train opendet3d/zoo/gdino3d/experiments/geometry_backend_ablation.py:get_unidepth_head_config

    # Run UniDepthHead + DINOv2 (all blocks, grouped)
    python -m vis4d.engine.run train opendet3d/zoo/gdino3d/experiments/geometry_backend_ablation.py:get_unidepth_head_dino_config

    # Run DetAny3D geometry backend (last 4 blocks)
    python -m vis4d.engine.run train opendet3d/zoo/gdino3d/experiments/geometry_backend_ablation.py:get_detany3d_config

    # Run UniDepthV2 geometry backend (all blocks, grouped)
    python -m vis4d.engine.run train opendet3d/zoo/gdino3d/experiments/geometry_backend_ablation.py:get_unidepth_v2_config
"""

from __future__ import annotations

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.zoo.base import get_default_cfg

from opendet3d.zoo.gdino3d.base.callback import (
    get_callback_cfg,
    get_omni3d_evaluator_cfg,
)
from opendet3d.zoo.gdino3d.base.connector import get_data_connector_cfg
from opendet3d.zoo.gdino3d.base.data import get_data_cfg
from opendet3d.zoo.gdino3d.base.dataset.omni3d import (
    get_omni3d_test_cfg,
    get_omni3d_train_cfg,
)
from opendet3d.zoo.gdino3d.base.dataset.open import (
    get_av2_data_cfg,
    get_scannet_data_cfg,
)
from opendet3d.zoo.gdino3d.base.loss import get_loss_cfg
from opendet3d.zoo.gdino3d.base.model import (
    get_gdino3d_hyperparams_cfg,
    get_gdino3d_with_geometry_backend_cfg,
    GeometryBackendType,
)
from opendet3d.zoo.gdino3d.base.optim import get_optim_cfg
from opendet3d.zoo.gdino3d.base.pl import get_pl_cfg


def _build_geometry_backend_experiment(
    exp_name: str,
    geometry_backend_type: GeometryBackendType,
    use_geom_backend_loss: bool = False,
    detach_depth_latents: bool = True,
    dino_model: str = "vit_small",
    dino_pretrained: str = "",
    detany3d_decoder_pretrained: str = "",
    detany3d_decoder_dino_model: str | None = None,
    unidepth_v2_version: str = "v2-vits14",
    unidepth_v2_pretrained: str | None = None,
    unidepth_v2_encoder_pretrained: str | None = None,
    unidepth_v2_decoder_pretrained: str | None = None,
    use_mini_dataset: bool = False,
    mini_dataset_size: int = 100,
    # Geometry Backend Learning Rates (optional overrides)
    geom_encoder_lr_mult: float | None = None,
    geom_decoder_lr_mult: float | None = None,
    geom_projector_lr_mult: float | None = None,
) -> ExperimentConfig:
    """Build experiment config for geometry backend ablation.

    Args:
        exp_name: Experiment name.
        geometry_backend_type: Type of geometry backend.
        use_geom_backend_loss: Whether to use geometry backend's internal losses.
            Set True for detany3d and unidepth_v2 to use their native losses.
        detach_depth_latents: Whether to detach depth latents.
        dino_model: DINOv2 model variant for unidepth_head_dino and detany3d.
        dino_pretrained: Path to DINOv2 pretrained weights.
        detany3d_decoder_pretrained: Path to DetAny3D decoder weights.
        detany3d_decoder_dino_model: DINO model variant that decoder expects.
        unidepth_v2_version: UniDepthV2 version for unidepth_v2.
        unidepth_v2_pretrained: Path to full UniDepthV2 checkpoint.
        unidepth_v2_encoder_pretrained: Path to UniDepthV2 encoder-only weights.
        unidepth_v2_decoder_pretrained: Path to UniDepthV2 decoder-only weights.
        use_mini_dataset: If True, use mini dataset (cache_omni3d50_miniN) for fast testing.
        mini_dataset_size: Size of mini dataset (default: 100).
        geom_encoder_lr_mult: Learning rate multiplier for geometry backend encoder.
            If None, uses default from params (0.0 = freeze).
        geom_decoder_lr_mult: Learning rate multiplier for geometry backend decoder.
            If None, uses default from params (1.0 = full lr).
        geom_projector_lr_mult: Learning rate multiplier for geometry backend projector.
            If None, uses default from params (1.0 = full lr).

    Returns:
        ExperimentConfig.
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name=exp_name)
    config.use_checkpoint = True

    # High level hyper parameters
    params = get_gdino3d_hyperparams_cfg()

    # Override geometry backend LR if provided
    if geom_encoder_lr_mult is not None:
        params.geom_encoder_lr_mult = geom_encoder_lr_mult
    if geom_decoder_lr_mult is not None:
        params.geom_decoder_lr_mult = geom_decoder_lr_mult
    if geom_projector_lr_mult is not None:
        params.geom_projector_lr_mult = geom_projector_lr_mult

    # Convert FieldReference to actual values
    # FieldReference objects have a .get() method to retrieve the actual value
    geom_encoder_lr = params.geom_encoder_lr_mult.get() if hasattr(params.geom_encoder_lr_mult, 'get') else params.geom_encoder_lr_mult
    geom_decoder_lr = params.geom_decoder_lr_mult.get() if hasattr(params.geom_decoder_lr_mult, 'get') else params.geom_decoder_lr_mult
    geom_projector_lr = params.geom_projector_lr_mult.get() if hasattr(params.geom_projector_lr_mult, 'get') else params.geom_projector_lr_mult

    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_backend = class_config(HDF5Backend)

    test_datasets_cfg = []

    # Omni3D
    omni3d_data_root = "data/omni3d"
    omni3d_test_datasets = (
        "KITTI_test",
        "nuScenes_test",
        "SUNRGBD_test",
        "Hypersim_test",
        "ARKitScenes_test",
        "Objectron_test",
    )

    omni3d_train_data_cfg = get_omni3d_train_cfg(
        data_root=omni3d_data_root,
        data_backend=data_backend,
        use_mini_dataset=use_mini_dataset,
        mini_dataset_size=mini_dataset_size,
    )

    omni3d_test_data_cfg = get_omni3d_test_cfg(
        data_root=omni3d_data_root,
        test_datasets=omni3d_test_datasets,
        data_backend=data_backend,
        use_mini_dataset=use_mini_dataset,
        mini_dataset_size=mini_dataset_size,
    )

    test_datasets_cfg.append(omni3d_test_data_cfg)

    # Open Datasets
    test_datasets_cfg += [
        get_av2_data_cfg(data_backend=data_backend),
        get_scannet_data_cfg(data_backend=data_backend),
    ]

    config.data = get_data_cfg(
        train_datasets=omni3d_train_data_cfg,
        test_datasets=test_datasets_cfg,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
    )

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    # Prepare backend-specific kwargs
    backend_kwargs = {}
    if geometry_backend_type in ["unidepth_head_dino", "detany3d"]:
        backend_kwargs["dino_model"] = dino_model
        backend_kwargs["dino_pretrained"] = dino_pretrained
        if geometry_backend_type == "detany3d":
            backend_kwargs["detany3d_decoder_pretrained"] = detany3d_decoder_pretrained
            backend_kwargs["detany3d_decoder_dino_model"] = detany3d_decoder_dino_model
    elif geometry_backend_type == "unidepth_v2":
        backend_kwargs["unidepth_v2_version"] = unidepth_v2_version
        backend_kwargs["unidepth_v2_pretrained"] = unidepth_v2_pretrained
        backend_kwargs["unidepth_v2_encoder_pretrained"] = unidepth_v2_encoder_pretrained
        backend_kwargs["unidepth_v2_decoder_pretrained"] = unidepth_v2_decoder_pretrained

    print(f"\n[DEBUG CONFIG] Building geometry backend experiment:")
    print(f"  exp_name: {exp_name}")
    print(f"  geometry_backend_type: {geometry_backend_type}")
    print(f"  backend_kwargs: {backend_kwargs}")

    config.model, box_coder = get_gdino3d_with_geometry_backend_cfg(
        params=params,
        geometry_backend_type=geometry_backend_type,
        pretrained=None,  # Don't load pretrained weights - we'll load from checkpoint
        use_checkpoint=config.use_checkpoint,
        detach_depth_latents=detach_depth_latents,
        **backend_kwargs,
    )

    print(f"[DEBUG CONFIG] Model config created: {type(config.model)}")

    # Loss configuration
    # - For unidepth_head (original): use aux_depth_loss=True (external SILog via LossConnector)
    # - For all geometry backends: use use_geom_backend_loss=True (losses from backend.forward_train())
    #   - unidepth_head_dino: returns {"depth_loss": SILog}
    #   - detany3d: returns {"depth_loss": SILog, "loss_phi": SILog, "loss_theta": SILog}
    #   - unidepth_v2: returns UniDepthV2's native losses
    aux_depth_loss = geometry_backend_type == "unidepth_head"
    config.loss = get_loss_cfg(
        params,
        box_coder,
        aux_depth_loss=aux_depth_loss,
        use_geom_backend_loss=use_geom_backend_loss,
    )

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    # Train Swin backbone and language model at reduced learning rate
    # Also train GroundingDINO head + Geometry backend
    #
    # Note: We use lr_mult to control learning rates for different components
    param_groups = [
        # Train language model and backbone at 10% lr
        {"custom_keys": ["language_model"], "lr_mult": 0.1},
        {"custom_keys": ["backbone"], "lr_mult": 0.1},

        # Geometry Backend Encoder (DINOv2 / pixel_encoder)
        # Default: freeze (lr_mult=0.0)
        {
            "custom_keys": [
                "geometry_backend.dino_encoder",                    # UniDepthHeadDino, DetAny3D
                "geometry_backend.unidepth_model.pixel_encoder",    # UniDepthV2
            ],
            "lr_mult": geom_encoder_lr,
        },

        # Geometry Backend Decoder
        # Default: train at full lr (lr_mult=1.0)
        {
            "custom_keys": [
                "geometry_backend.depth_head",                      # UniDepthHeadDino
                "geometry_backend.depth_decoder",                   # DetAny3D
                "geometry_backend.unidepth_model.decoder",          # UniDepthV2
            ],
            "lr_mult": geom_decoder_lr,
        },

        # Geometry Backend Projector (DetAny3D only, UniDepthV2 latent_proj)
        # Default: train at full lr (lr_mult=1.0)
        {
            "custom_keys": [
                "geometry_backend.feature_projector",               # DetAny3D
                "geometry_backend.latent_proj",                     # UniDepthV2
            ],
            "lr_mult": geom_projector_lr,
        },
    ]
    config.optimizers = get_optim_cfg(params, param_groups=param_groups)

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector, config.test_data_connector = (
        get_data_connector_cfg()
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    omni3d_evaluator_cfg = get_omni3d_evaluator_cfg(
        data_root=omni3d_data_root,
        omni3d50=True,
        test_datasets=omni3d_test_datasets,
    )

    open_test_datasets = ["Argoverse_val", "ScanNet_val"]

    callbacks = get_callback_cfg(
        output_dir=config.output_dir,
        omni3d_evaluator=omni3d_evaluator_cfg,
        open_test_datasets=open_test_datasets,
    )

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    config.pl_trainer = get_pl_cfg(config, params)

    return config.value_mode()


# ============================================================================
# Experiment Configurations
# ============================================================================


def get_unidepth_head_config(*args, **kwargs) -> ExperimentConfig:
    """Get config for UniDepthHead baseline (original 3D-MOOD).

    This is the original 3D-MOOD configuration with:
    - Swin-T backbone for visual features
    - FPN for depth features
    - UniDepthHead for depth prediction
    - External SILog loss
    """
    return _build_geometry_backend_experiment(
        exp_name="gdino3d_geom-ablation_unidepth-head",
        geometry_backend_type="unidepth_head",
        use_geom_backend_loss=False,  # Use external SILog loss
    )


def get_unidepth_head_dino_config(*args, **kwargs) -> ExperimentConfig:
    """Get config for UniDepthHead + DINOv2-Small backend.

    This configuration uses:
    - Swin-T backbone for visual features (detection)
    - DINOv2-Small (all 12 blocks, grouped into 4 scales) for depth features
    - UniDepthHead for depth prediction (same as original 3D-MOOD)
    - External SILog loss
    - Pretrained DINOv2-Small weights from UniDepthV2

    This is a hybrid approach combining:
    - UniDepthV2's feature extraction (all DINOv2 blocks with grouping)
    - 3D-MOOD's depth head architecture
    """
    return _build_geometry_backend_experiment(
        exp_name="gdino3d_geom-ablation_unidepth-head-dino-vits",
        geometry_backend_type="unidepth_head_dino",
        use_geom_backend_loss=True,  # Use geometry backend loss (returns only depth_loss)
        dino_model="vit_small",  # DINOv2-Small (12 blocks, 384 embed_dim)
        dino_pretrained="checkpoints/dinov2_backbones/unidepth_v2_s_dinov2_backbone.pth",
    )


def get_detany3d_config(*args, **kwargs) -> ExperimentConfig:
    """Get config for DetAny3D geometry backend with DINOv2-Small.

    This configuration uses:
    - Swin-T backbone for visual features (detection)
    - DINOv2-Small (frozen) for depth features
    - Linear projector (384 → 1024) to match decoder's expected dimension
    - DetAny3D's Unidepth_Decoder for depth prediction (pretrained with vit_large)
    - DetAny3D's native SILog losses (depth + phi + theta)
    - Pretrained DINOv2-Small encoder weights from UniDepthV2 (for fair comparison)
    - Pretrained DetAny3D decoder weights (extracted from vit_large checkpoint)

    Note: The decoder was trained with vit_large (1024-dim), so we add a projector
    to adapt vit_small (384-dim) features. This allows us to use pretrained decoder
    weights while only training a small projector (~400K params).
    """
    return _build_geometry_backend_experiment(
        exp_name="gdino3d_geom-ablation_detany3d-vits",
        geometry_backend_type="detany3d",
        use_geom_backend_loss=True,  # Use DetAny3D's internal losses
        dino_model="vit_small",  # DINOv2-Small (12 blocks, 384 embed_dim)
        dino_pretrained="checkpoints/dinov2_backbones/unidepth_v2_s_dinov2_backbone.pth",  # Use UniDepthV2's DINOv2 for fair comparison
        detany3d_decoder_pretrained="checkpoints/depth_heads/detany3d_decoder.pth",
        detany3d_decoder_dino_model="vit_large",  # Decoder expects vit_large (1024-dim)
    )


def get_unidepth_v2_config(*args, **kwargs) -> ExperimentConfig:
    """Get config for UniDepthV2 geometry backend with DINOv2-Small.

    This configuration uses:
    - Swin-T backbone for visual features (detection)
    - DINOv2-Small (frozen) for depth features
    - UniDepthV2 Decoder for depth prediction
    - UniDepthV2's native losses (depth + camera + invariance + SSI + confidence)
    - Loads encoder and decoder separately from extracted weights

    Note: We use extracted weights instead of downloading from HuggingFace:
    - Encoder: checkpoints/dinov2_backbones/unidepth_v2_s_dinov2_backbone.pth
    - Decoder: checkpoints/depth_heads/unidepth_v2_decoder_vits.pth
    """
    return _build_geometry_backend_experiment(
        exp_name="gdino3d_geom-ablation_unidepth-v2-vits",
        geometry_backend_type="unidepth_v2",
        use_geom_backend_loss=True,  # Use UniDepthV2's internal losses
        unidepth_v2_version="v2-vits14",  # UniDepthV2-Small
        unidepth_v2_pretrained=None,  # Don't load full checkpoint
        unidepth_v2_encoder_pretrained="checkpoints/dinov2_backbones/unidepth_v2_s_dinov2_backbone.pth",
        unidepth_v2_decoder_pretrained="checkpoints/depth_heads/unidepth_v2_decoder_vits.pth",
    )


# ============================================================================
# Mini Dataset Configurations (for fast testing)
# ============================================================================


def get_detany3d_mini_config(*args, **kwargs) -> ExperimentConfig:
    """Get config for DetAny3D with mini dataset (100 samples per dataset).

    This is useful for fast testing and debugging. Uses the same model
    configuration as get_detany3d_config() but with mini datasets.

    Total samples: ~1,200 (vs ~175,000 for full dataset)
    Loading time: ~10x faster
    """
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


# Default config function for vis4d CLI
def get_config(config_name: str = "detany3d", *args, **kwargs) -> ExperimentConfig:
    """Get config by name.

    Args:
        config_name: Name of the config to load. Options:
            - "unidepth_head": UniDepthHead (original 3D-MOOD baseline)
            - "unidepth_head_dino": UniDepthHead + DINOv2
            - "detany3d": DetAny3D (DINOv2 + DetAny3D decoder) [default]
            - "unidepth_v2": UniDepthV2 (DINOv2 + UniDepthV2 decoder)

    Returns:
        ExperimentConfig for the specified geometry backend.
    """
    # Strip leading colon if present (vis4d adds it when using : syntax)
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

