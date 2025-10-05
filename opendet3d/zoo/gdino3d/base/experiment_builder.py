"""Unified experiment configuration builder."""

from __future__ import annotations
from typing import Callable

from ml_collections import ConfigDict
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
    get_gdino3d_swin_tiny_cfg,
)
from opendet3d.zoo.gdino3d.base.model_cut3r import (
    get_gdino3d_swin_tiny_cut3r_cfg,
)
from opendet3d.zoo.gdino3d.base.optim import get_optim_cfg
from opendet3d.zoo.gdino3d.base.pl import get_pl_cfg


def build_experiment(
    exp_name: str,
    fusion_mixin: Callable | None = None,
    lr_mixin: Callable | None = None,
    backbone_mixin: Callable | None = None,
    wandb_mixin: Callable | None = None,
    use_checkpoint: bool = True,
    **extra_kwargs
) -> ExperimentConfig:
    """
    Build experiment configuration by composing mixins.
    
    Args:
        exp_name: Experiment name
        fusion_mixin: Fusion strategy mixin function
        lr_mixin: Learning rate mixin function
        backbone_mixin: Backbone config mixin function
        wandb_mixin: WandB config mixin function
        use_checkpoint: Whether to use gradient checkpointing
        **extra_kwargs: Additional keyword arguments
    
    Returns:
        Complete experiment configuration
    
    Example:
        >>> from opendet3d.zoo.gdino3d.mixins.fusion_strategies import full_fusion_all_mixin
        >>> from opendet3d.zoo.gdino3d.mixins.wandb_configs import wandb_fusion_strategy_mixin
        >>> config = build_experiment(
        ...     exp_name="full_fusion_all",
        ...     fusion_mixin=full_fusion_all_mixin,
        ...     wandb_mixin=wandb_fusion_strategy_mixin,
        ... )
    """
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name=exp_name)
    config.use_checkpoint = use_checkpoint
    
    # Hyperparameters
    params = get_gdino3d_hyperparams_cfg()
    
    # Apply learning rate mixin
    if lr_mixin is not None:
        lr_config = lr_mixin()
        params.lr = lr_config['lr']
    
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
        data_root=omni3d_data_root, data_backend=data_backend
    )

    omni3d_test_data_cfg = get_omni3d_test_cfg(
        data_root=omni3d_data_root,
        test_datasets=omni3d_test_datasets,
        data_backend=data_backend,
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
    model_kwargs = {}
    use_fusion = False
    
    # Apply fusion mixin
    if fusion_mixin is not None:
        fusion_config = fusion_mixin()
        model_kwargs.update(fusion_config)
        
        # Check if fusion is enabled
        if fusion_config.get('fusion_levels') is not None:
            use_fusion = True
    
    # Apply backbone mixin
    if backbone_mixin is not None:
        backbone_config = backbone_mixin()
        model_kwargs.update(backbone_config)
    
    # Build model
    if use_fusion:
        # CUT3R fusion model
        config.model, box_coder = get_gdino3d_swin_tiny_cut3r_cfg(
            params=params,
            pretrained="mm_gdino_swin_tiny_obj365_goldg_grit9m_v3det",
            use_checkpoint=use_checkpoint,
            **model_kwargs,
            **extra_kwargs
        )
    else:
        # Baseline model (no fusion)
        config.model, box_coder = get_gdino3d_swin_tiny_cfg(
            params=params,
            pretrained="mm_gdino_swin_tiny_obj365_goldg_grit9m_v3det",
            use_checkpoint=use_checkpoint,
        )
    
    # Loss
    config.loss = get_loss_cfg(params, box_coder, aux_depth_loss=True)
    
    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = get_optim_cfg(params)
    
    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector, config.test_data_connector = (
        get_data_connector_cfg()
    )
    
    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    # Omni3D Evaluator
    omni3d_evaluator_cfg = get_omni3d_evaluator_cfg(
        data_root=omni3d_data_root,
        omni3d50=True,
        test_datasets=omni3d_test_datasets,
    )

    # Open Detect3D Evaluator
    open_test_datasets = ["Argoverse_val", "ScanNet_val"]

    callbacks = get_callback_cfg(
        output_dir=config.output_dir,
        omni3d_evaluator=omni3d_evaluator_cfg,
        open_test_datasets=open_test_datasets,
    )
    
    config.callbacks = callbacks
    
    ######################################################
    ##                     WANDB                        ##
    ######################################################
    # WandB configuration (optional)
    if wandb_mixin is not None:
        wandb_config = wandb_mixin(exp_name)
        config.wandb = ConfigDict(wandb_config)
    
    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    config.pl_trainer = get_pl_cfg(config, params)
    
    return config.value_mode()

