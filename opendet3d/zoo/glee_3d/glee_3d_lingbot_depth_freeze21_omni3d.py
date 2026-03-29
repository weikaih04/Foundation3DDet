"""GLEE_3D + LingbotDepth (freeze21) on Omni3D.

First training config for GLEE_3D. Uses:
- GLEE Swin-L (Plus) Stage3 Scaleup pretrained
- LingbotDepth geometry backend (encoder freeze 21/24 blocks)
- 1024x1024 fixed square input
- Omni3D dataset (same transforms as SAM3_3D, GLEE_3D collator)
- Canonical rotation
"""

from __future__ import annotations

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig
from vis4d.data.io import HDF5Backend

from opendet3d.zoo.gdino3d.base.dataset.omni3d import (
    get_omni3d_test_cfg,
    get_omni3d_train_cfg,
)
from opendet3d.zoo.gdino3d.base.pl import get_pl_cfg
from opendet3d.zoo.gdino3d.base.callback import get_callback_cfg
from opendet3d.zoo.gdino3d.base.evaluator import get_omni3d_evaluator_cfg
from opendet3d.zoo.gdino3d.base.default_config import get_default_cfg

from opendet3d.zoo.glee_3d.base.model import (
    get_glee_3d_cfg,
    get_glee_3d_hyperparams_cfg,
)
from opendet3d.zoo.glee_3d.base.data import (
    get_glee_3d_data_cfg,
    get_glee_3d_data_connector_cfg,
)
from opendet3d.zoo.glee_3d.base.optim import get_glee_3d_optim_cfg
from opendet3d.zoo.glee_3d.base.loss import get_glee_3d_loss_cfg


def get_config() -> ExperimentConfig:
    """GLEE_3D + LingbotDepth freeze21 on Omni3D (canonical rotation)."""
    # ================================================================
    # General config
    # ================================================================
    config = get_default_cfg(
        exp_name="glee_3d_lingbot_f21_omni3d_canonical"
    )
    config.use_checkpoint = True

    # ================================================================
    # Hyperparameters
    # ================================================================
    params = get_glee_3d_hyperparams_cfg(
        num_epochs=12,
        samples_per_gpu=2,  # GLEE Swin-L is larger, use smaller batch
        workers_per_gpu=4,
        base_lr=1e-4,
    )
    config.params = params

    # Image shape for GLEE_3D (1024x1024 square, divisible by 32)
    glee_image_shape = (1024, 1024)

    # ================================================================
    # Datasets (reuse SAM3_3D dataset configs + transforms)
    # ================================================================
    data_backend = class_config(HDF5Backend)

    omni3d_train_data_cfg = get_omni3d_train_cfg(
        data_backend=data_backend,
        shape=glee_image_shape,
        truncation_thres=0.50,
    )

    omni3d_test_data_cfg = get_omni3d_test_cfg(
        data_backend=data_backend,
        shape=glee_image_shape,
    )

    # Data config: same datasets/transforms as SAM3_3D, GLEE_3D collator
    config.data = get_glee_3d_data_cfg(
        train_datasets=omni3d_train_data_cfg,
        test_datasets=[omni3d_test_data_cfg],
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
        image_shape=glee_image_shape,
    )

    # ================================================================
    # Model
    # ================================================================
    config.model, box_coder = get_glee_3d_cfg(
        params=params,
        glee_checkpoint="pretrained/glee/GLEE_Plus_scaleup.pth",
        geometry_backend_type="lingbot_depth",
        lingbot_pretrained="pretrained/lingbot-depth/postrain-dc-vitl14/model.pt",
        lingbot_encoder_freeze_blocks=21,
        backbone_freeze_blocks=0,
        canonical_rotation=True,
    )

    # ================================================================
    # Loss
    # ================================================================
    config.loss = get_glee_3d_loss_cfg(
        params=params,
        box_coder_cfg=box_coder,
        loss_3d_scale=1.0,
        loss_geom_scale=5.0,
    )

    # ================================================================
    # Data Connectors
    # ================================================================
    config.train_data_connector, config.loss_connector = (
        get_glee_3d_data_connector_cfg()
    )

    # ================================================================
    # Optimizer
    # ================================================================
    config.optimizers = get_glee_3d_optim_cfg(
        params,
        freeze_backbone=params.freeze_backbone,
        freeze_all_pretrained=params.freeze_all_pretrained,
    )

    # ================================================================
    # Callbacks
    # ================================================================
    omni3d_evaluator_cfg = get_omni3d_evaluator_cfg(
        params=params,
        test_data_cfg=omni3d_test_data_cfg,
    )
    config.callbacks = get_callback_cfg(
        output_dir=config.output_dir,
        omni3d_evaluator=omni3d_evaluator_cfg,
    )

    # ================================================================
    # PyTorch Lightning
    # ================================================================
    config.pl_trainer = get_pl_cfg(config, params)

    return config.value_mode()
