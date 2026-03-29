"""GLEE_3D + LingbotDepth on Omni3D Mini Dataset.

For fast profiling and debugging. Uses mini datasets (100 samples).

Usage:
    conda activate glee3d
    export PYTHONPATH=...  # see CLAUDE.md
    vis4d fit --config opendet3d/zoo/glee_3d/glee_3d_lingbot_depth_freeze21_omni3d_mini.py --gpus 1
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
from opendet3d.zoo.gdino3d.base.dataset.omni3d import (
    get_omni3d_test_cfg,
    get_omni3d_train_cfg,
)
from opendet3d.zoo.gdino3d.base.pl import get_pl_cfg

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
    """GLEE_3D + LingbotDepth mini config for debugging."""
    # ================================================================
    # General
    # ================================================================
    config = get_default_cfg(
        exp_name="glee_3d_lingbot_f21_omni3d_mini"
    )
    config.use_checkpoint = True

    params = get_glee_3d_hyperparams_cfg(
        num_epochs=2,
        samples_per_gpu=2,
        workers_per_gpu=2,
        base_lr=1e-4,
    )
    config.params = params

    # ================================================================
    # Datasets (mini = 100 samples)
    # ================================================================
    data_backend = class_config(HDF5Backend)
    glee_image_shape = (1024, 1024)

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
        shape=glee_image_shape,
        use_mini_dataset=True,
        mini_dataset_size=100,
    )

    omni3d_test_data_cfg = get_omni3d_test_cfg(
        data_root=omni3d_data_root,
        test_datasets=omni3d_test_datasets,
        data_backend=data_backend,
        shape=glee_image_shape,
        with_depth=True,
        use_mini_dataset=True,
        mini_dataset_size=100,
    )

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
    config.train_data_connector, config.test_data_connector = (
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
        data_root=omni3d_data_root,
        omni3d50=True,
        test_datasets=omni3d_test_datasets,
        use_mini_dataset=True,
    )
    config.callbacks = get_callback_cfg(
        output_dir=config.output_dir,
        omni3d_evaluator=omni3d_evaluator_cfg,
        open_test_datasets=[],
    )

    # ================================================================
    # PL CLI
    # ================================================================
    config.pl_trainer = get_pl_cfg(config, params)

    return config.value_mode()
