"""3D-MOOD with Swin-T + CUT3R Fusion."""

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
)
from opendet3d.zoo.gdino3d.base.optim import get_optim_cfg
from opendet3d.zoo.gdino3d.base.pl import get_pl_cfg

# Import CUT3R Fusion model config
from opendet3d.zoo.gdino3d.base.model_cut3r import get_gdino3d_swin_tiny_cut3r_cfg


def get_config() -> ExperimentConfig:
    """Returns the 3D-MOOD with Swin-T + CUT3R Fusion."""
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="gdino3d_swin-t_omni3d_cut3r")

    config.use_checkpoint = True

    # High level hyper parameters
    params = get_gdino3d_hyperparams_cfg()

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
    # Use CUT3R Fusion model
    config.model, box_coder = get_gdino3d_swin_tiny_cut3r_cfg(
        params=params,
        pretrained="mm_gdino_swin_tiny_obj365_goldg_grit9m_v3det",
        use_checkpoint=config.use_checkpoint,
        # CUT3R Fusion parameters
        cut3r_checkpoint="CUT3R/src/cut3r_512_dpt_4_64.pth",
        cut3r_freeze=True,
        fusion_levels=[0, 1, 2, 3],  # All 4 levels
        fusion_strategies={
            0: {'type': 'full'},  # Full cross-attention
            1: {'type': 'full'},  # Full cross-attention
            2: {'type': 'full'},  # Full cross-attention
            3: {'type': 'full'},  # Full cross-attention
        },
        fusion_num_heads=8,
        fusion_dropout=0.1,
        use_relative_pos_bias=False,
        # Optional: Load pretrained 3D-MOOD checkpoint
        # Uncomment to load from pretrained 3D-MOOD model
        # pretrained_3dmood_checkpoint="checkpoints/gdino3d_swin-t_120e_omni3d_699f69.pt",
    )

    config.loss = get_loss_cfg(params, box_coder, aux_depth_loss=True)

    ######################################################
    ##                    Optimizer                     ##
    ######################################################
    config.optimizers = [get_optim_cfg(params)]

    ######################################################
    ##                    Callbacks                     ##
    ######################################################
    config.callbacks = get_callback_cfg(
        params=params,
        evaluators=[
            get_omni3d_evaluator_cfg(
                data_root=omni3d_data_root,
                test_datasets=omni3d_test_datasets,
            )
        ],
    )

    ######################################################
    ##                  Data Connector                  ##
    ######################################################
    config.train_data_connector = get_data_connector_cfg()
    config.test_data_connector = get_data_connector_cfg()

    ######################################################
    ##                  PL Trainer                      ##
    ######################################################
    config.pl_trainer = get_pl_cfg(params)

    return config

