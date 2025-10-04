"""3D-MOOD with Swin-T for ScanNet evaluation."""

from __future__ import annotations

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.zoo.base import get_default_cfg

from opendet3d.zoo.gdino3d.base.callback import get_callback_cfg
from opendet3d.zoo.gdino3d.base.connector import get_data_connector_cfg
from opendet3d.zoo.gdino3d.base.data import get_data_cfg
from opendet3d.zoo.gdino3d.base.dataset.open import get_scannet_data_cfg
from opendet3d.zoo.gdino3d.base.loss import get_loss_cfg
from opendet3d.zoo.gdino3d.base.model import (
    get_gdino3d_hyperparams_cfg,
    get_gdino3d_swin_tiny_cfg,
)
from opendet3d.zoo.gdino3d.base.optim import get_optim_cfg
from opendet3d.zoo.gdino3d.base.pl import get_pl_cfg


def get_config() -> ExperimentConfig:
    """Returns the 3D-MOOD with Swin-T for ScanNet evaluation."""
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="gdino3d_swin-t_scannet")

    config.use_checkpoint = True

    # High level hyper parameters
    params = get_gdino3d_hyperparams_cfg()

    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_backend = class_config(HDF5Backend)

    # Only ScanNet dataset for evaluation
    test_datasets_cfg = [get_scannet_data_cfg(data_backend=data_backend)]

    config.data = get_data_cfg(
        train_datasets=None,  # No training for evaluation-only config
        test_datasets=test_datasets_cfg,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
    )

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    config.model, box_coder = get_gdino3d_swin_tiny_cfg(
        params=params,
        pretrained="mm_gdino_swin_tiny_obj365_goldg_grit9m_v3det",
        use_checkpoint=config.use_checkpoint,
    )

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
    # Only ScanNet evaluation
    open_test_datasets = ["ScanNet_val"]

    callbacks = get_callback_cfg(
        output_dir=config.output_dir,
        omni3d_evaluator=None,  # No Omni3D evaluation
        open_test_datasets=open_test_datasets,
    )

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    config.pl_trainer = get_pl_cfg(config, params)

    return config.value_mode()
