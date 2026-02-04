"""SAM3_3D with Omni3D dataset - TEXT PROMPTS ONLY (no geometry prompts).

This configuration is identical to sam3_3d_omni3d.py but with:
- use_geometry_prompts=False (no geometry/visual prompts)
- Only text prompts are used for training

This is faster because:
- Each category creates only ONE query (text) instead of TWO (text + geometry)
- Reduces prompt count by ~50%

Expected speedup: ~1.5-2x compared to full config with geometry prompts.

Usage:
    vis4d fit --config opendet3d/zoo/sam3_3d/sam3_3d_omni3d_text_only.py --gpus 8
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

from opendet3d.zoo.sam3_3d.base.optim import get_sam3_3d_optim_cfg

from opendet3d.zoo.sam3_3d.base.model import (
    get_sam3_3d_cfg,
    get_sam3_3d_hyperparams_cfg,
)
from opendet3d.zoo.sam3_3d.base.loss import get_sam3_3d_loss_cfg
from opendet3d.zoo.sam3_3d.base.connector import get_sam3_3d_data_connector_cfg
from opendet3d.zoo.sam3_3d.base.data import get_sam3_3d_data_cfg


def get_config() -> ExperimentConfig:
    """Returns the SAM3_3D with Omni3D configuration (text prompts only)."""
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="sam3_3d_omni3d_text_only")

    config.use_checkpoint = True

    # High level hyper parameters
    params = get_sam3_3d_hyperparams_cfg(
        num_epochs=12,
        samples_per_gpu=4,
        workers_per_gpu=4,
        base_lr=1e-4,
    )

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

    # SAM3 expects 1008x1008 images (pre-trained with img_size=1008, patch_size=14)
    sam3_image_shape = (1008, 1008)

    omni3d_train_data_cfg = get_omni3d_train_cfg(
        data_root=omni3d_data_root,
        data_backend=data_backend,
        shape=sam3_image_shape,
    )

    omni3d_test_data_cfg = get_omni3d_test_cfg(
        data_root=omni3d_data_root,
        test_datasets=omni3d_test_datasets,
        data_backend=data_backend,
        shape=sam3_image_shape,
        with_depth=True,
    )

    test_datasets_cfg.append(omni3d_test_data_cfg)

    # Use SAM3_3D custom data config with SAM3_3DCollator
    # TEXT PROMPTS ONLY - no geometry prompts
    # This is faster because each category creates only ONE query
    config.data = get_sam3_3d_data_cfg(
        train_datasets=omni3d_train_data_cfg,
        test_datasets=test_datasets_cfg,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
        max_prompts_per_image=50,
        use_text_prompts=True,
        # KEY DIFFERENCE: No geometry prompts
        use_geometry_prompts=False,
    )

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    config.model, box_coder = get_sam3_3d_cfg(
        params=params,
        sam3_checkpoint="pretrained/sam3/sam3_detector.pt",
        geometry_backend_type="unidepth_v2",
    )

    config.loss = get_sam3_3d_loss_cfg(params, box_coder)

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = get_sam3_3d_optim_cfg(
        params,
        freeze_backbone=params.freeze_backbone,
        freeze_all_pretrained=params.freeze_all_pretrained,
    )

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector, config.test_data_connector = (
        get_sam3_3d_data_connector_cfg()
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    omni3d_evaluator_cfg = get_omni3d_evaluator_cfg(
        data_root=omni3d_data_root,
        omni3d50=True,
        test_datasets=omni3d_test_datasets,
    )

    callbacks = get_callback_cfg(
        output_dir=config.output_dir,
        omni3d_evaluator=omni3d_evaluator_cfg,
        open_test_datasets=[],
    )

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    config.pl_trainer = get_pl_cfg(config, params)

    return config.value_mode()
