"""SAM3_3D with Omni3D Mini Dataset - For Fast Testing.

This configuration is for quick testing and debugging with mini datasets
(100 samples per dataset). Use this to verify the pipeline works before
running full training.

Total samples: ~1,200 (vs ~175,000 for full dataset)
Loading time: ~10x faster

Usage:
    vis4d fit --config opendet3d/zoo/sam3_3d/sam3_3d_omni3d_mini.py --gpus 2

For even smaller batch to avoid OOM:
    vis4d fit --config opendet3d/zoo/sam3_3d/sam3_3d_omni3d_mini.py --gpus 2 \
        --config.params.samples_per_gpu=1
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
    """Returns the SAM3_3D with Omni3D Mini Dataset configuration."""
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="sam3_3d_omni3d_mini100")

    config.use_checkpoint = True

    # High level hyper parameters
    # Reduced batch size for mini testing (avoid OOM with SAM3 1008x1008)
    # Note: Learning rate control is handled by param_groups in optim.py
    params = get_sam3_3d_hyperparams_cfg(
        num_epochs=12,
        samples_per_gpu=2,  # Reduced from 4 for OOM safety
        workers_per_gpu=4,
        base_lr=1e-4,
        accumulate_grad_batches=2,  # Maintain effective batch size
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
    # 1008 / 14 = 72 tokens per dimension, matching RoPE freqs_cis
    sam3_image_shape = (1008, 1008)

    # Use mini dataset (100 samples per dataset) for fast testing
    omni3d_train_data_cfg = get_omni3d_train_cfg(
        data_root=omni3d_data_root,
        data_backend=data_backend,
        shape=sam3_image_shape,
        use_mini_dataset=True,
        mini_dataset_size=100,
    )

    omni3d_test_data_cfg = get_omni3d_test_cfg(
        data_root=omni3d_data_root,
        test_datasets=omni3d_test_datasets,
        data_backend=data_backend,
        shape=sam3_image_shape,
        with_depth=True,  # Must match training config to avoid transform errors
        use_mini_dataset=True,
        mini_dataset_size=100,
    )

    test_datasets_cfg.append(omni3d_test_data_cfg)

    # Use SAM3_3D custom data config with SAM3_3DCollator
    # This converts per-image data to per-prompt batch (SAM3_3DBatchedInputs)
    # SAM3 original design: per-category queries with multi-instance targets
    config.data = get_sam3_3d_data_cfg(
        train_datasets=omni3d_train_data_cfg,
        test_datasets=test_datasets_cfg,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
        max_prompts_per_image=50,  # Max categories per image
        use_text_prompts=True,  # Include class names as text prompts
        # Text/Visual query ratio (SAM3 original design)
        text_query_prob=0.7,  # SAM3 recommended: 70% text, 30% visual
    )

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    config.model, box_coder = get_sam3_3d_cfg(
        params=params,
        sam3_checkpoint="pretrained/sam3/sam3_detector.pt",  # Use extracted SAM3 detector weights
        geometry_backend_type="unidepth_v2",
    )

    config.loss = get_sam3_3d_loss_cfg(params, box_coder)

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    # SAM3_3D-specific param_groups controlled by params
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
    # Omni3D Evaluator
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
