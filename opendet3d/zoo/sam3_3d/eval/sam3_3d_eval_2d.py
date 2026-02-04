"""SAM3_3D 2D Detection Evaluation Only.

This configuration is for testing SAM3's 2D detection capability without training.
It uses pretrained SAM3 weights to verify box/point/text prompt -> 2D bbox works.

Key features:
- train_datasets=None (no training)
- Mini test dataset (100 samples) for fast verification
- Uses vis4d framework with full dataloader pipeline

Usage:
    # Set environment first
    source /weka/oe-training-default/jieyuz2/improve_segments/miniconda3/bin/activate && \
    conda activate opendet3d && \
    export PYTHONPATH=/weka/oe-training-default/weikaih/3d_boundingbox_detection/Foundation3DDet/sam3_da3/Foundation3DDet:/weka/oe-training-default/weikaih/3d_boundingbox_detection/Foundation3DDet/sam3_da3/Foundation3DDet/UniDepth:/weka/oe-training-default/weikaih/3d_boundingbox_detection/Foundation3DDet/sam3_da3/Foundation3DDet/sam3:$PYTHONPATH && \
    cd /weka/oe-training-default/weikaih/3d_boundingbox_detection/Foundation3DDet/sam3_da3/Foundation3DDet && \

    # Run evaluation
    vis4d test --config opendet3d/zoo/sam3_3d/eval/sam3_3d_eval_2d.py --gpus 1
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
)
from opendet3d.zoo.gdino3d.base.optim import get_optim_cfg
from opendet3d.zoo.gdino3d.base.pl import get_pl_cfg

from opendet3d.zoo.sam3_3d.base.model import (
    get_sam3_3d_cfg,
    get_sam3_3d_hyperparams_cfg,
)
from opendet3d.zoo.sam3_3d.base.loss import get_sam3_3d_loss_cfg
from opendet3d.zoo.sam3_3d.base.connector import get_sam3_3d_data_connector_cfg
from opendet3d.zoo.sam3_3d.base.data import get_sam3_3d_data_cfg


def get_config() -> ExperimentConfig:
    """Returns the SAM3_3D 2D Detection Evaluation configuration."""
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="sam3_3d_eval_2d")

    config.use_checkpoint = True

    # High level hyper parameters
    # Using mini batch for evaluation
    params = get_sam3_3d_hyperparams_cfg(
        num_epochs=1,  # Not used for eval
        samples_per_gpu=1,  # Small batch to avoid OOM
        workers_per_gpu=2,
        base_lr=1e-4,  # Not used for eval
        freeze_sam3_backbone=True,
        freeze_geometry_backend_encoder=True,
    )

    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_backend = class_config(HDF5Backend)

    # Omni3D test configuration
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
    omni3d_test_data_cfg = get_omni3d_test_cfg(
        data_root=omni3d_data_root,
        test_datasets=omni3d_test_datasets,
        data_backend=data_backend,
        shape=sam3_image_shape,
        use_mini_dataset=True,
        mini_dataset_size=100,
    )

    test_datasets_cfg = [omni3d_test_data_cfg]

    # Use SAM3_3D custom data config with SAM3_3DCollator
    # train_datasets=None for evaluation only
    config.data = get_sam3_3d_data_cfg(
        train_datasets=None,  # No training data for eval
        test_datasets=test_datasets_cfg,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
        max_prompts_per_image=50,  # Max categories per image
        use_text_prompts=True,  # Include class names as text prompts
        # For evaluation, use pure text queries to test open-vocab detection
        text_query_prob=1.0,  # All text queries for eval
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
    # Still needed by vis4d framework even for eval
    config.optimizers = get_optim_cfg(params)

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector, config.test_data_connector = (
        get_sam3_3d_data_connector_cfg()
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    # Omni3D Evaluator for 2D/3D metrics
    omni3d_evaluator_cfg = get_omni3d_evaluator_cfg(
        data_root=omni3d_data_root,
        omni3d50=True,
        test_datasets=omni3d_test_datasets,
    )

    callbacks = get_callback_cfg(
        output_dir=config.output_dir,
        omni3d_evaluator=omni3d_evaluator_cfg,
        open_test_datasets=[],  # No open datasets evaluation
    )

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    config.pl_trainer = get_pl_cfg(config, params)

    return config.value_mode()
