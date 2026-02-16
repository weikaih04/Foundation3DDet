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

from opendet3d.zoo.gdino3d.base.callback import get_omni3d_evaluator_cfg
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
from opendet3d.zoo.sam3_3d.base.connector import (
    get_sam3_3d_data_connector_cfg,
    SAM3_3DEvalConnector,
    SAM3_3DVisConnector,
)
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
    # IMPORTANT: use_mini_dataset=True to match the test data
    # Otherwise evaluator uses full GT (9314 images) while predictions are for 100 images
    omni3d_evaluator_cfg = get_omni3d_evaluator_cfg(
        data_root=omni3d_data_root,
        omni3d50=True,
        test_datasets=omni3d_test_datasets,
        use_mini_dataset=True,  # Match test data subset
    )

    # SAM3_3D uses custom connectors since SAM3_3DBatchedInputs is a dataclass,
    # not a dict. vis4d's CallbackConnector (dict access) doesn't work.
    from vis4d.data.const import AxisMode
    from vis4d.engine.callbacks import EvaluatorCallback, VisualizerCallback
    from vis4d.vis.image.bbox3d_visualizer import BoundingBox3DVisualizer
    from vis4d.vis.image.canvas import PillowCanvasBackend
    from vis4d.zoo.base import get_default_callbacks_cfg

    callbacks = get_default_callbacks_cfg()

    # Evaluator with SAM3_3D-specific connector
    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=omni3d_evaluator_cfg,
            metrics_to_eval=["3D"],
            save_predictions=True,
            output_dir=config.output_dir,
            save_prefix="detection",
            test_connector=class_config(SAM3_3DEvalConnector),
        )
    )

    # Visualizer with SAM3_3D-specific connector
    callbacks.append(
        class_config(
            VisualizerCallback,
            visualizer=class_config(
                BoundingBox3DVisualizer,
                axis_mode=AxisMode.OPENCV,
                width=4,
                camera_near_clip=0.01,
                plot_heading=False,
                vis_freq=1,
                plot_trajectory=False,
                canvas=class_config(PillowCanvasBackend, font_size=16),
                save_boxes3d=True,
            ),
            output_dir=config.output_dir,
            save_prefix="box3d",
            test_connector=class_config(SAM3_3DVisConnector, score_threshold=0.1),
        )
    )

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    config.pl_trainer = get_pl_cfg(config, params)

    return config.value_mode()
