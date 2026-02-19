"""SAM3_3D with LingBot-Depth on Omni3D - GT Box Oracle Evaluation.

Same as sam3_3d_lingbot_depth_freeze21_omni3d.py but with oracle_eval=True:
each GT 2D box is fed as a geometry prompt, model predicts 3D for each box
independently (one-to-one). Measures pure 3D regression quality without
detection noise (no NMS, no score filter).

Freeze settings:
- LingBot encoder: freeze 21/24 blocks, train blocks[21..23]
- SAM3 ViT backbone: freeze 28/32 blocks, train blocks[28..31]
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
    """Returns oracle eval config for SAM3_3D + LingBot-Depth on full Omni3D."""
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(
        exp_name="sam3_3d_lingbot_depth_freeze21_omni3d_oracle"
    )

    config.use_checkpoint = True

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

    omni3d_data_root = "data/omni3d"
    omni3d_test_datasets = (
        "KITTI_test",
        "nuScenes_test",
        "SUNRGBD_test",
        "Hypersim_test",
        "ARKitScenes_test",
        "Objectron_test",
    )

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

    # Oracle mode: test collator creates per-GT-box geometry prompts
    config.data = get_sam3_3d_data_cfg(
        train_datasets=omni3d_train_data_cfg,
        test_datasets=test_datasets_cfg,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
        max_prompts_per_image=50,
        use_text_prompts=True,
        use_geometry_prompts=True,
        oracle_eval=True,  # Each GT box = one geometry prompt
    )

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    config.model, box_coder = get_sam3_3d_cfg(
        params=params,
        sam3_checkpoint="pretrained/sam3/sam3_detector.pt",
        geometry_backend_type="lingbot_depth",
        lingbot_encoder_freeze_blocks=21,
        backbone_freeze_blocks=28,
        oracle_eval=True,  # Top-1 per prompt, no NMS
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

    from vis4d.engine.callbacks import EvaluatorCallback, VisualizerCallback
    from vis4d.data.const import AxisMode
    from vis4d.vis.image.bbox3d_visualizer import BoundingBox3DVisualizer
    from vis4d.vis.image.canvas import PillowCanvasBackend
    from vis4d.zoo.base import get_default_callbacks_cfg

    callbacks = get_default_callbacks_cfg()

    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=omni3d_evaluator_cfg,
            metrics_to_eval=["2D", "3D"],
            save_predictions=True,
            output_dir=config.output_dir,
            save_prefix="detection",
            test_connector=class_config(SAM3_3DEvalConnector),
        )
    )

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
            test_connector=class_config(
                SAM3_3DVisConnector, score_threshold=0.0
            ),
        )
    )

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    config.pl_trainer = get_pl_cfg(config, params)

    return config.value_mode()
