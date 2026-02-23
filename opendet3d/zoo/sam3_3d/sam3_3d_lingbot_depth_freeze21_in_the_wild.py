"""SAM3_3D with LingBot-Depth on InTheWild - Text mode evaluation.

Evaluates SAM3_3D + LingBot-Depth (freeze21, 12e) on the InTheWild benchmark:
  800+ open-vocabulary categories from COCO/LVIS/Objects365 images
  with human-annotated 3D bounding boxes.

Mode: text prompts, monocular depth (no GT depth input).
"""

from __future__ import annotations

import os

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig
from vis4d.data.data_pipe import DataPipe
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.zoo.base import get_default_cfg

from opendet3d.zoo.gdino3d.base.callback import get_in_the_wild_eval_callbacks
from opendet3d.zoo.gdino3d.base.dataset.omni3d import get_omni3d_train_cfg
from opendet3d.zoo.gdino3d.base.dataset.transform import get_test_transforms_cfg
from opendet3d.zoo.gdino3d.base.pl import get_pl_cfg

from opendet3d.zoo.sam3_3d.base.optim import get_sam3_3d_optim_cfg
from opendet3d.zoo.sam3_3d.base.model import (
    get_sam3_3d_cfg,
    get_sam3_3d_hyperparams_cfg,
)
from opendet3d.zoo.sam3_3d.base.loss import get_sam3_3d_loss_cfg
from opendet3d.zoo.sam3_3d.base.connector import (
    get_sam3_3d_data_connector_cfg,
    SAM3_3DDetect3DEvalConnector,
    SAM3_3DVisConnector,
)
from opendet3d.zoo.sam3_3d.base.data import get_sam3_3d_data_cfg
from opendet3d.data.datasets.in_the_wild import (
    InTheWild3DDataset,
    load_in_the_wild_class_map,
)


def get_config() -> ExperimentConfig:
    """Returns SAM3_3D + LingBot text eval config for InTheWild."""
    config = get_default_cfg(
        exp_name="sam3_3d_lingbot_depth_freeze21_in_the_wild"
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
    sam3_image_shape = (1008, 1008)
    omni3d_data_root = "data/omni3d"
    in_the_wild_data_root = "data/in_the_wild"

    omni3d_train_data_cfg = get_omni3d_train_cfg(
        data_root=omni3d_data_root,
        data_backend=data_backend,
        shape=sam3_image_shape,
    )

    annotation_path = os.path.join(
        in_the_wild_data_root, "annotations/InTheWild_val.json"
    )
    class_map = load_in_the_wild_class_map(annotation_path)

    itw_test_data_cfg = class_config(
        DataPipe,
        datasets=class_config(
            InTheWild3DDataset,
            data_root=in_the_wild_data_root,
            dataset_name="InTheWild_val",
            class_map=class_map,
            with_depth=False,
            data_backend=None,
            cache_as_binary=True,
            cached_file_path="data/in_the_wild/val.pkl",
        ),
        preprocess_fn=get_test_transforms_cfg(shape=sam3_image_shape, with_depth=False),
    )

    config.data = get_sam3_3d_data_cfg(
        train_datasets=omni3d_train_data_cfg,
        test_datasets=[itw_test_data_cfg],
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
        max_prompts_per_image=50,
        use_text_prompts=True,
        use_geometry_prompts=True,
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
    from vis4d.data.const import AxisMode
    from vis4d.engine.callbacks import VisualizerCallback
    from vis4d.vis.image.bbox3d_visualizer import BoundingBox3DVisualizer
    from vis4d.vis.image.canvas import PillowCanvasBackend
    from vis4d.zoo.base import get_default_callbacks_cfg

    callbacks = get_default_callbacks_cfg()

    callbacks.extend(
        get_in_the_wild_eval_callbacks(
            data_root=in_the_wild_data_root,
            output_dir=config.output_dir,
            test_connector=class_config(SAM3_3DDetect3DEvalConnector),
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
                vis_freq=50,
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
