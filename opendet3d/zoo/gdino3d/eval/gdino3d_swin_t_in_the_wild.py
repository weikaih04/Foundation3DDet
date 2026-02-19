"""3D-MOOD with Swin-T on InTheWild evaluation.

Evaluates GDino3D (Swin-T, 120e) on the InTheWild benchmark:
  800+ open-vocabulary categories from COCO/LVIS/Objects365 images
  with human-annotated 3D bounding boxes.

Mode: text prompts, monocular depth from UniDepth backend.
"""

from __future__ import annotations

import os

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig
from vis4d.data.data_pipe import DataPipe
from vis4d.engine.callbacks import EvaluatorCallback, VisualizerCallback
from vis4d.engine.connectors import CallbackConnector
from vis4d.zoo.base import get_default_cfg, get_default_callbacks_cfg

from opendet3d.zoo.gdino3d.base.callback import (
    get_in_the_wild_evaluator_cfg,
    get_visualizer_callback_cfg,
)
from opendet3d.zoo.gdino3d.base.connector import (
    CONN_COCO_DET3D_EVAL,
    get_data_connector_cfg,
)
from opendet3d.zoo.gdino3d.base.data import get_data_cfg
from opendet3d.zoo.gdino3d.base.dataset.transform import get_test_transforms_cfg
from opendet3d.zoo.gdino3d.base.loss import get_loss_cfg
from opendet3d.zoo.gdino3d.base.model import (
    get_gdino3d_hyperparams_cfg,
    get_gdino3d_swin_tiny_cfg,
)
from opendet3d.zoo.gdino3d.base.optim import get_optim_cfg
from opendet3d.zoo.gdino3d.base.pl import get_pl_cfg
from opendet3d.data.datasets.in_the_wild import (
    InTheWild3DDataset,
    load_in_the_wild_class_map,
)


def get_config() -> ExperimentConfig:
    """Returns GDino3D Swin-T eval config for InTheWild."""
    config = get_default_cfg(exp_name="gdino3d_swin-t_in_the_wild")

    config.use_checkpoint = True

    params = get_gdino3d_hyperparams_cfg()

    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    in_the_wild_data_root = "data/in_the_wild"
    gdino3d_image_shape = (800, 1333)

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
        preprocess_fn=get_test_transforms_cfg(shape=gdino3d_image_shape),
    )

    config.data = get_data_cfg(
        train_datasets=None,
        test_datasets=[itw_test_data_cfg],
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
    )

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    config.model, box_coder = get_gdino3d_swin_tiny_cfg(
        params=params,
        pretrained=None,
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
    callbacks = get_default_callbacks_cfg()

    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=get_in_the_wild_evaluator_cfg(
                data_root=in_the_wild_data_root
            ),
            metrics_to_eval=["3D"],
            save_predictions=True,
            output_dir=config.output_dir,
            save_prefix="detection",
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_COCO_DET3D_EVAL
            ),
        )
    )

    callbacks.extend(get_visualizer_callback_cfg(output_dir=config.output_dir))

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    config.pl_trainer = get_pl_cfg(config, params)

    return config.value_mode()
