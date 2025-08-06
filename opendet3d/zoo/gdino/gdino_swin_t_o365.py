"""Grounding DINO Swin-T pre-train on Objects365."""

from __future__ import annotations

import lightning.pytorch as pl
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import MultiStepLR

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig, ExperimentParameters
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.engine.callbacks import EvaluatorCallback, VisualizerCallback
from vis4d.engine.connectors import (
    CallbackConnector,
    DataConnector,
    LossConnector,
)
from vis4d.engine.loss_module import LossModule
from vis4d.eval.coco import COCODetectEvaluator
from vis4d.vis.image import BoundingBoxVisualizer
from vis4d.zoo.base import (
    get_default_callbacks_cfg,
    get_default_cfg,
    get_default_pl_trainer_cfg,
    get_lr_scheduler_cfg,
    get_optimizer_cfg,
)
from vis4d.zoo.base.datasets.coco import CONN_COCO_BBOX_EVAL

from opendet3d.op.base.swin import SwinTransformer
from opendet3d.op.fpp.channel_mapper import ChannelMapper
from opendet3d.op.detect.grounding_dino.loss import GroundingDINOLoss
from opendet3d.zoo.gdino.base.connector import (
    CONN_GDINO_LOSS,
    CONN_BBOX_2D_TRAIN,
    CONN_BBOX_2D_TEST,
    CONN_BBOX_2D_VIS,
)
from opendet3d.zoo.gdino.base.data import get_coco_detection_cfg
from opendet3d.zoo.gdino.base.model import get_gdino_cfg


def get_config() -> ExperimentConfig:
    """Returns the config of Grounding DINO."""
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(exp_name="gdino_swin-t_o365")

    config.use_checkpoint = False

    # High level hyper parameters
    params = ExperimentParameters()
    params.samples_per_gpu = 4
    params.workers_per_gpu = 4
    params.lr = 0.0002  # bs=64, lr=0.0002
    params.num_epochs = 12
    params.accumulate_grad_batches = 1
    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_root = "data/coco"
    train_split = "train2017"
    test_split = "val2017"

    data_backend = class_config(HDF5Backend)

    config.data = get_coco_detection_cfg(
        data_root=data_root,
        train_split=train_split,
        test_split=test_split,
        data_backend=data_backend,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
    )

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    basemodel = class_config(
        SwinTransformer,
        convert_weights=True,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.2,
        out_indices=(1, 2, 3),
        with_cp=config.use_checkpoint,
        pretrained="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
    )

    neck = class_config(
        ChannelMapper,
        in_channels=[192, 384, 768],
        out_channels=256,
        num_outs=4,
        kernel_size=1,
        norm="GroupNorm",
        num_groups=32,
        activation=None,
        bias=True,
    )

    config.model = get_gdino_cfg(
        basemodel=basemodel,
        neck=neck,
        mm_gdino=True,
        pretrained="mm_gdino_swin_tiny_obj365_goldg_grit9m_v3det",
        # mm_gdino=False,
        # pretrained="gdino_swin_tiny_obj365_goldg_cap4m",
        use_checkpoint=config.use_checkpoint,
    )

    config.loss = class_config(
        LossModule,
        losses=[
            {
                "loss": class_config(GroundingDINOLoss),
                "connector": class_config(
                    LossConnector, key_mapping=CONN_GDINO_LOSS
                ),
            },
        ],
    )

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = [
        get_optimizer_cfg(
            optimizer=class_config(AdamW, lr=params.lr, weight_decay=0.0001),
            lr_schedulers=[
                get_lr_scheduler_cfg(
                    class_config(MultiStepLR, milestones=[8, 11], gamma=0.1),
                ),
            ],
            param_groups=[
                {"custom_keys": ["backbone"], "lr_mult": 0.1},
                {"custom_keys": ["language_model"], "lr_mult": 0.1},
                {"custom_keys": ["absolute_pos_embed"], "decay_mult": 0.0},
            ],
        )
    ]

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector = class_config(
        DataConnector, key_mapping=CONN_BBOX_2D_TRAIN
    )

    config.test_data_connector = class_config(
        DataConnector, key_mapping=CONN_BBOX_2D_TEST
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    # Logger and Checkpoint
    callbacks = get_default_callbacks_cfg(config.output_dir)

    # Visualizer
    callbacks.append(
        class_config(
            VisualizerCallback,
            visualizer=class_config(BoundingBoxVisualizer, vis_freq=1),
            output_dir=config.output_dir,
            save_prefix="box2d",
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_BBOX_2D_VIS
            ),
        )
    )

    # Evaluator
    callbacks.append(
        class_config(
            EvaluatorCallback,
            evaluator=class_config(
                COCODetectEvaluator, data_root=data_root, split=test_split
            ),
            metrics_to_eval=["Det"],
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_COCO_BBOX_EVAL
            ),
        )
    )

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    # PL Trainer args
    pl_trainer = get_default_pl_trainer_cfg(config)
    pl_trainer.max_epochs = params.num_epochs

    pl_trainer.accumulate_grad_batches = params.accumulate_grad_batches
    pl_trainer.gradient_clip_val = 0.1
    config.pl_trainer = pl_trainer

    # PL Callbacks
    pl_callbacks: list[pl.Callback] = []
    config.pl_callbacks = pl_callbacks

    return config.value_mode()
