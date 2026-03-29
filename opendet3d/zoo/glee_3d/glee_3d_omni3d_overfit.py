"""GLEE_3D Single Batch Overfit Test.

Copied from SAM3_3D's sam3_3d_omni3d_overfit.py.
Deterministic: no aug, no shuffle, seed=42, 2 samples, 500 epochs.

Usage:
    vis4d fit --config opendet3d/zoo/glee_3d/glee_3d_omni3d_overfit.py --gpus 1
"""

from __future__ import annotations

import pytorch_lightning as pl

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.zoo.base import get_default_cfg
from vis4d.data.data_pipe import DataPipe
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.data.transforms.base import compose

from opendet3d.zoo.gdino3d.base.dataset.omni3d import get_omni3d_dataset_cfg
from opendet3d.zoo.gdino3d.base.dataset.transform import (
    get_deterministic_train_transforms_cfg,
)
from opendet3d.zoo.glee_3d.base.model import (
    get_glee_3d_cfg,
    get_glee_3d_hyperparams_cfg,
)
from opendet3d.zoo.glee_3d.base.optim import get_glee_3d_optim_cfg
from opendet3d.zoo.glee_3d.base.loss import get_glee_3d_loss_cfg
from opendet3d.zoo.glee_3d.base.data import get_glee_3d_data_connector_cfg
from opendet3d.zoo.glee_3d.base.connector import GLEE_3DCollator

from vis4d.config.typing import DataConfig
from vis4d.zoo.base import get_train_dataloader_cfg


def get_config() -> ExperimentConfig:
    """GLEE_3D Single Batch Overfit (deterministic, no aug)."""
    # Fixed seed (same as SAM3_3D)
    pl.seed_everything(42, workers=True)

    config = get_default_cfg(exp_name="glee_3d_overfit_test")
    config.use_checkpoint = False
    config.log_every_n_steps = 1

    params = get_glee_3d_hyperparams_cfg(
        num_epochs=500,
        samples_per_gpu=1,
        workers_per_gpu=0,
        base_lr=1e-4,
    )
    config.params = params

    # ============================================================
    # Data: deterministic, no aug, mini dataset (2 samples)
    # ============================================================
    data_backend = class_config(HDF5Backend)
    glee_image_shape = (1024, 1024)

    # Dataset with deterministic transforms (no random resize/flip)
    train_dataset_cfg = get_omni3d_dataset_cfg(
        data_root="data/omni3d",
        datasets=("KITTI_train",),
        data_backend=data_backend,
        omni3d50=True,
        remove_empty=True,
        with_depth=True,
        cache_as_binary=True,
        use_mini_dataset=True,
        mini_dataset_size=2,
    )
    train_preprocess_cfg = get_deterministic_train_transforms_cfg(
        shape=glee_image_shape
    )
    train_data_cfg = class_config(
        DataPipe,
        datasets=train_dataset_cfg,
        preprocess_fn=train_preprocess_cfg,
    )

    # Deterministic collator: no visual prompt, no jitter
    _overfit_collator = GLEE_3DCollator(
        filter_empty_boxes=True,
        visual_prompt_prob=0.0,  # No visual prompts (deterministic)
        box_noise_std=0.0,       # No jitter
        object_text_prob=0.0,    # No "object" replacement
    )
    def glee_3d_collate_fn(batch, **kwargs):
        return _overfit_collator(batch)

    batchprocess_cfg = class_config(compose, transforms=[class_config(ToTensor)])

    data = DataConfig()
    data.train_dataloader = get_train_dataloader_cfg(
        datasets_cfg=train_data_cfg,
        samples_per_gpu=1,
        workers_per_gpu=0,
        batchprocess_cfg=batchprocess_cfg,
        collate_fn=glee_3d_collate_fn,
        shuffle=False,
    )
    data.test_dataloader = data.train_dataloader  # dummy
    config.data = data

    # ============================================================
    # Model
    # ============================================================
    config.model, box_coder = get_glee_3d_cfg(
        params=params,
        glee_checkpoint="pretrained/glee/GLEE_Plus_scaleup.pth",
        geometry_backend_type="lingbot_depth",
        lingbot_pretrained="pretrained/lingbot-depth/postrain-dc-vitl14/model.pt",
        lingbot_encoder_freeze_blocks=21,
        canonical_rotation=True,
    )

    # ============================================================
    # Loss
    # ============================================================
    config.loss = get_glee_3d_loss_cfg(
        params=params,
        box_coder_cfg=box_coder,
        loss_3d_scale=1.0,
        loss_geom_scale=5.0,
    )

    # ============================================================
    # Connectors
    # ============================================================
    config.train_data_connector, config.test_data_connector = (
        get_glee_3d_data_connector_cfg()
    )

    # ============================================================
    # Optimizer
    # ============================================================
    config.optimizers = get_glee_3d_optim_cfg(params)

    # ============================================================
    # Callbacks: minimal logging only
    # ============================================================
    from vis4d.engine.callbacks import LoggingCallback
    config.callbacks = [
        class_config(LoggingCallback, epoch_based=True, refresh_rate=1),
    ]

    # ============================================================
    # PL Trainer (copied from SAM3_3D overfit)
    # ============================================================
    from ml_collections import ConfigDict
    pl_trainer = ConfigDict()
    pl_trainer.work_dir = config.work_dir
    pl_trainer.exp_name = config.experiment_name
    pl_trainer.version = config.version
    pl_trainer.benchmark = config.benchmark
    pl_trainer.use_distributed_sampler = False
    pl_trainer.num_sanity_val_steps = 0
    pl_trainer.enable_progress_bar = False
    pl_trainer.log_every_n_steps = 1
    pl_trainer.find_unused_parameters = True
    pl_trainer.checkpoint_period = 9999
    pl_trainer.save_top_k = 1
    pl_trainer.wandb = False
    pl_trainer.max_epochs = 500
    pl_trainer.gradient_clip_val = 0.1
    pl_trainer.accumulate_grad_batches = 1
    pl_trainer.overfit_batches = 1
    pl_trainer.limit_train_batches = 1
    pl_trainer.check_val_every_n_epoch = 9999
    pl_trainer.limit_val_batches = 0

    config.pl_trainer = pl_trainer

    return config.value_mode()
