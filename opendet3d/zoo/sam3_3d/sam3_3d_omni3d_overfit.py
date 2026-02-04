"""SAM3_3D Single Batch Overfit Test.

This configuration is for sanity checking that the model and loss heads
are correctly set up. It trains on a single batch repeatedly until
all losses converge to near 0.

Expected behavior if model is correct:
- All 2D losses (loss_bbox, loss_giou, loss_cls) -> ~0
- All 3D losses (loss_delta_2d, loss_depth, loss_dim, loss_rot) -> ~0

Usage:
    vis4d fit --config opendet3d/zoo/sam3_3d/sam3_3d_omni3d_overfit.py --gpus 1

Key settings for deterministic overfit:
- shuffle=False: Same batch order every epoch
- seed=42: Fixed random seed for reproducibility
- workers_per_gpu=0: Disable multiprocessing
- box_noise_std=0: No box jittering
"""

from __future__ import annotations

import pytorch_lightning as pl

from vis4d.config import class_config
from vis4d.config.typing import ExperimentConfig
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.zoo.base import get_default_cfg, get_default_pl_trainer_cfg

from opendet3d.zoo.gdino3d.base.callback import (
    get_callback_cfg,
    get_omni3d_evaluator_cfg,
)
from vis4d.data.data_pipe import DataPipe
from opendet3d.zoo.gdino3d.base.dataset.omni3d import (
    get_omni3d_dataset_cfg,
    get_omni3d_test_cfg,
)
from opendet3d.zoo.gdino3d.base.dataset.transform import (
    get_deterministic_train_transforms_cfg,
)

from opendet3d.zoo.sam3_3d.base.optim import get_sam3_3d_optim_cfg

from opendet3d.zoo.sam3_3d.base.model import (
    get_sam3_3d_cfg,
    get_sam3_3d_hyperparams_cfg,
)
from opendet3d.zoo.sam3_3d.base.loss import get_sam3_3d_loss_cfg
from opendet3d.zoo.sam3_3d.base.connector import get_sam3_3d_data_connector_cfg
from opendet3d.zoo.sam3_3d.base.data import get_sam3_3d_data_cfg


def get_config() -> ExperimentConfig:
    """Returns the SAM3_3D Single Batch Overfit configuration."""
    ######################################################
    ##                    General Config                ##
    ######################################################
    # Set seed for deterministic behavior
    pl.seed_everything(42, workers=True)

    config = get_default_cfg(exp_name="sam3_3d_overfit_test")

    config.use_checkpoint = False  # No checkpointing needed for overfit test
    config.log_every_n_steps = 1  # Log every step for overfit test

    # High level hyper parameters
    params = get_sam3_3d_hyperparams_cfg(
        num_epochs=500,  # Train more epochs for deeper convergence
        samples_per_gpu=1,
        workers_per_gpu=0,  # Disable multiprocessing to avoid deadlock
        base_lr=1e-4,  # Stable lr for consistent overfit
        accumulate_grad_batches=1,  # No gradient accumulation
    )

    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_backend = class_config(HDF5Backend)

    test_datasets_cfg = []

    # Omni3D - use mini dataset for fast loading
    omni3d_data_root = "data/omni3d"
    omni3d_test_datasets = (
        "KITTI_test",  # Only one test dataset to speed up
    )

    # SAM3 expects 1008x1008 images
    sam3_image_shape = (1008, 1008)

    # Use ONLY KITTI with tiny dataset (2 samples) for overfit test
    # Use deterministic transforms (no random resize/flip) for reproducible overfit
    train_dataset_cfg = get_omni3d_dataset_cfg(
        data_root=omni3d_data_root,
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
        shape=sam3_image_shape
    )
    omni3d_train_data_cfg = class_config(
        DataPipe,
        datasets=train_dataset_cfg,
        preprocess_fn=train_preprocess_cfg,
    )

    omni3d_test_data_cfg = get_omni3d_test_cfg(
        data_root=omni3d_data_root,
        test_datasets=omni3d_test_datasets,
        data_backend=data_backend,
        shape=sam3_image_shape,
        use_mini_dataset=True,
        mini_dataset_size=10,  # Very small test set
    )

    test_datasets_cfg.append(omni3d_test_data_cfg)

    config.data = get_sam3_3d_data_cfg(
        train_datasets=omni3d_train_data_cfg,
        test_datasets=test_datasets_cfg,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
        max_prompts_per_image=50,
        use_text_prompts=True,
        text_query_prob=1.0,  # 100% text queries (deterministic)
        use_geometry_prompts=False,  # Disable geometry (uses random.choice)
        shuffle=False,  # Same batch order every epoch
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
    # For overfit test, only use LoggingCallback with refresh_rate=1
    from vis4d.engine.callbacks import LoggingCallback
    callbacks = [
        class_config(LoggingCallback, epoch_based=True, refresh_rate=1),
    ]

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    # Create pl_trainer config from scratch to control types
    # Don't use get_default_pl_trainer_cfg() which sets float defaults
    from ml_collections import ConfigDict
    pl_trainer = ConfigDict()

    # Required fields for vis4d PLTrainer
    pl_trainer.work_dir = config.work_dir
    pl_trainer.exp_name = config.experiment_name
    pl_trainer.version = config.version
    pl_trainer.benchmark = config.benchmark
    pl_trainer.use_distributed_sampler = False
    pl_trainer.num_sanity_val_steps = 0
    pl_trainer.enable_progress_bar = False  # Disable progress bar, use vis4d text logging
    pl_trainer.log_every_n_steps = 1  # Log every step
    pl_trainer.find_unused_parameters = False
    pl_trainer.checkpoint_period = 9999
    pl_trainer.save_top_k = 1
    pl_trainer.wandb = False

    # Training params
    pl_trainer.max_epochs = 500
    pl_trainer.gradient_clip_val = 0.1
    pl_trainer.accumulate_grad_batches = params.accumulate_grad_batches

    # Key: Set overfit_batches as INT from the start (not float!)
    pl_trainer.overfit_batches = 1  # int: use exactly 1 batch
    pl_trainer.limit_train_batches = 1  # int: same

    # Note: deterministic=True is incompatible with cumsum_cuda_kernel in UniDepth
    # We rely on shuffle=False + fixed seed for reproducibility

    # Skip validation
    pl_trainer.check_val_every_n_epoch = 9999
    pl_trainer.limit_val_batches = 0

    config.pl_trainer = pl_trainer

    return config.value_mode()
