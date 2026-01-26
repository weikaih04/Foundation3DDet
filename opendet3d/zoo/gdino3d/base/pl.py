"""3D Grounding DINO PyTorch Lightning config."""

from __future__ import annotations

import os

from ml_collections import ConfigDict
from vis4d.config.typing import ExperimentConfig, ExperimentParameters
from vis4d.zoo.base import get_default_pl_trainer_cfg


def get_pl_cfg(
    config: ExperimentConfig, params: ExperimentParameters
) -> ConfigDict:
    """Returns the PyTorch Lightning configuration."""
    # PL Trainer args
    pl_trainer = get_default_pl_trainer_cfg(config)

    # Mixed precision: check environment variable or default to 32-bit
    # Set MIXED_PRECISION=bf16 or MIXED_PRECISION=16 to enable
    precision = os.environ.get("MIXED_PRECISION", "32")
    if precision in ["bf16", "16", "32"]:
        pl_trainer.precision = precision if precision == "32" else f"{precision}-mixed"
    else:
        pl_trainer.precision = "32"

    pl_trainer.max_epochs = params.num_epochs
    pl_trainer.check_val_every_n_epoch = params.check_val_every_n_epoch

    # Validation frequency: 0.5 means validate twice per epoch
    pl_trainer.val_check_interval = 0.5

    # Checkpoint: save every epoch (vis4d uses every_n_epochs internally)
    # To sync with val_check_interval=0.5, override via YAML:
    #   --config.pl_trainer.epoch_based=False
    #   --config.pl_trainer.checkpoint_period=<steps_per_half_epoch>
    # Or just accept: val every 0.5 epoch, checkpoint every 1 epoch
    pl_trainer.checkpoint_period = 1

    pl_trainer.gradient_clip_val = 0.1
    pl_trainer.accumulate_grad_batches = params.accumulate_grad_batches

    # Enable find_unused_parameters for DDP to handle frozen/unused parameters
    # This is needed when using geometry backends with frozen encoders
    # NOTE: PLTrainer will create DDPStrategy with this parameter when devices > 1
    pl_trainer.find_unused_parameters = True

    # Strategy for multi-GPU training (can be overridden via command line)
    # Use "ddp_find_unused_parameters_true" for geometry backends with frozen encoders
    pl_trainer.strategy = "ddp_find_unused_parameters_true"

    return pl_trainer
