"""3D Grounding DINO PyTorch Lightning config."""

from __future__ import annotations

from ml_collections import ConfigDict
from vis4d.config.typing import ExperimentConfig, ExperimentParameters
from vis4d.zoo.base import get_default_pl_trainer_cfg


def get_pl_cfg(
    config: ExperimentConfig, params: ExperimentParameters
) -> ConfigDict:
    """Returns the PyTorch Lightning configuration."""
    # PL Trainer args
    pl_trainer = get_default_pl_trainer_cfg(config)

    pl_trainer.max_epochs = params.num_epochs
    pl_trainer.check_val_every_n_epoch = params.check_val_every_n_epoch

    # Initialize val_check_interval to allow command-line override
    # Default to 1.0 (validate every epoch), can be overridden to 0.5 for twice per epoch
    pl_trainer.val_check_interval = 1.0

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
