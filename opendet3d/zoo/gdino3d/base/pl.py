"""3D Grounding DINO PyTorch Lightning config."""

from __future__ import annotations

from ml_collections import ConfigDict
from vis4d.config.typing import ExperimentConfig, ExperimentParameters
from vis4d.zoo.base import get_default_pl_trainer_cfg


def get_pl_cfg(
    config: ExperimentConfig,
    params: ExperimentParameters,
    epoch_based: bool = True,
) -> ConfigDict:
    """Returns the PyTorch Lightning configuration."""
    # PL Trainer args
    pl_trainer = get_default_pl_trainer_cfg(config)

    pl_trainer.epoch_based = epoch_based

    if epoch_based:
        pl_trainer.max_epochs = params.num_epochs
        pl_trainer.check_val_every_n_epoch = params.check_val_every_n_epoch
    else:
        pl_trainer.max_steps = params.num_iters
        pl_trainer.check_val_every_n_epoch = None
        pl_trainer.val_check_interval = (
            params.val_freq * params.accumulate_grad_batches
        )
        pl_trainer.checkpoint_period = (
            params.val_freq * params.accumulate_grad_batches
        )

    pl_trainer.gradient_clip_val = 0.1
    pl_trainer.accumulate_grad_batches = params.accumulate_grad_batches

    return pl_trainer
