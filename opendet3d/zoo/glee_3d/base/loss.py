"""GLEE_3D loss configuration."""

from __future__ import annotations

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.engine.loss_module import LossModule

from opendet3d.op.detect3d.glee_3d.loss import (
    GLEE_3DLoss,
    GLEE_3DLossConfig,
    build_glee_criterion,
)
from opendet3d.zoo.glee_3d.base.connector import GLEE_3DLossConnector


def get_glee_3d_loss_cfg(
    params,
    box_coder_cfg: ConfigDict,
    loss_3d_scale: float = 1.0,
    loss_geom_scale: float = 5.0,
    task: str = "glee3d",
) -> ConfigDict:
    """Build GLEE_3D loss config wrapped in LossModule."""
    # Resolve FieldReference to int
    num_dec = int(getattr(params.num_decoder_layers, 'get', lambda: params.num_decoder_layers)())

    loss_config = class_config(
        GLEE_3DLossConfig,
        loss_3d_scale=loss_3d_scale,
        loss_geom_scale=loss_geom_scale,
        num_decoder_layers=num_dec,
    )

    criterion_2d = class_config(
        build_glee_criterion,
        config=loss_config,
    )

    loss = class_config(
        GLEE_3DLoss,
        config=loss_config,
        box_coder=box_coder_cfg,
        criterion_2d=criterion_2d,
        task=task,
    )

    # Wrap in LossModule with connector (same pattern as SAM3_3D)
    loss_dict = {
        "loss": loss,
        "connector": class_config(GLEE_3DLossConnector),
    }

    return class_config(
        LossModule,
        losses=[loss_dict],
    )
