"""Grounding DINO 3D loss."""

from __future__ import annotations

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.engine.connectors import LossConnector
from vis4d.engine.loss_module import LossModule

from opendet3d.op.detect3d.grounding_dino_3d import (
    GroundingDINO3DLoss,
)
from opendet3d.op.loss.geom_loss_aggregator import GeomLossAggregator
from opendet3d.op.loss.silog_loss import SILogLoss
from opendet3d.zoo.gdino3d.base.connector import (
    CONN_DEPTH_LOSS,
    CONN_GDINO3D_LOSS,
    CONN_GEOM_LOSS,
)


def get_loss_cfg(
    params: ConfigDict,
    box_coder: ConfigDict,
    aux_depth_loss: bool = False,
    use_geom_backend_loss: bool = False,
    loss_2d_scale: float = 1.0,
    loss_3d_scale: float = 1.0,
) -> ConfigDict:
    """Returns the loss configuration.

    Args:
        params: Experiment parameters.
        box_coder: Box coder configuration.
        aux_depth_loss: Whether to use auxiliary SILog depth loss (external).
        use_geom_backend_loss: Whether to use geometry backend internal losses.
        loss_2d_scale: Scale factor for 2D losses (cls, bbox, iou).
        loss_3d_scale: Scale factor for 3D losses (center, depth, dim, rot).
    """
    # Box 3D loss
    box3d_loss = {
        "loss": class_config(
            GroundingDINO3DLoss,
            box_coder=box_coder,
            loss_center_weight=params.loss_center_weight,
            loss_depth_weight=params.loss_depth_weight,
            loss_dim_weight=params.loss_dim_weight,
            loss_rot_weight=params.loss_rot_weight,
            loss_2d_scale=loss_2d_scale,
            loss_3d_scale=loss_3d_scale,
        ),
        "connector": class_config(
            LossConnector, key_mapping=CONN_GDINO3D_LOSS
        ),
    }

    losses = [box3d_loss]

    # Auxiliary depth loss (external SILog)
    if aux_depth_loss:
        depth_loss = {
            "loss": class_config(SILogLoss),
            "weight": params.si_log_weight,
            "connector": class_config(
                LossConnector, key_mapping=CONN_DEPTH_LOSS
            ),
        }
        losses.append(depth_loss)

    # Geometry backend internal losses (DetAny3D, UniDepthV2)
    if use_geom_backend_loss:
        geom_loss = {
            "loss": class_config(GeomLossAggregator, weight=1.0),
            "connector": class_config(
                LossConnector, key_mapping=CONN_GEOM_LOSS
            ),
        }
        losses.append(geom_loss)

    return class_config(LossModule, losses=losses)
