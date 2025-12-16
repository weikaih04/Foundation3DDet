"""Geometry Loss Aggregator.

This module provides a loss class that aggregates geometry losses from
the model output (geom_losses dict from GeometryBackend).
"""

from __future__ import annotations

from torch import Tensor
from vis4d.common.typing import ArgsType
from vis4d.op.loss.base import Loss


class GeomLossAggregator(Loss):
    """Aggregates geometry losses from model output.

    This loss class takes the geom_losses dict from the model output
    and returns the sum of all losses. Each individual loss is also
    logged separately.

    Args:
        weight: Global weight multiplier for all geometry losses.
    """

    def __init__(
        self,
        *args: ArgsType,
        weight: float = 1.0,
        **kwargs: ArgsType,
    ) -> None:
        """Initialize the GeomLossAggregator."""
        super().__init__(*args, **kwargs)
        self.weight = weight

    def forward(
        self,
        geom_losses: dict[str, Tensor] | None,
    ) -> dict[str, Tensor]:
        """Forward function.

        Args:
            geom_losses: Dictionary of geometry losses from the model.

        Returns:
            Dictionary of weighted losses.
        """
        if geom_losses is None or len(geom_losses) == 0:
            return {}

        weighted_losses = {}
        for name, loss in geom_losses.items():
            weighted_losses[f"geom_{name}"] = loss * self.weight

        return weighted_losses

