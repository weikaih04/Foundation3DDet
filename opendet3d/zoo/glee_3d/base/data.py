"""GLEE_3D data configuration.

Standalone data config (no dependency on sam3_3d.base).
Uses vis4d.zoo.base helpers for proper dataloader ConfigDict format.
"""

from __future__ import annotations

from vis4d.config import class_config
from vis4d.config.typing import DataConfig
from vis4d.data.data_pipe import DataPipe
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.zoo.base import (
    get_inference_dataloaders_cfg,
    get_train_dataloader_cfg,
)

from opendet3d.zoo.glee_3d.base.connector import (
    GLEE_3DCollator,
    GLEE_3DPassthroughConnector,
)


# Module-level collator singletons
_default_collator = GLEE_3DCollator(filter_empty_boxes=True)
_test_collator = GLEE_3DCollator(filter_empty_boxes=False)


def glee_3d_collate_fn(batch, **kwargs):
    """Default collate function for GLEE_3D training."""
    return _default_collator(batch)


def glee_3d_test_collate_fn(batch, **kwargs):
    """Collate function for GLEE_3D evaluation."""
    return _test_collator(batch)


def get_glee_3d_data_cfg(
    train_datasets,
    test_datasets,
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 4,
    image_shape: tuple[int, int] = (1024, 1024),
):
    """Build GLEE_3D data config using vis4d.zoo.base helpers."""
    batchprocess_cfg = class_config(
        compose, transforms=[class_config(ToTensor)]
    )

    data = DataConfig()

    # Train dataloader
    data.train_dataloader = get_train_dataloader_cfg(
        datasets_cfg=train_datasets,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        batchprocess_cfg=batchprocess_cfg,
        collate_fn=glee_3d_collate_fn,
    )

    # Test dataloaders
    if not isinstance(test_datasets, list):
        test_datasets = [test_datasets]

    data.test_dataloader = get_inference_dataloaders_cfg(
        datasets_cfg=test_datasets,
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
        batchprocess_cfg=batchprocess_cfg,
        collate_fn=glee_3d_test_collate_fn,
    )

    return data


def get_glee_3d_data_connector_cfg():
    """Get data connector configs for GLEE_3D.

    Returns:
        (train_connector, test_connector) - both passthrough.
        Loss connector is inside LossModule (see loss.py).
    """
    train_connector = class_config(GLEE_3DPassthroughConnector)
    test_connector = class_config(GLEE_3DPassthroughConnector)
    return train_connector, test_connector
