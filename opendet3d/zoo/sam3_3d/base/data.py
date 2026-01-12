"""SAM3_3D data config with custom collator.

This module provides data configuration for SAM3_3D that uses SAM3_3DCollator
to convert per-image DataLoader output to SAM3_3DBatchedInputs (per-prompt batch).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import List

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.config.typing import DataConfig
from vis4d.data.data_pipe import DataPipe
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.zoo.base import (
    get_inference_dataloaders_cfg,
    get_train_dataloader_cfg,
)

from .connector import SAM3_3DCollator


def create_sam3_3d_collate_fn(
    max_prompts_per_image: int = 50,
    use_text_prompts: bool = True,
):
    """Create a configured SAM3_3D collate function.

    Returns a callable that can be used as collate_fn in DataLoader.
    The returned function has proper __name__ and __module__ attributes
    for vis4d's class_config to work correctly.

    Args:
        max_prompts_per_image: Max prompts (GT boxes) per image
        use_text_prompts: Whether to include text with geometric prompts

    Returns:
        Configured collate function
    """
    collator = SAM3_3DCollator(
        max_prompts_per_image=max_prompts_per_image,
        use_text_prompts=use_text_prompts,
    )

    def collate_fn(batch: List[dict]):
        """Collate function for SAM3_3D."""
        return collator(batch)

    # Set name for vis4d compatibility
    collate_fn.__name__ = "sam3_3d_collate_fn"
    collate_fn.__module__ = __name__

    return collate_fn


def get_sam3_3d_data_cfg(
    train_datasets: ConfigDict | Sequence[ConfigDict],
    test_datasets: ConfigDict | Sequence[ConfigDict],
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 2,
    max_prompts_per_image: int = 50,
    use_text_prompts: bool = True,
) -> DataConfig:
    """Get the data config for SAM3_3D with custom collator.

    This differs from gdino3d's get_data_cfg by using SAM3_3DCollator
    to convert per-image data to per-prompt batch (SAM3_3DBatchedInputs).

    Args:
        train_datasets: Training dataset configuration
        test_datasets: Test dataset configuration
        samples_per_gpu: Batch size (number of images per GPU)
        workers_per_gpu: Number of data loading workers per GPU
        max_prompts_per_image: Max prompts (GT boxes) per image
        use_text_prompts: Whether to include text with geometric prompts

    Returns:
        DataConfig with train and test dataloaders
    """
    data = DataConfig()

    # Create collate function with configured parameters
    collate_fn = create_sam3_3d_collate_fn(
        max_prompts_per_image=max_prompts_per_image,
        use_text_prompts=use_text_prompts,
    )

    # Train dataloader
    train_batchprocess_cfg = class_config(
        compose, transforms=[class_config(ToTensor)]
    )

    data.train_dataloader = get_train_dataloader_cfg(
        datasets_cfg=train_datasets,
        batchprocess_cfg=train_batchprocess_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        collate_fn=collate_fn,
    )

    # Test dataloader
    # For inference, we use default collation since we may not have GT boxes
    test_batchprocess_cfg = class_config(
        compose, transforms=[class_config(ToTensor)]
    )

    if isinstance(test_datasets, list):
        test_datasets_cfg = class_config(DataPipe, datasets=test_datasets)
    else:
        test_datasets_cfg = test_datasets

    data.test_dataloader = get_inference_dataloaders_cfg(
        datasets_cfg=test_datasets_cfg,
        batchprocess_cfg=test_batchprocess_cfg,
        samples_per_gpu=1,  # Use batch size 1 for inference
        workers_per_gpu=workers_per_gpu,
    )

    return data


def get_sam3_3d_data_cfg_with_custom_collator(
    train_datasets: ConfigDict | Sequence[ConfigDict],
    test_datasets: ConfigDict | Sequence[ConfigDict],
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 2,
    collate_fn=None,
) -> DataConfig:
    """Get the data config for SAM3_3D with custom collator function.

    This version accepts a pre-created collate function.
    Use create_sam3_3d_collate_fn() to create a configured collate function.

    Args:
        train_datasets: Training dataset configuration
        test_datasets: Test dataset configuration
        samples_per_gpu: Batch size (number of images per GPU)
        workers_per_gpu: Number of data loading workers per GPU
        collate_fn: Optional collate function (must have __name__ attribute)

    Returns:
        DataConfig with train and test dataloaders
    """
    data = DataConfig()

    # Use provided collate_fn or create default
    if collate_fn is None:
        collate_fn = create_sam3_3d_collate_fn()

    # Train dataloader
    train_batchprocess_cfg = class_config(
        compose, transforms=[class_config(ToTensor)]
    )

    data.train_dataloader = get_train_dataloader_cfg(
        datasets_cfg=train_datasets,
        batchprocess_cfg=train_batchprocess_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        collate_fn=collate_fn,
    )

    # Test dataloader (default collation)
    test_batchprocess_cfg = class_config(
        compose, transforms=[class_config(ToTensor)]
    )

    if isinstance(test_datasets, list):
        test_datasets_cfg = class_config(DataPipe, datasets=test_datasets)
    else:
        test_datasets_cfg = test_datasets

    data.test_dataloader = get_inference_dataloaders_cfg(
        datasets_cfg=test_datasets_cfg,
        batchprocess_cfg=test_batchprocess_cfg,
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
    )

    return data

