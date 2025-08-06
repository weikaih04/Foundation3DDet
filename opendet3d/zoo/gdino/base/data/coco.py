"""Grounding DINO COCO config."""

from __future__ import annotations

from collections.abc import Sequence
from ml_collections import ConfigDict

from vis4d.config import class_config
from vis4d.config.typing import DataConfig
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.coco import COCO

from .util import get_train_dataloader, get_test_dataloader


def get_coco_detection_cfg(
    data_root: str = "data/coco",
    train_split: str = "train2017",
    train_keys_to_load: Sequence[str] = (
        K.images,
        K.original_images,
        K.boxes2d,
        K.boxes2d_classes,
    ),
    train_cached_file_path: str | None = "data/coco/train.pkl",
    test_split: str = "val2017",
    test_keys_to_load: Sequence[str] = (
        K.images,
        K.original_images,
        K.boxes2d,
        K.boxes2d_classes,
    ),
    test_cached_file_path: str | None = "data/coco/val.pkl",
    cache_as_binary: bool = True,
    data_backend: None | ConfigDict = None,
    pad_stride: int = 1,
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 2,
) -> DataConfig:
    """Get the default config for COCO detection."""
    data = DataConfig()

    # Train Dataset
    train_dataset = class_config(
        COCO,
        keys_to_load=train_keys_to_load,
        data_root=data_root,
        split=train_split,
        remove_empty=True,
        data_backend=data_backend,
        cache_as_binary=cache_as_binary,
        cached_file_path=train_cached_file_path,
    )

    data.train_dataloader = get_train_dataloader(
        train_dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        pad_stride=pad_stride,
    )

    test_dataset = class_config(
        COCO,
        keys_to_load=test_keys_to_load,
        data_root=data_root,
        split=test_split,
        data_backend=data_backend,
        cache_as_binary=cache_as_binary,
        cached_file_path=test_cached_file_path,
    )

    data.test_dataloader = get_test_dataloader(
        test_dataset,
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
        pad_stride=pad_stride,
    )

    return data
