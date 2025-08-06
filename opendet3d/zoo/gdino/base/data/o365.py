"""Grounding DINO Object365 config."""

from __future__ import annotations

from collections.abc import Sequence
from ml_collections import ConfigDict

from vis4d.config import class_config
from vis4d.config.typing import DataConfig
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.coco import COCO

from opendet3d.data.datasets.odvg import ODVGDataset

from .util import get_train_dataloader, get_test_dataloader


def get_objects365v1_detection_cfg(
    data_root: str = "data/coco",
    train_cached_file_path: str | None = "data/objects365v1/train.pkl",
    test_keys_to_load: Sequence[str] = (
        K.images,
        K.original_images,
        K.boxes2d,
        K.boxes2d_classes,
    ),
    test_split: str = "val2017",
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
        ODVGDataset,
        data_root="data/objects365v1/",
        ann_file="o365v1_train_odvg.json",
        label_map_file="o365v1_label_map.json",
        dataset_prefix="train",
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
        random_sample_neg_pos=True,
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
        pad_stride=pad_stride,
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
    )

    return data
