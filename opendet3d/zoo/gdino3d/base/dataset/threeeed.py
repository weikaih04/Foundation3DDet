"""3EED dataset config for training and evaluation."""

from __future__ import annotations

import os
from collections.abc import Sequence

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.data.data_pipe import DataPipe

from opendet3d.data.datasets.threeeed import (
    ThreeEEDDataset,
    get_threeeed_class_map,
    get_threeeed_det_map,
)

from .transform import get_test_transforms_cfg, get_train_transforms_cfg


def get_threeeed_dataset_cfg(
    data_root: str,
    datasets: Sequence[str],
    data_backend: None | ConfigDict = None,
    remove_empty: bool = False,
    with_depth: bool = False,
    cache_as_binary: bool = False,
    use_mini_dataset: bool = False,
    mini_dataset_size: int = 100,
) -> list[ConfigDict]:
    """Get dataset configs for 3EED.

    Args:
        data_root: Root directory for 3EED data.
        datasets: List of dataset names
            (e.g., "3EED_det_train", "3EED_ref_train").
        data_backend: Data backend configuration.
        remove_empty: Whether to remove empty samples.
        with_depth: Whether to load depth maps.
        cache_as_binary: Whether to cache as binary.
        use_mini_dataset: If True, use mini dataset cache.
        mini_dataset_size: Size of mini dataset.
    """
    if use_mini_dataset:
        cached_dir = os.path.join(
            data_root, f"cache_mini{mini_dataset_size}"
        )
    else:
        cached_dir = os.path.join(data_root, "cache")

    dataset_cfg_list = []
    for dataset in datasets:
        det_map = get_threeeed_det_map(
            dataset_name=dataset, data_root=data_root
        )
        class_map = get_threeeed_class_map(
            dataset_name=dataset, data_root=data_root
        )

        dataset_cfg = class_config(
            ThreeEEDDataset,
            class_map=class_map,
            data_backend=data_backend,
            data_root=data_root,
            dataset_name=dataset,
            det_map=det_map,
            with_depth=with_depth,
            remove_empty=remove_empty,
            data_prefix="data",
            cache_as_binary=cache_as_binary,
            cached_file_path=os.path.join(
                cached_dir, f"{dataset}.pkl"
            ),
        )

        dataset_cfg_list.append(dataset_cfg)

    return dataset_cfg_list


def get_threeeed_train_cfg(
    data_root: str = "data/3eed",
    train_datasets: Sequence[str] = ("3EED_det_train",),
    data_backend: None | ConfigDict = None,
    shape: tuple[int, int] = (800, 1333),
    cache_as_binary: bool = True,
    use_mini_dataset: bool = False,
    mini_dataset_size: int = 100,
) -> ConfigDict:
    """Get train config for 3EED.

    Args:
        data_root: Root directory for 3EED data.
        train_datasets: List of training dataset names.
        data_backend: Data backend configuration.
        shape: Input image shape (H, W).
        cache_as_binary: Whether to cache as binary.
        use_mini_dataset: If True, use mini dataset cache.
        mini_dataset_size: Size of mini dataset.
    """
    train_dataset_cfg = get_threeeed_dataset_cfg(
        data_root=data_root,
        datasets=train_datasets,
        data_backend=data_backend,
        remove_empty=True,
        with_depth=True,
        cache_as_binary=cache_as_binary,
        use_mini_dataset=use_mini_dataset,
        mini_dataset_size=mini_dataset_size,
    )

    train_preprocess_cfg = get_train_transforms_cfg(shape=shape)

    return class_config(
        DataPipe,
        datasets=train_dataset_cfg,
        preprocess_fn=train_preprocess_cfg,
    )
