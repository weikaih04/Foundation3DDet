"""CubifyAnything (CA-1M) dataset config."""

from __future__ import annotations

import os
from collections.abc import Sequence

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.data.data_pipe import DataPipe

from opendet3d.data.datasets.cubifyanything import (
    CubifyAnything,
    get_cubifyanything_class_map,
    get_cubifyanything_det_map,
)

from .transform import get_test_transforms_cfg, get_train_transforms_cfg


def get_ca1m_dataset_cfg(
    data_root: str,
    datasets: Sequence[str],
    data_backend: None | ConfigDict = None,
    remove_empty: bool = False,
    with_depth: bool = False,
    cache_as_binary: bool = False,
) -> list[ConfigDict]:
    """Get the dataset configs for CubifyAnything.

    Args:
        data_root: Root directory for CubifyAnything data.
        datasets: List of dataset names
            (e.g., "CubifyAnything_train", "CubifyAnything_val").
        data_backend: Data backend configuration.
        remove_empty: Whether to remove empty samples.
        with_depth: Whether to load depth maps.
        cache_as_binary: Whether to cache as binary.
    """
    cached_dir = os.path.join(data_root, "cache")

    dataset_cfg_list = []
    for dataset in datasets:
        det_map = get_cubifyanything_det_map(
            dataset_name=dataset, data_root=data_root
        )
        class_map = get_cubifyanything_class_map(
            dataset_name=dataset, data_root=data_root
        )

        dataset_cfg = class_config(
            CubifyAnything,
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


def get_ca1m_train_cfg(
    data_root: str = "data/cubifyanything",
    train_datasets: Sequence[str] = ("CubifyAnything_train",),
    data_backend: None | ConfigDict = None,
    shape: tuple[int, int] = (800, 1333),
    cache_as_binary: bool = True,
) -> ConfigDict:
    """Get the train config for CubifyAnything (CA-1M).

    Args:
        data_root: Root directory for CubifyAnything data.
        train_datasets: List of training dataset names.
        data_backend: Data backend configuration.
        shape: Input image shape (H, W).
        cache_as_binary: Whether to cache as binary.
    """
    train_dataset_cfg = get_ca1m_dataset_cfg(
        data_root=data_root,
        datasets=train_datasets,
        data_backend=data_backend,
        remove_empty=True,
        with_depth=True,
        cache_as_binary=cache_as_binary,
    )

    train_preprocess_cfg = get_train_transforms_cfg(shape=shape)

    return class_config(
        DataPipe,
        datasets=train_dataset_cfg,
        preprocess_fn=train_preprocess_cfg,
    )


def get_ca1m_test_cfg(
    data_root: str = "data/cubifyanything",
    test_datasets: Sequence[str] = ("CubifyAnything_val",),
    data_backend: None | ConfigDict = None,
    with_depth: bool = False,
    shape: tuple[int, int] = (800, 1333),
    cache_as_binary: bool = True,
) -> ConfigDict:
    """Get the test config for CubifyAnything (CA-1M).

    Args:
        data_root: Root directory for CubifyAnything data.
        test_datasets: List of test dataset names.
        data_backend: Data backend configuration.
        with_depth: Whether to load depth maps.
        shape: Input image shape (H, W).
        cache_as_binary: Whether to cache as binary.
    """
    test_dataset_cfg = get_ca1m_dataset_cfg(
        data_root=data_root,
        datasets=test_datasets,
        data_backend=data_backend,
        with_depth=with_depth,
        cache_as_binary=cache_as_binary,
    )

    test_preprocess_cfg = get_test_transforms_cfg(shape=shape)

    return class_config(
        DataPipe,
        datasets=test_dataset_cfg,
        preprocess_fn=test_preprocess_cfg,
    )
