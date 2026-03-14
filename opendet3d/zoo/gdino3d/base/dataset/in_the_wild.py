"""InTheWild dataset config for training and testing."""

from __future__ import annotations

import os

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.data.data_pipe import DataPipe

from opendet3d.data.datasets.in_the_wild import (
    InTheWild3DDataset,
    load_in_the_wild_class_map,
)

from .transform import get_test_transforms_cfg, get_train_transforms_cfg


def get_in_the_wild_dataset_cfg(
    data_root: str,
    dataset_name: str,
    data_backend: None | ConfigDict = None,
    with_depth: bool = False,
    cache_as_binary: bool = True,
    use_mini_dataset: bool = False,
    mini_dataset_size: int = 100,
    per_image_categories: bool = False,
    depth_confidence_threshold: int = 0,
    truncation_thres: float = 0.33333333,
    visibility_thres: float = 0.33333333,
    min_height_thres: float = 0.0625,
    max_height_thres: float = 1.50,
) -> ConfigDict:
    """Get dataset config for a single InTheWild split.

    Args:
        data_root: Root directory for InTheWild data.
        dataset_name: Dataset name (e.g., "InTheWild_train").
        data_backend: Data backend (HDF5Backend or None for FileBackend).
        with_depth: Whether to load depth maps.
        cache_as_binary: Whether to cache as binary.
        use_mini_dataset: If True, use mini dataset cache.
        mini_dataset_size: Size of mini dataset (default: 100).
        per_image_categories: If True, boxes2d_names only contains
            GT categories per image (for GDino eval).
        depth_confidence_threshold: Min MoGe2 confidence (uint8,
            0-255) for valid depth pixels. 0 = no masking.
        truncation_thres: Truncation threshold for is_ignore.
        visibility_thres: Visibility threshold for is_ignore.
        min_height_thres: Min box height ratio for is_ignore.
        max_height_thres: Max box height ratio for is_ignore.
    """
    annotation_path = os.path.join(
        data_root, "annotations", f"{dataset_name}.json"
    )
    class_map = load_in_the_wild_class_map(annotation_path)

    split = dataset_name.split("_")[-1]  # "train" or "val"
    depth_suffix = "_depth" if with_depth else ""
    if use_mini_dataset:
        cached_file_path = os.path.join(
            data_root,
            f"cache_mini{mini_dataset_size}",
            f"{split}{depth_suffix}.pkl",
        )
    else:
        cached_file_path = os.path.join(
            data_root, f"{split}{depth_suffix}.pkl"
        )

    return class_config(
        InTheWild3DDataset,
        data_root=data_root,
        dataset_name=dataset_name,
        class_map=class_map,
        with_depth=with_depth,
        per_image_categories=per_image_categories,
        depth_confidence_threshold=depth_confidence_threshold,
        data_backend=data_backend,
        data_prefix=data_root,
        cache_as_binary=cache_as_binary,
        cached_file_path=cached_file_path,
        truncation_thres=truncation_thres,
        visibility_thres=visibility_thres,
        min_height_thres=min_height_thres,
        max_height_thres=max_height_thres,
    )


def get_in_the_wild_train_cfg(
    data_root: str = "data/in_the_wild",
    train_dataset: str = "InTheWild_train",
    data_backend: None | ConfigDict = None,
    shape: tuple[int, int] = (800, 1333),
    cache_as_binary: bool = True,
    use_mini_dataset: bool = False,
    mini_dataset_size: int = 100,
    depth_confidence_threshold: int = 0,
    truncation_thres: float = 0.50,
    visibility_thres: float = 0.1,
    min_height_thres: float = 0.0,
    max_height_thres: float = 1.50,
) -> ConfigDict:
    """Get the train config for InTheWild.

    Args:
        data_root: Root directory for InTheWild data.
        train_dataset: Training dataset name.
        data_backend: Data backend (HDF5Backend or None for FileBackend).
        shape: Input image shape (H, W).
        cache_as_binary: Whether to cache as binary.
        use_mini_dataset: If True, use mini dataset for fast testing.
        mini_dataset_size: Size of mini dataset (default: 100).
        depth_confidence_threshold: Min MoGe2 confidence (uint8,
            0-255) for valid depth pixels. 0 = no masking.
        truncation_thres: Truncation threshold (default 0.99, relaxed).
        visibility_thres: Visibility threshold (default 0.0, relaxed).
        min_height_thres: Min height ratio (default 0.0, relaxed).
        max_height_thres: Max height ratio (default 1.50).
    """
    train_dataset_cfg = get_in_the_wild_dataset_cfg(
        data_root=data_root,
        dataset_name=train_dataset,
        data_backend=data_backend,
        with_depth=True,
        cache_as_binary=cache_as_binary,
        use_mini_dataset=use_mini_dataset,
        mini_dataset_size=mini_dataset_size,
        depth_confidence_threshold=depth_confidence_threshold,
        truncation_thres=truncation_thres,
        visibility_thres=visibility_thres,
        min_height_thres=min_height_thres,
        max_height_thres=max_height_thres,
    )

    train_preprocess_cfg = get_train_transforms_cfg(shape=shape)

    return class_config(
        DataPipe,
        datasets=train_dataset_cfg,
        preprocess_fn=train_preprocess_cfg,
    )


def get_in_the_wild_test_cfg(
    data_root: str = "data/in_the_wild",
    test_dataset: str = "InTheWild_val_human_filtered",
    data_backend: None | ConfigDict = None,
    with_depth: bool = False,
    shape: tuple[int, int] = (800, 1333),
    cache_as_binary: bool = True,
    per_image_categories: bool = False,
) -> ConfigDict:
    """Get the test config for InTheWild.

    Args:
        data_root: Root directory for InTheWild data.
        test_dataset: Test dataset name.
        data_backend: Data backend (HDF5Backend or None for FileBackend).
        with_depth: Whether to load depth maps.
        shape: Input image shape (H, W).
        cache_as_binary: Whether to cache as binary.
        per_image_categories: If True, per-image GT category filtering
            (for GDino/3D-MOOD eval).
    """
    test_dataset_cfg = get_in_the_wild_dataset_cfg(
        data_root=data_root,
        dataset_name=test_dataset,
        data_backend=data_backend,
        with_depth=with_depth,
        cache_as_binary=cache_as_binary,
        per_image_categories=per_image_categories,
    )

    test_preprocess_cfg = get_test_transforms_cfg(
        shape=shape, with_depth=with_depth
    )

    return class_config(
        DataPipe,
        datasets=test_dataset_cfg,
        preprocess_fn=test_preprocess_cfg,
    )
