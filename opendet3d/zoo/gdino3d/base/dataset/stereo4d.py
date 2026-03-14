"""Stereo4D tinyval dataset config for testing."""

from __future__ import annotations

import os

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.data.data_pipe import DataPipe

from opendet3d.data.datasets.stereo4d import (
    Stereo4D3DDataset,
    load_stereo4d_class_map,
)

from .transform import get_test_transforms_cfg


def get_stereo4d_dataset_cfg(
    data_root: str,
    dataset_name: str,
    data_backend: None | ConfigDict = None,
    with_depth: bool = False,
    cache_as_binary: bool = True,
    per_image_categories: bool = False,
) -> ConfigDict:
    """Get dataset config for Stereo4D tinyval.

    Args:
        data_root: Root directory for Stereo4D data.
        dataset_name: Dataset name (e.g., "Stereo4D_tinyval").
        data_backend: Data backend (HDF5Backend or None for FileBackend).
        with_depth: Whether to load depth maps.
        cache_as_binary: Whether to cache as binary.
        per_image_categories: If True, boxes2d_names only contains
            GT categories per image.
    """
    annotation_path = os.path.join(
        data_root, "annotations", f"{dataset_name}.json"
    )
    class_map = load_stereo4d_class_map(annotation_path)

    depth_suffix = "_depth" if with_depth else ""
    cached_file_path = os.path.join(
        data_root, f"tinyval{depth_suffix}.pkl"
    )

    return class_config(
        Stereo4D3DDataset,
        data_root=data_root,
        dataset_name=dataset_name,
        class_map=class_map,
        with_depth=with_depth,
        per_image_categories=per_image_categories,
        data_backend=data_backend,
        data_prefix=data_root,
        cache_as_binary=cache_as_binary,
        cached_file_path=cached_file_path,
    )


def get_stereo4d_test_cfg(
    data_root: str = "data/in_the_wild",
    test_dataset: str = "Stereo4D_tinyval",
    data_backend: None | ConfigDict = None,
    with_depth: bool = False,
    shape: tuple[int, int] = (800, 1333),
    cache_as_binary: bool = True,
    per_image_categories: bool = False,
) -> ConfigDict:
    """Get the test config for Stereo4D tinyval.

    Args:
        data_root: Root directory for data (same as in_the_wild).
        test_dataset: Test dataset name.
        data_backend: Data backend (HDF5Backend or None for FileBackend).
        with_depth: Whether to load depth maps.
        shape: Input image shape (H, W).
        cache_as_binary: Whether to cache as binary.
        per_image_categories: If True, per-image GT category filtering.
    """
    test_dataset_cfg = get_stereo4d_dataset_cfg(
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
