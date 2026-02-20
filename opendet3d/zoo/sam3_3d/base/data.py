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


# Default collator instances (module-level for vis4d compatibility).
# vis4d's DelayedInstantiator resolves functions by module path + __name__,
# so module-level functions must exist for each variant.
# IMPORTANT: create_sam3_3d_collate_fn() creates closures with __name__ set
# to one of these module-level function names. At runtime vis4d resolves the
# stored class_path back to the actual module-level function, so the closure
# parameters are IGNORED. Parameters must be baked into each module-level fn.
_default_collator = None
_test_collator = None
_oracle_collator = None
_5mode_collator = None


def sam3_3d_collate_fn(batch: List[dict], **kwargs):
    """Default SAM3_3D collate function (module-level for vis4d).

    This function is callable directly from the module for vis4d's class_config.
    Uses default parameters: max_prompts_per_image=50, use_text_prompts=True,
    text_query_prob=0.7 (SAM3 recommended: 70% text, 30% visual).

    For custom parameters, use create_sam3_3d_collate_fn() instead.

    Args:
        batch: List of data samples
        **kwargs: Additional arguments from vis4d (e.g., collate_keys)

    Returns:
        Collated batch data
    """
    global _default_collator
    if _default_collator is None:
        _default_collator = SAM3_3DCollator(
            max_prompts_per_image=50,
            use_text_prompts=True,
            text_query_prob=0.7,  # SAM3 recommended: 70% text, 30% visual
        )
    # Ignore kwargs (like collate_keys) that vis4d passes but we don't use
    return _default_collator(batch)


def sam3_3d_test_collate_fn(batch: List[dict], **kwargs):
    """Test-time SAM3_3D collate function (module-level for vis4d).

    Pure text queries, no filtering of empty-box images.
    Used for standard test-time evaluation (non-oracle).

    Args:
        batch: List of data samples
        **kwargs: Additional arguments from vis4d (e.g., collate_keys)

    Returns:
        Collated batch data
    """
    global _test_collator
    if _test_collator is None:
        _test_collator = SAM3_3DCollator(
            max_prompts_per_image=50,
            use_text_prompts=True,
            text_query_prob=1.0,  # Pure text at test time
            filter_empty_boxes=False,  # Keep all images at test time
        )
    return _test_collator(batch)


def sam3_3d_oracle_collate_fn(batch: List[dict], **kwargs):
    """Oracle SAM3_3D collate function (module-level for vis4d).

    Each GT 2D box becomes its own geometry prompt (one-to-one mapping).
    Used for measuring pure 3D regression quality with GT box prompts.

    Args:
        batch: List of data samples
        **kwargs: Additional arguments from vis4d (e.g., collate_keys)

    Returns:
        Collated batch data
    """
    global _oracle_collator
    if _oracle_collator is None:
        _oracle_collator = SAM3_3DCollator(
            max_prompts_per_image=50,
            use_text_prompts=True,
            oracle_eval=True,
            filter_empty_boxes=False,  # Oracle eval is always test-time
        )
    return _oracle_collator(batch)


def sam3_3d_5mode_collate_fn(batch: List[dict], **kwargs):
    """5-mode SAM3_3D collate function (module-level for vis4d).

    Per category creates 2 queries:
      Branch 1 (o2m): TEXT / VISUAL / VISUAL+LABEL
      Branch 2 (o2o): GEOMETRY / GEOMETRY+LABEL

    Args:
        batch: List of data samples
        **kwargs: Additional arguments from vis4d

    Returns:
        Collated batch data
    """
    global _5mode_collator
    if _5mode_collator is None:
        _5mode_collator = SAM3_3DCollator(
            max_prompts_per_image=50,
            use_text_prompts=True,
            use_geometry_prompts=True,
            text_only_prob=0.5,
            use_label_prob=1/3,
        )
    return _5mode_collator(batch)


def create_sam3_3d_collate_fn(
    max_prompts_per_image: int = 50,
    use_text_prompts: bool = True,
    text_query_prob: float = 0.7,
    keep_text_for_visual: bool = False,
    # Geometry prompt options (NEW: text + geometry training)
    use_geometry_prompts: bool = False,
    geometric_query_str: str = "geometric",
    box_noise_std: float = 0.0,
    box_noise_max: float | None = 20.0,
    # Oracle evaluation mode
    oracle_eval: bool = False,
    # Training vs inference filtering
    filter_empty_boxes: bool = True,
):
    """Create a configured SAM3_3D collate function.

    Returns a callable that can be used as collate_fn in DataLoader.
    The returned function has proper __name__ and __module__ attributes
    for vis4d's class_config to work correctly.

    Args:
        max_prompts_per_image: Max prompts (categories) per image
        use_text_prompts: Whether to include text with geometric prompts
        text_query_prob: Probability of text-only queries (SAM3 recommended: 0.7)
            1.0 = all text queries (pure text training)
            0.7 = 70% text, 30% visual (SAM3 mixed training)
            0.0 = all visual queries (DetAny3D style)
        keep_text_for_visual: If True, visual queries keep category text
            If False (default), visual queries use "visual" as text
        use_geometry_prompts: If True, create geometry queries per category
            This implements text + geometry training (SAM3 style):
            - Each category gets 1 TEXT query (one-to-many targets)
            - Each category gets 1 GEOMETRY query (one-to-one target)
        geometric_query_str: Text for geometry queries (default "geometric")
        box_noise_std: Noise std for box jittering (0 = no noise)
        box_noise_max: Max noise in pixels
        oracle_eval: If True, each GT 2D box becomes its own geometry
            prompt for measuring 3D regression quality in isolation.

    Returns:
        Configured collate function
    """
    collator = SAM3_3DCollator(
        max_prompts_per_image=max_prompts_per_image,
        use_text_prompts=use_text_prompts,
        text_query_prob=text_query_prob,
        keep_text_for_visual=keep_text_for_visual,
        use_geometry_prompts=use_geometry_prompts,
        geometric_query_str=geometric_query_str,
        box_noise_std=box_noise_std,
        box_noise_max=box_noise_max,
        oracle_eval=oracle_eval,
        filter_empty_boxes=filter_empty_boxes,
    )

    def collate_fn(batch: List[dict], **kwargs):
        """Collate function for SAM3_3D."""
        return collator(batch)

    # Set name for vis4d compatibility.
    # vis4d's DelayedInstantiator resolves by module path + __name__,
    # so the name must match a module-level function.
    if oracle_eval:
        collate_fn.__name__ = "sam3_3d_oracle_collate_fn"
    else:
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
    # Text/Visual query ratio (SAM3 original design)
    text_query_prob: float = 0.7,
    keep_text_for_visual: bool = False,
    # Geometry prompt options (5-mode training)
    use_geometry_prompts: bool = False,
    geometric_query_str: str = "geometric",
    text_only_prob: float = 0.5,
    use_label_prob: float = 1/3,
    box_noise_std: float = 0.0,
    box_noise_max: float | None = 20.0,
    # Dataloader options
    shuffle: bool = True,
    # Test-specific collator options (for pure text evaluation like GDino3D)
    test_text_query_prob: float = 1.0,
    test_use_geometry_prompts: bool = False,
    # Oracle evaluation mode
    oracle_eval: bool = False,
) -> DataConfig:
    """Get the data config for SAM3_3D with custom collator.

    This differs from gdino3d's get_data_cfg by using SAM3_3DCollator
    to convert per-image data to per-prompt batch (SAM3_3DBatchedInputs).

    IMPORTANT: Training and test use SEPARATE collators by default:
    - Training: Uses configured use_geometry_prompts with 5-mode training
    - Testing: Uses pure text for fair evaluation

    Args:
        train_datasets: Training dataset configuration
        test_datasets: Test dataset configuration
        samples_per_gpu: Batch size (number of images per GPU)
        workers_per_gpu: Number of data loading workers per GPU
        max_prompts_per_image: Max prompts (categories) per image
        use_text_prompts: Whether to include text with geometric prompts
        text_query_prob: Probability of text-only queries (legacy 2-mode)
        keep_text_for_visual: If True, visual queries keep category text
        use_geometry_prompts: If True, 5-mode training with 2 queries per
            category (Branch 1 o2m + Branch 2 o2o)
        geometric_query_str: Text for geometry queries (default "geometric")
        text_only_prob: Branch 1 P(TEXT) (only when use_geometry_prompts=True)
        use_label_prob: P(+LABEL) for box-based queries (both branches)
        box_noise_std: Noise std for box jittering (0 = no noise)
        box_noise_max: Max noise in pixels
        shuffle: Whether to shuffle training data (default True)
        test_text_query_prob: Probability of text queries during test
        test_use_geometry_prompts: Whether to use geometry prompts during test
        oracle_eval: If True, test collator uses oracle mode

    Returns:
        DataConfig with train and test dataloaders
    """
    data = DataConfig()

    # Training collate function: select based on use_geometry_prompts.
    # Use module-level functions so vis4d's DelayedInstantiator can resolve.
    if use_geometry_prompts:
        train_collate_fn = sam3_3d_5mode_collate_fn
    else:
        train_collate_fn = sam3_3d_collate_fn

    # Test collate function: use a module-level function so vis4d resolves it
    # correctly.  create_sam3_3d_collate_fn() closures do NOT work because
    # vis4d's DelayedInstantiator resolves by __module__ + __name__ back to
    # the module-level function, ignoring the closure parameters.
    if oracle_eval:
        test_collate_fn = sam3_3d_oracle_collate_fn
    else:
        test_collate_fn = sam3_3d_test_collate_fn

    # Train dataloader
    train_batchprocess_cfg = class_config(
        compose, transforms=[class_config(ToTensor)]
    )

    data.train_dataloader = get_train_dataloader_cfg(
        datasets_cfg=train_datasets,
        batchprocess_cfg=train_batchprocess_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        collate_fn=train_collate_fn,
        shuffle=shuffle,
    )

    # Test dataloader with SEPARATE pure-text collator
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
        collate_fn=test_collate_fn,  # Use SEPARATE test collator (pure text)
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

    # Test dataloader (use same collator as training)
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
        collate_fn=collate_fn,
    )

    return data

