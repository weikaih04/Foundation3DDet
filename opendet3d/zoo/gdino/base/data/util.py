"""Grounding DINO data config."""

from __future__ import annotations

from ml_collections import ConfigDict

from vis4d.config import class_config
from vis4d.data.data_pipe import DataPipe
from vis4d.data.transforms.base import RandomApply, compose
from vis4d.data.transforms.crop import (
    GenCropParameters,
    CropBoxes2D,
    CropImages,
)
from vis4d.data.transforms.flip import FlipBoxes2D, FlipImages
from vis4d.data.transforms.normalize import NormalizeImages
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.resize import (
    GenResizeParameters,
    ResizeBoxes2D,
    ResizeImages,
)
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.zoo.base import (
    get_inference_dataloaders_cfg,
    get_train_dataloader_cfg,
)

from opendet3d.data.transforms.language import RandomSamplingNegPos


def get_train_dataloader(
    train_dataset_cfg: ConfigDict,
    samples_per_gpu: int,
    workers_per_gpu: int,
    pad_stride: int = 1,
    random_sample_neg_pos: bool = False,
) -> ConfigDict:
    """Get train dataloader config."""
    preprocess_transforms = [
        class_config(
            RandomApply,
            transforms=[
                class_config(
                    GenResizeParameters,
                    shape=[
                        (400, 4200),
                        (500, 4200),
                        (600, 4200),
                    ],
                    multiscale_mode="list",
                    keep_ratio=True,
                    align_long_edge=True,
                ),
                class_config(ResizeImages),
                class_config(ResizeBoxes2D),
                class_config(
                    GenCropParameters,
                    shape=(384, 600),
                    allow_empty_crops=False,
                ),
                class_config(CropImages),
                class_config(CropBoxes2D),
            ],
            probability=0.5,
        )
    ]

    preprocess_transforms += [
        class_config(
            GenResizeParameters,
            shape=[
                (480, 1333),
                (512, 1333),
                (544, 1333),
                (576, 1333),
                (608, 1333),
                (640, 1333),
                (672, 1333),
                (704, 1333),
                (736, 1333),
                (768, 1333),
                (800, 1333),
            ],
            multiscale_mode="list",
            keep_ratio=True,
            align_long_edge=True,
        ),
        class_config(ResizeImages),
        class_config(ResizeBoxes2D),
    ]

    preprocess_transforms.append(
        class_config(
            RandomApply,
            transforms=[
                class_config(FlipImages),
                class_config(FlipBoxes2D),
            ],
            probability=0.5,
        )
    )

    if random_sample_neg_pos:
        preprocess_transforms.append(class_config(RandomSamplingNegPos))

    preprocess_transforms.append(class_config(NormalizeImages))

    train_preprocess_cfg = class_config(
        compose, transforms=preprocess_transforms
    )

    train_batchprocess_cfg = class_config(
        compose,
        transforms=[
            class_config(PadImages, stride=pad_stride),
            class_config(ToTensor),
        ],
    )

    return get_train_dataloader_cfg(
        preprocess_cfg=train_preprocess_cfg,
        dataset_cfg=train_dataset_cfg,
        batchprocess_cfg=train_batchprocess_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
    )


def get_test_dataloader(
    test_dataset: ConfigDict,
    pad_stride: int = 1,
    samples_per_gpu: int = 1,
    workers_per_gpu: int = 2,
) -> ConfigDict:
    """Get test dataloader config."""
    preprocess_transforms = [
        class_config(
            GenResizeParameters,
            shape=(800, 1333),
            keep_ratio=True,
            align_long_edge=True,
            fixed_scale=True,
        ),
        class_config(ResizeImages),
    ]

    preprocess_transforms.append(class_config(NormalizeImages))

    test_preprocess_cfg = class_config(
        compose, transforms=preprocess_transforms
    )

    test_batchprocess_cfg = class_config(
        compose,
        transforms=[
            class_config(PadImages, stride=pad_stride),
            class_config(ToTensor),
        ],
    )

    test_dataset_cfg = class_config(
        DataPipe, datasets=test_dataset, preprocess_fn=test_preprocess_cfg
    )

    return get_inference_dataloaders_cfg(
        datasets_cfg=test_dataset_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        batchprocess_cfg=test_batchprocess_cfg,
    )
