"""SAM3_3D + LingBot-Depth + Omni3D + CA-1M + Waymo + 3EED + FoundationPose.

Mixed training with dataset-ratio sampling. Full config with all 7 data
sources and HDF5 backend.

Dataset mixing:
- Omni3D:         ~100K images (6 datasets, train+val splits), HDF5 backend
- CA-1M:          ~206K images (CubifyAnything_train), HDF5 backend
- Waymo:          ~99K images (Waymo_train, frame_interval=2), HDF5 backend
- 3EED_det:       ~12K images (detection, all objects per frame)
- 3EED_ref:       ~12K images (referring, target + short phrases)
- FoundationPose: ~424K train (synthetic GSO, 438 categories), HDF5 backend
- Sampling: Omni3D 70%, CA-1M 8%, Waymo 5%, 3EED_det 4%, 3EED_ref 3%, FP 10%

HDF5 conversion (run once before training):
    python -m vis4d.data.io.to_hdf5 -p data/waymo/images
    python -m vis4d.data.io.to_hdf5 -p data/waymo/depth
    python -m vis4d.data.io.to_hdf5 -p data/cubifyanything/data
    python -m vis4d.data.io.to_hdf5 -p data/cubifyanything/depth_gt
    # FoundationPose images: custom script (symlink, needs followlinks=True)
    # FoundationPose depth: python -m vis4d.data.io.to_hdf5 -p data/foundationpose/depth

Usage:
    vis4d fit --config opendet3d/zoo/sam3_3d/sam3_3d_lingbot_depth_freeze21_omni3d_ca1m_waymo_3eed_fp.py --gpus 8
"""

from __future__ import annotations

from vis4d.config import class_config
from vis4d.config.typing import DataConfig, ExperimentConfig
from vis4d.data.data_pipe import DataPipe
from vis4d.data.io.hdf5 import HDF5Backend
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.zoo.base import get_default_cfg
from vis4d.zoo.base.callable import get_callable_cfg
from vis4d.zoo.base.dataloader import get_inference_dataloaders_cfg

from opendet3d.data.samplers import build_train_dataloader_with_ratios
from opendet3d.zoo.gdino3d.base.callback import (
    get_callback_cfg,
    get_omni3d_evaluator_cfg,
)
from opendet3d.zoo.gdino3d.base.dataset.cubifyanything import (
    get_ca1m_train_cfg,
)
from opendet3d.zoo.gdino3d.base.dataset.foundationpose import (
    get_foundationpose_train_cfg,
)
from opendet3d.zoo.gdino3d.base.dataset.omni3d import (
    get_omni3d_test_cfg,
    get_omni3d_train_cfg,
)
from opendet3d.zoo.gdino3d.base.dataset.threeeed import (
    get_threeeed_train_cfg,
)
from opendet3d.zoo.gdino3d.base.dataset.waymo import get_waymo_train_cfg
from opendet3d.zoo.gdino3d.base.pl import get_pl_cfg
from opendet3d.zoo.sam3_3d.base.connector import (
    get_sam3_3d_data_connector_cfg,
)
from opendet3d.zoo.sam3_3d.base.data import (
    sam3_3d_5mode_collate_fn,
    sam3_3d_test_collate_fn,
)
from opendet3d.zoo.sam3_3d.base.loss import get_sam3_3d_loss_cfg
from opendet3d.zoo.sam3_3d.base.model import (
    get_sam3_3d_cfg,
    get_sam3_3d_hyperparams_cfg,
)
from opendet3d.zoo.sam3_3d.base.optim import get_sam3_3d_optim_cfg


def get_config() -> ExperimentConfig:
    """Returns SAM3_3D + LingBot + all datasets config."""
    ######################################################
    ##                    General Config                ##
    ######################################################
    config = get_default_cfg(
        exp_name=(
            "sam3_3d_lingbot_depth_freeze21"
            "_omni3d_ca1m_waymo_3eed_fp"
        )
    )

    config.use_checkpoint = True

    params = get_sam3_3d_hyperparams_cfg(
        num_epochs=12,
        samples_per_gpu=4,
        workers_per_gpu=4,
        base_lr=1e-4,
    )

    config.params = params

    ######################################################
    ##          Datasets with augmentations             ##
    ######################################################
    data_backend = class_config(HDF5Backend)

    sam3_image_shape = (1008, 1008)

    # --- Omni3D (HDF5 backend) ---
    omni3d_data_root = "data/omni3d"
    omni3d_test_datasets = (
        "KITTI_test",
        "nuScenes_test",
        "SUNRGBD_test",
        "Hypersim_test",
        "ARKitScenes_test",
        "Objectron_test",
    )

    omni3d_train_data_cfg = get_omni3d_train_cfg(
        data_root=omni3d_data_root,
        data_backend=data_backend,
        shape=sam3_image_shape,
        # Disable annotation filtering
        truncation_thres=1.1,
        visibility_thres=-1.0,
        min_height_thres=0.0,
        max_height_thres=1e6,
    )

    omni3d_test_data_cfg = get_omni3d_test_cfg(
        data_root=omni3d_data_root,
        test_datasets=omni3d_test_datasets,
        data_backend=data_backend,
        shape=sam3_image_shape,
        with_depth=True,
    )

    # --- CubifyAnything (CA-1M, HDF5 backend) ---
    ca1m_train_data_cfg = get_ca1m_train_cfg(
        data_root="data/cubifyanything",
        data_backend=data_backend,
        shape=sam3_image_shape,
        cache_as_binary=True,
    )

    # --- Waymo (HDF5 backend) ---
    waymo_train_data_cfg = get_waymo_train_cfg(
        data_root="data/waymo",
        train_datasets=("Waymo_train",),
        data_backend=data_backend,
        shape=sam3_image_shape,
        cache_as_binary=True,
    )

    # --- 3EED detection (HDF5 backend) ---
    threeeed_det_train_data_cfg = get_threeeed_train_cfg(
        data_root="data/3eed",
        train_datasets=("3EED_det_train",),
        data_backend=data_backend,
        shape=sam3_image_shape,
        cache_as_binary=True,
    )

    # --- 3EED referring (HDF5 backend) ---
    threeeed_ref_train_data_cfg = get_threeeed_train_cfg(
        data_root="data/3eed",
        train_datasets=("3EED_ref_train",),
        data_backend=data_backend,
        shape=sam3_image_shape,
        cache_as_binary=True,
    )

    # --- FoundationPose (HDF5 backend) ---
    fp_train_data_cfg = get_foundationpose_train_cfg(
        data_root="data/foundationpose",
        train_datasets=("FoundationPose_train",),
        data_backend=data_backend,
        shape=sam3_image_shape,
        cache_as_binary=True,
    )

    # --- Build data config ---
    combined_train_datasets = class_config(
        DataPipe,
        datasets=[
            omni3d_train_data_cfg,       # 0: Omni3D
            ca1m_train_data_cfg,         # 1: CA-1M
            waymo_train_data_cfg,        # 2: Waymo
            threeeed_det_train_data_cfg, # 3: 3EED det
            threeeed_ref_train_data_cfg, # 4: 3EED ref
            fp_train_data_cfg,           # 5: FoundationPose
        ],
    )

    train_batchprocess_cfg = class_config(
        compose, transforms=[class_config(ToTensor)]
    )

    data = DataConfig()

    # Sampling ratios (7 datasets)
    # Omni3D 70%, CA-1M 8%, Waymo 5%, 3EED_det 4%, 3EED_ref 3%, FP 10%
    data.train_dataloader = class_config(
        build_train_dataloader_with_ratios,
        dataset=combined_train_datasets,
        target_proportions=[0.70, 0.08, 0.05, 0.04, 0.03, 0.10],
        epoch_dataset_idx=0,
        samples_per_gpu=params.samples_per_gpu,
        workers_per_gpu=params.workers_per_gpu,
        batchprocess_fn=train_batchprocess_cfg,
        collate_fn=get_callable_cfg(sam3_3d_5mode_collate_fn),
    )

    test_datasets_cfg = [omni3d_test_data_cfg]
    test_datasets_pipe = class_config(
        DataPipe, datasets=test_datasets_cfg
    )

    test_batchprocess_cfg = class_config(
        compose, transforms=[class_config(ToTensor)]
    )

    data.test_dataloader = get_inference_dataloaders_cfg(
        datasets_cfg=test_datasets_pipe,
        batchprocess_cfg=test_batchprocess_cfg,
        samples_per_gpu=1,
        workers_per_gpu=params.workers_per_gpu,
        collate_fn=sam3_3d_test_collate_fn,
    )

    config.data = data

    ######################################################
    ##                  MODEL & LOSS                    ##
    ######################################################
    config.model, box_coder = get_sam3_3d_cfg(
        params=params,
        sam3_checkpoint="pretrained/sam3/sam3_detector.pt",
        geometry_backend_type="lingbot_depth",
        lingbot_encoder_freeze_blocks=21,
        backbone_freeze_blocks=28,
    )

    config.loss = get_sam3_3d_loss_cfg(
        params, box_coder, use_3d_conf=True, use_ignore_suppress=True
    )

    ######################################################
    ##                    OPTIMIZERS                    ##
    ######################################################
    config.optimizers = get_sam3_3d_optim_cfg(
        params,
        freeze_backbone=params.freeze_backbone,
        freeze_all_pretrained=params.freeze_all_pretrained,
    )

    ######################################################
    ##                  DATA CONNECTOR                  ##
    ######################################################
    config.train_data_connector, config.test_data_connector = (
        get_sam3_3d_data_connector_cfg()
    )

    ######################################################
    ##                     CALLBACKS                    ##
    ######################################################
    omni3d_evaluator_cfg = get_omni3d_evaluator_cfg(
        data_root=omni3d_data_root,
        omni3d50=True,
        test_datasets=omni3d_test_datasets,
    )

    callbacks = get_callback_cfg(
        output_dir=config.output_dir,
        omni3d_evaluator=omni3d_evaluator_cfg,
        open_test_datasets=[],
    )

    config.callbacks = callbacks

    ######################################################
    ##                     PL CLI                       ##
    ######################################################
    config.pl_trainer = get_pl_cfg(config, params)

    return config.value_mode()
