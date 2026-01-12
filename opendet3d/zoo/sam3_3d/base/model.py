"""SAM3_3D model configuration."""

from __future__ import annotations

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.config.typing import ExperimentParameters

from opendet3d.model.detect3d.sam3_3d import SAM3_3D
from opendet3d.op.detect3d.grounding_dino_3d import (
    GroundingDINO3DCoder,
    GroundingDINO3DHead,
    RoI2Det3D,
)
from opendet3d.op.detect3d.geometry import UniDepthV2GeometryBackend


def get_sam3_3d_hyperparams_cfg(
    # Training
    num_epochs: int = 12,
    samples_per_gpu: int = 4,
    workers_per_gpu: int = 4,
    base_lr: float = 1e-4,
    weight_decay: float = 0.0001,
    accumulate_grad_batches: int = 1,
    check_val_every_n_epoch: int = 1,

    # Learning rate schedule (for MultiStepLR)
    step_1: int | None = None,  # If None, defaults to 2/3 of num_epochs
    step_2: int | None = None,  # If None, defaults to 5/6 of num_epochs

    # SAM3 specific
    num_queries: int = 100,
    hidden_dim: int = 256,
    num_decoder_layers: int = 6,

    # 3D specific
    depth_scale: float = 100.0,
    depth_output_scales: tuple[int, ...] = (4, 8, 16, 32),

    # Freeze settings
    freeze_sam3_backbone: bool = True,
    freeze_geometry_backend_encoder: bool = True,
) -> ExperimentParameters:
    """Get SAM3_3D hyperparameters.
    
    Args:
        num_epochs: Number of training epochs.
        samples_per_gpu: Batch size per GPU.
        workers_per_gpu: Number of data loading workers per GPU.
        base_lr: Base learning rate.
        weight_decay: Weight decay for optimizer.
        num_queries: Number of object queries.
        hidden_dim: Hidden dimension for transformer.
        num_decoder_layers: Number of decoder layers.
        depth_scale: Scale factor for depth prediction.
        depth_output_scales: Output scales for depth head.
        freeze_sam3_backbone: Whether to freeze SAM3 backbone.
        freeze_geometry_backend_encoder: Whether to freeze geometry encoder.
        
    Returns:
        ExperimentParameters with all hyperparameters.
    """
    params = ExperimentParameters()

    # Training
    params.num_epochs = num_epochs
    params.samples_per_gpu = samples_per_gpu
    params.workers_per_gpu = workers_per_gpu
    params.base_lr = base_lr
    params.lr = base_lr  # Alias for get_optim_cfg compatibility
    params.weight_decay = weight_decay
    params.accumulate_grad_batches = accumulate_grad_batches
    params.check_val_every_n_epoch = check_val_every_n_epoch

    # Learning rate schedule (for MultiStepLR)
    # Default to 2/3 and 5/6 of num_epochs if not specified
    params.step_1 = step_1 if step_1 is not None else int(num_epochs * 2 / 3)
    params.step_2 = step_2 if step_2 is not None else int(num_epochs * 5 / 6)

    # SAM3 specific
    params.num_queries = num_queries
    params.hidden_dim = hidden_dim
    params.num_decoder_layers = num_decoder_layers

    # 3D specific
    params.depth_scale = depth_scale
    params.depth_output_scales = depth_output_scales

    # Freeze settings
    params.freeze_sam3_backbone = freeze_sam3_backbone
    params.freeze_geometry_backend_encoder = freeze_geometry_backend_encoder

    return params


def get_sam3_3d_cfg(
    params: ExperimentParameters,
    sam3_checkpoint: str | None = None,
    geometry_backend_type: str = "unidepth_v2",
) -> tuple[ConfigDict, ConfigDict]:
    """Get SAM3_3D model configuration.
    
    Args:
        params: Experiment parameters.
        sam3_checkpoint: Path to SAM3 checkpoint.
        geometry_backend_type: Type of geometry backend.
        
    Returns:
        Tuple of (model_cfg, box_coder_cfg).
    """
    # Box coder
    box_coder = class_config(GroundingDINO3DCoder)
    
    # 3D Head
    bbox3d_head = class_config(
        GroundingDINO3DHead,
        hidden_dim=params.hidden_dim,
        num_layers=params.num_decoder_layers,
    )
    
    # Geometry backend
    if geometry_backend_type == "unidepth_v2":
        geometry_backend = class_config(
            UniDepthV2GeometryBackend,
            depth_scale=params.depth_scale,
        )
    else:
        geometry_backend = None
    
    # RoI2Det3D for inference
    roi2det3d = class_config(RoI2Det3D, box_coder=box_coder)
    
    # SAM3_3D model
    # Note: sam3_model will be built from checkpoint in __init__
    model = class_config(
        SAM3_3D,
        sam3_model=None,  # Will be built in __init__
        sam3_checkpoint=sam3_checkpoint,  # Pass checkpoint path
        bbox3d_head=bbox3d_head,
        box_coder=box_coder,
        geometry_backend=geometry_backend,
        roi2det3d=roi2det3d,
        freeze_sam3_backbone=params.freeze_sam3_backbone,
        freeze_geometry_backend_encoder=params.freeze_geometry_backend_encoder,
    )

    return model, box_coder

