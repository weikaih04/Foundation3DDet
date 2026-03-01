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
from opendet3d.op.detect3d.early_depth_fusion import (
    EarlyDepthFusionLingbot,
    EarlyDepthFusionUniDepthV2,
)


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

    # Optimizer freeze settings (for get_sam3_3d_optim_cfg)
    freeze_backbone: bool = False,
    freeze_all_pretrained: bool = False,

    # NMS settings for evaluation (following 3D-MOOD's RoI2Det3D design)
    nms: bool = True,  # Whether to apply NMS during eval (enabled by default)
    class_agnostic_nms: bool = False,  # If False, NMS only within same category (recommended)
    nms_iou_threshold: float = 0.6,  # IoU threshold for NMS
    score_threshold: float = 0.0,  # Score threshold before NMS (0 = no filtering)
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
        freeze_backbone: If True, freeze vision + depth backbones (0.0x lr).
        freeze_all_pretrained: If True, freeze all pretrained, only train 3D head.

    Returns:
        ExperimentParameters with all hyperparameters.

    Note:
        Learning rate control is handled by param_groups in optim.py:
        - Default: All pretrained use 0.1x lr (like GDino3D)
        - freeze_backbone=True: Freeze vision + depth backbones
        - freeze_all_pretrained=True: Only train bbox3d_head + early_depth_fusion
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

    # Optimizer freeze settings
    params.freeze_backbone = freeze_backbone
    params.freeze_all_pretrained = freeze_all_pretrained

    # NMS settings for evaluation
    params.nms = nms
    params.class_agnostic_nms = class_agnostic_nms
    params.nms_iou_threshold = nms_iou_threshold
    params.score_threshold = score_threshold

    return params


def get_sam3_3d_cfg(
    params: ExperimentParameters,
    sam3_checkpoint: str | None = None,
    sam3_model: ConfigDict | None = None,
    geometry_backend_type: str = "unidepth_v2",
    lingbot_encoder_freeze_blocks: int = 0,
    backbone_freeze_blocks: int = 0,
    oracle_eval: bool = False,
    use_depth_input_test: bool = False,
    eval_3d_conf_weight: float = 0.5,
) -> tuple[ConfigDict, ConfigDict]:
    """Get SAM3_3D model configuration.

    Args:
        params: Experiment parameters.
        sam3_checkpoint: Path to SAM3 checkpoint. Used if sam3_model is None.
        sam3_model: Pre-built SAM3 model config (e.g., EfficientSAM3).
            If provided, sam3_checkpoint is ignored.
        geometry_backend_type: Type of geometry backend.
        backbone_freeze_blocks: Number of SAM3 ViT blocks to freeze (0=none, 30=last 2 trainable).
        oracle_eval: If True, use oracle evaluation mode (top-1 per prompt,
            no NMS) for measuring 3D regression with GT box prompts.

    Returns:
        Tuple of (model_cfg, box_coder_cfg).
    """
    # Box coder
    box_coder = class_config(GroundingDINO3DCoder)

    # Note: bbox3d_head is NOT created here - let SAM3_3D create it automatically
    # based on geometry_backend.is_ray_aware to get the correct use_camera_prompt setting.
    # For ray-aware backends (UniDepthV2, DetAny3D), use_camera_prompt=False.
    # For non-ray-aware backends (UniDepthHead v1), use_camera_prompt=True.

    # Geometry backend
    if geometry_backend_type == "unidepth_v2":
        # Use UniDepthV2-Small (v2-vits14) - memory efficient
        # output_scales=1 (default): 1/8 resolution, 256 channels, ~128x128
        # Higher resolution (output_scales=2,3) has fewer channels (128, 64)
        geometry_backend = class_config(
            UniDepthV2GeometryBackend,
            version="v2-vits14",
            encoder_pretrained="checkpoints/dinov2_backbones/unidepth_v2_s_dinov2_backbone.pth",
            decoder_pretrained="checkpoints/depth_heads/unidepth_v2_decoder_vits.pth",
            # output_scales=1 (default): 256 channels, latent_proj=Identity
        )
        # UniDepthV2-Small outputs 256-dim latents at 1/8 scale, no projection needed
        depth_latent_dim = 256
    elif geometry_backend_type == "lingbot_depth":
        from opendet3d.op.detect3d.geometry.lingbot_depth_backend import (
            LingbotDepthBackend,
        )
        geometry_backend = class_config(
            LingbotDepthBackend,
            pretrained_model="robbyant/lingbot-depth-postrain-dc-vitl14",
            num_tokens=2400,
            target_latent_dim=256,
            depth_loss_weight=1.0,
            silog_loss_weight=0.5,
            affine_global_weight=1.0,
            affine_local_weight=1.0,
            edge_loss_weight=1.0,
            mask_loss_weight=0.1,
            monocular_prob=0.7,
            masked_prob=0.2,
            mask_ratio_range=(0.6, 0.9),
            mask_patch_size=14,
            camera_loss_weight=1.0,
            detach_depth_latents=True,
            encoder_freeze_blocks=lingbot_encoder_freeze_blocks,
        )
        # Neck level 1 outputs 256-dim, same as UniDepthV2
        depth_latent_dim = 256
    else:
        geometry_backend = None
        depth_latent_dim = 128  # Default

    # EarlyDepthFusion: pick variant based on geometry backend
    if geometry_backend_type == "lingbot_depth":
        # ControlNet-style: LayerNorm + depth-only projection
        early_depth_fusion = class_config(
            EarlyDepthFusionLingbot,
            visual_dim=params.hidden_dim,
            depth_dim=depth_latent_dim,
            zero_init=True,
        )
    else:
        # Concat-Add: [visual; depth] -> project -> residual add
        early_depth_fusion = class_config(
            EarlyDepthFusionUniDepthV2,
            visual_dim=params.hidden_dim,
            depth_dim=depth_latent_dim,
            fusion_type="concat_add",
            zero_init=True,
        )

    # RoI2Det3D for inference (with NMS support)
    # Note: max_per_img not used - SAM3_3D already limits to 100 proposals per category
    roi2det3d = class_config(
        RoI2Det3D,
        box_coder=box_coder,
        nms=params.nms,
        class_agnostic_nms=params.class_agnostic_nms,
        iou_threshold=params.nms_iou_threshold,
        score_threshold=params.score_threshold,
    )

    # SAM3_3D model
    # Learning rate control is handled by param_groups in optimizer config,
    # not by freezing parameters. See optim.py for SAM3_3D-specific param_groups.
    # bbox3d_head=None - let SAM3_3D create it based on is_ray_aware and depth_latent_dim
    model = class_config(
        SAM3_3D,
        sam3_model=sam3_model,  # Pre-built model (e.g., EfficientSAM3) or None
        sam3_checkpoint=sam3_checkpoint if sam3_model is None else None,
        box_coder=box_coder,
        geometry_backend=geometry_backend,
        roi2det3d=roi2det3d,
        early_depth_fusion=early_depth_fusion,
        backbone_freeze_blocks=backbone_freeze_blocks,
        oracle_eval=oracle_eval,
        use_depth_input_test=use_depth_input_test,
        eval_3d_conf_weight=eval_3d_conf_weight,
    )

    return model, box_coder

