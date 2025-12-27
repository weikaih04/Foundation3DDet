"""3D-MOOD model config."""

from __future__ import annotations

from typing import Literal

from ml_collections import ConfigDict, FieldReference
from vis4d.config import class_config
from vis4d.config.typing import ExperimentParameters
from vis4d.op.fpp.fpn import FPN

from opendet3d.model.detect3d.grounding_dino_3d import GroundingDINO3D
from opendet3d.op.base.swin import SwinTransformer
from opendet3d.op.detect3d.grounding_dino_3d import (
    GroundingDINO3DCoder,
    GroundingDINO3DHead,
    RoI2Det3D,
    UniDepthHead,
)
from opendet3d.op.detect3d.geometry import (
    UniDepthHeadBackend,
    DetAny3DGeometryBackend,
    UniDepthV2GeometryBackend,
)
from opendet3d.op.fpp.channel_mapper import ChannelMapper
from opendet3d.zoo.gdino.base.model import GDINO_MODEL_WEIGHTS


# ============================================================================
# Geometry Backend Configuration Helpers
# ============================================================================

GeometryBackendType = Literal["unidepth_head", "unidepth_head_dino", "detany3d", "unidepth_v2"]


def get_unidepth_head_backend_cfg(
    params: ExperimentParameters,
    depth_fpn: ConfigDict,
    depth_loss_weight: float = 10.0,
    detach_depth_latents: bool = True,
) -> ConfigDict:
    """Get UniDepthHead geometry backend config (original 3D-MOOD).

    This backend uses Swin + FPN features with UniDepthHead decoder.

    Args:
        params: Experiment parameters (for depth_scale, depth_output_scales).
        depth_fpn: FPN config for depth feature extraction.
        depth_loss_weight: Weight for SILog depth loss.
        detach_depth_latents: Whether to detach latents from gradient graph.

    Returns:
        ConfigDict for UniDepthHeadBackend.
    """
    depth_head = class_config(
        UniDepthHead,
        depth_scale=params.depth_scale,
        input_dims=[256, 256, 256, 256],
        output_scales=params.depth_output_scales,
    )

    return class_config(
        UniDepthHeadBackend,
        depth_head=depth_head,
        depth_loss_weight=depth_loss_weight,
        detach_depth_latents=detach_depth_latents,
    ), depth_fpn


def get_unidepth_head_dino_backend_cfg(
    params: ExperimentParameters,
    dino_model: str = "vit_large",
    dino_pretrained: str = "",
    freeze_dino: bool = False,  # Changed to False - freezing now handled by optimizer lr_mult=0.0
    detach_depth_latents: bool = True,
    depth_loss_weight: float = 10.0,
    stacking_mode: str = "mean",
    target_embed_dims: int = 256,
) -> tuple[ConfigDict, None]:
    """Get UniDepthHead + DINOv2 geometry backend config.

    This backend uses:
    - DINOv2 for feature extraction (all blocks, grouped like UniDepthV2)
    - Feature projection from DINOv2 embed_dim to target_embed_dims
    - UniDepthHead for depth prediction (like original 3D-MOOD)
    - External SILog loss

    Args:
        params: Experiment parameters (for depth_scale, depth_output_scales).
        dino_model: DINOv2 model variant ("vit_large", "vit_base", "vit_small").
        dino_pretrained: Path to pretrained DINO weights, or "" for hub weights.
        freeze_dino: Whether to freeze DINOv2 weights.
        detach_depth_latents: Whether to detach latents from gradient graph.
        depth_loss_weight: Weight for SILog depth loss.
        stacking_mode: How to aggregate blocks ("mean", "max", "last").
        target_embed_dims: Target embedding dimension for UniDepthHead (default: 256).

    Returns:
        Tuple of (backend_config, None). None because no FPN is used.
    """
    from opendet3d.op.detect3d.geometry.unidepth_head_dino_backend import (
        UniDepthHeadDinoBackend,
    )

    # Create UniDepthHead with target_embed_dims
    # UniDepthHead expects all input_dims to be equal to embed_dims
    depth_head = class_config(
        UniDepthHead,
        embed_dims=target_embed_dims,
        depth_scale=params.depth_scale,
        input_dims=[target_embed_dims] * 4,  # 4 grouped features
        output_scales=params.depth_output_scales,
    )

    return class_config(
        UniDepthHeadDinoBackend,
        dino_model=dino_model,
        dino_pretrained=dino_pretrained,
        depth_head=depth_head,
        depth_loss_weight=depth_loss_weight,
        freeze_dino=freeze_dino,
        detach_depth_latents=detach_depth_latents,
        stacking_mode=stacking_mode,
        target_embed_dims=target_embed_dims,
    ), None  # No FPN needed


def get_detany3d_backend_cfg(
    params: ExperimentParameters,
    dino_model: str = "vit_large",
    dino_pretrained: str = "",
    decoder_pretrained: str = "",
    decoder_dino_model: str | None = None,
    freeze_dino: bool = False,  # Changed to False - freezing now handled by optimizer lr_mult=0.0
    detach_depth_latents: bool = True,
    target_latent_dim: int = 128,
) -> ConfigDict:
    """Get DetAny3D geometry backend config (DINOv2 + Unidepth_Decoder).

    This backend uses DINOv2 features with DetAny3D's depth decoder.

    Args:
        params: Experiment parameters (for depth_output_scales).
        dino_model: DINO model variant (vit_large, vit_base, vit_small).
        dino_pretrained: Path to pretrained DINO weights, or "" for hub weights.
        decoder_pretrained: Path to pretrained decoder weights, or "" for random init.
        decoder_dino_model: DINO model variant that decoder expects (for dimension matching).
            If None, assumes decoder expects same dimension as dino_model.
            If different from dino_model, a linear projector will be added.
        freeze_dino: Whether to freeze DINO encoder.
        detach_depth_latents: Whether to detach latents from gradient graph.
        target_latent_dim: Target dimension for depth_latents (default 128).

    Returns:
        ConfigDict for DetAny3DGeometryBackend.
    """
    return class_config(
        DetAny3DGeometryBackend,
        dino_model=dino_model,
        dino_pretrained=dino_pretrained,
        decoder_pretrained=decoder_pretrained,
        decoder_dino_model=decoder_dino_model,
        freeze_dino=freeze_dino,
        output_scales=params.depth_output_scales,
        target_latent_dim=target_latent_dim,
        detach_depth_latents=detach_depth_latents,
    ), None  # DetAny3D does not use FPN


def get_unidepth_v2_backend_cfg(
    params: ExperimentParameters,
    version: str = "v2-vitl14",
    pretrained_path: str | None = None,
    encoder_pretrained: str | None = None,
    decoder_pretrained: str | None = None,
    freeze_encoder: bool = False,  # Changed to False - freezing now handled by optimizer lr_mult=0.0
    detach_depth_latents: bool = True,
    target_latent_dim: int = 128,
) -> ConfigDict:
    """Get UniDepthV2 geometry backend config (original UniDepthV2).

    This backend uses UniDepthV2's complete system with all native losses.

    Args:
        params: Experiment parameters (for depth_output_scales).
        version: UniDepthV2 version (v2-vitl14, v2-vitb14, v2-vits14).
        pretrained_path: Path to local full checkpoint. If None, loads from HuggingFace.
        encoder_pretrained: Path to encoder-only weights (DINOv2 backbone).
        decoder_pretrained: Path to decoder-only weights.
        freeze_encoder: Whether to freeze pixel encoder.
        detach_depth_latents: Whether to detach latents from gradient graph.
        target_latent_dim: Target dimension for depth_latents (default 128).

    Returns:
        ConfigDict for UniDepthV2GeometryBackend.
    """
    return class_config(
        UniDepthV2GeometryBackend,
        version=version,
        pretrained_path=pretrained_path,
        encoder_pretrained=encoder_pretrained,
        decoder_pretrained=decoder_pretrained,
        freeze_encoder=freeze_encoder,
        output_scales=params.depth_output_scales,
        target_latent_dim=target_latent_dim,
        detach_depth_latents=detach_depth_latents,
    ), None  # UniDepthV2 does not use FPN


def get_gdino3d_hyperparams_cfg() -> ExperimentParameters:
    """Get the hyperparameters for 3D-MOOD."""
    params = ExperimentParameters()

    # Training
    params.samples_per_gpu = 2
    params.workers_per_gpu = 4
    params.accumulate_grad_batches = 1
    params.lr = 0.0004  # bs=128, lr=0.0004
    params.weight_decay = 0.0001

    # Learning rate schedule
    params.num_epochs = 120
    params.step_1 = 80
    params.step_2 = 110
    params.check_val_every_n_epoch = 1

    # Grounding DINO 3D Coder
    params.center_scale = 10.0
    params.depth_scale = 2.0
    params.dim_scale = 2.0
    params.orientation = "rotation_6d"

    # Grounding DINO 3D Loss
    params.loss_center_weight = 1.0
    params.loss_depth_weight = 1.0
    params.loss_dim_weight = 1.0
    params.loss_rot_weight = 1.0

    # Aux Depth Loss
    params.si_log_weight = 10.0

    # RoI2Det3D
    params.nms = False
    params.class_agnostic_nms = False
    params.max_per_img = 100
    params.score_threshold = 0.0
    params.iou_threshold = 0.5

    # Depth Head
    params.depth_output_scales = 1

    # Geometry Backend Learning Rates (for ablation experiments)
    # These control the learning rate multipliers for geometry backend components
    # All components train at 10% of base lr (0.0004 * 0.1 = 0.00004)
    params.geom_encoder_lr_mult = 0.1   # Train encoder at 10% lr
    params.geom_decoder_lr_mult = 0.1   # Train decoder at 10% lr
    params.geom_projector_lr_mult = 0.1 # Train projector at 10% lr

    return params


def get_gdino3d_head_cfg(params: ExperimentParameters) -> ConfigDict:
    """Get the G-DINO 3D head config."""
    box_coder = class_config(
        GroundingDINO3DCoder,
        center_scale=params.center_scale,
        depth_scale=params.depth_scale,
        dim_scale=params.dim_scale,
        orientation=params.orientation,
    )

    bbox3d_head = class_config(
        GroundingDINO3DHead,
        box_coder=box_coder,
        depth_output_scales=params.depth_output_scales,
    )

    roi2det3d = class_config(
        RoI2Det3D,
        nms=params.nms,
        max_per_img=params.max_per_img,
        class_agnostic_nms=params.class_agnostic_nms,
        score_threshold=params.score_threshold,
        iou_threshold=params.iou_threshold,
        box_coder=box_coder,
    )

    return bbox3d_head, roi2det3d, box_coder


def get_gdino3d_cfg(
    params: ExperimentParameters,
    basemodel: ConfigDict,
    neck: ConfigDict,
    depth_fpn: ConfigDict,
    num_feature_levels: int = 4,
    chunked_size: int = -1,
    cat_mapping: dict[str, int] | None = None,
    pretrained: str | None = None,
    use_checkpoint: bool | FieldReference = False,
) -> ConfigDict:
    """Get the Grounding DINO with Swin-B model config."""
    # UniDepth Head
    depth_head = class_config(
        UniDepthHead,
        depth_scale=params.depth_scale,
        input_dims=[256, 256, 256, 256],
        output_scales=params.depth_output_scales,
    )

    bbox3d_head, roi2det3d, box_coder = get_gdino3d_head_cfg(params=params)

    if pretrained is not None:
        weights = GDINO_MODEL_WEIGHTS[pretrained]
    else:
        weights = None

    model = class_config(
        GroundingDINO3D,
        basemodel=basemodel,
        neck=neck,
        num_feature_levels=num_feature_levels,
        bbox3d_head=bbox3d_head,
        roi2det3d=roi2det3d,
        fpn=depth_fpn,
        depth_head=depth_head,
        use_checkpoint=use_checkpoint,
        weights=weights,
        chunked_size=chunked_size,
        cat_mapping=cat_mapping,
    )

    return model, box_coder


def get_gdino3d_swin_tiny_cfg(
    params: ExperimentParameters,
    chunked_size: int = -1,
    cat_mapping: dict[str, int] | None = None,
    pretrained: str | None = None,
    use_checkpoint: bool | FieldReference = False,
) -> ConfigDict:
    """Get the config of Swin-Tiny."""
    basemodel = class_config(
        SwinTransformer,
        convert_weights=True,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.2,
        out_indices=(0, 1, 2, 3),
        with_cp=use_checkpoint,
        pretrained="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
    )

    neck = class_config(
        ChannelMapper,
        in_channels=[192, 384, 768],
        out_channels=256,
        num_outs=4,
        kernel_size=1,
        norm="GroupNorm",
        num_groups=32,
        activation=None,
        bias=True,
    )

    depth_fpn = class_config(
        FPN,
        in_channels_list=[96, 192, 384, 768],
        out_channels=256,
        extra_blocks=None,
        start_index=0,
    )

    return get_gdino3d_cfg(
        params,
        basemodel=basemodel,
        neck=neck,
        depth_fpn=depth_fpn,
        chunked_size=chunked_size,
        cat_mapping=cat_mapping,
        pretrained=pretrained,
        use_checkpoint=use_checkpoint,
    )


def get_gdino3d_swin_base_cfg(
    params: ExperimentParameters,
    chunked_size: int = -1,
    cat_mapping: dict[str, int] | None = None,
    pretrained: str | None = None,
    use_checkpoint: bool | FieldReference = False,
) -> ConfigDict:
    """Get the config of Swin-Base."""
    basemodel = class_config(
        SwinTransformer,
        convert_weights=True,
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        drop_path_rate=0.3,
        out_indices=(0, 1, 2, 3),
        with_cp=use_checkpoint,
        pretrained="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth",
    )

    neck = class_config(
        ChannelMapper,
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_outs=4,
        kernel_size=1,
        norm="GroupNorm",
        num_groups=32,
        activation=None,
        bias=True,
    )

    depth_fpn = class_config(
        FPN,
        in_channels_list=[128, 256, 512, 1024],
        out_channels=256,
        extra_blocks=None,
        start_index=0,
    )

    return get_gdino3d_cfg(
        params,
        basemodel=basemodel,
        neck=neck,
        depth_fpn=depth_fpn,
        chunked_size=chunked_size,
        cat_mapping=cat_mapping,
        pretrained=pretrained,
        use_checkpoint=use_checkpoint,
    )


# ============================================================================
# Geometry Backend Ablation Configurations
# ============================================================================


def get_gdino3d_with_geometry_backend_cfg(
    params: ExperimentParameters,
    geometry_backend_type: GeometryBackendType = "unidepth_head",
    chunked_size: int = -1,
    cat_mapping: dict[str, int] | None = None,
    pretrained: str | None = None,
    use_checkpoint: bool | FieldReference = False,
    # Geometry backend options
    detach_depth_latents: bool = True,
    freeze_dino: bool = False,  # Changed to False - freezing now handled by optimizer lr_mult=0.0
    freeze_encoder: bool = False,  # Changed to False - freezing now handled by optimizer lr_mult=0.0
    target_latent_dim: int = 128,
    depth_loss_weight: float = 10.0,
    dino_model: str = "vit_small",
    dino_pretrained: str = "",
    detany3d_decoder_pretrained: str = "",
    detany3d_decoder_dino_model: str | None = None,
    unidepth_v2_version: str = "v2-vits14",
    unidepth_v2_pretrained: str | None = None,
    unidepth_v2_encoder_pretrained: str | None = None,
    unidepth_v2_decoder_pretrained: str | None = None,
) -> tuple[ConfigDict, ConfigDict]:
    """Get 3D-MOOD config with specified geometry backend.

    This function allows ablation studies comparing different geometry backends:
    - unidepth_head: Original 3D-MOOD (Swin + FPN + UniDepthHead)
    - unidepth_head_dino: UniDepthHead + DINOv2 (all blocks, grouped)
    - detany3d: DetAny3D (DINOv2 + Unidepth_Decoder)
    - unidepth_v2: UniDepthV2 (DINOv2 + UniDepthV2 Decoder)

    Args:
        params: Experiment parameters.
        geometry_backend_type: Type of geometry backend to use.
        chunked_size: Chunked size for text processing.
        cat_mapping: Category mapping.
        pretrained: Pretrained model name.
        use_checkpoint: Whether to use gradient checkpointing.
        detach_depth_latents: Whether to detach latents from gradient graph.
        freeze_dino: Whether to freeze DINO (for unidepth_head_dino, detany3d).
        freeze_encoder: Whether to freeze encoder (for unidepth_v2).
        target_latent_dim: Target dimension for depth_latents.
        depth_loss_weight: Weight for depth loss (for unidepth_head).
        dino_model: DINOv2 model variant (for unidepth_head_dino, detany3d).
        dino_pretrained: Path to DINOv2 pretrained weights (for unidepth_head_dino, detany3d).
        detany3d_decoder_pretrained: Path to DetAny3D decoder weights (for detany3d).
        unidepth_v2_version: UniDepthV2 version (for unidepth_v2).
        unidepth_v2_pretrained: Path to UniDepthV2 full checkpoint (for unidepth_v2).
        unidepth_v2_encoder_pretrained: Path to UniDepthV2 encoder-only weights (for unidepth_v2).
        unidepth_v2_decoder_pretrained: Path to UniDepthV2 decoder-only weights (for unidepth_v2).

    Returns:
        Tuple of (model_config, box_coder_config).
    """
    # Swin-Tiny backbone (shared for visual features)
    basemodel = class_config(
        SwinTransformer,
        convert_weights=True,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.2,
        out_indices=(0, 1, 2, 3),
        with_cp=use_checkpoint,
        pretrained="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
    )

    neck = class_config(
        ChannelMapper,
        in_channels=[192, 384, 768],
        out_channels=256,
        num_outs=4,
        kernel_size=1,
        norm="GroupNorm",
        num_groups=32,
        activation=None,
        bias=True,
    )

    # Get geometry backend config based on type
    if geometry_backend_type == "unidepth_head":
        depth_fpn = class_config(
            FPN,
            in_channels_list=[96, 192, 384, 768],
            out_channels=256,
            extra_blocks=None,
            start_index=0,
        )
        geometry_backend, fpn = get_unidepth_head_backend_cfg(
            params=params,
            depth_fpn=depth_fpn,
            depth_loss_weight=depth_loss_weight,
            detach_depth_latents=detach_depth_latents,
        )
    elif geometry_backend_type == "unidepth_head_dino":
        # UniDepthHead + DINOv2: DINOv2 (all blocks, grouped) + UniDepthHead
        geometry_backend, fpn = get_unidepth_head_dino_backend_cfg(
            params=params,
            dino_model=dino_model,
            dino_pretrained=dino_pretrained,
            freeze_dino=freeze_dino,
            detach_depth_latents=detach_depth_latents,
            depth_loss_weight=depth_loss_weight,
        )
    elif geometry_backend_type == "detany3d":
        geometry_backend, fpn = get_detany3d_backend_cfg(
            params=params,
            dino_model=dino_model,
            dino_pretrained=dino_pretrained,
            decoder_pretrained=detany3d_decoder_pretrained,
            decoder_dino_model=detany3d_decoder_dino_model,
            freeze_dino=freeze_dino,
            detach_depth_latents=detach_depth_latents,
            target_latent_dim=target_latent_dim,
        )
    elif geometry_backend_type == "unidepth_v2":
        geometry_backend, fpn = get_unidepth_v2_backend_cfg(
            params=params,
            version=unidepth_v2_version,
            pretrained_path=unidepth_v2_pretrained,
            encoder_pretrained=unidepth_v2_encoder_pretrained,
            decoder_pretrained=unidepth_v2_decoder_pretrained,
            freeze_encoder=freeze_encoder,
            detach_depth_latents=detach_depth_latents,
            target_latent_dim=target_latent_dim,
        )
    else:
        raise ValueError(f"Unknown geometry backend type: {geometry_backend_type}")

    # 3D detection head
    bbox3d_head, roi2det3d, box_coder = get_gdino3d_head_cfg(params=params)

    if pretrained is not None:
        weights = GDINO_MODEL_WEIGHTS[pretrained]
    else:
        weights = None

    model = class_config(
        GroundingDINO3D,
        basemodel=basemodel,
        neck=neck,
        num_feature_levels=4,
        bbox3d_head=bbox3d_head,
        roi2det3d=roi2det3d,
        fpn=fpn,
        depth_head=None,  # Use geometry_backend instead
        geometry_backend=geometry_backend,
        use_checkpoint=use_checkpoint,
        weights=weights,
        chunked_size=chunked_size,
        cat_mapping=cat_mapping,
    )

    return model, box_coder
