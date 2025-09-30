"""3D-MOOD model config with CUT3R Fusion."""

from __future__ import annotations

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
from opendet3d.op.fpp.channel_mapper import ChannelMapper
from opendet3d.zoo.gdino.base.model import GDINO_MODEL_WEIGHTS


def get_gdino3d_cut3r_cfg(
    params: ExperimentParameters,
    basemodel: ConfigDict,
    neck: ConfigDict,
    depth_fpn: ConfigDict,
    num_feature_levels: int = 4,
    chunked_size: int = -1,
    cat_mapping: dict[str, int] | None = None,
    pretrained: str | None = None,
    use_checkpoint: bool | FieldReference = False,
    # CUT3R Fusion parameters
    cut3r_checkpoint: str | None = None,
    cut3r_freeze: bool = True,
    fusion_levels: list[int] | None = None,
    fusion_strategies: dict[int, dict] | None = None,
    fusion_num_heads: int = 8,
    fusion_dropout: float = 0.1,
    use_relative_pos_bias: bool = False,
    # Optional: Load pretrained 3D-MOOD checkpoint
    pretrained_3dmood_checkpoint: str | None = None,
) -> ConfigDict:
    """Get the Grounding DINO 3D with CUT3R Fusion model config."""
    # UniDepth Head
    depth_head = class_config(
        UniDepthHead,
        depth_scale=params.depth_scale,
        input_dims=[256, 256, 256, 256],
        output_scales=params.depth_output_scales,
    )

    # 3D Head
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

    # Pretrained weights
    if pretrained is not None:
        weights = GDINO_MODEL_WEIGHTS[pretrained]
    else:
        weights = None

    # Override with pretrained 3D-MOOD checkpoint if provided
    if pretrained_3dmood_checkpoint is not None:
        weights = pretrained_3dmood_checkpoint

    # Build model with CUT3R Fusion
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
        # CUT3R Fusion parameters
        cut3r_checkpoint=cut3r_checkpoint,
        cut3r_freeze=cut3r_freeze,
        fusion_levels=fusion_levels,
        fusion_strategies=fusion_strategies,
        fusion_num_heads=fusion_num_heads,
        fusion_dropout=fusion_dropout,
        use_relative_pos_bias=use_relative_pos_bias,
    )

    return model, box_coder


def get_gdino3d_swin_tiny_cut3r_cfg(
    params: ExperimentParameters,
    chunked_size: int = -1,
    cat_mapping: dict[str, int] | None = None,
    pretrained: str | None = None,
    use_checkpoint: bool | FieldReference = False,
    # CUT3R Fusion parameters
    cut3r_checkpoint: str | None = None,
    cut3r_freeze: bool = True,
    fusion_levels: list[int] | None = None,
    fusion_strategies: dict[int, dict] | None = None,
    fusion_num_heads: int = 8,
    fusion_dropout: float = 0.1,
    use_relative_pos_bias: bool = False,
    # Optional: Load pretrained 3D-MOOD checkpoint
    pretrained_3dmood_checkpoint: str | None = None,
) -> ConfigDict:
    """Get the config of Swin-Tiny with CUT3R Fusion."""
    basemodel = class_config(
        SwinTransformer,
        pretrain_img_size=224,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        frozen_stages=-1,
    )

    neck = class_config(
        ChannelMapper,
        in_channels=[96, 192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg={"type": "GN", "num_groups": 32},
        num_outs=4,
    )

    depth_fpn = class_config(
        FPN,
        in_channels_list=[96, 192, 384, 768],
        out_channels=256,
    )

    return get_gdino3d_cut3r_cfg(
        params=params,
        basemodel=basemodel,
        neck=neck,
        depth_fpn=depth_fpn,
        num_feature_levels=4,
        chunked_size=chunked_size,
        cat_mapping=cat_mapping,
        pretrained=pretrained,
        use_checkpoint=use_checkpoint,
        # CUT3R Fusion parameters
        cut3r_checkpoint=cut3r_checkpoint,
        cut3r_freeze=cut3r_freeze,
        fusion_levels=fusion_levels,
        fusion_strategies=fusion_strategies,
        fusion_num_heads=fusion_num_heads,
        fusion_dropout=fusion_dropout,
        use_relative_pos_bias=use_relative_pos_bias,
        # Optional: Load pretrained 3D-MOOD checkpoint
        pretrained_3dmood_checkpoint=pretrained_3dmood_checkpoint,
    )

