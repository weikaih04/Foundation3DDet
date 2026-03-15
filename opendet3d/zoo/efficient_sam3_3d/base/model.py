"""EfficientSAM3_3D model configuration."""

from __future__ import annotations

from ml_collections import ConfigDict
from vis4d.config import class_config

from opendet3d.model.detect3d.efficient_sam3_3d import EfficientSAM3_3D
from opendet3d.op.detect3d.grounding_dino_3d import (
    GroundingDINO3DCoder,
    GroundingDINO3DHead,
    RoI2Det3D,
)
from opendet3d.zoo.sam3_3d.base.efficient_sam3_builder import (
    build_efficientsam3_tiny_mobileclip_ft,
)

# Reuse hyperparams from SAM3_3D (same structure)
from opendet3d.zoo.sam3_3d.base.model import get_sam3_3d_hyperparams_cfg


def get_efficient_sam3_3d_cfg(
    params,
    sam3_3d_checkpoint: str | None = None,
    lingbot_pretrained: str = "pretrained/lingbot-depth/postrain-dc-vitl14/model.pt",
    efficient_sam3_checkpoint_dir: str = "pretrained/efficient_sam3",
    oracle_eval: bool = False,
    use_depth_input_test: bool = False,
    eval_3d_conf_weight: float = 0.5,
    canonical_rotation: bool = False,
) -> tuple[ConfigDict, ConfigDict]:
    """Get EfficientSAM3_3D model configuration.

    Args:
        params: Experiment parameters (from get_sam3_3d_hyperparams_cfg).
        sam3_3d_checkpoint: Path to trained SAM3_3D checkpoint for loading
            depth head + 3D head weights.
        lingbot_pretrained: Path to LingBot pretrained model (fallback if
            sam3_3d_checkpoint not provided).
        efficient_sam3_checkpoint_dir: Directory with EfficientSAM3 checkpoints.
        oracle_eval: Use oracle evaluation mode.
        use_depth_input_test: Use depth input at test time.
        eval_3d_conf_weight: Weight for 3D confidence in eval score.
        canonical_rotation: Use canonical rotation for 3D box encoding.

    Returns:
        (model_cfg, box_coder_cfg) tuple.
    """
    # Box coder
    box_coder = class_config(
        GroundingDINO3DCoder,
        ambiguous_rotation=False,
        canonical_rotation=canonical_rotation,
    )

    # 3D head
    bbox3d_head = class_config(
        GroundingDINO3DHead,
        embed_dims=params.hidden_dim,
        num_decoder_layer=params.num_decoder_layers,
        box_coder=box_coder,
        use_camera_prompt=True,  # LingBot is NOT ray-aware
        use_depth_prompt=True,
        depth_latent_dim=256,
    )

    # ROI post-processor
    roi2det3d = class_config(
        RoI2Det3D,
        box_coder=box_coder,
        nms=params.nms,
        class_agnostic_nms=params.class_agnostic_nms,
        iou_threshold=params.nms_iou_threshold,
        score_threshold=params.score_threshold,
    )

    # EfficientSAM3 model (TinyViT-11M + MobileCLIP-S1 + Geo-FT)
    efficient_sam3_model = class_config(
        build_efficientsam3_tiny_mobileclip_ft,
        checkpoint_dir=efficient_sam3_checkpoint_dir,
    )

    # Full model
    model = class_config(
        EfficientSAM3_3D,
        sam3_model=efficient_sam3_model,
        bbox3d_head=bbox3d_head,
        box_coder=box_coder,
        roi2det3d=roi2det3d,
        sam3_3d_checkpoint=sam3_3d_checkpoint,
        lingbot_pretrained=lingbot_pretrained,
        oracle_eval=oracle_eval,
        use_depth_input_test=use_depth_input_test,
        eval_3d_conf_weight=eval_3d_conf_weight,
    )

    return model, box_coder
