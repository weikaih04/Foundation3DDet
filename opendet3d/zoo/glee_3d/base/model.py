"""GLEE_3D model configuration."""

from __future__ import annotations

import os

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.config.typing import ExperimentParameters

from opendet3d.model.detect3d.glee_3d import GLEE_3D
from opendet3d.op.detect3d.grounding_dino_3d import (
    GroundingDINO3DCoder,
    GroundingDINO3DHead,
)
from opendet3d.op.detect3d.early_depth_fusion import EarlyDepthFusionLingbot


# Path to GLEE repo (relative to project root)
GLEE_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "GLEE"
)
GLEE_SWINL_CONFIG = os.path.join(
    GLEE_ROOT,
    "projects/GLEE/configs/images/Plus/Stage3_scaleup_CLIPteacher_SwinL.yaml",
)


def get_glee_3d_hyperparams_cfg(
    # Training
    num_epochs: int = 12,
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 4,
    base_lr: float = 1e-4,
    weight_decay: float = 0.0001,
    accumulate_grad_batches: int = 1,
    check_val_every_n_epoch: int = 1,
    # LR schedule (MultiStepLR)
    step_1: int | None = None,
    step_2: int | None = None,
    # GLEE specific
    num_queries: int = 300,
    hidden_dim: int = 256,
    num_decoder_layers: int = 9,
    # Freeze settings
    freeze_backbone: bool = False,
    freeze_all_pretrained: bool = False,
    # NMS
    nms: bool = True,
    nms_iou_threshold: float = 0.6,
    score_threshold: float = 0.0,
) -> ExperimentParameters:
    """Get GLEE_3D hyperparameters."""
    params = ExperimentParameters()
    params.num_epochs = num_epochs
    params.samples_per_gpu = samples_per_gpu
    params.workers_per_gpu = workers_per_gpu
    params.base_lr = base_lr
    params.weight_decay = weight_decay
    params.accumulate_grad_batches = accumulate_grad_batches
    params.check_val_every_n_epoch = check_val_every_n_epoch

    # LR schedule
    params.step_1 = step_1 or int(num_epochs * 2 / 3)
    params.step_2 = step_2 or int(num_epochs * 5 / 6)

    # Alias for gdino3d optim compatibility (expects params.lr)
    params.lr = base_lr

    # GLEE specific
    params.num_queries = num_queries
    params.hidden_dim = hidden_dim
    params.num_decoder_layers = num_decoder_layers

    # Freeze
    params.freeze_backbone = freeze_backbone
    params.freeze_all_pretrained = freeze_all_pretrained

    # NMS
    params.nms = nms
    params.nms_iou_threshold = nms_iou_threshold
    params.score_threshold = score_threshold

    return params


def build_glee_model_from_cfg(
    glee_checkpoint: str = "pretrained/glee/GLEE_Plus_scaleup.pth",
) -> "GLEE_Model":  # noqa: F821
    """Build GLEE_Model using detectron2 config system.

    This is called at model instantiation time (not config time).
    Returns a GLEE_Model instance with backbone, pixel_decoder, predictor.
    Loads pretrained GLEE checkpoint if provided.
    """
    from detectron2.config import get_cfg
    from glee.config import add_glee_config
    from glee.models.glee_model import GLEE_Model
    from glee.models.matcher import HungarianMatcher

    cfg = get_cfg()
    add_glee_config(cfg)
    cfg.merge_from_file(GLEE_SWINL_CONFIG)

    # Override for GLEE_3D usage
    cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES = 300
    cfg.MODEL.MaskDINO.DEC_LAYERS = 9
    cfg.MODEL.MaskDINO.DN = "standard"
    cfg.MODEL.MaskDINO.DN_NUM = 100
    cfg.MODEL.VISUAL_PROMPT = True  # Enable sot_fuser for box/point prompts
    cfg.MODEL.SWIN.PRETRAINED_WEIGHT = ""  # Skip backbone pretrained (load full GLEE ckpt later)
    cfg.FIND_UNUSED_PARAMETERS = True

    # Fix CLIP tokenizer path (GLEE hardcodes relative path)
    clip_path = os.path.join(GLEE_ROOT, "projects/GLEE/clip_vit_base_patch32")
    if not os.path.exists(clip_path):
        clip_path = os.path.join(
            os.path.dirname(GLEE_ROOT), "projects/GLEE/clip_vit_base_patch32"
        )
    cfg.MODEL.TEXT.CLIP_PATH = clip_path

    matcher = HungarianMatcher(
        cost_class=4.0,
        cost_mask=0.0,
        cost_dice=0.0,
        cost_box=5.0,
        cost_giou=2.0,
        num_points=0,
    )

    model = GLEE_Model(
        cfg, matcher, device="cpu",
        video_info={"bz": 1, "len": 1},
        contras_mean=False,
    )

    # Load pretrained GLEE checkpoint
    if glee_checkpoint:
        import torch
        print(f"[GLEE_3D] Loading GLEE checkpoint: {glee_checkpoint}")
        ckpt = torch.load(glee_checkpoint, map_location="cpu")
        # Handle different checkpoint formats
        state_dict = ckpt.get("model", ckpt)
        # Strip 'glee.' prefix if present (from wrapped models)
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("glee."):
                cleaned[k[5:]] = v
            else:
                cleaned[k] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"[GLEE_3D] Loaded checkpoint: {len(cleaned)} keys")
        if missing:
            print(f"[GLEE_3D] Missing keys: {len(missing)} "
                  f"(first 5: {missing[:5]})")
        if unexpected:
            print(f"[GLEE_3D] Unexpected keys: {len(unexpected)} "
                  f"(first 5: {unexpected[:5]})")

    return model


def get_glee_3d_cfg(
    params: ExperimentParameters,
    glee_checkpoint: str = "pretrained/glee/GLEE_Plus_scaleup.pth",
    # Geometry backend
    geometry_backend_type: str = "lingbot_depth",
    lingbot_pretrained: str = "pretrained/lingbot-depth/postrain-dc-vitl14/model.pt",
    lingbot_encoder_freeze_blocks: int = 21,
    # Backbone freeze
    backbone_freeze_blocks: int = 0,
    # 3D box encoding
    canonical_rotation: bool = True,
    ambiguous_rotation: bool = False,
    # Task
    task: str = "glee3d",
) -> tuple[ConfigDict, ConfigDict]:
    """Build GLEE_3D model config.

    Returns:
        (model_cfg, box_coder_cfg) tuple.
    """
    # Box coder
    box_coder = class_config(
        GroundingDINO3DCoder,
        canonical_rotation=canonical_rotation,
        ambiguous_rotation=ambiguous_rotation,
    )

    # Geometry backend
    if geometry_backend_type == "lingbot_depth":
        from opendet3d.op.detect3d.geometry.lingbot_depth_backend import (
            LingbotDepthBackend,
        )

        geometry_backend = class_config(
            LingbotDepthBackend,
            pretrained_model=lingbot_pretrained,
            encoder_freeze_blocks=lingbot_encoder_freeze_blocks,
            target_latent_dim=256,
        )
        early_depth_fusion = class_config(
            EarlyDepthFusionLingbot,
            visual_dim=256,
            depth_dim=256,
        )
    else:
        raise ValueError(
            f"Unknown geometry_backend_type: {geometry_backend_type}"
        )

    # GLEE model (built from detectron2 cfg at instantiation time)
    glee_model = class_config(
        build_glee_model_from_cfg,
        glee_checkpoint=glee_checkpoint,
    )

    # 3D Head: 10 layers (GLEE two-stage: 1 initial + 9 decoder)
    num_3d_layers = params.num_decoder_layers + 1
    bbox3d_head = class_config(
        GroundingDINO3DHead,
        embed_dims=params.hidden_dim,
        num_decoder_layer=num_3d_layers,
        use_camera_prompt=True,  # LingbotDepth is NOT ray-aware
        depth_latent_dim=256,
    )

    # Compose GLEE_3D
    model = class_config(
        GLEE_3D,
        glee_model=glee_model,
        geometry_backend=geometry_backend,
        bbox3d_head=bbox3d_head,
        box_coder=box_coder,
        early_depth_fusion=early_depth_fusion,
        backbone_freeze_blocks=backbone_freeze_blocks,
        freeze_text_encoder=True,
        task=task,
    )

    return model, box_coder
