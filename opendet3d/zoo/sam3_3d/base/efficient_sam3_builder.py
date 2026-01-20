"""EfficientSAM3 Model Builder for SAM3_3D integration.

This module provides helper functions to build EfficientSAM3 models
that are compatible with the SAM3_3D pipeline.

The main entry point is build_efficientsam3_for_3d(), which automatically
selects the appropriate checkpoint and builds a Sam3Image model instance.

Example:
    >>> # Build TinyViT-11M with MobileCLIP-S1 and Geo FT (recommended)
    >>> model = build_efficientsam3_for_3d(
    ...     backbone_type="tinyvit",
    ...     model_size="11m",
    ...     use_mobileclip=True,
    ...     use_geo_ft=True,
    ... )
    >>> # Use with SAM3_3D
    >>> sam3_3d = SAM3_3D(sam3_model=model, ...)
"""

from __future__ import annotations

import os
from typing import Literal


# Checkpoint mapping for easy selection
# Format: (backbone_type, model_size, use_mobileclip, geo_ft) -> checkpoint_name
EFFICIENT_SAM3_CHECKPOINTS = {
    # Type 1: SAM3 Text Encoder + EfficientSAM3 Image Encoder
    ("repvit", "m0_9", False, False): "efficient_sam3_repvit_s.pt",
    ("repvit", "m1_1", False, False): "efficient_sam3_repvit_m.pt",
    ("repvit", "m1_1", False, True): "efficient_sam3_repvit_m_geo_ft.pt",
    ("repvit", "m2_3", False, False): "efficient_sam3_repvit_l.pt",

    ("tinyvit", "5m", False, False): "efficient_sam3_tinyvit_s.pt",
    ("tinyvit", "11m", False, False): "efficient_sam3_tinyvit_m.pt",
    ("tinyvit", "11m", False, True): "efficient_sam3_tinyvit_m_geo_ft.pt",  # Recommended for geometry
    ("tinyvit", "21m", False, False): "efficient_sam3_tinyvit_l.pt",

    ("efficientvit", "b0", False, False): "efficient_sam3_efficientvit_s.pt",
    ("efficientvit", "b1", False, False): "efficient_sam3_efficientvit_m.pt",
    ("efficientvit", "b1", False, True): "efficient_sam3_efficientvit_m_geo_ft.pt",
    ("efficientvit", "b2", False, False): "efficient_sam3_efficientvit_l.pt",

    # Type 2: MobileCLIP Text + EfficientSAM3 Image (Both distilled + Geo FT)
    ("repvit", "m0_9", True, False): "efficient_sam3_repvit-m0_9_mobileclip_s1.pth",
    ("repvit", "m1_1", True, False): "efficient_sam3_repvit-m1_1_mobileclip_s1.pth",
    ("repvit", "m1_1", True, True): "efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth",
    ("repvit", "m2_3", True, False): "efficient_sam3_repvit-m2_3_mobileclip_s1.pth",

    ("tinyvit", "5m", True, False): "efficient_sam3_tinyvit_5m_mobileclip_s1.pth",
    ("tinyvit", "11m", True, False): "efficient_sam3_tinyvit_11m_mobileclip_s1.pth",
    ("tinyvit", "11m", True, True): "efficient_sam3_tiny_vit_11m_mobileclip_s1_ft.pth",  # ⭐ RECOMMENDED
    ("tinyvit", "21m", True, False): "efficient_sam3_tinyvit_21m_mobileclip_s1.pth",

    ("efficientvit", "b0", True, False): "efficient_sam3_efficientvit-b0_mobileclip_s1.pth",
    ("efficientvit", "b1", True, False): "efficient_sam3_efficientvit-b1_mobileclip_s1.pth",
    ("efficientvit", "b1", True, True): "efficient_sam3_efficientvit_b1_mobileclip_s1_ft.pth",
    ("efficientvit", "b2", True, False): "efficient_sam3_efficientvit-b2_mobileclip_s1.pth",
}


def get_checkpoint_path(
    backbone_type: Literal["repvit", "tinyvit", "efficientvit"],
    model_size: str,
    use_mobileclip: bool = True,
    use_geo_ft: bool = True,
    checkpoint_dir: str = "pretrained/efficient_sam3",
) -> str:
    """Get EfficientSAM3 checkpoint path based on configuration.

    Args:
        backbone_type: Vision backbone type (repvit/tinyvit/efficientvit)
        model_size: Model size variant:
            - RepViT: m0_9, m1_1, m2_3
            - TinyViT: 5m, 11m, 21m
            - EfficientViT: b0, b1, b2
        use_mobileclip: Whether to use MobileCLIP text encoder
            - True: Use MobileCLIP-S1 (63.56M, 94.7% performance)
            - False: Use SAM3 text encoder (353M, 100% performance)
        use_geo_ft: Whether to use geometry fine-tuned version
            - True: Fine-tuned on geometry prompts (recommended for 3D tasks)
            - False: Base distilled version
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Full path to checkpoint file

    Raises:
        ValueError: If configuration is not available
        FileNotFoundError: If checkpoint file doesn't exist

    Example:
        >>> path = get_checkpoint_path("tinyvit", "11m", True, True)
        >>> print(path)
        pretrained/efficient_sam3/efficient_sam3_tiny_vit_11m_mobileclip_s1_ft.pth
    """
    key = (backbone_type, model_size, use_mobileclip, use_geo_ft)

    if key not in EFFICIENT_SAM3_CHECKPOINTS:
        # Fallback: try without geo_ft
        key_fallback = (backbone_type, model_size, use_mobileclip, False)
        if key_fallback in EFFICIENT_SAM3_CHECKPOINTS:
            print(f"[Warning] Geometry fine-tuned checkpoint not found, using standard version")
            checkpoint_name = EFFICIENT_SAM3_CHECKPOINTS[key_fallback]
        else:
            available_configs = [
                f"{bt}-{ms} (mobileclip={mc}, geo_ft={gf})"
                for bt, ms, mc, gf in EFFICIENT_SAM3_CHECKPOINTS.keys()
                if bt == backbone_type
            ]
            raise ValueError(
                f"Checkpoint configuration not found: {backbone_type}-{model_size} "
                f"(mobileclip={use_mobileclip}, geo_ft={use_geo_ft})\n"
                f"Available {backbone_type} configurations:\n  " + "\n  ".join(available_configs)
            )
    else:
        checkpoint_name = EFFICIENT_SAM3_CHECKPOINTS[key]

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    # Return path without checking file existence
    # The check will be done in build function to provide better error messages
    return checkpoint_path


def build_efficientsam3_for_3d(
    backbone_type: Literal["repvit", "tinyvit", "efficientvit"] = "tinyvit",
    model_size: str = "11m",
    use_mobileclip: bool = True,
    use_geo_ft: bool = True,
    checkpoint_path: str | None = None,
    checkpoint_dir: str = "pretrained/efficient_sam3",
    device: str = "cpu",
):
    """Build EfficientSAM3 model for SAM3_3D integration.

    This function creates an EfficientSAM3 model that is fully compatible
    with SAM3_3D pipeline. The returned model is a Sam3Image instance.

    Args:
        backbone_type: Vision backbone type
            - "tinyvit": Balanced efficiency and performance (recommended)
            - "repvit": Mobile-optimized for highest throughput
            - "efficientvit": Ultra-lightweight for minimal latency
        model_size: Model size variant (depends on backbone_type)
            - TinyViT: "5m" (5.07M), "11m" (10.55M), "21m" (20.62M)
            - RepViT: "m0_9" (4.72M), "m1_1" (7.77M), "m2_3" (22.40M)
            - EfficientViT: "b0" (0.68M), "b1" (4.64M), "b2" (14.98M)
        use_mobileclip: Use MobileCLIP text encoder (lighter) or SAM3 text encoder
            - True: MobileCLIP-S1 (63.56M, 94.7% performance, 85.7% total reduction)
            - False: SAM3 Text (353M, 100% performance, 52.1% total reduction)
        use_geo_ft: Use geometry fine-tuned version (recommended for 3D tasks)
            - True: Fine-tuned on box/point prompts (better for 3D detection)
            - False: Base distilled version
        checkpoint_path: Explicit checkpoint path (overrides auto-selection)
        checkpoint_dir: Directory containing checkpoints
        device: Device to load model on

    Returns:
        Sam3Image model instance (compatible with SAM3_3D)

    Example:
        >>> # Recommended: TinyViT-11M + MobileCLIP-S1 + Geo FT
        >>> model = build_efficientsam3_for_3d(
        ...     backbone_type="tinyvit",
        ...     model_size="11m",
        ...     use_mobileclip=True,
        ...     use_geo_ft=True,
        ... )
        >>> # Use with SAM3_3D
        >>> from opendet3d.model.detect3d.sam3_3d import SAM3_3D
        >>> sam3_3d = SAM3_3D(sam3_model=model)
    """
    # EfficientSAM3 has nested structure: efficientsam3/sam3/sam3/model_builder.py
    # We need to temporarily override sam3 module to use efficientsam3's version
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    efficientsam3_sam3_path = os.path.join(project_root, "efficientsam3", "sam3")

    # Save original sam3 module if exists
    original_sam3 = sys.modules.get("sam3", None)
    original_sam3_model = sys.modules.get("sam3.model", None)
    original_sam3_model_builder = sys.modules.get("sam3.model_builder", None)

    # Remove sam3 from modules cache to force reimport from efficientsam3
    modules_to_remove = [k for k in sys.modules.keys() if k == "sam3" or k.startswith("sam3.")]
    for mod in modules_to_remove:
        del sys.modules[mod]

    # Add efficientsam3/sam3 to front of path
    sys.path.insert(0, efficientsam3_sam3_path)

    # Now import from efficientsam3
    from sam3.model_builder import build_efficientsam3_image_model

    # Restore path (keep efficientsam3 for model's internal imports)
    # Don't restore original sam3 modules - efficientsam3 needs its own sam3

    # Auto-select checkpoint if not provided
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(
            backbone_type=backbone_type,
            model_size=model_size,
            use_mobileclip=use_mobileclip,
            use_geo_ft=use_geo_ft,
            checkpoint_dir=checkpoint_dir,
        )

    # Check if checkpoint file exists
    if not os.path.exists(checkpoint_path):
        checkpoint_name = os.path.basename(checkpoint_path)
        raise FileNotFoundError(
            f"[Error] Checkpoint file not found: {checkpoint_path}\n\n"
            f"Please download from Hugging Face:\n"
            f"  mkdir -p {checkpoint_dir}\n"
            f"  wget https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/stage1_all_converted/{checkpoint_name} \\\n"
            f"       -O {checkpoint_path}\n\n"
            f"Or visit:\n"
            f"  https://huggingface.co/Simon7108528/EfficientSAM3/tree/main/stage1_all_converted"
        )

    print(f"\n{'='*80}")
    print(f"[EfficientSAM3] Building Model")
    print(f"{'='*80}")
    print(f"  Backbone: {backbone_type}-{model_size}")
    print(f"  Text Encoder: {'MobileCLIP-S1 (63.56M)' if use_mobileclip else 'SAM3 (353M)'}")
    print(f"  Geometry FT: {'Yes (optimized for box/point prompts)' if use_geo_ft else 'No'}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")

    # Build EfficientSAM3 model
    print("Loading model...")
    model = build_efficientsam3_image_model(
        checkpoint_path=checkpoint_path,
        device=device,
        eval_mode=True,
        backbone_type=backbone_type,
        model_name=model_size,
        text_encoder_type="MobileCLIP-S1" if use_mobileclip else None,
    )

    print(f"\n[EfficientSAM3] Model built successfully!")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Hidden dim: {model.hidden_dim}")

    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params / 1e6:.2f}M")
    print(f"{'='*80}\n")

    return model


# Preset configurations for common use cases

def build_efficientsam3_tiny_geo_ft(**kwargs):
    """Build TinyViT-11M + SAM3 Text + Geo FT.

    Best for: Geometry tasks where text quality is critical
    - Vision: TinyViT-11M (10.55M)
    - Text: SAM3 (353M, 100% performance)
    - Total: 414M (52.1% reduction from SAM3 865M)
    - Geometry: Fine-tuned on box/point prompts
    """
    return build_efficientsam3_for_3d(
        backbone_type="tinyvit",
        model_size="11m",
        use_mobileclip=False,
        use_geo_ft=True,
        **kwargs
    )


def build_efficientsam3_tiny_mobileclip_ft(**kwargs):
    """Build TinyViT-11M + MobileCLIP-S1 + Geo FT. ⭐ RECOMMENDED

    Best for: All-around performance (text + geometry optimization)
    - Vision: TinyViT-11M (10.55M, geometry fine-tuned)
    - Text: MobileCLIP-S1 (63.56M, 94.7% performance)
    - Total: 124M (85.7% reduction from SAM3 865M)
    - Geometry: Fine-tuned on box/point prompts
    - Speed: 4-6x faster than SAM3

    This is the recommended configuration for 3D detection tasks.
    """
    return build_efficientsam3_for_3d(
        backbone_type="tinyvit",
        model_size="11m",
        use_mobileclip=True,
        use_geo_ft=True,
        **kwargs
    )


def build_efficientsam3_repvit_mobileclip_ft(**kwargs):
    """Build RepViT-M1.1 + MobileCLIP-S1 + Geo FT.

    Best for: Fastest inference speed
    - Vision: RepViT-M1.1 (7.77M, geometry fine-tuned)
    - Text: MobileCLIP-S1 (63.56M, 94.7% performance)
    - Total: 121M (86.0% reduction from SAM3 865M)
    - Geometry: Fine-tuned on box/point prompts
    - Speed: 5-7x faster than SAM3 (mobile-optimized)
    """
    return build_efficientsam3_for_3d(
        backbone_type="repvit",
        model_size="m1_1",
        use_mobileclip=True,
        use_geo_ft=True,
        **kwargs
    )


def build_efficientsam3_efficientvit_mobileclip_ft(**kwargs):
    """Build EfficientViT-B1 + MobileCLIP-S1 + Geo FT.

    Best for: Minimal model size and latency
    - Vision: EfficientViT-B1 (4.64M, geometry fine-tuned)
    - Text: MobileCLIP-S1 (63.56M, 94.7% performance)
    - Total: 118M (86.4% reduction from SAM3 865M)
    - Geometry: Fine-tuned on box/point prompts
    - Speed: Ultra-lightweight for edge devices
    """
    return build_efficientsam3_for_3d(
        backbone_type="efficientvit",
        model_size="b1",
        use_mobileclip=True,
        use_geo_ft=True,
        **kwargs
    )
