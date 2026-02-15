"""3D-MOOD."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from vis4d.op.base import BaseModel
from vis4d.op.fpp.fpn import FPN

from opendet3d.model.detect.grounding_dino import GroundingDINO
from opendet3d.model.language.mm_bert import BertModel
from opendet3d.op.detect3d.grounding_dino_3d import (
    GroundingDINO3DHead,
    RoI2Det3D,
)
from opendet3d.op.detect.grounding_dino import GroundingDINOHead, RoI2Det
from opendet3d.op.fpp.channel_mapper import ChannelMapper
from opendet3d.op.detect3d.geometry import GeometryBackendBase, UniDepthHeadBackend
from opendet3d.op.detect3d.depth_memory_fusion import DepthMemoryFusion

# CUT3R Fusion imports - DISABLED for geometry ablation experiments
# from opendet3d.model.cut3r_tower import CUT3RTower
# from opendet3d.model.cut3r_fusion import MultiScaleCUT3RFusion


class Det3DOut(NamedTuple):
    """Output of the detection model.

    boxes (list[Tensor]): 2D bounding boxes of shape [N, 4] in xyxy format.
    boxes3d (list[Tensor]): 3D bounding boxes of shape [N, 10].
    scores (list[Tensor]): confidence scores of shape [N,].
    class_ids (list[Tensor]): class ids of shape [N,].
    depth_maps (list[Tensor] | None): depth maps for each image.
    categories (list[list[str]] | None): category names for each detection.
    predicted_intrinsics (Tensor | None): predicted camera intrinsics (B, 3, 3).
    """

    boxes: list[Tensor]
    boxes3d: list[Tensor]
    scores: list[Tensor]
    class_ids: list[Tensor]
    depth_maps: list[Tensor] | None
    categories: list[list[str]] | None = None
    predicted_intrinsics: Tensor | None = None


class GroundingDINO3DOut(NamedTuple):
    """Output of the Grounding DINO model."""

    all_layers_cls_scores: list[Tensor]
    all_layers_bbox_preds: list[Tensor]
    all_layers_bbox_3d_preds: list[Tensor]
    enc_outputs_class: Tensor
    enc_outputs_coord: Tensor
    enc_outputs_3d: Tensor
    text_token_mask: Tensor
    dn_meta: dict[str, Tensor]
    positive_maps: list[Tensor]
    depth_maps: Tensor | None
    geom_losses: dict[str, Tensor] | None = None


class GroundingDINO3D(GroundingDINO):
    """Grounding DINO."""

    def __init__(
        self,
        basemodel: BaseModel,
        neck: ChannelMapper,
        texts: list[str] | None = None,
        custom_entities: bool = True,
        chunked_size: int = -1,
        num_queries: int = 900,
        num_feature_levels: int = 4,
        use_checkpoint: bool = False,
        bbox_head: GroundingDINOHead | None = None,
        language_model: BertModel | None = None,
        roi2det: RoI2Det | None = None,
        bbox3d_head: GroundingDINO3DHead | None = None,
        roi2det3d: RoI2Det3D | None = None,
        depth_head: nn.Module | None = None,
        fpn: FPN | None = None,
        freeze_detector: bool = False,
        weights: str | None = None,
        cat_mapping: dict[str, int] | None = None,
        # Geometry Backend (new unified interface)
        geometry_backend: GeometryBackendBase | None = None,
        depth_loss_weight: float = 10.0,
        # Depth-Memory Fusion (fuses depth latents into encoder memory)
        depth_memory_fusion: DepthMemoryFusion | None = None,
        # CUT3R Fusion parameters
        cut3r_checkpoint: str | None = None,
        cut3r_freeze: bool = True,
        fusion_levels: list[int] | None = None,
        fusion_strategies: dict[int, dict] | None = None,
        fusion_num_heads: int = 8,
        fusion_dropout: float = 0.1,
        use_relative_pos_bias: bool = False,
        freeze_backbone: bool = False,
    ) -> None:
        """Create the Grounding DINO model."""
        super().__init__(
            basemodel=basemodel,
            neck=neck,
            texts=texts,
            custom_entities=custom_entities,
            chunked_size=chunked_size,
            num_queries=num_queries,
            num_feature_levels=num_feature_levels,
            use_checkpoint=use_checkpoint,
            bbox_head=bbox_head,
            roi2det=roi2det,
            language_model=language_model,
            weights=weights,
        )

        # Depth Head / Geometry Backend
        self.fpn = fpn
        self.depth_head = depth_head

        # Setup geometry backend FIRST (before bbox3d_head)
        # If geometry_backend is provided, use it directly
        # Otherwise, wrap depth_head in UniDepthHeadBackend for backward compat
        print(f"\n[DEBUG __init__] geometry_backend parameter: {type(geometry_backend) if geometry_backend is not None else None}")
        print(f"[DEBUG __init__] depth_head parameter: {type(depth_head) if depth_head is not None else None}")

        if geometry_backend is not None:
            print(f"[DEBUG __init__] Using provided geometry_backend: {type(geometry_backend).__name__}")
            self.geometry_backend = geometry_backend
        elif depth_head is not None:
            # Backward compatibility: wrap depth_head in UniDepthHeadBackend
            print(f"[DEBUG __init__] Wrapping depth_head in UniDepthHeadBackend")
            self.geometry_backend = UniDepthHeadBackend(
                depth_head=depth_head,
                depth_loss_weight=depth_loss_weight,
            )
        else:
            print(f"[DEBUG __init__] No geometry backend or depth_head provided")
            self.geometry_backend = None

        # Determine use_camera_prompt based on geometry_backend.is_ray_aware
        # If backend is ray-aware (UniDepthV2, DetAny3D), depth_latents already
        # contain ray info, so we don't need the camera prompt branch.
        # If backend is NOT ray-aware (UniDepthHead v1), we need camera prompt.
        if self.geometry_backend is not None:
            use_camera_prompt = not self.geometry_backend.is_ray_aware
            print(f"[DEBUG __init__] geometry_backend.is_ray_aware={self.geometry_backend.is_ray_aware}, use_camera_prompt={use_camera_prompt}")
        else:
            use_camera_prompt = True  # Default to True for safety

        # Create bbox3d_head with appropriate use_camera_prompt setting
        if bbox3d_head is not None:
            self.bbox3d_head = bbox3d_head
        else:
            self.bbox3d_head = GroundingDINO3DHead(use_camera_prompt=use_camera_prompt)

        self.roi2det3d = roi2det3d or RoI2Det3D()

        # ========== Depth-Memory Fusion Setup ==========
        self.depth_memory_fusion = depth_memory_fusion

        # ========== CUT3R Fusion Setup ==========
        # DISABLED for geometry ablation experiments
        # Save freeze setting
        self._cut3r_freeze = cut3r_freeze

        # Always disable CUT3R for now
        self.cut3r_tower = None
        self.cut3r_fusion = None

        # if cut3r_checkpoint is not None:
        #     # Create CUT3R Tower and load pretrained weights from fixed path
        #     self.cut3r_tower = CUT3RTower(freeze=cut3r_freeze)
        #     # ... fusion setup ...
        # else:
        #     self.cut3r_tower = None
        #     self.cut3r_fusion = None

        if freeze_detector:
            self._freeze_detector()

        if freeze_backbone:
            self._freeze_backbone()

        # Print trainable parameter statistics
        self._print_trainable_params()

        # Save weights path for later use in summary
        self._weights_path = weights

        # Note: Weight loading summary will be printed in load_state_dict()
        # after all weights are actually loaded

        self.cat_mapping = cat_mapping

    def on_load_checkpoint(self, checkpoint):
        """
        PyTorch Lightning hook called when loading a checkpoint.

        This is called BEFORE load_state_dict, so we can:
        1. Load geometry backend pretrained weights first
        2. Filter out incompatible keys from the checkpoint
        3. Let PyTorch Lightning load the filtered checkpoint
        """
        print("\n" + "="*80)
        print("📦 CHECKPOINT LOADING (PyTorch Lightning Hook)")
        print("="*80)

        # Get the state_dict from checkpoint
        state_dict = checkpoint.get('state_dict', {})

        # Analyze checkpoint content
        has_geometry_backend = any('geometry_backend' in key for key in state_dict.keys())
        has_depth_head = any(key.startswith('depth_head.') for key in state_dict.keys())
        has_cut3r = any('cut3r_tower' in key for key in state_dict.keys())

        # Determine if this is resume training or first training
        is_resume = has_geometry_backend

        if is_resume:
            # Resume training: load everything from checkpoint
            print("\n📌 Mode: Resume Training")
            print("Loading complete checkpoint (all components)")
            print(f"  Resuming from epoch {checkpoint.get('epoch', 'unknown')}")
            print(f"  Resuming from global_step {checkpoint.get('global_step', 'unknown')}")

        else:
            # First training: load pretrained weights + 3D-MOOD checkpoint
            print("\n📌 Mode: First Training (Fine-tuning)")

            # Step 1: Load geometry backend pretrained weights
            if self.geometry_backend is not None and hasattr(self.geometry_backend, 'load_pretrained_weights'):
                print("\n[Step 1/2] Loading geometry backend pretrained weights...")
                self.geometry_backend.load_pretrained_weights()

            # Step 2: Filter out incompatible keys from checkpoint
            print("\n[Step 2/2] Filtering 3D-MOOD checkpoint...")

            if has_depth_head:
                print("  Filtering out old depth_head/fpn keys (using geometry_backend instead)")
                original_count = len(state_dict)
                filtered_state_dict = {
                    k: v for k, v in state_dict.items()
                    if not k.startswith('depth_head.') and not k.startswith('fpn.')
                }
                filtered_count = original_count - len(filtered_state_dict)
                print(f"  Loaded: {len(filtered_state_dict)} keys")
                print(f"  Filtered: {filtered_count} keys (depth_head/fpn)")

                # Update checkpoint with filtered state_dict
                checkpoint['state_dict'] = filtered_state_dict

            # Step 3: Reset training state (epoch, step, optimizer)
            # This ensures we start from epoch 0 when fine-tuning from 3D-MOOD
            print("\n[Step 3/3] Resetting training state for fine-tuning...")
            if 'epoch' in checkpoint:
                old_epoch = checkpoint['epoch']
                checkpoint['epoch'] = 0
                print(f"  Reset epoch: {old_epoch} → 0")

            if 'global_step' in checkpoint:
                old_step = checkpoint['global_step']
                checkpoint['global_step'] = 0
                print(f"  Reset global_step: {old_step} → 0")

            # Remove optimizer states (they won't match our new optimizer config)
            if 'optimizer_states' in checkpoint:
                del checkpoint['optimizer_states']
                print(f"  Removed optimizer_states (will initialize fresh)")

            # Remove lr_scheduler states
            if 'lr_schedulers' in checkpoint:
                del checkpoint['lr_schedulers']
                print(f"  Removed lr_schedulers (will initialize fresh)")

        # Store resume status for later use in summary
        self._is_resume_training = is_resume

        # Let PyTorch Lightning continue with the (possibly filtered) checkpoint
        # No need to call super() - PyTorch Lightning will handle the rest

    def load_state_dict(self, state_dict, strict=True):
        """
        Load state dict and print weight loading summary.

        This method is called by vis4d's checkpoint loading system.
        It handles:
        1. Filtering out incompatible keys (e.g., depth_head when using geometry_backend)
        2. Loading checkpoint weights (Swin, BERT, GroundingDINO heads)
        3. Loading geometry backend pretrained weights (DINOv2, decoder) ONLY if not in checkpoint
        """
        print(f"\n[DEBUG] load_state_dict called")
        print(f"[DEBUG] self.depth_head: {type(self.depth_head) if self.depth_head is not None else None}")
        print(f"[DEBUG] self.geometry_backend: {type(self.geometry_backend)}")

        # Mark that we're loading from checkpoint
        # This is used by the weight loading summary to show correct status
        self._checkpoint_loaded = True

        # Check if checkpoint contains geometry_backend weights (resume training)
        has_geometry_backend_weights = any('geometry_backend' in k for k in state_dict.keys())
        print(f"[DEBUG] Checkpoint has geometry_backend weights: {has_geometry_backend_weights}")

        # Filter out depth_head keys if we're using a different geometry backend
        # This prevents checkpoint's depth_head/fpn from overwriting our geometry_backend
        if self.geometry_backend is not None and not isinstance(self.geometry_backend, UniDepthHeadBackend):
            # We're using a custom geometry backend (DetAny3D, UniDepthV2, etc.)
            # Remove depth_head and fpn keys from checkpoint to avoid conflicts
            depth_head_keys = [k for k in state_dict.keys() if k.startswith('depth_head.')]
            fpn_keys = [k for k in state_dict.keys() if k.startswith('fpn.')]

            if depth_head_keys or fpn_keys:
                print(f"\n[DEBUG] Filtering out incompatible keys from checkpoint:")
                if depth_head_keys:
                    print(f"[DEBUG]   - {len(depth_head_keys)} depth_head.* keys")
                    for k in depth_head_keys:
                        del state_dict[k]
                if fpn_keys:
                    print(f"[DEBUG]   - {len(fpn_keys)} fpn.* keys")
                    for k in fpn_keys:
                        del state_dict[k]
                print(f"[DEBUG] (Using custom geometry backend: {type(self.geometry_backend).__name__})")

        # Call parent's load_state_dict to load checkpoint weights
        result = super().load_state_dict(state_dict, strict=False)

        # Load geometry backend pretrained weights ONLY if checkpoint doesn't have them
        # This is important: if we're resuming training, we should use the trained weights
        # from the checkpoint, not overwrite them with pretrained weights!
        print(f"\n[DEBUG] After loading checkpoint:")
        print(f"[DEBUG] geometry_backend type: {type(self.geometry_backend)}")
        print(f"[DEBUG] hasattr load_pretrained_weights: {hasattr(self.geometry_backend, 'load_pretrained_weights')}")

        if has_geometry_backend_weights:
            print("\n" + "="*80)
            print("✅ Using geometry_backend weights from checkpoint (resume training)")
            print("   NOT loading pretrained weights - using trained weights instead")
            print("="*80 + "\n")
        elif hasattr(self.geometry_backend, 'load_pretrained_weights'):
            print("\n" + "="*80)
            print("🔧 Loading Geometry Backend Pretrained Weights (first training)")
            print("="*80)
            self.geometry_backend.load_pretrained_weights()
            print("="*80 + "\n")
        else:
            print(f"[DEBUG] geometry_backend does not have load_pretrained_weights method")
            print(f"[DEBUG] Available methods: {[m for m in dir(self.geometry_backend) if not m.startswith('_')][:20]}")

        # DEBUG: Check depth_memory_fusion weights after loading
        if self.depth_memory_fusion is not None:
            print("\n" + "="*80)
            print("🔍 DEBUG: Checking depth_memory_fusion weights after loading")
            print("="*80)
            for name, param in self.depth_memory_fusion.named_parameters():
                if 'weight' in name and len(param.shape) == 4:  # Conv weights
                    is_zero = (param.abs().max().item() < 1e-6)
                    print(f"  {name}: shape={tuple(param.shape)}, mean={param.mean().item():.6f}, std={param.std().item():.6f}, is_zero={is_zero}")
            print("="*80 + "\n")

        # Print weight loading summary after loading is complete
        is_resume = has_geometry_backend_weights
        self._print_weight_loading_summary_after_load(is_resume)

        return result

    def _freeze_cut3r(self):
        """Freeze CUT3R tower."""
        if not hasattr(self, 'cut3r_tower') or self.cut3r_tower is None:
            return

        print("[INFO] Freezing CUT3R tower")
        self.cut3r_tower.eval()
        for param in self.cut3r_tower.parameters():
            param.requires_grad = False

    def _freeze_backbone(self):
        """Freeze only the backbone (for fusion training)."""
        print("[INFO] Freezing backbone")
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _freeze_detector(self):
        """Freeze the detector (everything except CUT3R fusion)."""
        print("[INFO] Freezing detector (backbone, neck, encoder, decoder, 2D head, 3D head, depth head)")

        # Freeze all detector components
        modules_to_freeze = [
            self.backbone,
            self.neck,
            self.encoder,
            self.positional_encoding,
            self.memory_trans_fc,
            self.memory_trans_norm,
            self.decoder,
            self.bbox_head,        # 2D detection head
            self.bbox3d_head,      # 3D detection head
            self.roi2det3d,        # 3D RoI to detection converter
            self.dn_query_generator,
            self.language_model,
            self.text_feat_map,
        ]

        # Add depth-related modules if they exist
        if self.geometry_backend is not None:
            modules_to_freeze.append(self.geometry_backend)
        elif self.depth_head is not None:
            modules_to_freeze.append(self.depth_head)
        if self.fpn is not None:
            modules_to_freeze.append(self.fpn)

        for model in modules_to_freeze:
            # Skip if not a nn.Module (e.g., RoI2Det3D is not a module)
            if hasattr(model, 'eval') and hasattr(model, 'parameters'):
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False

        # Freeze embeddings
        self.level_embed.requires_grad = False
        # For nn.Embedding, need to freeze the weight parameter
        self.query_embedding.weight.requires_grad = False

    def _print_trainable_params(self):
        """Print trainable parameter statistics."""
        total_params = 0
        trainable_params = 0

        # Count by component
        component_stats = {}

        for name, param in self.named_parameters():
            total_params += param.numel()

            # Determine component
            if 'backbone' in name:
                component = 'backbone'
            elif 'neck' in name:
                component = 'neck'
            elif 'encoder' in name:
                component = 'encoder'
            elif 'decoder' in name:
                component = 'decoder'
            elif 'bbox_head' in name or 'bbox3d_head' in name:
                component = 'bbox_head'
            elif 'cut3r_tower' in name:
                component = 'cut3r_tower'
            elif 'cut3r_fusion' in name:
                component = 'cut3r_fusion'
            else:
                component = 'other'

            if component not in component_stats:
                component_stats[component] = {'total': 0, 'trainable': 0}

            component_stats[component]['total'] += param.numel()

            if param.requires_grad:
                trainable_params += param.numel()
                component_stats[component]['trainable'] += param.numel()

        print("\n" + "="*80)
        print("📊 Trainable Parameter Statistics")
        print("="*80)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        print(f"Frozen parameters: {total_params - trainable_params:,} ({100*(total_params - trainable_params)/total_params:.2f}%)")

        print("\nBy Component:")
        print(f"{'Component':<20} {'Total':<15} {'Trainable':<15} {'%':<10} {'Status'}")
        print("-"*80)

        for component in sorted(component_stats.keys()):
            stats = component_stats[component]
            total = stats['total']
            trainable = stats['trainable']
            pct = 100 * trainable / total if total > 0 else 0

            if trainable == 0:
                status = "❄️  FROZEN"
            elif trainable == total:
                status = "✅ TRAINABLE"
            else:
                status = "⚠️  PARTIAL"

            print(f"{component:<20} {total:>12,}   {trainable:>12,}   {pct:>6.2f}%   {status}")

        print("="*80 + "\n")

    def _get_component_name(self, param_name: str) -> str:
        """Get component name from parameter name for detailed statistics."""
        if 'backbone' in param_name:
            return 'backbone'
        elif 'neck' in param_name:
            return 'neck'
        elif 'language_model' in param_name or 'bert' in param_name:
            return 'language_model'
        elif 'geometry_backend' in param_name:
            # Further breakdown geometry backend
            if 'dino_encoder' in param_name or 'pixel_encoder' in param_name:
                return 'geom_encoder'
            elif 'depth_decoder' in param_name or 'depth_head' in param_name or 'unidepth_model.decoder' in param_name:
                return 'geom_decoder'
            elif 'feature_projector' in param_name or 'latent_proj' in param_name:
                return 'geom_projector'
            else:
                return 'geometry_backend_other'
        elif 'encoder' in param_name:
            return 'encoder'
        elif 'decoder' in param_name:
            return 'decoder'
        elif 'bbox_head' in param_name or 'bbox3d_head' in param_name:
            return 'bbox_head'
        elif 'cut3r_tower' in param_name:
            return 'cut3r_tower'
        elif 'cut3r_fusion' in param_name:
            return 'cut3r_fusion'
        else:
            return 'other'

    def print_trainable_params_with_lr(self, optimizer):
        """Print trainable parameter statistics with learning rate info.

        This function shows which parameters are actually trainable (lr > 0) vs
        frozen by lr=0.0 vs frozen by requires_grad=False.

        Args:
            optimizer: The optimizer instance with param_groups.
        """
        # Build param -> lr mapping from optimizer
        param_to_lr = {}
        for param_group in optimizer.param_groups:
            lr = param_group['lr']  # base_lr * lr_mult
            for param in param_group['params']:
                param_to_lr[id(param)] = lr

        total_params = 0
        actually_trainable = 0  # lr > 0
        frozen_by_lr = 0        # lr == 0
        frozen_by_grad = 0      # requires_grad == False

        component_stats = {}

        for name, param in self.named_parameters():
            total_params += param.numel()

            # Determine component
            component = self._get_component_name(name)
            if component not in component_stats:
                component_stats[component] = {
                    'total': 0,
                    'trainable': 0,      # lr > 0
                    'frozen_lr': 0,      # lr == 0
                    'frozen_grad': 0,    # requires_grad == False
                }

            component_stats[component]['total'] += param.numel()

            # Check freeze status
            if not param.requires_grad:
                # Frozen by requires_grad=False
                frozen_by_grad += param.numel()
                component_stats[component]['frozen_grad'] += param.numel()
            elif id(param) in param_to_lr:
                lr = param_to_lr[id(param)]
                if lr == 0.0:
                    # Frozen by lr=0.0
                    frozen_by_lr += param.numel()
                    component_stats[component]['frozen_lr'] += param.numel()
                else:
                    # Actually trainable
                    actually_trainable += param.numel()
                    component_stats[component]['trainable'] += param.numel()
            else:
                # Not in optimizer (shouldn't happen, but count as trainable)
                print(f"  ⚠️  WARNING: Parameter {name} not found in optimizer!")
                actually_trainable += param.numel()
                component_stats[component]['trainable'] += param.numel()

        # Print statistics
        print("\n" + "="*105)
        print("📊 Trainable Parameter Statistics (with Learning Rate)")
        print("="*105)
        print(f"Total parameters: {total_params:,}")
        print(f"Actually trainable (lr > 0): {actually_trainable:,} ({100*actually_trainable/total_params:.2f}%)")
        print(f"Frozen by lr=0.0: {frozen_by_lr:,} ({100*frozen_by_lr/total_params:.2f}%)")
        print(f"Frozen by requires_grad=False: {frozen_by_grad:,} ({100*frozen_by_grad/total_params:.2f}%)")

        print("\nBy Component:")
        print(f"{'Component':<25} {'Total':<15} {'Trainable':<15} {'Frozen(lr=0)':<17} {'Frozen(grad=F)':<17} {'Status'}")
        print("-"*105)

        for component in sorted(component_stats.keys()):
            stats = component_stats[component]
            total = stats['total']
            trainable = stats['trainable']
            frozen_lr = stats['frozen_lr']
            frozen_grad = stats['frozen_grad']

            if frozen_lr + frozen_grad == total:
                status = "🔒 FROZEN"
            elif trainable == total:
                status = "✅ TRAINABLE"
            else:
                status = "⚠️  PARTIAL"

            print(f"{component:<25} {total:<15,} {trainable:<15,} {frozen_lr:<17,} {frozen_grad:<17,} {status}")

        print("="*105)
        print()

    def _print_weight_loading_summary_after_load(self, is_resume: bool):
        """Print weight loading summary AFTER all weights are loaded.

        Args:
            is_resume: Whether this is resume training (True) or first training (False)
        """
        print("\n" + "=" * 80)
        print("🔍 WEIGHT LOADING SUMMARY (FINAL)")
        print("=" * 80)

        if is_resume:
            print("\n📌 Mode: Resume Training")
            print("All components loaded from unified checkpoint")
            print("No separate pretrained weights needed")
            print("\n" + "=" * 80)
            print("✅ Weight Loading Summary Complete")
            print("=" * 80 + "\n")
            return

        print("\n📌 Mode: First Training (Fine-tuning)")
        print("-" * 80)

        # Check if checkpoint was loaded (via --ckpt flag)
        checkpoint_loaded = getattr(self, '_checkpoint_loaded', False)

        # 1. Backbone (Swin Transformer)
        print(f"\n📦 Backbone (Swin Transformer):")
        if checkpoint_loaded or self._weights_path:
            source = "checkpoint (--ckpt)" if checkpoint_loaded else self._weights_path
            print(f"   ✅ Loaded from: {source}")
            print(f"   └─ Status: Pretrained (3D-MOOD)")
        else:
            print(f"   ⚠️  Random initialization")

        # 2. Language Model
        print(f"\n📝 Language Model (BERT):")
        if checkpoint_loaded or self._weights_path:
            source = "checkpoint (--ckpt)" if checkpoint_loaded else self._weights_path
            print(f"   ✅ Loaded from: {source}")
            print(f"   └─ Status: Pretrained (3D-MOOD)")
        else:
            print(f"   ⚠️  Random initialization")

        # 3. GroundingDINO Head
        print(f"\n🎯 GroundingDINO Head (2D Detection):")
        if checkpoint_loaded or self._weights_path:
            source = "checkpoint (--ckpt)" if checkpoint_loaded else self._weights_path
            print(f"   ✅ Loaded from: {source}")
            print(f"   └─ Status: Pretrained (3D-MOOD)")
        else:
            print(f"   ⚠️  Random initialization")

        # 4. GroundingDINO3D Head
        print(f"\n🎯 GroundingDINO3D Head (3D Detection):")
        if checkpoint_loaded or self._weights_path:
            source = "checkpoint (--ckpt)" if checkpoint_loaded else self._weights_path
            print(f"   ✅ Loaded from: {source}")
            print(f"   └─ Status: Pretrained (3D-MOOD)")
        else:
            print(f"   ⚠️  Random initialization")

        # 5. Geometry Backend
        if self.geometry_backend is not None:
            backend_type = type(self.geometry_backend).__name__
            print(f"\n🌍 Geometry Backend ({backend_type}):")

            if backend_type == "UniDepthHeadBackend":
                if self._weights_path:
                    print(f"   ✅ UniDepthHead decoder loaded from: {self._weights_path}")
                    print(f"   └─ Status: Pretrained (3D-MOOD checkpoint)")
                else:
                    print(f"   ⚠️  Random initialization")

            elif backend_type == "UniDepthHeadDinoBackend":
                backend = self.geometry_backend
                dino_model = getattr(backend, 'dino_model', 'unknown')
                dino_pretrained = getattr(backend, '_dino_pretrained_path', None)

                print(f"   📌 DINOv2 Encoder ({dino_model}):")
                if dino_pretrained:
                    print(f"      ✅ Loaded from: {dino_pretrained}")
                    print(f"      └─ Status: Pretrained (frozen)")
                else:
                    print(f"      ⚠️  Random initialization or hub weights")

                print(f"   📌 Feature Projection:")
                print(f"      ⚠️  Random init (~1M params, adapts DINOv2 to UniDepthHead)")

                print(f"   📌 UniDepthHead Decoder:")
                if self._weights_path:
                    print(f"      ✅ Loaded from: {self._weights_path}")
                    print(f"      └─ Status: Pretrained (finetuned from Swin-based)")
                else:
                    print(f"      ⚠️  Random initialization")

            elif backend_type == "DetAny3DGeometryBackend":
                backend = self.geometry_backend
                dino_model = getattr(backend, 'dino_model', 'unknown')
                dino_pretrained = getattr(backend, '_dino_pretrained_path', None)
                decoder_pretrained = getattr(backend, '_decoder_pretrained_path', None)
                has_projector = getattr(backend, '_has_projector', False)

                print(f"   📌 DINOv2 Encoder ({dino_model}):")
                if dino_pretrained:
                    print(f"      ✅ Loaded from: {dino_pretrained}")
                    print(f"      └─ Status: Pretrained (frozen)")
                else:
                    print(f"      ⚠️  Random initialization")

                if has_projector:
                    # Get projector dimensions
                    projector = backend.feature_projector
                    if hasattr(projector, 'in_features') and hasattr(projector, 'out_features'):
                        in_dim = projector.in_features
                        out_dim = projector.out_features
                        num_params = in_dim * out_dim + out_dim  # weights + bias
                        print(f"   📌 Feature Projector:")
                        print(f"      ⚠️  Random init ({in_dim} → {out_dim}, ~{num_params/1000:.0f}K params)")
                        print(f"      └─ Purpose: Adapt encoder to decoder dimension")

                print(f"   📌 DetAny3D Decoder (UnidepthDecoderDinoOnly):")
                if decoder_pretrained:
                    print(f"      ✅ Loaded from: {decoder_pretrained}")
                    print(f"      └─ Status: Pretrained (trainable)")
                else:
                    print(f"      ⚠️  Random initialization")

            elif backend_type == "UniDepthV2GeometryBackend":
                backend = self.geometry_backend
                encoder_pretrained = getattr(backend, '_encoder_pretrained_path', None)
                decoder_pretrained = getattr(backend, '_decoder_pretrained_path', None)
                pretrained_path = getattr(backend, '_pretrained_path', None)

                print(f"   📌 UniDepthV2 Model:")
                if encoder_pretrained and decoder_pretrained:
                    print(f"      ✅ Encoder (DINOv2) loaded from: {encoder_pretrained}")
                    print(f"      ✅ Decoder loaded from: {decoder_pretrained}")
                    print(f"      └─ Status: Pretrained (trainable)")
                elif pretrained_path:
                    print(f"      ✅ Full model loaded from: {pretrained_path}")
                    print(f"      └─ Status: Pretrained (trainable)")
                else:
                    print(f"      ✅ Loaded from HuggingFace")
                    print(f"      └─ Status: Pretrained (trainable)")

        print("\n" + "=" * 80)
        print("✅ Weight Loading Summary Complete")
        print("=" * 80 + "\n")

    def forward_transformer(
        self,
        feat_flatten: Tensor,
        lvl_pos_embed_flatten: Tensor,
        memory_mask: Tensor | None,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        text_dict: dict[str, Tensor],
        boxes: Tensor | None = None,
        class_ids: Tensor | None = None,
        input_hw: list[tuple[int, int]] | None = None,
        ray_embeddings: Tensor | None = None,
        depth_latents: Tensor | None = None,
        depth_latents_hw: tuple[int, int] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
        """Forward function for the transformer."""
        text_token_mask = text_dict["text_token_mask"]

        memory, memory_text = self.encoder(
            query=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            key_padding_mask=memory_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            memory_text=text_dict["embedded"],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict["position_ids"],
            text_self_attention_masks=text_dict["masks"],
        )

        # ========== Depth-Memory Fusion ==========
        # Fuse depth latents into encoder memory before decoder
        if (
            self.depth_memory_fusion is not None
            and depth_latents is not None
            and depth_latents_hw is not None
        ):
            memory = self.depth_memory_fusion(
                depth_latents=depth_latents,
                depth_latents_hw=depth_latents_hw,
                memory=memory,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
            )

        bs = memory.shape[0]

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes
        )

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers
        ](output_memory, memory_text, text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers
        ].max_text_len
        enc_outputs_coord_unact = (
            self.bbox_head.reg_branches[self.decoder.num_layers](output_memory)
            + output_proposals
        )

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1
        )[1]

        topk_score = torch.gather(
            enc_outputs_class,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features),
        )
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4),
        )
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        # Top-k 3D proposals
        topk_output_memory = torch.gather(
            output_memory, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 256)
        )

        topk_output_3d = self.bbox3d_head.single_forward(
            self.decoder.num_layers,
            topk_output_memory,
            ray_embeddings,
            depth_latents,
        )

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)

        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = (
                self.dn_query_generator(boxes, class_ids, input_hw)
            )
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat(
                [dn_bbox_query, topk_coords_unact], dim=1
            )
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None

        reference_points = reference_points.sigmoid()

        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        if self.training:
            enc_outputs_class = topk_score
            enc_outputs_coord = topk_coords

        hidden_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )

        if len(query) == self.num_queries:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            hidden_states[0] += (
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0
            )

        if self.training:
            return (
                memory_text,
                text_token_mask,
                hidden_states,
                list(references),
                enc_outputs_class,
                enc_outputs_coord,
                topk_output_3d,
                dn_meta,
            )

        return (
            memory_text,
            text_token_mask,
            hidden_states,
            list(references),
        )

    def _extract_image_features(
        self, images: Tensor
    ) -> tuple[list[Tensor], list[Tensor] | None]:
        """Extract image features."""
        # Step 1: Backbone
        visual_feats = self.backbone(images)[2:]
        # visual_feats: List of 4 tensors
        # - Level 0: [B, 96, 200, 333]
        # - Level 1: [B, 192, 100, 166]
        # - Level 2: [B, 384, 50, 83]
        # - Level 3: [B, 768, 25, 41]

        # ========== Step 2: CUT3R Fusion ==========
        if hasattr(self, 'cut3r_tower') and self.cut3r_tower is not None and hasattr(self, 'cut3r_fusion') and self.cut3r_fusion is not None:
            # Extract CUT3R features
            camera_tokens, patch_tokens = self.cut3r_tower(images)
            # camera_tokens: [B, 1, 768]
            # patch_tokens: [B, 729, 768] (27x27 patches for 432x432 input)

            # Concatenate camera and patch tokens
            cut3r_features = torch.cat([camera_tokens, patch_tokens], dim=1)
            # cut3r_features: [B, 730, 768] (1 camera + 729 patches)

            # Fuse with visual features
            visual_feats, _ = self.cut3r_fusion(visual_feats, cut3r_features)

        # ========== Step 3: FPN (for Depth Head) ==========
        if self.fpn is not None:
            depth_feats = self.fpn(visual_feats)

        # Step 4: Adjust visual_feats for Neck if needed
        if len(visual_feats) > len(self.neck.in_channels):
            start_index = len(visual_feats) - len(self.neck.in_channels)
            # NOTE: This is to make sure the number of input channels is the
            # same as the number of input channels of the neck.
            visual_feats = visual_feats[start_index:]

        # Step 5: Neck (Channel Mapper)
        visual_feats = self.neck(visual_feats)

        # Step 6: Fallback for depth_feats
        if self.fpn is None:
            depth_feats = visual_feats

        return visual_feats, depth_feats

    def _forward_train(
        self,
        images: Tensor,
        input_texts: list[list[str]] | None,
        boxes2d: list[Tensor],
        boxes2d_classes: list[Tensor],
        input_hw: list[tuple[int, int]],
        intrinsics: Tensor,
        input_tokens_positive: list[list[int, int]] | list[None] | None = None,
        padding: list[list[int]] | None = None,
        depth_gt: Tensor | None = None,
        depth_mask: Tensor | None = None,
    ) -> GroundingDINO3DOut:
        """Forward function for training."""
        batch_size = images.shape[0]

        new_text_prompts = []
        positive_maps = []
        if input_tokens_positive is not None:
            for tokens_positive_dict, text_prompt, gt_label in zip(
                input_tokens_positive, input_texts, boxes2d_classes
            ):
                if tokens_positive_dict is not None:
                    tokenized = self.language_model.tokenizer(
                        [text_prompt], padding="longest", return_tensors="pt"
                    )

                    new_text_prompts.append(text_prompt)

                    new_tokens_positive = [
                        tokens_positive_dict[label.item()]
                        for label in gt_label
                    ]
                else:
                    tokenized, caption_string, tokens_positive, _ = (
                        self.get_tokens_and_prompts(text_prompt)
                    )

                    new_text_prompts.append(caption_string)

                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]

                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive
                )
                positive_maps.append(positive_map)
        else:
            # All the text prompts are the same or using the self.texts,
            # so there is no need to calculate them multiple times.
            if (
                input_texts is None
                or len(set(["".join(t) for t in input_texts])) == 1
            ):
                if input_texts is None:
                    assert self.texts is not None, "Texts should be provided."
                    text_prompt = self.texts
                else:
                    text_prompt = input_texts[0]

                tokenized, caption_string, tokens_positive, _ = (
                    self.get_tokens_and_prompts(text_prompt)
                )
                new_text_prompts = [caption_string] * batch_size
                for gt_label in boxes2d_classes:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive
                    )
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(input_texts, boxes2d_classes):
                    tokenized, caption_string, tokens_positive, _ = (
                        self.get_tokens_and_prompts(text_prompt)
                    )

                    new_text_prompts.append(caption_string)

                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive
                    )

                    positive_maps.append(positive_map)

        for i in range(batch_size):
            positive_maps[i] = (
                positive_maps[i].to(images.device).bool().float()
            )

        text_dict = self.language_model(new_text_prompts)

        if self.text_feat_map is not None:
            text_dict["embedded"] = self.text_feat_map(text_dict["embedded"])

        text_token_masks = []
        for i in range(batch_size):
            text_token_masks.append(
                text_dict["text_token_mask"][i]
                .unsqueeze(0)
                .repeat(len(positive_map), 1)
            )

        visual_feats, depth_feats = self._extract_image_features(images)

        batch_input_img_h, batch_input_img_w = images.shape[-2:]
        batch_input_shape = (batch_input_img_h, batch_input_img_w)

        (
            feat_flatten,
            lvl_pos_embed_flatten,
            memory_mask,
            spatial_shapes,
            level_start_index,
            valid_ratios,
        ) = self.pre_transformer(
            visual_feats, input_hw, batch_input_shape, padding=padding
        )

        # Geometry Backend / Depth Head
        # Run this first to get ray_intrinsics/ray_image_hw/ray_downsample
        geom_losses = {}
        depth_latents_hw = None  # For depth-memory fusion
        if self.geometry_backend is not None:
            geom_out = self.geometry_backend(
                images=images,
                depth_feats=depth_feats,
                intrinsics=intrinsics,
                image_hw=batch_input_shape,
                depth_gt=depth_gt,
                depth_mask=depth_mask,
            )
            depth_preds = geom_out["depth_map"].squeeze(1)  # [B, H, W]
            depth_latents = geom_out["depth_latents"]
            geom_losses = geom_out.get("losses", {})

            # Use backend's ray parameters for consistent space
            ray_intrinsics = geom_out["ray_intrinsics"]
            ray_image_hw = geom_out["ray_image_hw"]
            ray_downsample = geom_out["ray_downsample"]

            # Get depth_latents_hw for depth-memory fusion (from aux dict)
            aux = geom_out.get("aux", {})
            if aux and "depth_latents_hw" in aux:
                depth_latents_hw = aux["depth_latents_hw"]
        elif self.depth_head is not None:
            # Fallback to direct depth_head call (backward compat)
            depth_preds, depth_latents = self.depth_head(
                depth_feats, intrinsics, batch_input_shape
            )
            # Legacy depth_head uses original space with 1/16 resolution
            ray_intrinsics = intrinsics
            ray_image_hw = batch_input_shape
            ray_downsample = 16
        else:
            raise ValueError("Either geometry_backend or depth_head must be set")

        # Generate ray embeddings only if camera prompt is enabled
        # For ray-aware backends (UniDepthV2, DetAny3D), depth_latents already
        # contain ray info, so we skip the camera prompt branch.
        if self.bbox3d_head.use_camera_prompt:
            ray_embeddings = self.bbox3d_head.get_camera_embeddings(
                ray_intrinsics, ray_image_hw, ray_downsample
            )
        else:
            ray_embeddings = None

        (
            memory_text,
            text_token_mask,
            hidden_states,
            references,
            enc_outputs_class,
            enc_outputs_coord,
            enc_outputs_3d,
            dn_meta,
        ) = self.forward_transformer(
            feat_flatten,
            lvl_pos_embed_flatten,
            memory_mask,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            text_dict,
            boxes2d,
            boxes2d_classes,
            input_hw,
            ray_embeddings,
            depth_latents,
            depth_latents_hw,
        )

        all_layers_cls_scores, all_layers_bbox_preds = self.bbox_head(
            hidden_states, references, memory_text, text_token_mask
        )

        # Not using denoising for 3D
        hidden_states_3d = hidden_states[
            :, :, dn_meta["num_denoising_queries"] :, :
        ]

        all_layers_outputs_3d = self.bbox3d_head(
            hidden_states_3d, ray_embeddings, depth_latents
        )

        return GroundingDINO3DOut(
            all_layers_cls_scores,
            all_layers_bbox_preds,
            all_layers_outputs_3d,
            enc_outputs_class,
            enc_outputs_coord,
            enc_outputs_3d,
            text_token_mask,
            dn_meta,
            positive_maps,
            depth_maps=depth_preds,
            geom_losses=geom_losses,
        )

    def _forward_test(
        self,
        images: Tensor,
        input_texts: list[str] | None,
        text_prompt_mapping: list[dict[str, dict[str, str]]] | None,
        input_hw: list[tuple[int, int]],
        original_hw: list[tuple[int, int]],
        intrinsics: list[Tensor] | None,
        padding: list[list[int]] | None,
    ) -> Det3DOut:
        """Forward."""
        batch_size = images.shape[0]

        token_positive_maps = []
        text_prompts = []
        entities = []
        for i in range(batch_size):
            if self.texts is not None:
                text_prompt = self.texts
            else:
                text_prompt = input_texts[i]

            if text_prompt_mapping is not None:
                prompt_mapping = text_prompt_mapping[i]
            else:
                prompt_mapping = None

            token_positive_map, captions, _, _entities = (
                self.get_tokens_positive_and_prompts(
                    text_prompt,
                    self.custom_entities,
                    text_prompt_mapping=prompt_mapping,
                )
            )

            token_positive_maps.append(token_positive_map)
            text_prompts.append(captions)
            entities.append(_entities)

        # image feature extraction
        batch_input_img_h, batch_input_img_w = images.shape[-2:]

        visual_feats, depth_feats = self._extract_image_features(images)

        batch_input_shape = (batch_input_img_h, batch_input_img_w)

        if isinstance(text_prompts[0], list):
            assert batch_size == 1, "Batch size should be 1 for chunked text."
            assert (
                self.cat_mapping is not None
            ), "Category mapping should be provided."

            boxes = []
            boxes3d = []
            scores = []
            class_ids = []
            categories = []
            for i, captions in enumerate(text_prompts[0]):
                token_positive_map = token_positive_maps[0][i]
                cur_entities = entities[0][i]

                text_dict = self.language_model([captions])

                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict["embedded"] = self.text_feat_map(
                        text_dict["embedded"]
                    )

                (
                    feat_flatten,
                    lvl_pos_embed_flatten,
                    memory_mask,
                    spatial_shapes,
                    level_start_index,
                    valid_ratios,
                ) = self.pre_transformer(
                    copy.deepcopy(visual_feats), input_hw, batch_input_shape
                )

                # Geometry Backend / Depth Head
                depth_latents_hw = None  # For depth-memory fusion
                if self.geometry_backend is not None:
                    geom_out = self.geometry_backend(
                        images=images,
                        depth_feats=depth_feats,
                        intrinsics=intrinsics,
                        image_hw=batch_input_shape,
                    )
                    depth_preds = geom_out["depth_map"].squeeze(1)  # [B, H, W]
                    depth_latents = geom_out["depth_latents"]

                    # Use backend's ray parameters for consistent space
                    ray_intrinsics = geom_out["ray_intrinsics"]
                    ray_image_hw = geom_out["ray_image_hw"]
                    ray_downsample = geom_out["ray_downsample"]

                    # Get depth_latents_hw for depth-memory fusion (from aux dict)
                    aux = geom_out.get("aux", {})
                    if aux and "depth_latents_hw" in aux:
                        depth_latents_hw = aux["depth_latents_hw"]
                elif self.depth_head is not None:
                    depth_preds, depth_latents = self.depth_head(
                        depth_feats, intrinsics, batch_input_shape
                    )
                    # Legacy depth_head uses original space with 1/16 resolution
                    ray_intrinsics = intrinsics
                    ray_image_hw = batch_input_shape
                    ray_downsample = 16
                else:
                    raise ValueError("Either geometry_backend or depth_head must be set")

                # Generate ray embeddings only if camera prompt is enabled
                if self.bbox3d_head.use_camera_prompt:
                    ray_embeddings = self.bbox3d_head.get_camera_embeddings(
                        ray_intrinsics, ray_image_hw, ray_downsample
                    )
                else:
                    ray_embeddings = None

                depth_maps = []
                for i, depth_pred in enumerate(depth_preds):
                    if padding is not None:
                        pad_left, pad_right, pad_top, pad_bottom = padding[i]

                        depth_pred = depth_pred[
                            pad_top : batch_input_img_h - pad_bottom,
                            pad_left : batch_input_img_w - pad_right,
                        ]

                    depth_maps.append(
                        F.interpolate(
                            depth_pred.unsqueeze(0).unsqueeze(0),
                            size=original_hw[i],
                            mode="bilinear",
                            align_corners=False,
                            antialias=True,
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )

                (
                    memory_text,
                    text_token_mask,
                    hidden_states,
                    references,
                ) = self.forward_transformer(
                    feat_flatten,
                    lvl_pos_embed_flatten,
                    memory_mask,
                    spatial_shapes,
                    level_start_index,
                    valid_ratios,
                    text_dict,
                    ray_embeddings=ray_embeddings,
                    depth_latents=depth_latents,
                    depth_latents_hw=depth_latents_hw,
                )

                all_layers_cls_scores, all_layers_bbox_preds = self.bbox_head(
                    hidden_states, references, memory_text, text_token_mask
                )

                all_layers_outputs_3d = self.bbox3d_head(
                    hidden_states, ray_embeddings, depth_latents=depth_latents
                )

                cls_scores = all_layers_cls_scores[-1]
                bbox_preds = all_layers_bbox_preds[-1]
                boxes3d_preds = all_layers_outputs_3d[-1]

                cls_score = cls_scores[0]
                det_bboxes, det_scores, det_labels, det_bboxes3d = (
                    self.roi2det3d(
                        cls_score,
                        bbox_preds[0],
                        token_positive_map,
                        input_hw[0],
                        original_hw[0],
                        boxes3d_preds[0],
                        intrinsics[0],
                        padding[0] if padding is not None else None,
                    )
                )

                boxes.append(det_bboxes)
                scores.append(det_scores)
                boxes3d.append(det_bboxes3d)

                # Get the categories text and class ids
                cur_class_ids = []
                cur_categories = []
                for label in det_labels:
                    cur_class_ids.append(self.cat_mapping[cur_entities[label]])
                    cur_categories.append(cur_entities[label])

                class_ids.append(cur_class_ids)
                categories.append(cur_categories)

            boxes = [torch.cat([b for b in boxes])]
            boxes3d = [torch.cat([b for b in boxes3d])]
            scores = [torch.cat([s for s in scores])]
            class_ids = [sum(class_ids, [])]
            categories = [sum(categories, [])]
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))

            # text feature map layer
            if self.text_feat_map is not None:
                text_dict["embedded"] = self.text_feat_map(
                    text_dict["embedded"]
                )

            (
                feat_flatten,
                lvl_pos_embed_flatten,
                memory_mask,
                spatial_shapes,
                level_start_index,
                valid_ratios,
            ) = self.pre_transformer(visual_feats, input_hw, batch_input_shape)

            # Geometry Backend / Depth Head
            # Run this first to get ray_intrinsics/ray_image_hw/ray_downsample
            depth_latents_hw = None  # For depth-memory fusion
            if self.geometry_backend is not None:
                geom_out = self.geometry_backend(
                    images=images,
                    depth_feats=depth_feats,
                    intrinsics=intrinsics,
                    image_hw=batch_input_shape,
                )
                depth_preds = geom_out["depth_map"].squeeze(1)  # [B, H, W]
                depth_latents = geom_out["depth_latents"]

                # Use backend's ray parameters for consistent space
                ray_intrinsics = geom_out["ray_intrinsics"]
                ray_image_hw = geom_out["ray_image_hw"]
                ray_downsample = geom_out["ray_downsample"]

                # Get depth_latents_hw for depth-memory fusion (from aux dict)
                aux = geom_out.get("aux", {})
                if aux and "depth_latents_hw" in aux:
                    depth_latents_hw = aux["depth_latents_hw"]
            elif self.depth_head is not None:
                depth_preds, depth_latents = self.depth_head(
                    depth_feats, intrinsics, batch_input_shape
                )
                # Legacy depth_head uses original space with 1/16 resolution
                ray_intrinsics = intrinsics
                ray_image_hw = batch_input_shape
                ray_downsample = 16
            else:
                raise ValueError("Either geometry_backend or depth_head must be set")

            # Generate ray embeddings only if camera prompt is enabled
            if self.bbox3d_head.use_camera_prompt:
                ray_embeddings = self.bbox3d_head.get_camera_embeddings(
                    ray_intrinsics, ray_image_hw, ray_downsample
                )
            else:
                ray_embeddings = None

            depth_maps = []
            for i, depth_pred in enumerate(depth_preds):
                if padding is not None:
                    pad_left, pad_right, pad_top, pad_bottom = padding[i]

                    depth_pred = depth_pred[
                        pad_top : batch_input_img_h - pad_bottom,
                        pad_left : batch_input_img_w - pad_right,
                    ]

                depth_maps.append(
                    F.interpolate(
                        depth_pred.unsqueeze(0).unsqueeze(0),
                        size=original_hw[i],
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

            (
                memory_text,
                text_token_mask,
                hidden_states,
                references,
            ) = self.forward_transformer(
                feat_flatten,
                lvl_pos_embed_flatten,
                memory_mask,
                spatial_shapes,
                level_start_index,
                valid_ratios,
                text_dict,
                ray_embeddings=ray_embeddings,
                depth_latents=depth_latents,
                depth_latents_hw=depth_latents_hw,
            )

            all_layers_cls_scores, all_layers_bbox_preds = self.bbox_head(
                hidden_states, references, memory_text, text_token_mask
            )

            all_layers_outputs_3d = self.bbox3d_head(
                hidden_states, ray_embeddings, depth_latents=depth_latents
            )

            cls_scores = all_layers_cls_scores[-1]
            bbox_preds = all_layers_bbox_preds[-1]
            boxes3d_preds = all_layers_outputs_3d[-1]

            boxes = []
            boxes3d = []
            scores = []
            class_ids = []
            categories = []
            for i, bbox_pred in enumerate(bbox_preds):
                cls_score = cls_scores[i]
                det_bboxes, det_scores, det_labels, det_bboxes3d = (
                    self.roi2det3d(
                        cls_score,
                        bbox_pred,
                        token_positive_maps[i],
                        input_hw[i],
                        original_hw[i],
                        boxes3d_preds[i],
                        intrinsics[i],
                        padding[i] if padding is not None else None,
                    )
                )

                boxes.append(det_bboxes)
                scores.append(det_scores)
                class_ids.append(det_labels)
                boxes3d.append(det_bboxes3d)

                # Get the categories text
                cur_categories = []
                for label in det_labels:
                    cur_categories.append(entities[i][label])

                categories.append(cur_categories)

        return Det3DOut(
            boxes,
            boxes3d,
            scores,
            class_ids,
            depth_maps=depth_maps,
            categories=categories,
            predicted_intrinsics=None,
        )

    def forward(
        self,
        images: Tensor,
        input_hw: list[tuple[int, int]],
        intrinsics: Tensor,
        boxes2d: Tensor | None = None,
        boxes2d_classes: Tensor | None = None,
        original_hw: list[tuple[int, int]] | None = None,
        input_texts: Sequence[str] | str | None = None,
        input_tokens_positive: list[dict[int, list[int, int]]] | None = None,
        text_prompt_mapping: dict[str, dict[str, str]] | None = None,
        padding: list[list[int]] | None = None,
        **kwargs,
    ) -> GroundingDINO3DOut | Det3DOut:
        """Forward function."""
        if self.training:
            assert boxes2d is not None and boxes2d_classes is not None
            return self._forward_train(
                images,
                input_texts,
                boxes2d,
                boxes2d_classes,
                input_hw,
                intrinsics,
                input_tokens_positive=input_tokens_positive,
                padding=padding,
                **kwargs,
            )

        assert original_hw is not None
        return self._forward_test(
            images,
            input_texts,
            text_prompt_mapping,
            input_hw,
            original_hw,
            intrinsics,
            padding,
        )
