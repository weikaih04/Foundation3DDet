"""Export SAM3_3D to ONNX with raw outputs (before _forward_test).

Outputs fixed-shape raw predictions:
  pred_logits: (N_prompts, S, 1)
  pred_boxes_2d: (N_prompts, S, 4) normalized xyxy [0,1]
  pred_boxes_3d: (N_prompts, S, 12) encoded params
  pred_conf_3d: (N_prompts, S, 1)
  depth_map: (1, 1, 1008, 1008)
  K_pred: (1, 3, 3)

No NMS, no box decode, no rescaling inside model.
All postprocessing done in Swift (identical to API Python code).

Run in opendet3d env:
    conda activate opendet3d
    export CUDA_VISIBLE_DEVICES=1
    export PYTHONPATH=...(all paths)...:$PYTHONPATH
    export SAM3_COMPILE=0 SAM3_DISABLE_ACT_CKPT=1
    python test_scripts/coreml_conversion/step9_onnx_raw_output.py
"""

import os, sys, math, time, contextlib, inspect
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

os.environ["SAM3_COMPILE"] = "0"
os.environ["SAM3_DISABLE_ACT_CKPT"] = "1"
sys.setrecursionlimit(50000)

project_root = Path(__file__).parent.parent.parent
OUTPUT_DIR = project_root / "test_scripts" / "coreml_conversion" / "exported_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = "vis4d-workspace/sam3_3d_lingbot_f21_itw_canonical_4node/v1/checkpoints/epoch=2-step=6450.ckpt"


# ============================================================
# RoPE patch (same as all previous steps)
# ============================================================

def apply_rotary_enc_real(xq, xk, freqs_cis, repeat_freqs_k=False):
    fc, fs = freqs_cis[..., 0], freqs_cis[..., 1]
    ndim = xq.ndim
    shape = [1] * (ndim - 2) + list(fc.shape[-2:])
    fc, fs = fc.view(*shape), fs.view(*shape)
    xq_f = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xq_re, xq_im = xq_f[..., 0], xq_f[..., 1]
    xq_out = torch.stack([xq_re * fc - xq_im * fs, xq_re * fs + xq_im * fc], dim=-1).flatten(-2)
    if xk.shape[-2] == 0:
        return xq_out.type_as(xq), xk
    xk_f = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xk_re, xk_im = xk_f[..., 0], xk_f[..., 1]
    if repeat_freqs_k:
        r = xk_re.shape[-2] // xq_re.shape[-2]
        fc = fc.repeat(*([1] * (fc.ndim - 2)), r, 1)
        fs = fs.repeat(*([1] * (fs.ndim - 2)), r, 1)
    xk_out = torch.stack([xk_re * fc - xk_im * fs, xk_re * fs + xk_im * fc], dim=-1).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def get_abs_pos_fixed(abs_pos, has_cls_token, hw, retain_cls_token=False, tiling=False):
    if retain_cls_token: assert has_cls_token
    h, w = hw
    if has_cls_token:
        cls_pos = abs_pos[:, :1]; abs_pos = abs_pos[:, 1:]
    size = round(math.sqrt(abs_pos.shape[1]))
    if size != h or size != w:
        new_abs_pos = abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2)
        if tiling:
            new_abs_pos = new_abs_pos.tile([1, 1] + [x // y + 1 for x, y in zip((h, w), new_abs_pos.shape[2:])])[:, :, :h, :w]
        else:
            new_abs_pos = F.interpolate(new_abs_pos, size=(h, w), mode="bicubic", align_corners=False)
        if not retain_cls_token: return new_abs_pos.permute(0, 2, 3, 1)
        return torch.cat([cls_pos, new_abs_pos.permute(0, 2, 3, 1).reshape(1, h * w, -1)], dim=1)
    else:
        if not retain_cls_token: return abs_pos.reshape(1, h, w, -1)
        return torch.cat([cls_pos, abs_pos], dim=1)


# ============================================================
# Raw output export wrapper
# ============================================================

class SAM3_3DRawExport(nn.Module):
    """Exports SAM3_3D with raw outputs before _forward_test.

    Replicates SAM3_3D.forward steps 1-6 but returns raw tensors
    instead of decoded/NMS-filtered results.

    Fixed shapes: N_prompts=5, S=200 (SAM3 query count).
    """

    def __init__(self, sam3_3d, exportable_depth, text_features, text_mask):
        super().__init__()
        self.sam3_3d = sam3_3d
        self.exportable_depth = exportable_depth
        self.register_buffer('tf', text_features)
        self.register_buffer('tm', text_mask)

    def forward(self, image, depth, intrinsics):
        """
        Args:
            image: (1, 3, 1008, 1008) ImageNet normalized
            depth: (1, 1, 1008, 1008) meters (0=monocular)
            intrinsics: (1, 3, 3) padded-space K (OpenCV convention)

        Returns:
            pred_logits: (N_prompts, S, 1) raw objectness logits
            pred_boxes_2d: (N_prompts, S, 4) normalized xyxy [0,1]
            pred_boxes_3d: (N_prompts, S, 12) encoded 3D params
            pred_conf_3d: (N_prompts, S, 1) 3D confidence logits
            presence_logits: (N_prompts, 1) category presence logits
            depth_map: (1, 1, 1008, 1008) predicted depth
            K_pred: (1, 3, 3) predicted intrinsics
        """
        N_texts = self.tf.shape[1]
        device = image.device
        H, W = 1008, 1008

        # ========== Step 1: Depth backend (parallel with backbone in original) ==========
        depth_latents, depth_map, K_pred, ray_intrinsics, confidence_map = \
            self.exportable_depth(image, depth, intrinsics)

        # ========== Step 2: SAM3 Backbone ==========
        images_sam3 = self.sam3_3d._convert_imagenet_to_sam3_norm(image)
        backbone_out = {"img_batch_all_stages": image}
        backbone_out.update(self.sam3_3d.sam3.backbone.forward_image(images_sam3))

        # Inject baked text features
        backbone_out["language_features"] = self.tf
        backbone_out["language_mask"] = self.tm

        # ========== Step 2.5: Early Depth Fusion ==========
        if self.sam3_3d.early_depth_fusion is not None and depth_latents is not None:
            depth_latents_hw = (49, 49)  # static for 1008x1008
            if "backbone_fpn" in backbone_out:
                backbone_fpn = backbone_out["backbone_fpn"]
                if not isinstance(backbone_fpn, list):
                    backbone_fpn = [backbone_fpn]
                fused_fpn = self.sam3_3d.early_depth_fusion(
                    visual_feats=backbone_fpn,
                    depth_latents=depth_latents,
                    depth_latents_hw=depth_latents_hw,
                )
                if len(fused_fpn) == 1:
                    backbone_out["backbone_fpn"] = fused_fpn[0]
                else:
                    backbone_out["backbone_fpn"] = fused_fpn

        # ========== Step 3: Build SAM3 inputs (text-only) ==========
        from opendet3d.model.detect3d.sam3_3d import SAM3_3DBatchedInputs
        from sam3.model.data_misc import FindStage
        from sam3.model.geometry_encoders import Prompt

        img_ids = torch.zeros(N_texts, dtype=torch.long, device=device)
        text_ids = torch.arange(N_texts, dtype=torch.long, device=device)

        find_input = FindStage(
            img_ids=img_ids, text_ids=text_ids,
            input_boxes=torch.zeros(1, N_texts, 4, device=device),
            input_boxes_mask=torch.ones(N_texts, 1, dtype=torch.bool, device=device),
            input_boxes_label=torch.zeros(1, N_texts, dtype=torch.long, device=device),
            input_points=torch.zeros(1, N_texts, 2, device=device),
            input_points_mask=torch.ones(N_texts, 1, dtype=torch.bool, device=device),
        )
        geometric_prompt = Prompt(
            box_embeddings=torch.zeros(1, N_texts, 4, device=device),
            box_mask=torch.ones(N_texts, 1, dtype=torch.bool, device=device),
            box_labels=torch.zeros(1, N_texts, dtype=torch.long, device=device),
            point_embeddings=torch.zeros(1, N_texts, 2, device=device),
            point_mask=torch.ones(N_texts, 1, dtype=torch.bool, device=device),
            point_labels=torch.zeros(1, N_texts, dtype=torch.long, device=device),
        )

        # ========== Step 4: SAM3 forward_grounding ==========
        sam3_out = self.sam3_3d.sam3.forward_grounding(
            backbone_out=backbone_out,
            find_input=find_input,
            find_target=None,
            geometric_prompt=geometric_prompt,
        )

        # ========== Step 5: Extract raw outputs ==========
        pred_logits = sam3_out["pred_logits"]       # (N_prompts, S, 1)
        pred_boxes_xyxy = sam3_out["pred_boxes_xyxy"]  # (N_prompts, S, 4) normalized
        queries = sam3_out.get("queries")            # (N_prompts, S, d_model)
        presence_logits = sam3_out.get("presence_logit_dec")  # (N_prompts, 1) or None
        if presence_logits is None:
            presence_logits = torch.zeros(N_texts, 1, device=device)

        # ========== Step 6: 3D Head ==========
        pred_boxes_3d = torch.zeros(N_texts, queries.shape[1], 12, device=device)
        pred_conf_3d = torch.zeros(N_texts, queries.shape[1], 1, device=device)

        if self.sam3_3d.bbox3d_head is not None and queries is not None:
            # Ray embeddings from intrinsics
            ray_embeddings = None
            if self.sam3_3d.bbox3d_head.use_camera_prompt:
                ray_embeddings = self.sam3_3d.bbox3d_head.get_camera_embeddings(
                    ray_intrinsics, (686, 686), 14
                )

            # Index to per-prompt (all prompts see image 0)
            if ray_embeddings is not None:
                ray_embeddings = ray_embeddings[img_ids]
            depth_latents_per_prompt = depth_latents[img_ids]

            # Run 3D head (final layer only)
            hidden_states = queries.unsqueeze(0)  # (1, N_prompts, S, C)
            all_boxes_3d, all_conf_3d = self.sam3_3d.bbox3d_head(
                hidden_states=hidden_states,
                ray_embeddings=ray_embeddings,
                depth_latents=depth_latents_per_prompt,
            )
            pred_boxes_3d = all_boxes_3d.squeeze(0)  # (N_prompts, S, 12)
            pred_conf_3d = all_conf_3d.squeeze(0)    # (N_prompts, S, 1)

        return pred_logits, pred_boxes_xyxy, pred_boxes_3d, pred_conf_3d, presence_logits, depth_map, K_pred


def apply_all_patches(sam3_3d):
    """Apply all ONNX-compatibility patches to SAM3_3D."""
    # RoPE: complex -> real
    import sam3.model.vitdet as vit
    vit.apply_rotary_enc = apply_rotary_enc_real
    vit.get_abs_pos = get_abs_pos_fixed
    c = 0
    for _, m in sam3_3d.named_modules():
        if hasattr(m, 'freqs_cis') and m.freqs_cis is not None and torch.is_complex(m.freqs_cis):
            m.freqs_cis = torch.stack([m.freqs_cis.real, m.freqs_cis.imag], dim=-1); c += 1
    print(f"  RoPE: {c} converted")

    # Profiler + pin_memory: no-op
    torch.profiler.record_function = lambda n: contextlib.nullcontext()
    torch.Tensor.pin_memory = lambda self, *a, **kw: self

    # Geometry encoder: no-op (text-only mode)
    d = sam3_3d.sam3.hidden_dim
    def _noop(*args, **kwargs):
        img_feats = args[1] if len(args) > 1 else kwargs.get('img_feats')
        bs = img_feats[-1].shape[1]
        return (torch.zeros(0, bs, d, device=img_feats[-1].device),
                torch.ones(bs, 0, dtype=torch.bool, device=img_feats[-1].device))
    sam3_3d.sam3.geometry_encoder.forward = _noop

    # DINOv2 xformers -> sdpa
    p = 0
    for _, m in sam3_3d.named_modules():
        if type(m).__name__ == 'MemEffAttention':
            try: src = inspect.getsource(m.forward)
            except: src = ""
            if 'memory_efficient_attention' in src:
                def _mk(mod):
                    def _f(x):
                        B,N,C = x.shape
                        qkv = mod.qkv(x).reshape(B,N,3,mod.num_heads,C//mod.num_heads).permute(2,0,3,1,4)
                        q,k,v = qkv.unbind(0)
                        x = F.scaled_dot_product_attention(q,k,v)
                        return mod.proj_drop(mod.proj(x.transpose(1,2).reshape(B,N,C)))
                    return _f
                m.forward = _mk(m); p += 1
    print(f"  xformers: {p} patched")

    # Antialias: disabled
    _real_interp = F.interpolate
    def _no_aa(*args, **kwargs):
        kwargs.pop('antialias', None)
        return _real_interp(*args, **kwargs)
    F.interpolate = _no_aa
    torch.nn.functional.interpolate = _no_aa
    print("  antialias: disabled")


def main():
    print("=" * 60)
    print("Step 9: ONNX Raw Output Export")
    print(f"Checkpoint: {CHECKPOINT}")
    print("=" * 60)
    device = torch.device("cuda:0")

    # 1. Load model
    print("\n[1/5] Loading SAM3_3D...")
    from scripts.demo_sam3_3d import get_sam3_3d_model
    demo = get_sam3_3d_model(
        checkpoint=CHECKPOINT, score_threshold=0.0,
        device=str(device), nms=False,
        use_depth_input_test=True,
        canonical_rotation=True,
    )
    sam3_3d = demo.sam3_3d
    print(f"  {sum(p.numel() for p in sam3_3d.parameters())/1e6:.1f}M params")

    # 2. Apply patches
    print("\n[2/5] Applying patches...")
    apply_all_patches(sam3_3d)

    # Create exportable depth backend
    from opendet3d.op.detect3d.geometry.lingbot_depth_exportable import LingbotDepthExportable
    exportable_depth = LingbotDepthExportable(sam3_3d.geometry_backend).to(device)
    print("  exportable depth: ready")

    sam3_3d.eval()

    # 3. Text embeddings
    print("\n[3/5] Text embeddings...")
    prompts = ["monitor", "keyboard", "table", "chair", "computer"]
    with torch.no_grad():
        text_out = sam3_3d.sam3.backbone.forward_text(prompts, device=device)
    tf, tm = text_out["language_features"], text_out["language_mask"]
    print(f"  shape: {tf.shape}")

    # Permanently replace forward_text
    sam3_3d.sam3.backbone.forward_text = lambda *a, **kw: {
        "language_features": tf, "language_mask": tm,
    }

    # 4. Test forward
    print("\n[4/5] Testing raw output forward...")
    model = SAM3_3DRawExport(sam3_3d, exportable_depth, tf, tm).eval()
    dummy_img = torch.randn(1, 3, 1008, 1008, device=device)
    dummy_depth = torch.zeros(1, 1, 1008, 1008, device=device)
    dummy_K = torch.tensor([[1008.,0.,504.],[0.,1008.,504.],[0.,0.,1.]], device=device).unsqueeze(0)

    with torch.no_grad():
        logits, boxes2d, boxes3d, conf3d, presence, depth_map, K_pred = model(dummy_img, dummy_depth, dummy_K)
    print(f"  pred_logits: {logits.shape}")
    print(f"  pred_boxes_2d: {boxes2d.shape}")
    print(f"  pred_boxes_3d: {boxes3d.shape}")
    print(f"  pred_conf_3d: {conf3d.shape}")
    print(f"  presence_logits: {presence.shape}")
    print(f"  depth_map: {depth_map.shape}")
    print(f"  K_pred: {K_pred.shape}")

    # Show top scores per prompt
    scores = logits.sigmoid()
    for p in range(logits.shape[0]):
        top3 = scores[p, :, 0].topk(3).values.cpu().numpy()
        print(f"  '{prompts[p]}': top scores = {top3}")

    # 5. ONNX export
    print("\n[5/5] ONNX export...")
    onnx_dir = OUTPUT_DIR / "sam3_3d_raw_onnx"
    onnx_dir.mkdir(exist_ok=True)
    onnx_path = onnx_dir / "model.onnx"
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_img, dummy_depth, dummy_K),
            str(onnx_path),
            input_names=["image", "depth", "intrinsics"],
            output_names=["pred_logits", "pred_boxes_2d", "pred_boxes_3d",
                          "pred_conf_3d", "presence_logits", "depth_map", "K_pred"],
            opset_version=17,
            do_constant_folding=True,
        )
    export_time = time.time() - t0

    # Check total size
    total_size = sum(os.path.getsize(os.path.join(str(onnx_dir), f))
                     for f in os.listdir(str(onnx_dir)))
    print(f"  Export: {export_time:.1f}s")
    print(f"  Total: {total_size/(1024**3):.2f}GB")

    # Validate with ONNX Runtime
    print("\n  Validating with ONNX Runtime...")
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    ort_out = sess.run(None, {
        "image": dummy_img.cpu().numpy(),
        "depth": dummy_depth.cpu().numpy(),
        "intrinsics": dummy_K.cpu().numpy(),
    })
    print(f"  ORT pred_logits: {ort_out[0].shape}")
    print(f"  ORT pred_boxes_3d: {ort_out[2].shape}")

    # Compare
    logits_diff = np.abs(logits.cpu().numpy() - ort_out[0]).max()
    boxes3d_diff = np.abs(boxes3d.cpu().numpy() - ort_out[2]).max()
    print(f"  PyTorch vs ORT: logits_diff={logits_diff:.6f}, boxes3d_diff={boxes3d_diff:.6f}")

    print(f"\n{'='*60}")
    print(f"SUCCESS!")
    print(f"  ONNX: {onnx_dir}")
    print(f"  Inputs: image (1,3,1008,1008) + depth (1,1,1008,1008) + intrinsics (1,3,3)")
    print(f"  Outputs: raw logits + boxes2d + boxes3d + conf3d + depth_map + K_pred")
    print(f"  Checkpoint: {CHECKPOINT}")


if __name__ == "__main__":
    main()
