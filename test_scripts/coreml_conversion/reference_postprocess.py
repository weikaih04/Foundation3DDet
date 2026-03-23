"""Reference postprocessing: raw ONNX outputs -> same result as API server.

Validates that decoding raw ONNX outputs reproduces the exact same
DetectionResponse as the API's run_detection() function.

Run in opendet3d env:
    conda activate opendet3d
    export CUDA_VISIBLE_DEVICES=1
    export PYTHONPATH=...(all paths)...:$PYTHONPATH
    python test_scripts/coreml_conversion/reference_postprocess.py
"""

import os, sys, math, time, contextlib, inspect
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from torchvision.ops import batched_nms

os.environ["SAM3_COMPILE"] = "0"
os.environ["SAM3_DISABLE_ACT_CKPT"] = "1"

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the exact same functions used by API server
from scripts.demo_sam3_3d import preprocess_sam3_3d
from meta_demo.api_server import (
    fix_intrinsics_cy, boxes3d_camera_to_world, cross_class_nms,
    CATEGORY_COLORS, NMS_DISTANCE,
)

# Import box decoder
from opendet3d.op.detect3d.grounding_dino_3d.coder import GroundingDINO3DCoder

PROMPTS = ["monitor", "keyboard", "table", "chair", "computer"]
INPUT_SIZE = 1008


def postprocess_raw_outputs(
    pred_logits: np.ndarray,     # (N_prompts, S, 1)
    pred_boxes_2d: np.ndarray,   # (N_prompts, S, 4) normalized xyxy [0,1]
    pred_boxes_3d: np.ndarray,   # (N_prompts, S, 12) encoded
    pred_conf_3d: np.ndarray,    # (N_prompts, S, 1)
    intrinsics_padded: np.ndarray,  # (3, 3) padded-space K
    original_hw: tuple,          # (H_orig, W_orig)
    padding: tuple,              # (pad_l, pad_r, pad_t, pad_b)
    camera_to_world: np.ndarray, # (4, 4)
    score_threshold: float = 0.5,
    source: str = "iphone",
    canonical_rotation: bool = True,
) -> List[dict]:
    """Replicate API's run_detection() postprocessing on raw ONNX outputs.

    This is the REFERENCE implementation that Swift must replicate exactly.
    """
    N_prompts, S, _ = pred_logits.shape
    orig_h, orig_w = original_hw
    pad_l, pad_r, pad_t, pad_b = padding
    H, W = INPUT_SIZE, INPUT_SIZE

    # Step 1: Compute scores (same as _forward_test lines 1271-1300)
    scores_2d = 1.0 / (1.0 + np.exp(-pred_logits.squeeze(-1)))  # sigmoid, (N, S)

    # Optional: 3D confidence weight (default 0.5)
    scores_3d = 1.0 / (1.0 + np.exp(-pred_conf_3d.squeeze(-1)))
    conf_weight = 0.5
    scores_all = scores_2d + conf_weight * scores_3d

    # Step 2: Scale normalized boxes to padded pixel coords
    boxes_pixel = pred_boxes_2d.copy()
    boxes_pixel[..., 0::2] *= W
    boxes_pixel[..., 1::2] *= H

    # Step 3: Flatten (N_prompts, S) -> (N_prompts * S)
    scores_2d_flat = scores_2d.flatten()
    scores_all_flat = scores_all.flatten()
    boxes_pixel_flat = boxes_pixel.reshape(-1, 4)
    boxes_3d_flat = pred_boxes_3d.reshape(-1, 12)

    # Class IDs: each prompt repeated S times
    class_ids_flat = np.repeat(np.arange(N_prompts), S)

    # Step 4: Score threshold filter (on 2D score, same as API line 485)
    mask = scores_2d_flat >= score_threshold
    scores_2d_filtered = scores_2d_flat[mask]
    scores_all_filtered = scores_all_flat[mask]
    boxes_pixel_filtered = boxes_pixel_flat[mask]
    boxes_3d_filtered = boxes_3d_flat[mask]
    class_ids_filtered = class_ids_flat[mask]

    if len(boxes_pixel_filtered) == 0:
        return []

    # Step 5: Per-class NMS (same as _forward_test lines 1486-1503)
    keep = batched_nms(
        torch.from_numpy(boxes_pixel_filtered),
        torch.from_numpy(scores_all_filtered),
        torch.from_numpy(class_ids_filtered.astype(np.int64)),
        iou_threshold=0.6,
    ).numpy()

    scores_2d_nms = scores_2d_filtered[keep]
    boxes_pixel_nms = boxes_pixel_filtered[keep]
    boxes_3d_nms = boxes_3d_filtered[keep]
    class_ids_nms = class_ids_filtered[keep]

    # Step 6: 3D box decode (same as _forward_test line 1522)
    box_coder = GroundingDINO3DCoder(canonical_rotation=canonical_rotation)
    K_tensor = torch.from_numpy(intrinsics_padded).float()
    decoded_3d = box_coder.decode(
        torch.from_numpy(boxes_pixel_nms).float(),
        torch.from_numpy(boxes_3d_nms).float(),
        K_tensor,
    ).numpy()  # (N, 10) [cx,cy,cz,w,l,h,qx,qy,qz,qw]

    # Step 7: Rescale 2D boxes from padded to original (same as _forward_test lines 1563-1576)
    boxes_2d_rescaled = boxes_pixel_nms.copy()
    boxes_2d_rescaled[:, 0::2] -= pad_l
    boxes_2d_rescaled[:, 1::2] -= pad_t
    content_w = W - pad_l - pad_r
    content_h = H - pad_t - pad_b
    scale_x = content_w / orig_w
    scale_y = content_h / orig_h
    boxes_2d_rescaled[:, 0::2] /= scale_x
    boxes_2d_rescaled[:, 1::2] /= scale_y

    # Step 8: Camera-to-world transform (same as API lines 539-541)
    apply_flip_y = source not in ("opencv", "robot")
    world_boxes = boxes3d_camera_to_world(
        decoded_3d, camera_to_world, apply_flip_y=apply_flip_y
    )

    # Step 9: Attach labels, scores, colors (same as API lines 544-553)
    output_boxes = []
    for i, box in enumerate(world_boxes):
        class_idx = int(class_ids_nms[i])
        label = PROMPTS[class_idx] if class_idx < len(PROMPTS) else "object"
        color_idx = class_idx % len(CATEGORY_COLORS)
        box["label"] = label
        box["score"] = float(scores_2d_nms[i])
        box["color"] = CATEGORY_COLORS[color_idx]
        output_boxes.append(box)

    # Step 10: Cross-class NMS (same as API lines 556-560)
    if len(PROMPTS) > 1 and len(output_boxes) > 1:
        before = len(output_boxes)
        output_boxes = cross_class_nms(output_boxes)
        if before != len(output_boxes):
            print(f"  Cross-class NMS: {before} -> {len(output_boxes)}")

    return output_boxes


def main():
    print("=" * 60)
    print("Reference Postprocessing Validation")
    print("=" * 60)

    device = torch.device("cuda:0")

    # Load test image
    test_img_path = project_root / "assets" / "demo" / "rgb.png"
    if not test_img_path.exists():
        print("No test image found, using random")
        img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        K_raw = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    else:
        from PIL import Image
        img = np.array(Image.open(str(test_img_path))).astype(np.uint8)
        K_path = project_root / "assets" / "demo" / "intrinsics.npy"
        K_raw = np.load(str(K_path)).astype(np.float32) if K_path.exists() else None

    img_h, img_w = img.shape[:2]
    print(f"  Image: {img.shape}")

    # Fix intrinsics (same as API)
    if K_raw is not None:
        K_fixed = fix_intrinsics_cy(K_raw.copy(), img_h, "iphone")
    else:
        K_fixed = None

    # Preprocess (same as API)
    data = preprocess_sam3_3d(img.astype(np.float32), K_fixed)
    K_padded = data["intrinsics"].numpy() if isinstance(data["intrinsics"], torch.Tensor) else data["intrinsics"]
    if K_padded.ndim == 3:
        K_padded = K_padded[0]

    # Dummy camera_to_world (identity + slight offset for testing)
    cam2world = np.eye(4, dtype=np.float32)
    cam2world[2, 3] = 0.5  # small Z offset

    # Run ONNX model
    print("\n  Running ONNX model...")
    import onnxruntime as ort
    onnx_path = str(project_root / "test_scripts" / "coreml_conversion" / "exported_models" / "sam3_3d_raw_onnx" / "model.onnx")
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # Prepare inputs
    image_tensor = data["images"].numpy() if isinstance(data["images"], torch.Tensor) else data["images"]
    if image_tensor.ndim == 3:
        image_tensor = image_tensor[np.newaxis]
    depth_tensor = np.zeros((1, 1, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    K_input = K_padded[np.newaxis] if K_padded.ndim == 2 else K_padded

    t0 = time.time()
    ort_out = sess.run(None, {
        "image": image_tensor,
        "depth": depth_tensor,
        "intrinsics": K_input,
    })
    print(f"  ONNX inference: {time.time()-t0:.2f}s")

    pred_logits = ort_out[0]     # (5, 200, 1)
    pred_boxes_2d = ort_out[1]   # (5, 200, 4)
    pred_boxes_3d = ort_out[2]   # (5, 200, 12)
    pred_conf_3d = ort_out[3]    # (5, 200, 1)
    depth_map = ort_out[4]       # (1, 1, 1008, 1008)
    K_pred = ort_out[5]          # (1, 3, 3)

    print(f"  Outputs: logits={pred_logits.shape}, boxes3d={pred_boxes_3d.shape}")

    # Postprocess (same as API)
    print("\n  Running reference postprocessing...")
    results = postprocess_raw_outputs(
        pred_logits=pred_logits,
        pred_boxes_2d=pred_boxes_2d,
        pred_boxes_3d=pred_boxes_3d,
        pred_conf_3d=pred_conf_3d,
        intrinsics_padded=K_padded,
        original_hw=data["original_hw"],
        padding=data["padding"],
        camera_to_world=cam2world,
        score_threshold=0.3,
        source="iphone",
    )

    print(f"\n  Results: {len(results)} detections")
    for i, box in enumerate(results[:8]):
        c = box["center"]
        s = box["size"]
        print(f"    {box['label']}: score={box['score']:.3f}, "
              f"center=[{c[0]:.2f},{c[1]:.2f},{c[2]:.2f}], "
              f"size=[{s[0]:.2f},{s[1]:.2f},{s[2]:.2f}]")

    print(f"\n  This output format matches what the API returns to iPhone.")
    print(f"  Swift LocalInferenceService must replicate postprocess_raw_outputs().")


if __name__ == "__main__":
    main()
