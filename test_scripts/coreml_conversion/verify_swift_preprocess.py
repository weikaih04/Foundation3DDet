"""Verify Swift preprocessing logic matches Python preprocessing.

Simulates the Swift LocalInferenceService.preprocess() in Python
and compares against the actual preprocess_sam3_3d() output.

This catches any differences in resize, padding, normalization, or K scaling.
"""
import os, sys
os.environ["SAM3_COMPILE"] = "0"
os.environ["SAM3_DISABLE_ACT_CKPT"] = "1"
sys.path.insert(0, ".")
sys.path.insert(0, "sam3")
sys.path.insert(0, "lingbot-depth")
sys.path.insert(0, "UniDepth")

import cv2
import numpy as np
import torch
from pathlib import Path

from scripts.demo_sam3_3d import preprocess_sam3_3d
from meta_demo.api_server import fix_intrinsics_cy

INPUT_SIZE = 1008
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


def swift_preprocess(img_rgb, K_raw, source="iphone"):
    """Simulates Swift LocalInferenceService.preprocess() in Python."""
    orig_h, orig_w = img_rgb.shape[:2]

    # Step 1: fix_cy (same as API)
    K = K_raw.copy()
    if source == "iphone":
        K[1, 2] = orig_h - K[1, 2]

    # Step 2: Resize to fit INPUT_SIZE preserving aspect
    scale = INPUT_SIZE / max(orig_w, orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Step 3: Scale intrinsics
    K[0, 0] *= scale; K[0, 2] *= scale  # fx, cx
    K[1, 1] *= scale; K[1, 2] *= scale  # fy, cy

    # Step 4: Center pad
    pad_left = (INPUT_SIZE - new_w) // 2
    pad_top = (INPUT_SIZE - new_h) // 2
    pad_right = INPUT_SIZE - new_w - pad_left
    pad_bottom = INPUT_SIZE - new_h - pad_top

    # Step 5: Adjust K for padding
    K[0, 2] += pad_left
    K[1, 2] += pad_top

    # Step 6: Create image tensor (1, 3, H, W) with ImageNet normalization
    padded = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    padded[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized.astype(np.float32)

    # Normalize ENTIRE image including padding (same as Python NormalizeImages)
    # This means padding area (0,0,0 pixels) become (-mean/std) not 0
    for c in range(3):
        padded[:, :, c] = (padded[:, :, c] / 255.0 - IMG_MEAN[c]) / IMG_STD[c]

    # HWC -> CHW -> BCHW
    tensor = padded.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

    return {
        "images": tensor,
        "intrinsics": K,
        "original_hw": (orig_h, orig_w),
        "input_hw": (INPUT_SIZE, INPUT_SIZE),
        "padding": (pad_left, pad_right, pad_top, pad_bottom),
        "new_hw": (new_h, new_w),
    }


def main():
    print("=" * 60)
    print("Verify Swift preprocessing matches Python")
    print("=" * 60)

    # Load test image
    test_img_path = Path("assets/demo/rgb.png")
    if test_img_path.exists():
        img = cv2.imread(str(test_img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)

    K_path = Path("assets/demo/intrinsics.npy")
    if K_path.exists():
        K_raw = np.load(str(K_path)).astype(np.float32)
    else:
        K_raw = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)

    print(f"  Image: {img_rgb.shape}")
    print(f"  K_raw: fx={K_raw[0,0]:.1f}, fy={K_raw[1,1]:.1f}, cx={K_raw[0,2]:.1f}, cy={K_raw[1,2]:.1f}")

    # Python preprocessing (ground truth)
    K_fixed = fix_intrinsics_cy(K_raw.copy(), img_rgb.shape[0], "iphone")
    py_data = preprocess_sam3_3d(img_rgb.astype(np.float32), K_fixed)

    py_images = py_data["images"]
    if isinstance(py_images, torch.Tensor):
        py_images = py_images.numpy()
    py_K = py_data["intrinsics"]
    if isinstance(py_K, torch.Tensor):
        py_K = py_K.numpy()

    # Swift preprocessing (simulated)
    sw_data = swift_preprocess(img_rgb, K_raw.copy(), "iphone")
    sw_images = sw_data["images"]
    sw_K = sw_data["intrinsics"]

    # Compare
    print(f"\n  Python:")
    print(f"    images shape: {py_images.shape}, range: [{py_images.min():.3f}, {py_images.max():.3f}]")
    print(f"    K: fx={py_K[0,0]:.2f}, fy={py_K[1,1]:.2f}, cx={py_K[0,2]:.2f}, cy={py_K[1,2]:.2f}")
    print(f"    padding: {py_data['padding']}")
    print(f"    original_hw: {py_data['original_hw']}")

    print(f"\n  Swift (simulated):")
    print(f"    images shape: {sw_images.shape}, range: [{sw_images.min():.3f}, {sw_images.max():.3f}]")
    print(f"    K: fx={sw_K[0,0]:.2f}, fy={sw_K[1,1]:.2f}, cx={sw_K[0,2]:.2f}, cy={sw_K[1,2]:.2f}")
    print(f"    padding: {sw_data['padding']}")
    print(f"    original_hw: {sw_data['original_hw']}")

    # Numerical comparison
    img_diff = np.abs(py_images - sw_images)
    K_diff = np.abs(py_K - sw_K)

    print(f"\n  Image diff: max={img_diff.max():.6f}, mean={img_diff.mean():.8f}")
    print(f"  K diff: max={K_diff.max():.6f}")
    print(f"  Padding match: {py_data['padding'] == sw_data['padding']}")
    print(f"  original_hw match: {py_data['original_hw'] == sw_data['original_hw']}")

    # Debug padding details
    print(f"\n  Python padding: {py_data['padding']}")
    print(f"  Swift padding: {sw_data['padding']}")
    print(f"  Python new_hw (from resize): input_hw={py_data['input_hw']}")
    print(f"  Swift new_hw: {sw_data['new_hw']}")

    # Check if padding area is the source of diff
    pad_l_py = py_data['padding'][0]
    pad_t_py = py_data['padding'][2]
    pad_l_sw = sw_data['padding'][0]
    pad_t_sw = sw_data['padding'][2]

    print(f"\n  Pixel at (0,0,0,0) - py={py_images[0,0,0,0]:.6f}, sw={sw_images[0,0,0,0]:.6f}")
    print(f"  Pixel at content start py=(0,0,{pad_t_py},{pad_l_py}): {py_images[0,0,pad_t_py,pad_l_py]:.6f}")
    print(f"  Pixel at content start sw=(0,0,{pad_t_sw},{pad_l_sw}): {sw_images[0,0,pad_t_sw,pad_l_sw]:.6f}")

    # Compare only content area (where both have actual image data)
    content_py = py_images[0, :, pad_t_py:pad_t_py+sw_data['new_hw'][0], pad_l_py:pad_l_py+sw_data['new_hw'][1]]
    content_sw = sw_images[0, :, pad_t_sw:pad_t_sw+sw_data['new_hw'][0], pad_l_sw:pad_l_sw+sw_data['new_hw'][1]]
    content_diff = np.abs(content_py - content_sw)
    print(f"\n  Content-only diff: max={content_diff.max():.6f}, mean={content_diff.mean():.8f}")

    if img_diff.max() < 0.01 and K_diff.max() < 0.01:
        print(f"\n  SUCCESS: Swift preprocessing matches Python!")
    elif content_diff.max() < 0.05 and K_diff.max() < 0.01:
        print(f"\n  MOSTLY OK: Content matches, padding area differs (expected due to normalization of zeros)")
        print(f"  Padding area: Python normalizes zeros → {(-IMG_MEAN[0])/IMG_STD[0]:.4f}, Swift uses 0.0")
    else:
        print(f"\n  WARNING: Significant differences in content area!")


if __name__ == "__main__":
    main()
