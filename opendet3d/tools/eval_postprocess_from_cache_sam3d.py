"""Evaluate SAM3D-style depth-aligned 3D boxes from exported cache (offline).

This does NOT rerun the model. It reads npz files exported during inference and
applies SAM3D-style one-shot scaling + translation to each predicted 3D box.

Example:
  python -m opendet3d.tools.eval_postprocess_from_cache_sam3d \
    --cache_root vis4d-workspace/.../postprocess_cache \
    --data_root data/omni3d \
    --output_dir vis4d-workspace/.../postprocess_sam3d_eval \
    --use-mini-dataset
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
from glob import glob
from typing import Sequence

import numpy as np
import torch

from opendet3d.eval.omni3d import Omni3DEvaluator
from opendet3d.postprocess.depth_align_sam3d import (
    Sam3DParams,
    align_boxes3d_sam3d_from_depth,
)
from opendet3d.data.datasets.omni3d.hypersim import Hypersim
from opendet3d.data.datasets.omni3d.sunrgbd import SUNRGBD
from opendet3d.data.datasets.omni3d.arkitscenes import ARKitScenes
from opendet3d.data.datasets.omni3d.util import get_dataset_det_map
from vis4d.data.datasets.util import im_decode
import cv2


def _list_datasets(cache_root: str) -> list[str]:
    ds = []
    for name in os.listdir(cache_root):
        path = os.path.join(cache_root, name)
        if os.path.isdir(path):
            ds.append(name)
    return sorted(ds)


def _iter_cache_files(cache_root: str, dataset_name: str) -> Sequence[str]:
    pattern = os.path.join(cache_root, dataset_name, "*.npz")
    return sorted(glob(pattern))


class GTDepthLoader:
    """Helper class to load GT depth maps efficiently."""

    def __init__(self, data_root: str = "data/omni3d"):
        self.data_root = data_root
        self.ann_cache = {}  # Cache loaded annotations
        self.img_info_cache = {}  # Cache image info by (dataset_name, image_id)

    def _load_annotation(self, dataset_name: str) -> dict | None:
        """Load annotation file for a dataset (with caching)."""
        if dataset_name in self.ann_cache:
            return self.ann_cache[dataset_name]

        ann_file = os.path.join(self.data_root, "annotations", f"{dataset_name}.json")
        if not os.path.exists(ann_file):
            self.ann_cache[dataset_name] = None
            return None

        import json
        print(f"Loading annotation file: {ann_file}")
        with open(ann_file, 'r') as f:
            ann_data = json.load(f)

        # Build image ID lookup
        img_lookup = {img['id']: img for img in ann_data['images']}
        self.ann_cache[dataset_name] = img_lookup
        return img_lookup

    def _get_image_info(self, dataset_name: str, image_id: int) -> dict | None:
        """Get image info for a specific image ID."""
        cache_key = (dataset_name, image_id)
        if cache_key in self.img_info_cache:
            return self.img_info_cache[cache_key]

        img_lookup = self._load_annotation(dataset_name)
        if img_lookup is None:
            self.img_info_cache[cache_key] = None
            return None

        img_info = img_lookup.get(image_id)
        self.img_info_cache[cache_key] = img_info
        return img_info

    def load_gt_depth(self, dataset_name: str, image_id: int) -> np.ndarray | None:
        """Load GT depth map for a given image.

        Args:
            dataset_name: Dataset name (e.g., "KITTI_test", "SUNRGBD_test")
            image_id: Image ID in the dataset

        Returns:
            GT depth map as numpy array, or None if not available
        """
        # Only certain datasets have GT depth
        if "KITTI" in dataset_name or "nuScenes" in dataset_name or "Objectron" in dataset_name:
            # These datasets don't have dense GT depth
            return None

        img_info = self._get_image_info(dataset_name, image_id)
        if img_info is None:
            return None

        # Get depth file path based on dataset type
        try:
            if "Hypersim" in dataset_name:
                # Path format: hypersim/scene/images/cam/frame.jpg
                parts = img_info["file_path"].split("/")
                if len(parts) == 5:
                    _, scene, _, img_dir, img_name = parts
                else:
                    # Fallback for different format
                    scene = parts[1] if len(parts) > 1 else ""
                    img_dir = parts[-2] if len(parts) > 1 else ""
                    img_name = parts[-1] if len(parts) > 0 else ""

                depth_path = os.path.join(
                    self.data_root,
                    "data/hypersim_depth",
                    scene,
                    "images",
                    img_dir,
                    img_name.replace("jpg", "png").replace("tonemap.", "geometry_hdf5."),
                )
                if os.path.exists(depth_path):
                    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                    depth = depth / 1000.0  # Hypersim depth scale
                    return depth

            elif "SUNRGBD" in dataset_name:
                img_info["file_path"] = img_info["file_path"].replace("//", "/")
                data_dir = img_info["file_path"].split("/image")[0]
                depth_dir = os.path.join(self.data_root, data_dir, "depth")

                if os.path.exists(depth_dir):
                    depth_files = os.listdir(depth_dir)
                    if len(depth_files) == 1:
                        depth_path = os.path.join(depth_dir, depth_files[0])
                        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                        # SUNRGBD specific decoding
                        depth = depth >> 3 | depth << (16 - 3)
                        depth = depth / 1000.0  # SUNRGBD depth scale
                        return depth

            elif "ARKitScenes" in dataset_name:
                _, _, split, video_id, image_name = img_info["file_path"].split("/")
                depth_path = os.path.join(
                    self.data_root,
                    "data/ARKitScenes_depth",
                    split,
                    video_id,
                    image_name.replace("jpg", "png"),
                )
                if os.path.exists(depth_path):
                    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                    depth = depth / 1000.0  # ARKitScenes depth scale
                    return depth

        except Exception as e:
            print(f"Warning: Failed to load GT depth for {dataset_name} image {image_id}: {e}")
            return None

        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data/omni3d")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated dataset names. If empty, infer from cache_root subdirs.",
    )
    parser.add_argument("--mini_limit", type=int, default=-1)
    parser.add_argument("--progress_every", type=int, default=50)
    parser.add_argument(
        "--use-mini-dataset",
        action="store_true",
        help="Use mini100 dataset for GT annotations",
    )

    # SAM3D params
    parser.add_argument("--filter_percentile", type=float, default=90.0)
    parser.add_argument("--min_points", type=int, default=10)
    parser.add_argument("--use-v2", action="store_true", help="Use SAM3D v2 (uniform depth scaling)")
    parser.add_argument("--use-v3", action="store_true", help="Use SAM3D v3 (height scaling + translation, default)")
    parser.add_argument("--scale-min", type=float, default=0.5, help="Min scale for v2")
    parser.add_argument("--scale-max", type=float, default=1.5, help="Max scale for v2")
    parser.add_argument("--trim-ratio", type=float, default=0.3, help="Trim ratio (default: 0.3 = trim front/back 30%)")
    parser.add_argument("--min-depth", type=float, default=0.0, help="Only apply SAM3D to boxes with depth >= this value (default: 0 = all boxes)")
    parser.add_argument("--use-gt-depth", action="store_true", help="Use GT depth instead of predicted depth (upper bound experiment)")
    args = parser.parse_args()

    # Default to v3 if neither v2 nor v3 specified
    use_v3 = args.use_v3 or (not args.use_v2)

    cache_root = args.cache_root
    os.makedirs(args.output_dir, exist_ok=True)

    if args.datasets.strip():
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = _list_datasets(cache_root)

    if not datasets:
        raise ValueError(f"No datasets found under cache_root={cache_root}")

    use_mini_dataset = getattr(args, 'use_mini_dataset', False)

    print(f"[eval_postprocess_from_cache_sam3d] cache_root={cache_root}")
    print(f"[eval_postprocess_from_cache_sam3d] datasets={datasets}")
    print(f"[eval_postprocess_from_cache_sam3d] use_mini_dataset={use_mini_dataset}")
    for dn in datasets:
        n = len(_iter_cache_files(cache_root, dn))
        print(f"[eval_postprocess_from_cache_sam3d]  {dn}: {n} files")
    print(f"[eval_postprocess_from_cache_sam3d] mini_limit={args.mini_limit}")
    print(f"[eval_postprocess_from_cache_sam3d] progress_every={args.progress_every}")
    print(
        f"[eval_postprocess_from_cache_sam3d] filter_percentile={args.filter_percentile} min_points={args.min_points}"
    )
    print("[eval_postprocess_from_cache_sam3d] starting...")

    evaluator = Omni3DEvaluator(
        data_root=args.data_root,
        omni3d50=True,
        datasets=tuple(datasets),
        per_class_eval=True,
    )

    # Override annotation paths to use mini100 if requested (same as compute_metrics_mini100.py)
    if use_mini_dataset:
        from opendet3d.data.datasets.coco3d import COCO3D
        from opendet3d.data.datasets.omni3d.util import get_dataset_det_map

        mini_ann_dir = "data/omni3d/annotations_mini100"
        if not os.path.exists(mini_ann_dir):
            raise FileNotFoundError(f"{mini_ann_dir} not found. Generate mini100 annotations first.")

        print(f"[eval_postprocess_from_cache_sam3d] Overriding annotations to mini100...")
        for dataset_name in evaluator.dataset_names:
            mini_ann_file = os.path.join(mini_ann_dir, f"{dataset_name}.json")
            if not os.path.exists(mini_ann_file):
                raise FileNotFoundError(f"{mini_ann_file} not found")

            det_map = get_dataset_det_map(dataset_name=dataset_name, omni3d50=True)
            category_names = sorted(det_map, key=det_map.get)

            with contextlib.redirect_stdout(io.StringIO()):
                evaluator.evaluators[dataset_name]._coco_gt = COCO3D(
                    [mini_ann_file], category_names
                )
            print(f"  ✓ Loaded mini100 GT for {dataset_name}")

    sam3d_params = Sam3DParams(
        filter_percentile=args.filter_percentile,
        min_points=args.min_points,
        use_v2=args.use_v2,
        use_v3=use_v3,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        trim_ratio=args.trim_ratio,
        min_depth_for_alignment=args.min_depth,
    )
    print(f"[eval_postprocess_from_cache_sam3d] use_v2={args.use_v2} use_v3={use_v3} trim_ratio={args.trim_ratio} min_depth={args.min_depth}")
    print(f"[eval_postprocess_from_cache_sam3d] use_gt_depth={args.use_gt_depth}")

    # Initialize GT depth loader if needed
    gt_depth_loader = GTDepthLoader(args.data_root) if args.use_gt_depth else None

    processed = 0
    gt_depth_loaded = 0
    gt_depth_failed = 0

    for dataset_name in datasets:
        files = _iter_cache_files(cache_root, dataset_name)
        for fp in files:
            with np.load(fp, allow_pickle=True) as data:
                boxes2d = data["boxes2d"].astype(np.float32)
                scores = data["scores"].astype(np.float32)
                class_ids = data["class_ids"].astype(np.int64)
                boxes3d_raw = data["boxes3d_raw"].astype(np.float32)
                depth_map = data["depth_map"].astype(np.float32)
                K = data["intrinsics"].astype(np.float32)
                meta = data["meta"].item()
                image_id = int(meta["image_id"])

            # Use GT depth if requested
            if gt_depth_loader is not None:
                gt_depth = gt_depth_loader.load_gt_depth(dataset_name, image_id)
                if gt_depth is not None:
                    # Resize GT depth to match predicted depth map size if needed
                    if gt_depth.shape != depth_map.shape:
                        gt_depth = cv2.resize(gt_depth, (depth_map.shape[1], depth_map.shape[0]), interpolation=cv2.INTER_NEAREST)
                    depth_map = gt_depth
                    gt_depth_loaded += 1
                else:
                    gt_depth_failed += 1

            boxes3d_post = align_boxes3d_sam3d_from_depth(
                boxes3d_raw=boxes3d_raw,
                boxes2d_xyxy=boxes2d,
                depth_map=depth_map,
                intrinsics=K,
                params=sam3d_params,
            )

            evaluator.process_batch(
                coco_image_id=[image_id],
                dataset_names=[dataset_name],
                pred_boxes=[torch.from_numpy(boxes2d)],
                pred_scores=[torch.from_numpy(scores)],
                pred_classes=[torch.from_numpy(class_ids)],
                pred_boxes3d=[torch.from_numpy(boxes3d_post)],
            )

            processed += 1
            if args.progress_every > 0 and (processed % args.progress_every) == 0:
                print(f"[eval_postprocess_from_cache_sam3d] processed={processed}")
            if args.mini_limit > 0 and processed >= args.mini_limit:
                break
        if args.mini_limit > 0 and processed >= args.mini_limit:
            break

    if gt_depth_loader is not None:
        print(f"\n[eval_postprocess_from_cache_sam3d] GT depth stats:")
        print(f"  GT depth loaded: {gt_depth_loaded}")
        print(f"  GT depth failed: {gt_depth_failed}")
        print(f"  Total processed: {processed}")

    log_dict, log_str = evaluator.evaluate("3D")
    print(log_str)
    print("Metrics:")
    for k in sorted(log_dict.keys()):
        print(f"{k}: {log_dict[k]}")

    evaluator.save("3D", args.output_dir)


if __name__ == "__main__":
    main()


