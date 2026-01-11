"""Evaluate depth-aligned (V3) 3D boxes from exported cache, without rerunning the model.

Usage (example):
  python -m opendet3d.tools.eval_postprocess_from_cache \
    --cache_root /path/to/postprocess_cache \
    --data_root data/omni3d \
    --output_dir vis4d-workspace/postprocess_v3_eval \
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
from opendet3d.postprocess.depth_align_v3 import (
    PseudoMaskParams,
    align_boxes3d_v3_from_depth,
)


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
    parser.add_argument("--mini_limit", type=int, default=-1, help="Limit total samples (for quick sanity).")
    parser.add_argument(
        "--progress_every",
        type=int,
        default=50,
        help="Print progress every N samples. Set <=0 to disable.",
    )
    parser.add_argument(
        "--use-mini-dataset",
        action="store_true",
        help="Use mini100 dataset for GT annotations",
    )

    # Pseudo-mask params
    parser.add_argument("--seed_frac", type=float, default=0.4)
    parser.add_argument("--depth_gate_alpha", type=float, default=0.15)
    parser.add_argument("--min_seed_points", type=int, default=30)
    parser.add_argument("--min_mask_points", type=int, default=50)
    args = parser.parse_args()

    cache_root = args.cache_root
    os.makedirs(args.output_dir, exist_ok=True)

    if args.datasets.strip():
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = _list_datasets(cache_root)

    if not datasets:
        raise ValueError(f"No datasets found under cache_root={cache_root}")

    use_mini_dataset = getattr(args, 'use_mini_dataset', False)

    # Helpful run header (especially when stdout is buffered/piped).
    print(f"[eval_postprocess_from_cache] cache_root={cache_root}")
    print(f"[eval_postprocess_from_cache] datasets={datasets}")
    print(f"[eval_postprocess_from_cache] use_mini_dataset={use_mini_dataset}")
    for dn in datasets:
        n = len(_iter_cache_files(cache_root, dn))
        print(f"[eval_postprocess_from_cache]  {dn}: {n} files")
    print(f"[eval_postprocess_from_cache] mini_limit={args.mini_limit}")
    print(f"[eval_postprocess_from_cache] progress_every={args.progress_every}")
    print("[eval_postprocess_from_cache] starting...")

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

        print(f"[eval_postprocess_from_cache] Overriding annotations to mini100...")
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

    pseudo_params = PseudoMaskParams(
        seed_frac=args.seed_frac,
        depth_gate_alpha=args.depth_gate_alpha,
        min_seed_points=args.min_seed_points,
        min_mask_points=args.min_mask_points,
    )

    processed = 0
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

            boxes3d_post = align_boxes3d_v3_from_depth(
                boxes3d_raw=boxes3d_raw,
                boxes2d_xyxy=boxes2d,
                depth_map=depth_map,
                intrinsics=K,
                pseudo_mask_params=pseudo_params,
                v3_kwargs=None,
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
                print(f"[eval_postprocess_from_cache] processed={processed}")
            if args.mini_limit > 0 and processed >= args.mini_limit:
                break
        if args.mini_limit > 0 and processed >= args.mini_limit:
            break

    log_dict, log_str = evaluator.evaluate("3D")
    print(log_str)
    print("Metrics:")
    for k in sorted(log_dict.keys()):
        print(f"{k}: {log_dict[k]}")

    # Save predictions in COCO-format json (per dataset)
    evaluator.save("3D", args.output_dir)


if __name__ == "__main__":
    main()


