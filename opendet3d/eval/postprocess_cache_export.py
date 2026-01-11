"""Postprocess cache exporter (test-time).

This evaluator is used with vis4d's EvaluatorCallback to export per-image caches
needed for depth-based 3D box post-processing, without changing the normal
evaluation flow.

Cache layout:
  {cache_root}/{dataset_name}/{image_id}.npz

We intentionally store the full metric depth map (aligned to original_hw) to
avoid coordinate-system bugs from cropping.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from vis4d.common.array import array_to_numpy
from vis4d.common.typing import GenericFunc, MetricLogs, NDArrayNumber
from vis4d.eval.base import Evaluator


class PostprocessCacheExporter(Evaluator):
    """Exports model outputs needed for post-processing into .npz cache files."""

    def __init__(
        self,
        cache_root: str,
        compress: bool = True,
        overwrite: bool = False,
        depth_dtype: str = "float32",
    ) -> None:
        super().__init__()
        self.cache_root = cache_root
        self.compress = compress
        self.overwrite = overwrite
        if depth_dtype not in {"float16", "float32"}:
            raise ValueError(f"Unsupported depth_dtype: {depth_dtype}")
        self.depth_dtype = depth_dtype

        self._num_written = 0
        self._num_skipped = 0

    @property
    def metrics(self) -> list[str]:
        # Not a real evaluator; we only export.
        return []

    def reset(self) -> None:  # pragma: no cover
        self._num_written = 0
        self._num_skipped = 0

    def gather(self, gather_func: GenericFunc) -> None:  # pragma: no cover
        # Nothing to gather; each rank writes its own files (safe because image_id is unique).
        return

    def process_batch(
        self,
        coco_image_id: list[int],
        dataset_names: list[str],
        pred_boxes: list[NDArrayNumber],
        pred_scores: list[NDArrayNumber],
        pred_classes: list[NDArrayNumber],
        pred_boxes3d: list[NDArrayNumber] | None = None,
        pred_categories: list[list[str]] | None = None,
        depth_maps: list[torch.Tensor] | None = None,
        intrinsics: list[NDArrayNumber] | NDArrayNumber | None = None,
        original_hw: list[tuple[int, int]] | None = None,
    ) -> None:
        """Write one .npz per image."""
        if pred_boxes3d is None:
            # No 3D boxes -> nothing to export for depth alignment.
            print("[PostprocessCacheExporter] Skipping: pred_boxes3d is None")
            return
        if depth_maps is None:
            # Depth backend disabled -> nothing to export.
            print("[PostprocessCacheExporter] Skipping: depth_maps is None")
            return
        if intrinsics is None:
            print("[PostprocessCacheExporter] Skipping: intrinsics is None")
            return
        if original_hw is None:
            print("[PostprocessCacheExporter] Skipping: original_hw is None")
            return

        print(f"[PostprocessCacheExporter] Processing batch: {len(coco_image_id)} images")

        # Normalize intrinsics to per-sample list
        if torch.is_tensor(intrinsics):
            # intrinsics: Tensor [B, 3, 3] (may be on GPU)
            intrinsics_np = intrinsics.detach().cpu().numpy()
            intrinsics_list = [intrinsics_np[j] for j in range(intrinsics_np.shape[0])]
        elif isinstance(intrinsics, np.ndarray):
            # intrinsics: ndarray [3,3] or [B,3,3]
            if intrinsics.ndim == 2:
                intrinsics_list = [intrinsics for _ in range(len(coco_image_id))]
            else:
                intrinsics_list = [intrinsics[j] for j in range(intrinsics.shape[0])]
        else:
            # intrinsics: sequence of arrays/tensors
            intrinsics_list = list(intrinsics)

        for i, image_id in enumerate(coco_image_id):
            dataset_name = dataset_names[i]
            out_dir = os.path.join(self.cache_root, str(dataset_name))
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(out_dir, f"{int(image_id)}.npz")
            if (not self.overwrite) and os.path.exists(out_path):
                self._num_skipped += 1
                continue

            boxes2d = array_to_numpy(
                pred_boxes[i].to(torch.float32) if hasattr(pred_boxes[i], "to") else pred_boxes[i],
                n_dims=None,
                dtype=np.float32,
            )
            scores = array_to_numpy(
                pred_scores[i].to(torch.float32) if hasattr(pred_scores[i], "to") else pred_scores[i],
                n_dims=None,
                dtype=np.float32,
            )
            class_ids = array_to_numpy(
                pred_classes[i].to(torch.int64) if hasattr(pred_classes[i], "to") else pred_classes[i],
                n_dims=None,
                dtype=np.int64,
            )
            boxes3d = array_to_numpy(
                pred_boxes3d[i].to(torch.float32) if hasattr(pred_boxes3d[i], "to") else pred_boxes3d[i],
                n_dims=None,
                dtype=np.float32,
            )

            # depth_maps is list[Tensor] where each Tensor is [H, W] or [1, H, W]
            depth = depth_maps[i]
            if depth.ndim == 3 and depth.shape[0] == 1:
                depth = depth[0]
            depth_np = depth.detach().cpu().numpy()
            depth_np = depth_np.astype(np.float16 if self.depth_dtype == "float16" else np.float32)

            Ki = intrinsics_list[i]
            if torch.is_tensor(Ki):
                K = Ki.detach().cpu().numpy().astype(np.float32)
            else:
                K = np.asarray(Ki, dtype=np.float32)
            hw = original_hw[i]

            meta: dict[str, Any] = {
                "dataset_name": str(dataset_name),
                "image_id": int(image_id),
                "original_hw": np.asarray(hw, dtype=np.int32),
            }

            # Categories are variable-length strings; store as object array.
            if pred_categories is not None and i < len(pred_categories) and pred_categories[i] is not None:
                cats = np.asarray(pred_categories[i], dtype=object)
            else:
                cats = np.asarray([], dtype=object)

            save_fn = np.savez_compressed if self.compress else np.savez
            save_fn(
                out_path,
                boxes2d=boxes2d,
                scores=scores,
                class_ids=class_ids,
                boxes3d_raw=boxes3d,
                categories=cats,
                depth_map=depth_np,
                intrinsics=K,
                meta=np.asarray(meta, dtype=object),
            )
            self._num_written += 1

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        # No evaluation; return empty.
        return {}, f"PostprocessCacheExporter: wrote={self._num_written}, skipped={self._num_skipped}"

    def save(self, metric: str, output_dir: str, prefix: str | None = None) -> None:  # pragma: no cover
        # Nothing to save beyond the cache files.
        return


