"""Stereo4D tinyval 3D dataset (real stereo depth, 500 images)."""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

from vis4d.common.typing import ArgsType, DictStrAny
from vis4d.data.const import CommonKeys as K

from .coco3d import COCO3DDataset

# Stereo4D v3 depth directory (meters, 512x512 .npy files)
_STEREO4D_DEPTH_DIR = (
    "/weka/oe-training-default/weikaih/3d_boundingbox_detection"
    "/video_data/stereo4d_test/stereo4d_dataset_v3/depth"
)

# V3 annotation for image_id -> filename mapping (depth file lookup)
_V3_ANN_PATH = (
    "/weka/oe-training-default/weikaih/3d_boundingbox_detection"
    "/video_data/stereo4d_test/stereo4d_dataset_v3"
    "/annotations/stereo4d_test.json"
)

# Tinyval source directory (to recover original v3 image_ids)
_TINYVAL_DIR = (
    "/weka/oe-training-default/weikaih/3d_boundingbox_detection"
    "/single_frame_data/experiment/v4_score_merged_la3d/stereo4d/tinyval"
)

# Cached v3 id-to-stem mapping (built once, reused)
_v3_id_to_stem_cache: dict[int, str] | None = None
_tinyval_orig_ids_cache: list[int] | None = None


def _load_v3_id_to_stem() -> dict[int, str]:
    """Load v3 image_id -> file stem mapping for depth lookup."""
    global _v3_id_to_stem_cache
    if _v3_id_to_stem_cache is not None:
        return _v3_id_to_stem_cache
    with open(_V3_ANN_PATH) as f:
        v3 = json.load(f)
    _v3_id_to_stem_cache = {}
    for img in v3["images"]:
        stem = os.path.splitext(os.path.basename(img["file_name"]))[0]
        _v3_id_to_stem_cache[img["id"]] = stem
    return _v3_id_to_stem_cache


def _load_tinyval_orig_ids() -> list[int]:
    """Load tinyval original v3 image_ids in sorted order."""
    global _tinyval_orig_ids_cache
    if _tinyval_orig_ids_cache is not None:
        return _tinyval_orig_ids_cache
    files = sorted(
        f for f in os.listdir(_TINYVAL_DIR) if f.endswith(".json")
    )
    _tinyval_orig_ids_cache = []
    for f in files:
        img_id = int(f.split("_")[-1].replace(".json", ""))
        _tinyval_orig_ids_cache.append(img_id)
    return _tinyval_orig_ids_cache


def load_stereo4d_class_map(
    annotation_path: str,
) -> dict[str, int]:
    """Load class map from Stereo4D annotation file.

    Returns a mapping from category name to category ID.
    """
    cache_path = annotation_path.replace(".json", "_class_map.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    with open(annotation_path) as f:
        data = json.load(f)
    class_map = {cat["name"]: cat["id"] for cat in data["categories"]}
    with open(cache_path, "w") as f:
        json.dump(class_map, f)
    return class_map


class Stereo4D3DDataset(COCO3DDataset):
    """Stereo4D tinyval 3D dataset with real stereo depth.

    500 images from Stereo4D test set with human-reviewed 3D bounding
    boxes. Depth maps are real stereo depth (meters, 512x512).

    Key differences from InTheWild3DDataset:
      - Depth is real stereo depth (meters), not estimated depth (mm).
      - All images are 512x512.
      - No confidence masking needed (stereo depth is high quality).
    """

    def __init__(
        self,
        class_map: dict[str, int],
        max_depth: float = 100.0,
        per_image_categories: bool = False,
        **kwargs: ArgsType,
    ) -> None:
        """Creates an instance of the class.

        Args:
            class_map: Mapping from category name to category ID.
            max_depth: Maximum depth in meters (clip beyond this).
            per_image_categories: If True, boxes2d_names only contains
                the GT categories present in each image.
        """
        # Initialize depth mappings BEFORE super().__init__() because
        # _generate_data_mapping -> get_depth_filenames needs these.
        self.per_image_categories = per_image_categories
        self._v3_id_to_stem = _load_v3_id_to_stem()
        self._tinyval_orig_ids = _load_tinyval_orig_ids()

        super().__init__(
            class_map=class_map,
            det_map=class_map,
            max_depth=max_depth,
            **kwargs,
        )

    def __getitem__(self, idx: int):
        """Get single sample, optionally with per-image category filtering."""
        data_dict = super().__getitem__(idx)
        if self.per_image_categories:
            class_ids_in_img = data_dict[K.boxes2d_classes]
            if len(class_ids_in_img) > 0:
                unique_global_ids = sorted(set(class_ids_in_img.tolist()))
                data_dict[K.boxes2d_names] = [
                    self.categories[gid] for gid in unique_global_ids
                ]
            else:
                data_dict[K.boxes2d_names] = []
        return data_dict

    def get_depth_filenames(self, img: DictStrAny) -> str | None:
        """Return path to the .npy stereo depth file for this image.

        Maps: converted image index -> tinyval orig_id -> v3 stem -> depth .npy
        """
        img_id = img["id"]
        if img_id >= len(self._tinyval_orig_ids):
            return None
        orig_id = self._tinyval_orig_ids[img_id]
        stem = self._v3_id_to_stem.get(orig_id)
        if stem is None:
            return None
        depth_path = os.path.join(_STEREO4D_DEPTH_DIR, f"{stem}.npy")
        return depth_path if os.path.exists(depth_path) else None

    def get_depth_map(self, sample: DictStrAny) -> np.ndarray:
        """Load stereo depth .npy (meters, 512x512).

        No mm-to-meters conversion needed (already in meters).
        Resize to original resolution if needed.
        """
        depth = np.load(sample["depth_filename"])  # (H, W) float32, meters

        orig_h = sample["img"]["height"]
        orig_w = sample["img"]["width"]

        if depth.shape != (orig_h, orig_w):
            depth = cv2.resize(
                depth,
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )

        # Clip to max_depth
        depth[depth > self.max_depth] = 0.0

        return depth.astype(np.float32)
