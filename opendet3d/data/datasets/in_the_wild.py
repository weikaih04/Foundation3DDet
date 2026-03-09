"""In-The-Wild 3D dataset (COCO/LVIS/Objects365 with human-annotated 3D boxes)."""

from __future__ import annotations

import json
import os

import numpy as np
import cv2

from vis4d.common.typing import ArgsType, DictStrAny
from vis4d.data.const import CommonKeys as K

from .coco3d import COCO3DDataset

_V4_DEPTH_ROOT = (
    "/weka/oe-training-default/weikaih/3d_boundingbox_detection"
    "/single_frame_data/experiment/v4_depth"
)

# Depth directories per source (from v4_depth, SR 1024-long-edge .npy files)
_DEPTH_DIRS = {
    "coco/val": f"{_V4_DEPTH_ROOT}/coco/val/depth",
    "coco/train": f"{_V4_DEPTH_ROOT}/coco/train/depth",
    "obj365/val": f"{_V4_DEPTH_ROOT}/obj365/val/depth",
    "obj365/train": f"{_V4_DEPTH_ROOT}/obj365/train/depth",
}

# Confidence map directories (uint8 PNG, same resolution as depth)
_CONF_DIRS = {
    "coco/val": f"{_V4_DEPTH_ROOT}/coco/val/confidence",
    "coco/train": f"{_V4_DEPTH_ROOT}/coco/train/confidence",
    "obj365/val": f"{_V4_DEPTH_ROOT}/obj365/val/confidence",
    "obj365/train": f"{_V4_DEPTH_ROOT}/obj365/train/confidence",
}

# Depth values in the .npy files are in mm; convert to meters
_DEPTH_MM_TO_M = 1.0 / 1000.0


def _get_source_key_from_file_path(file_path: str) -> str:
    """Infer v4_depth source key from image file_path.

    Handles both absolute paths (legacy) and HDF5 relative paths:
      /weka/.../coco/train2017/X.jpg       -> "coco/train"
      images/coco_train/X.jpg              -> "coco/train"
    """
    if "coco/val2017" in file_path or "/coco_val/" in file_path:
        return "coco/val"
    elif "coco/train2017" in file_path or "/coco_train/" in file_path:
        return "coco/train"
    elif (
        ("obj365" in file_path and "/train/" in file_path)
        or "/obj365_train/" in file_path
    ):
        return "obj365/train"
    else:
        return "obj365/val"


def _get_formatted_id_from_file_path(file_path: str) -> str:
    """Extract zero-padded 12-digit image ID from file path."""
    basename = file_path.split("/")[-1]  # e.g. 000000000724.jpg
    return (
        basename.replace(".jpg", "")
        .replace("obj365_val_", "")
        .replace("obj365_train_", "")
    )


def load_in_the_wild_class_map(
    annotation_path: str = "data/in_the_wild/annotations/InTheWild_val.json",
) -> dict[str, int]:
    """Load class map from InTheWild annotation file.

    Returns a mapping from category name to category ID (0-indexed alphabetical).

    Args:
        annotation_path: Path to the InTheWild annotation JSON file.

    Returns:
        dict mapping category name to annotation category ID.
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


class InTheWild3DDataset(COCO3DDataset):
    """In-The-Wild 3D dataset with 800+ open-vocabulary categories.

    Human-annotated 3D bounding boxes on COCO val2017, LVIS (COCO train2017),
    and Objects365 val images.

    Annotations converted from human_annotated_val_full2d.json to Omni3D
    COCO3D format using scripts/in_the_wild/convert_in_the_wild.py.
    Camera intrinsics are scaled back to original image resolution (non-SR).

    Depth maps are from v4_depth (SR 1024-long-edge .npy, mm units),
    resized to original image resolution on load.
    """

    def __init__(
        self,
        class_map: dict[str, int],
        max_depth: float = 100.0,
        per_image_categories: bool = False,
        depth_confidence_threshold: int = 0,
        **kwargs: ArgsType,
    ) -> None:
        """Creates an instance of the class.

        Args:
            class_map: Mapping from category name to category ID.
            max_depth: Maximum depth in meters (clip beyond this).
            per_image_categories: If True, boxes2d_names only contains
                the GT categories present in each image. Required for
                GDino/3D-MOOD eval (avoids BERT truncation with 1246
                categories). Must be False for SAM3_3D (collator indexes
                boxes2d_names by global cat_id).
            depth_confidence_threshold: Minimum confidence (uint8, 0-255)
                for a depth pixel to be considered valid. Pixels below
                this threshold are set to 0 (invalid). Set to 0 to
                disable confidence masking. Only applies when confidence
                map exists for the image.
        """
        super().__init__(
            class_map=class_map,
            det_map=class_map,
            max_depth=max_depth,
            **kwargs,
        )
        self.per_image_categories = per_image_categories
        self.depth_confidence_threshold = depth_confidence_threshold

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
        """Return path to the .npy depth file for this image."""
        file_path = img["file_path"]
        source_key = _get_source_key_from_file_path(file_path)
        depth_dir = _DEPTH_DIRS[source_key]
        formatted_id = _get_formatted_id_from_file_path(file_path)
        depth_path = f"{depth_dir}/{formatted_id}_sr_1024_long.npy"
        return depth_path if os.path.exists(depth_path) else None

    def get_depth_map(self, sample: DictStrAny) -> np.ndarray:
        """Load .npy depth (mm) and resize to original image resolution.

        If depth_confidence_threshold > 0, loads the MoGe2 confidence
        map (uint8 PNG, same resolution as depth) and zeros out pixels
        where confidence < threshold.
        """
        depth_npy = np.load(sample["depth_filename"])  # (H_sr, W_sr) float32, mm

        # Apply MoGe2 confidence masking before resize
        if self.depth_confidence_threshold > 0:
            file_path = sample["img"]["file_path"]
            source_key = _get_source_key_from_file_path(file_path)
            conf_dir = _CONF_DIRS.get(source_key)
            if conf_dir is not None:
                formatted_id = _get_formatted_id_from_file_path(
                    file_path
                )
                conf_path = f"{conf_dir}/{formatted_id}.png"
                if os.path.exists(conf_path):
                    conf = cv2.imread(
                        conf_path, cv2.IMREAD_UNCHANGED
                    )  # uint8, same shape as depth
                    if conf.shape != depth_npy.shape:
                        conf = cv2.resize(
                            conf,
                            (depth_npy.shape[1], depth_npy.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    depth_npy[
                        conf < self.depth_confidence_threshold
                    ] = 0.0

        orig_h = sample["img"]["height"]
        orig_w = sample["img"]["width"]

        # Resize to original image size using nearest-neighbor to avoid
        # interpolation artifacts at depth discontinuities
        if depth_npy.shape != (orig_h, orig_w):
            depth_npy = cv2.resize(
                depth_npy,
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )

        # Convert mm -> meters
        depth = depth_npy * _DEPTH_MM_TO_M

        # Clip to max_depth
        depth[depth > self.max_depth] = 0.0

        return depth.astype(np.float32)
