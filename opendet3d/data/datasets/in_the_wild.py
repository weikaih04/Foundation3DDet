"""In-The-Wild 3D dataset (COCO/LVIS/Objects365 with human-annotated 3D boxes)."""

from __future__ import annotations

import json

import numpy as np
import cv2

from vis4d.common.typing import ArgsType, DictStrAny
from vis4d.data.const import CommonKeys as K

from .coco3d import COCO3DDataset

# Depth directories per source (from v4_depth, SR 1024-long-edge .npy files)
_DEPTH_DIRS = {
    "coco/val": (
        "/weka/oe-training-default/weikaih/3d_boundingbox_detection"
        "/single_frame_data/experiment/v4_depth/coco/val/depth"
    ),
    "coco/train": (
        "/weka/oe-training-default/weikaih/3d_boundingbox_detection"
        "/single_frame_data/experiment/v4_depth/coco/train/depth"
    ),
    "obj365/val": (
        "/weka/oe-training-default/weikaih/3d_boundingbox_detection"
        "/single_frame_data/experiment/v4_depth/obj365/val/depth"
    ),
}

# Depth values in the .npy files are in mm; convert to meters
_DEPTH_MM_TO_M = 1.0 / 1000.0


def _get_depth_dir_from_file_path(file_path: str) -> str:
    """Infer depth directory from image file_path."""
    if "coco/val2017" in file_path:
        return _DEPTH_DIRS["coco/val"]
    elif "coco/train2017" in file_path:
        return _DEPTH_DIRS["coco/train"]
    else:  # obj365
        return _DEPTH_DIRS["obj365/val"]


def _get_formatted_id_from_file_path(file_path: str) -> str:
    """Extract zero-padded 12-digit image ID from file path."""
    basename = file_path.split("/")[-1]  # e.g. 000000000724.jpg
    return basename.replace(".jpg", "").replace("obj365_val_", "")


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
    with open(annotation_path) as f:
        data = json.load(f)
    return {cat["name"]: cat["id"] for cat in data["categories"]}


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
        **kwargs: ArgsType,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__(
            class_map=class_map,
            det_map=class_map,
            max_depth=max_depth,
            **kwargs,
        )

    def __getitem__(self, idx: int):
        """Get single sample with per-image category prompts.

        Overrides COCO3DDataset to replace boxes2d_names with the unique
        GT categories present in this image instead of all 800+ categories.
        This avoids BERT token-length truncation at evaluation time.
        """
        data_dict = super().__getitem__(idx)
        # Use only the unique categories that appear in this image's GT.
        # self.categories is sorted by 0-based global ID, so indexing by
        # global ID gives the correct category name.
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
        depth_dir = _get_depth_dir_from_file_path(file_path)
        formatted_id = _get_formatted_id_from_file_path(file_path)
        depth_path = f"{depth_dir}/{formatted_id}_sr_1024_long.npy"
        import os
        return depth_path if os.path.exists(depth_path) else None

    def get_depth_map(self, sample: DictStrAny) -> np.ndarray:
        """Load .npy depth (mm) and resize to original image resolution."""
        depth_npy = np.load(sample["depth_filename"])  # (H_sr, W_sr) float32, mm

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
