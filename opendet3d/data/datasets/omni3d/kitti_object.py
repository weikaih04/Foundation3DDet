"""KITTI Object from Omni3D.

KITTI Object Labels:
Categories, -, -, alpha, x1, y1, x2, y2, h, w, l, x, botom_y, z, ry

KITTI Object Categories:
{
    "Pedestrian": "pedestrian",
    "Cyclist": "cyclist",
    "Car": "car",
    "Van": "car",
    "Truck": "truck",
    "Tram": "tram",
    "Person": "pedestrian",
    "Person_sitting": "pedestrian",
    "Misc": "misc",
    "DontCare": "dontcare",
}
"""

from __future__ import annotations

import os

from vis4d.common.typing import ArgsType, DictStrAny

from opendet3d.data.datasets.coco3d import COCO3DDataset

from .omni3d_classes import omni3d_class_map

kitti_train_det_map = kitti_test_det_map = {
    "car": 0,
    "cyclist": 1,
    "pedestrian": 2,
    "person": 3,
    "tram": 4,
    "truck": 5,
    "van": 6,
}

kitti_val_det_map = {
    "car": 0,
    "cyclist": 1,
    "pedestrian": 2,
    "tram": 3,
    "truck": 4,
}

# KITTI-Omni3D Mapping
omni3d_kitti_det_map = {
    "pedestrian": 0,
    "car": 1,
    "cyclist": 2,
    "van": 3,
    "truck": 4,
}


def get_kitti_det_map(split: str) -> dict[str, int]:
    """Get the KITTI detection map."""
    assert split in {"train", "val", "test"}, f"Invalid split: {split}"

    if split == "val":
        return kitti_val_det_map

    # Train and Test are the same
    return kitti_train_det_map


def get_kitti_mapping(
    object_data_root: str = "data/KITTI_object",
) -> tuple[list[str], list[str]]:
    """Get the KITTI object and raw data mapping."""
    with open(os.path.join(object_data_root, "train_rand.txt"), "r") as f:
        kitti_train_rand = f.readlines()[0].split(",")

    with open(os.path.join(object_data_root, "train_mapping.txt"), "r") as f:
        kitti_train_mapping = f.readlines()

    return kitti_train_rand, kitti_train_mapping


def get_kitti_depth_data(
    img_name: str,
    kitti_train_rand: list[str],
    kitti_train_mapping: list[str],
    depth_data_root: str = "data/kitti_depth",
) -> tuple[str, str]:
    """Get the raw KITTI data from Object dataset."""
    raw = kitti_train_mapping[
        int(kitti_train_rand[int(img_name)]) - 1
    ].replace("\n", "")
    date, seq_name, raw_img_name = raw.split()

    depth_file_path = os.path.join(
        depth_data_root,
        "input",
        date,
        seq_name,
        "image_00",
        "data",
        f"{raw_img_name}.png",
    )

    depth_gt_file_path = os.path.join(
        depth_data_root,
        "gt_depth",
        seq_name,
        "proj_depth",
        "groundtruth",
        "image_02",
        f"{raw_img_name}.png",
    )

    return depth_file_path, depth_gt_file_path


class KITTIObject(COCO3DDataset):
    """KITTI Object Dataset."""

    def __init__(
        self,
        class_map: dict[str, int] = omni3d_class_map,
        max_depth: float = 80.0,
        depth_scale: float = 256.0,
        **kwargs: ArgsType,
    ) -> None:
        """Creates an instance of the class."""
        self.kitti_train_rand, self.kitti_train_mapping = get_kitti_mapping()

        super().__init__(
            class_map=class_map,
            max_depth=max_depth,
            depth_scale=depth_scale,
            **kwargs,
        )

    def get_depth_filenames(self, img: DictStrAny) -> str | None:
        """Get the depth filenames.

        Since not every data has depth.
        """
        _, depth_filename = get_kitti_depth_data(
            img["file_path"].split("/")[-1].split(".")[0],
            self.kitti_train_rand,
            self.kitti_train_mapping,
        )
        return depth_filename
