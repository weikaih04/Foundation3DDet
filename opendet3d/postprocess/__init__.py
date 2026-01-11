"""Post-processing utilities (depth/geometry alignment, etc.)."""

from opendet3d.postprocess.depth_align_centroid import (
    CentroidAlignParams,
    align_boxes3d_centroid_v3,
    correct_3d_box_centroid_v3,
)
from opendet3d.postprocess.depth_align_sam3d import (
    Sam3DParams,
    align_boxes3d_sam3d_from_depth,
)
from opendet3d.postprocess.depth_align_v3 import (
    PseudoMaskParams,
    align_boxes3d_v3_from_depth,
)

__all__ = [
    # Centroid V3 (asymmetric trimming)
    "CentroidAlignParams",
    "align_boxes3d_centroid_v3",
    "correct_3d_box_centroid_v3",
    # SAM3D style
    "Sam3DParams",
    "align_boxes3d_sam3d_from_depth",
    # V3 with pseudo mask
    "PseudoMaskParams",
    "align_boxes3d_v3_from_depth",
]
