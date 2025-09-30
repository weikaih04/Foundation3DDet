"""
CUT3R Fusion Modules for 3D-MOOD.

This package provides flexible multi-scale fusion of CUT3R 3D geometric features
with visual features from the Swin Transformer backbone.
"""

from .gated_fusion import GatedCUT3RFusion
from .deformable_fusion import DeformableGatedFusion
from .multi_scale_fusion import MultiScaleCUT3RFusion

__all__ = [
    'GatedCUT3RFusion',
    'DeformableGatedFusion',
    'MultiScaleCUT3RFusion',
]

