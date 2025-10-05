"""Backbone freeze/unfreeze configurations as mixins."""

from __future__ import annotations


def freeze_backbone_mixin():
    """Freeze backbone (only train fusion + head)."""
    return {
        'freeze_backbone': True,
    }


def unfreeze_backbone_mixin():
    """Unfreeze backbone (train end-to-end)."""
    return {
        'freeze_backbone': False,
    }


# ============ Registry ============

BACKBONE_CONFIGS = {
    'freeze': freeze_backbone_mixin,
    'unfreeze': unfreeze_backbone_mixin,
}

