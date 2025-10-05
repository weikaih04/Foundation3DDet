"""Learning rate configurations as mixins."""

from __future__ import annotations


def lr_1e4_mixin():
    """Learning rate 1e-4."""
    return {'lr': 1e-4}


def lr_2e4_mixin():
    """Learning rate 2e-4 (half of default)."""
    return {'lr': 2e-4}


def lr_4e4_mixin():
    """Learning rate 4e-4 (default)."""
    return {'lr': 4e-4}


def lr_5e4_mixin():
    """Learning rate 5e-4."""
    return {'lr': 5e-4}


def lr_8e4_mixin():
    """Learning rate 8e-4 (2x default)."""
    return {'lr': 8e-4}


# ============ Registry ============

LR_CONFIGS = {
    '1e4': lr_1e4_mixin,
    '2e4': lr_2e4_mixin,
    '4e4': lr_4e4_mixin,
    '5e4': lr_5e4_mixin,
    '8e4': lr_8e4_mixin,
}

