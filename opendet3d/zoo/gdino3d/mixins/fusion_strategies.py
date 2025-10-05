"""Fusion strategy configurations as mixins."""

from __future__ import annotations


# ============ Fusion Strategy Mixins ============

def no_fusion_mixin():
    """No fusion (baseline)."""
    return {
        'cut3r_checkpoint': None,
        'cut3r_freeze': True,
        'fusion_levels': None,
        'fusion_strategies': None,
        'fusion_num_heads': 8,
        'fusion_dropout': 0.1,
        'use_relative_pos_bias': False,
    }


def full_fusion_all_mixin():
    """Full cross-attention fusion at all 4 levels (scale 1-4)."""
    return {
        'cut3r_checkpoint': "CUT3R/src/cut3r_512_dpt_4_64.pth",
        'cut3r_freeze': True,
        'fusion_levels': [0, 1, 2, 3],
        'fusion_strategies': {
            0: {'type': 'full'},
            1: {'type': 'full'},
            2: {'type': 'full'},
            3: {'type': 'full'},
        },
        'fusion_num_heads': 8,
        'fusion_dropout': 0.1,
        'use_relative_pos_bias': False,
    }


def full_fusion_34_mixin():
    """Full cross-attention fusion at levels 3-4 only."""
    return {
        'cut3r_checkpoint': "CUT3R/src/cut3r_512_dpt_4_64.pth",
        'cut3r_freeze': True,
        'fusion_levels': [2, 3],  # Only level 2 and 3
        'fusion_strategies': {
            2: {'type': 'full'},
            3: {'type': 'full'},
        },
        'fusion_num_heads': 8,
        'fusion_dropout': 0.1,
        'use_relative_pos_bias': False,
    }


def deform_12_full_34_mixin():
    """Deformable fusion at levels 1-2, full fusion at levels 3-4."""
    return {
        'cut3r_checkpoint': "CUT3R/src/cut3r_512_dpt_4_64.pth",
        'cut3r_freeze': True,
        'fusion_levels': [0, 1, 2, 3],
        'fusion_strategies': {
            0: {'type': 'deformable', 'n_points': 4},
            1: {'type': 'deformable', 'n_points': 4},
            2: {'type': 'full'},
            3: {'type': 'full'},
        },
        'fusion_num_heads': 8,
        'fusion_dropout': 0.1,
        'use_relative_pos_bias': False,
    }


def deform_all_mixin():
    """Deformable fusion at all 4 levels."""
    return {
        'cut3r_checkpoint': "CUT3R/src/cut3r_512_dpt_4_64.pth",
        'cut3r_freeze': True,
        'fusion_levels': [0, 1, 2, 3],
        'fusion_strategies': {
            0: {'type': 'deformable', 'n_points': 4},
            1: {'type': 'deformable', 'n_points': 4},
            2: {'type': 'deformable', 'n_points': 4},
            3: {'type': 'deformable', 'n_points': 4},
        },
        'fusion_num_heads': 8,
        'fusion_dropout': 0.1,
        'use_relative_pos_bias': False,
    }


def full_fusion_12_mixin():
    """Full cross-attention fusion at levels 1-2 only (middle layers)."""
    return {
        'cut3r_checkpoint': "CUT3R/src/cut3r_512_dpt_4_64.pth",
        'cut3r_freeze': True,
        'fusion_levels': [0, 1],  # Only level 0 and 1 (middle)
        'fusion_strategies': {
            0: {'type': 'full'},
            1: {'type': 'full'},
        },
        'fusion_num_heads': 8,
        'fusion_dropout': 0.1,
        'use_relative_pos_bias': False,
    }


def full_fusion_3_mixin():
    """Full cross-attention fusion at level 3 only (highest layer)."""
    return {
        'cut3r_checkpoint': "CUT3R/src/cut3r_512_dpt_4_64.pth",
        'cut3r_freeze': True,
        'fusion_levels': [3],  # Only level 3 (highest)
        'fusion_strategies': {
            3: {'type': 'full'},
        },
        'fusion_num_heads': 8,
        'fusion_dropout': 0.1,
        'use_relative_pos_bias': False,
    }


def full_fusion_0_mixin():
    """Full cross-attention fusion at level 0 only (lowest layer)."""
    return {
        'cut3r_checkpoint': "CUT3R/src/cut3r_512_dpt_4_64.pth",
        'cut3r_freeze': True,
        'fusion_levels': [0],  # Only level 0 (lowest)
        'fusion_strategies': {
            0: {'type': 'full'},
        },
        'fusion_num_heads': 8,
        'fusion_dropout': 0.1,
        'use_relative_pos_bias': False,
    }


# ============ Registry ============

FUSION_STRATEGIES = {
    'none': no_fusion_mixin,
    'full_all': full_fusion_all_mixin,
    'full_34': full_fusion_34_mixin,
    'full_12': full_fusion_12_mixin,
    'full_3': full_fusion_3_mixin,
    'full_0': full_fusion_0_mixin,
    'deform_12_full_34': deform_12_full_34_mixin,
    'deform_all': deform_all_mixin,
}

