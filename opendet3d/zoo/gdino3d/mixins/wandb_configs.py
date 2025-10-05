"""WandB logging configurations as mixins."""

from __future__ import annotations


def wandb_fusion_strategy_mixin(exp_name: str):
    """WandB config for fusion strategy ablations."""
    return {
        'project': '3D-MOOD-CUT3R',
        'group': 'fusion_strategy',
        'name': exp_name,
        'tags': ['fusion_ablation', exp_name],
    }


def wandb_lr_mixin(exp_name: str):
    """WandB config for learning rate ablations."""
    return {
        'project': '3D-MOOD-CUT3R',
        'group': 'learning_rate',
        'name': exp_name,
        'tags': ['lr_ablation', exp_name],
    }


def wandb_backbone_mixin(exp_name: str):
    """WandB config for backbone ablations."""
    return {
        'project': '3D-MOOD-CUT3R',
        'group': 'backbone_freeze',
        'name': exp_name,
        'tags': ['backbone_ablation', exp_name],
    }

