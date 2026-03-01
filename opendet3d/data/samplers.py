"""Dataset-ratio weighted sampler for multi-dataset training."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence

import torch
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from vis4d.common.distributed import get_rank, get_world_size
from vis4d.data.data_pipe import DataPipe
from vis4d.data.loader import build_train_dataloader
from vis4d.data.typing import DictData, DictDataOrList


class DatasetRatioSampler(Sampler[int]):
    """Weighted sampler that controls per-dataset sampling ratios.

    For a ConcatDataset with N sub-datasets, this sampler assigns each
    sample a weight based on which sub-dataset it belongs to, then
    performs weighted random sampling. This allows controlling the
    proportion each dataset appears during training without dropping
    any data.

    Two modes of specifying ratios:

    1. dataset_ratios (original): raw per-dataset weights.
       weight_i = ratio_i / size_i, proportion is derived.
       Example: dataset_ratios=[1.0, 1.0] for Omni3D(100K)+CA-1M(200K)
       -> 50/50 sampling proportion.

    2. target_proportions (new): directly specify desired proportions.
       Must sum to 1.0. Weights are computed automatically.
       Example: target_proportions=[0.5, 0.25, 0.25]
       -> Omni3D 50%, CA-1M 25%, Waymo 25%.

    epoch_dataset_idx: If set, one epoch = the specified dataset sees
    every sample once. num_samples is computed as:
        size[idx] / proportion[idx]

    Supports distributed training (splits indices across ranks).

    Args:
        dataset: A ConcatDataset (e.g., DataPipe with multiple datasets).
        dataset_ratios: Per-dataset sampling weight. Mutually exclusive
            with target_proportions.
        target_proportions: Per-dataset target proportion (must sum to 1).
            Mutually exclusive with dataset_ratios.
        epoch_dataset_idx: If set, one epoch = this dataset sees all its
            samples once. Overrides num_samples.
        num_samples: Total samples per epoch. If None and
            epoch_dataset_idx is None, uses sum of all dataset sizes.
        shuffle: Whether to shuffle indices each epoch.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: ConcatDataset,
        dataset_ratios: list[float] | None = None,
        target_proportions: list[float] | None = None,
        epoch_dataset_idx: int | None = None,
        num_samples: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        """Creates an instance of the class."""
        assert isinstance(dataset, ConcatDataset), (
            "dataset must be a ConcatDataset (e.g., DataPipe)"
        )
        assert (dataset_ratios is None) != (target_proportions is None), (
            "Exactly one of dataset_ratios or target_proportions "
            "must be provided"
        )
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        num_datasets = len(dataset.datasets)
        sizes = [len(d) for d in dataset.datasets]

        if target_proportions is not None:
            assert len(target_proportions) == num_datasets, (
                f"target_proportions length ({len(target_proportions)}) "
                f"must match number of sub-datasets ({num_datasets})"
            )
            assert abs(sum(target_proportions) - 1.0) < 1e-6, (
                f"target_proportions must sum to 1.0, "
                f"got {sum(target_proportions)}"
            )
            # weight per sample = proportion_i / size_i
            # Expected count: num_samples * (prop_i/size_i * size_i) / sum(prop) = num_samples * prop_i
            sample_weights = []
            for size, prop in zip(sizes, target_proportions):
                w = prop / size
                sample_weights.extend([w] * size)
            proportions = list(target_proportions)
        else:
            assert len(dataset_ratios) == num_datasets, (
                f"dataset_ratios length ({len(dataset_ratios)}) must "
                f"match number of sub-datasets ({num_datasets})"
            )
            # weight_i = ratio_i / size_i
            sample_weights = []
            for size, ratio in zip(sizes, dataset_ratios):
                w = ratio / size
                sample_weights.extend([w] * size)
            # Compute actual proportions for epoch_dataset_idx
            raw = [r / s for r, s in zip(dataset_ratios, sizes)]
            total = sum(raw)
            proportions = [r / total for r in raw]

        self.weights = torch.tensor(sample_weights, dtype=torch.float64)

        # Determine num_samples (epoch length)
        if epoch_dataset_idx is not None:
            assert 0 <= epoch_dataset_idx < num_datasets
            # 1 epoch = dataset[idx] sees all samples once
            self.num_samples = int(
                sizes[epoch_dataset_idx] / proportions[epoch_dataset_idx]
            )
            print(
                f"[DatasetRatioSampler] epoch_dataset_idx={epoch_dataset_idx}"
                f" ({sizes[epoch_dataset_idx]} samples,"
                f" {proportions[epoch_dataset_idx]:.1%} proportion)"
                f" -> {self.num_samples} samples/epoch"
            )
        elif num_samples is not None:
            self.num_samples = num_samples
        else:
            self.num_samples = len(dataset)

        # Log dataset info
        for i, (size, prop) in enumerate(zip(sizes, proportions)):
            expected = int(self.num_samples * prop)
            print(
                f"[DatasetRatioSampler] dataset[{i}]: "
                f"size={size}, proportion={prop:.1%}, "
                f"~{expected} samples/epoch"
            )

        # Distributed settings
        self.world_size = get_world_size()
        self.rank = get_rank()
        # Each rank gets an equal share
        self.num_samples_per_rank = math.ceil(
            self.num_samples / self.world_size
        )
        self.total_size = self.num_samples_per_rank * self.world_size

    def __iter__(self) -> Iterator[int]:
        """Generate sampled indices."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = torch.multinomial(
            self.weights,
            num_samples=self.total_size,
            replacement=True,
            generator=g,
        ).tolist()

        # Subsample for this rank
        indices = indices[self.rank::self.world_size]
        assert len(indices) == self.num_samples_per_rank

        return iter(indices)

    def __len__(self) -> int:
        """Return number of samples for this rank."""
        return self.num_samples_per_rank

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling (required for distributed training)."""
        self.epoch = epoch


def build_train_dataloader_with_ratios(
    dataset: DataPipe,
    dataset_ratios: list[float] | None = None,
    target_proportions: list[float] | None = None,
    epoch_dataset_idx: int | None = None,
    num_samples: int | None = None,
    **kwargs,
) -> DataLoader[DictDataOrList]:
    """Build training dataloader with per-dataset ratio sampling.

    Thin wrapper around vis4d's build_train_dataloader that creates a
    DatasetRatioSampler at runtime (when the dataset is instantiated).

    Two ways to specify dataset mixing:

    1. dataset_ratios: raw weights (original, for backwards compat).
       Example: dataset_ratios=[1.0, 1.0] -> equal weight per dataset.

    2. target_proportions: direct proportions (must sum to 1.0).
       Example: target_proportions=[0.5, 0.25, 0.25]

    Args:
        dataset: DataPipe (ConcatDataset) with multiple sub-datasets.
        dataset_ratios: Per-dataset sampling weight (mutually exclusive
            with target_proportions).
        target_proportions: Per-dataset target proportion, must sum to 1.
        epoch_dataset_idx: If set, 1 epoch = this dataset sees all its
            samples once. Overrides num_samples.
        num_samples: Total samples per epoch (overridden by
            epoch_dataset_idx).
        **kwargs: All other arguments forwarded to build_train_dataloader.
    """
    sampler = DatasetRatioSampler(
        dataset,
        dataset_ratios=dataset_ratios,
        target_proportions=target_proportions,
        epoch_dataset_idx=epoch_dataset_idx,
        num_samples=num_samples,
        shuffle=kwargs.pop("shuffle", True),
    )
    # shuffle must be False when using custom sampler (PyTorch requirement)
    return build_train_dataloader(
        dataset=dataset, sampler=sampler, shuffle=False, **kwargs
    )
