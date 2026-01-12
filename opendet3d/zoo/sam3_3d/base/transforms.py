"""SAM3_3D Data Transforms.

This module implements data augmentation transforms for SAM3_3D training,
particularly the SAM3DPromptSampler for mixed query training.

Key Components:
1. SAM3DPromptSampler: Implements mixed TEXT/GEOMETRIC/TEXT_GEOMETRIC training
2. Box jittering for geometric prompts
3. Point sampling from GT boxes
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor


QueryType = Literal["TEXT", "GEOMETRIC", "TEXT_GEOMETRIC"]


@dataclass
class SAM3DPromptSamplerConfig:
    """Configuration for SAM3DPromptSampler.

    This configuration implements the mixed training strategy from design doc:
    - 70% TEXT: Pure text queries
    - 15% GEOMETRIC: Pure geometric queries with "visual" placeholder
    - 15% TEXT_GEOMETRIC: Mixed text + geometric queries
    """
    # Query type probabilities
    visual_query_prob: float = 0.3  # 30% → convert to geometric
    keep_text_for_visual: float = 0.5  # 50% of geometric keep text

    # Geometric prompt settings
    use_box_vs_point: float = 0.5  # 50% box, 50% point

    # Box jittering (for geometric prompts)
    box_noise_std: float = 0.1  # Gaussian std for box jitter
    box_noise_max: float = 0.3  # Max jitter clamp

    # Point sampling (for geometric prompts)
    num_points: int = 3  # Number of points to sample per box
    point_sampling_mode: Literal["centered", "random_mask", "random_box"] = "centered"

    # Text placeholder for geometric queries
    visual_placeholder: str = "visual"


class SAM3DPromptSampler:
    """Transform to mix TEXT/GEOMETRIC/TEXT_GEOMETRIC queries.

    This implements the mixed training strategy from SAM3D design doc:
    - Per-sample decision on query type
    - Converts TEXT → GEOMETRIC or TEXT_GEOMETRIC based on probability
    - Generates geometric prompts (boxes or points) with augmentation

    Query Type Distribution:
    - 70% TEXT: Original text query, no geometric prompt
    - 15% GEOMETRIC: "visual" placeholder + geometric prompt
    - 15% TEXT_GEOMETRIC: Original text + geometric prompt

    Example:
        sampler = SAM3DPromptSampler(
            visual_query_prob=0.3,
            keep_text_for_visual=0.5,
        )

        sample = {
            "boxes_2d": tensor([[100, 100, 200, 200]]),
            "text_prompts": ["car"],
            "image_size": (480, 640),
        }

        # After sampling, sample will have:
        # - query_types: ["TEXT"] or ["GEOMETRIC"] or ["TEXT_GEOMETRIC"]
        # - text_prompts: ["car"] or ["visual"] or ["car"]
        # - geo_boxes: None or tensor([[0.15, 0.2, 0.32, 0.42]])
        # - geo_points: None or tensor([[0.25, 0.3], [0.28, 0.35], ...])
    """

    def __init__(
        self,
        visual_query_prob: float = 0.3,
        keep_text_for_visual: float = 0.5,
        use_box_vs_point: float = 0.5,
        box_noise_std: float = 0.1,
        box_noise_max: float = 0.3,
        num_points: int = 3,
        point_sampling_mode: Literal["centered", "random_mask", "random_box"] = "centered",
        visual_placeholder: str = "visual",
    ) -> None:
        """Initialize SAM3DPromptSampler.

        Args:
            visual_query_prob: Probability to convert TEXT → GEOMETRIC/TEXT_GEOMETRIC
            keep_text_for_visual: Probability to keep text when visual_query=True
            use_box_vs_point: Probability to use box prompts vs point prompts
            box_noise_std: Standard deviation for Gaussian box jitter
            box_noise_max: Maximum absolute jitter value (clamp)
            num_points: Number of points to sample per box
            point_sampling_mode: How to sample points ("centered", "random_mask", "random_box")
            visual_placeholder: Text placeholder for pure geometric queries
        """
        self.visual_query_prob = visual_query_prob
        self.keep_text_for_visual = keep_text_for_visual
        self.use_box_vs_point = use_box_vs_point
        self.box_noise_std = box_noise_std
        self.box_noise_max = box_noise_max
        self.num_points = num_points
        self.point_sampling_mode = point_sampling_mode
        self.visual_placeholder = visual_placeholder

    def __call__(self, sample: dict) -> dict:
        """Apply mixed query sampling to a single sample.

        Args:
            sample: Dictionary with:
                - boxes_2d: (N, 4) tensor in xyxy format (pixel coordinates)
                - text_prompts: List[str] of length N
                - image_size: (H, W) tuple
                - [optional] boxes_3d, intrinsics, etc.

        Returns:
            Modified sample with:
                - query_types: List[QueryType] of length N
                - text_prompts: Modified text prompts
                - geo_boxes: (N, K, 4) tensor or None (normalized cxcywh)
                - geo_points: (N, P, 2) tensor or None (normalized xy)
        """
        boxes_2d = sample["boxes_2d"]  # (N, 4) pixel xyxy
        text_prompts = sample["text_prompts"]  # List[str]
        image_size = sample["image_size"]  # (H, W)

        num_boxes = boxes_2d.shape[0]
        img_h, img_w = image_size

        # Normalize boxes to [0, 1] for geometric prompt generation
        boxes_2d_norm = boxes_2d.clone().float()
        boxes_2d_norm[:, [0, 2]] /= img_w
        boxes_2d_norm[:, [1, 3]] /= img_h

        # Initialize outputs
        query_types = []
        new_text_prompts = []
        geo_boxes_list = []
        geo_points_list = []

        # Per-box decision
        for i in range(num_boxes):
            box_xyxy = boxes_2d_norm[i]  # (4,) normalized xyxy
            text = text_prompts[i]

            # Decide query type
            use_visual = random.random() < self.visual_query_prob

            if use_visual:
                # 30% → GEOMETRIC or TEXT_GEOMETRIC
                keep_text = random.random() < self.keep_text_for_visual

                if keep_text:
                    # TEXT_GEOMETRIC: Keep text + add geometric prompt
                    query_type = "TEXT_GEOMETRIC"
                    new_text = text
                else:
                    # GEOMETRIC: Replace text with placeholder + add geometric prompt
                    query_type = "GEOMETRIC"
                    new_text = self.visual_placeholder

                # Generate geometric prompt (box or point)
                use_box = random.random() < self.use_box_vs_point

                if use_box:
                    # Generate jittered box
                    geo_box = self._jitter_box(box_xyxy)
                    geo_boxes_list.append(geo_box.unsqueeze(0))  # (1, 4)
                    geo_points_list.append(None)
                else:
                    # Generate points
                    geo_points = self._sample_points(box_xyxy)
                    geo_boxes_list.append(None)
                    geo_points_list.append(geo_points)  # (P, 2)
            else:
                # 70% → TEXT: No geometric prompt
                query_type = "TEXT"
                new_text = text
                geo_boxes_list.append(None)
                geo_points_list.append(None)

            query_types.append(query_type)
            new_text_prompts.append(new_text)

        # Aggregate geometric prompts
        # geo_boxes: (N, K, 4) where K=1 for boxes, 0 for no box
        # geo_points: (N, P, 2) where P=num_points for points, 0 for no points

        has_boxes = [b is not None for b in geo_boxes_list]
        has_points = [p is not None for p in geo_points_list]

        if any(has_boxes):
            # Create padded box tensor
            max_boxes = 1
            geo_boxes = torch.zeros(num_boxes, max_boxes, 4, dtype=torch.float32)
            for i, box in enumerate(geo_boxes_list):
                if box is not None:
                    geo_boxes[i] = box
            sample["geo_boxes"] = geo_boxes
        else:
            sample["geo_boxes"] = None

        if any(has_points):
            # Create padded point tensor
            max_points = self.num_points
            geo_points = torch.zeros(num_boxes, max_points, 2, dtype=torch.float32)
            for i, points in enumerate(geo_points_list):
                if points is not None:
                    num_pts = points.shape[0]
                    geo_points[i, :num_pts] = points
            sample["geo_points"] = geo_points
        else:
            sample["geo_points"] = None

        # Update sample
        sample["query_types"] = query_types
        sample["text_prompts"] = new_text_prompts

        return sample

    def _jitter_box(self, box_xyxy: Tensor) -> Tensor:
        """Apply Gaussian jitter to box coordinates.

        Args:
            box_xyxy: (4,) tensor in normalized xyxy format [0, 1]

        Returns:
            Jittered box in normalized cxcywh format [0, 1]
        """
        x1, y1, x2, y2 = box_xyxy
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # Apply Gaussian noise
        noise = torch.randn(4) * self.box_noise_std
        noise = torch.clamp(noise, -self.box_noise_max, self.box_noise_max)

        cx_jit = cx + noise[0] * w
        cy_jit = cy + noise[1] * h
        w_jit = w * (1 + noise[2])
        h_jit = h * (1 + noise[3])

        # Clamp to [0, 1]
        cx_jit = torch.clamp(cx_jit, 0.0, 1.0)
        cy_jit = torch.clamp(cy_jit, 0.0, 1.0)
        w_jit = torch.clamp(w_jit, 0.01, 1.0)
        h_jit = torch.clamp(h_jit, 0.01, 1.0)

        box_cxcywh = torch.stack([cx_jit, cy_jit, w_jit, h_jit])
        return box_cxcywh

    def _sample_points(self, box_xyxy: Tensor) -> Tensor:
        """Sample points from box based on sampling mode.

        Args:
            box_xyxy: (4,) tensor in normalized xyxy format [0, 1]

        Returns:
            Points tensor (P, 2) in normalized xy format [0, 1]
        """
        x1, y1, x2, y2 = box_xyxy

        if self.point_sampling_mode == "centered":
            # Sample from center region of box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1

            # Sample within 50% of box size around center
            points = []
            for _ in range(self.num_points):
                px = cx + (random.random() - 0.5) * w * 0.5
                py = cy + (random.random() - 0.5) * h * 0.5
                px = torch.clamp(torch.tensor(px), x1, x2)
                py = torch.clamp(torch.tensor(py), y1, y2)
                points.append([px.item(), py.item()])

        elif self.point_sampling_mode == "random_box":
            # Uniformly sample from entire box
            points = []
            for _ in range(self.num_points):
                px = x1 + random.random() * (x2 - x1)
                py = y1 + random.random() * (y2 - y1)
                points.append([px.item(), py.item()])

        elif self.point_sampling_mode == "random_mask":
            # For now, fall back to random_box
            # In future, could use instance masks if available
            points = []
            for _ in range(self.num_points):
                px = x1 + random.random() * (x2 - x1)
                py = y1 + random.random() * (y2 - y1)
                points.append([px.item(), py.item()])

        else:
            raise ValueError(f"Unknown point_sampling_mode: {self.point_sampling_mode}")

        return torch.tensor(points, dtype=torch.float32)


def build_sam3d_prompt_sampler(config: SAM3DPromptSamplerConfig | None = None) -> SAM3DPromptSampler:
    """Build SAM3DPromptSampler from config.

    Args:
        config: Configuration for prompt sampler

    Returns:
        Initialized SAM3DPromptSampler
    """
    if config is None:
        config = SAM3DPromptSamplerConfig()

    return SAM3DPromptSampler(
        visual_query_prob=config.visual_query_prob,
        keep_text_for_visual=config.keep_text_for_visual,
        use_box_vs_point=config.use_box_vs_point,
        box_noise_std=config.box_noise_std,
        box_noise_max=config.box_noise_max,
        num_points=config.num_points,
        point_sampling_mode=config.point_sampling_mode,
        visual_placeholder=config.visual_placeholder,
    )
