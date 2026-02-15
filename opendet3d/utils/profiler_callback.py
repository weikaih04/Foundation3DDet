"""vis4d callback for comprehensive training loop profiling.

This callback profiles the ENTIRE training loop:
- Data loading time (between batches)
- Forward pass time
- Loss computation time
- Backward pass time
- Optimizer step time

Usage:
    Set PROFILE_SAM3_3D=1 and PROFILE_INTERVAL=N (print every N steps)
    Set PROFILE_WARMUP=N to skip first N steps (default: 10)
"""

import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Union

import torch
import torch.distributed as dist

# Use vis4d's Callback class, not PyTorch Lightning's
from vis4d.engine.callbacks.base import Callback

from opendet3d.utils.profiler import profile_start, profile_stop, profiler


class EnhancedProfilingCallback(Callback):
    """Comprehensive training loop profiler using vis4d Callback.

    Measures all phases of training:
    1. data_loading: Time to fetch next batch from DataLoader
    2. forward_only: Model forward pass (without loss)
    3. loss_total: Loss computation
    4. backward: Backward pass (gradient computation)
    5. optimizer_and_sync: Optimizer step + DDP gradient sync

    Timing diagram:
    |-- data_loading --|-- forward_only --|-- loss_total --|-- backward --|-- optimizer_sync --|
    """

    def __init__(self):
        super().__init__()
        self._enabled = os.environ.get("PROFILE_SAM3_3D", "0") == "1"
        self._print_interval = int(os.environ.get("PROFILE_INTERVAL", "10"))
        self._warmup_steps = int(os.environ.get("PROFILE_WARMUP", "10"))

        # Timing state
        self._last_batch_end_time = None
        self._step_count = 0
        self._forward_start_time = None
        self._backward_start_time = None

        # Accumulated timings for averaging
        self._timings: Dict[str, List[float]] = defaultdict(list)

        if self._enabled:
            print(f"[Profiling] EnhancedProfilingCallback initialized "
                  f"(interval={self._print_interval}, warmup={self._warmup_steps})")

    def _sync_cuda(self) -> None:
        """Synchronize CUDA for accurate timing."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _record_timing(self, name: str, duration_ms: float) -> None:
        """Record a timing measurement."""
        self._timings[name].append(duration_ms)

    def _compute_stats(self, values: List[float]) -> dict:
        """Compute statistics with outlier detection."""
        if not values:
            return {"avg": 0, "min": 0, "max": 0, "p50": 0, "p95": 0}

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        return {
            "avg": sum(values) / n,
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
            "p50": sorted_vals[n // 2],
            "p95": sorted_vals[int(n * 0.95)] if n >= 2 else sorted_vals[-1],
        }

    def _is_rank_zero(self) -> bool:
        """Check if current process is rank 0."""
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def _print_summary(self, step: int) -> None:
        """Print timing summary (rank 0 only)."""
        if not self._timings or not self._is_rank_zero():
            return

        print("\n" + "=" * 80)
        print(f"[TrainingProfiler] Step {step} - Full Training Loop Timing")
        print("=" * 80)
        n_samples = len(self._timings.get("step_total", [1]))
        print(f"TRAINING STEP BREAKDOWN (avg over last {n_samples} steps, warmup={self._warmup_steps} skipped):")

        # Calculate total for percentages
        total_avg = 0
        if "step_total" in self._timings:
            total_avg = sum(self._timings["step_total"]) / len(self._timings["step_total"])

        # Define display order - now with forward_only and loss_total separated
        display_order = [
            "data_loading",
            "forward_only",
            "  backbone",
            "  geometry_backend",
            "  backbone+geom_parallel",
            "  sam3_grounding",
            "  3d_head",
            "loss_total",
            "  loss_targets",
            "  loss_cls",
            "  loss_2d_box",
            "  loss_o2m",
            "  loss_3d",
            "  loss_geom",
            "  loss_aux",
            "backward",
            "optimizer_and_sync",
            "step_total",
        ]

        for name in display_order:
            if name in self._timings and self._timings[name]:
                stats = self._compute_stats(self._timings[name])
                pct = (stats["avg"] / total_avg * 100) if total_avg > 0 else 0

                if name == "step_total":
                    print("-" * 80)
                    print(f"  {name:30s}: avg={stats['avg']:8.2f}ms  "
                          f"p50={stats['p50']:8.2f}ms  p95={stats['p95']:8.2f}ms")
                elif name.startswith("  "):
                    # Nested items (indented)
                    print(f"    {name:28s}: avg={stats['avg']:8.2f}ms  ({pct:5.1f}%)")
                else:
                    # Check for outliers (max > 2x avg)
                    outlier_flag = " [!]" if stats["max"] > stats["avg"] * 2 else ""
                    print(f"  {name:30s}: avg={stats['avg']:8.2f}ms  ({pct:5.1f}%)  "
                          f"p50={stats['p50']:8.2f}ms  p95={stats['p95']:8.2f}ms{outlier_flag}")

        # Print throughput estimate
        if total_avg > 0:
            steps_per_sec = 1000.0 / total_avg
            total_samples = 175573
            print("-" * 80)
            print(f"  Throughput: {steps_per_sec:.3f} steps/sec")
            # Estimate for different GPU counts
            for n_gpus in [1, 2, 8]:
                steps_per_epoch = total_samples // n_gpus
                epoch_hours = (steps_per_epoch * total_avg / 1000) / 3600
                print(f"  Estimated epoch time ({n_gpus} GPUs): {epoch_hours:.1f} hours")

        print("=" * 80 + "\n")

        # Clear timings after printing
        self._timings.clear()

    def on_train_batch_start(
        self,
        trainer: Any,
        pl_module: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called when batch starts - data loading is complete."""
        if not self._enabled:
            return

        self._sync_cuda()
        now = time.perf_counter()

        # Skip warmup steps
        if self._step_count >= self._warmup_steps:
            # Measure data loading time (time since last batch ended)
            if self._last_batch_end_time is not None:
                data_time_ms = (now - self._last_batch_end_time) * 1000
                self._record_timing("data_loading", data_time_ms)
                profiler().current_step_timings["data_loading"] = data_time_ms / 1000

        # Start timing forward+loss
        self._forward_start_time = now

    def on_before_backward(
        self,
        trainer: Any,
        pl_module: Any,
        loss: torch.Tensor,
    ) -> None:
        """Called before backward - forward+loss is done."""
        if not self._enabled:
            return

        self._sync_cuda()
        now = time.perf_counter()

        # Skip warmup steps
        if self._step_count >= self._warmup_steps:
            # Record forward+loss time
            if self._forward_start_time is not None:
                forward_and_loss_ms = (now - self._forward_start_time) * 1000
                self._record_timing("forward_and_loss", forward_and_loss_ms)

            # Copy nested timings from profiler
            p = profiler()

            # Forward components
            for key in ["forward_total", "  backbone", "  geometry_backend",
                        "  backbone+geom_parallel", "  sam3_grounding", "  3d_head",
                        "collator_total"]:
                if key in p.current_step_timings:
                    self._record_timing(key, p.current_step_timings[key] * 1000)

            # Loss components
            for key in ["loss_total", "  loss_targets", "  loss_cls",
                        "  loss_2d_box", "  loss_o2m", "  loss_3d",
                        "  loss_geom", "  loss_aux"]:
                if key in p.current_step_timings:
                    self._record_timing(key, p.current_step_timings[key] * 1000)

            # Compute forward_only = forward_and_loss - loss_total
            if "forward_total" in p.current_step_timings:
                forward_only_ms = p.current_step_timings["forward_total"] * 1000
                self._record_timing("forward_only", forward_only_ms)

        # Start backward timing
        self._backward_start_time = now

    def on_after_backward(
        self,
        trainer: Any,
        pl_module: Any,
    ) -> None:
        """Called after backward - gradients computed."""
        if not self._enabled:
            return

        self._sync_cuda()
        now = time.perf_counter()

        # Skip warmup steps
        if self._step_count >= self._warmup_steps:
            # Record backward time
            if self._backward_start_time is not None:
                backward_time_ms = (now - self._backward_start_time) * 1000
                self._record_timing("backward", backward_time_ms)

        # Start optimizer+sync timing (will be stopped in on_train_batch_end)
        self._optimizer_start_time = now

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Union[torch.Tensor, Mapping[str, Any], None],
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called when batch ends - everything complete."""
        if not self._enabled:
            return

        self._sync_cuda()
        now = time.perf_counter()

        # Skip warmup steps
        if self._step_count >= self._warmup_steps:
            # Record optimizer+sync time
            if hasattr(self, '_optimizer_start_time') and self._optimizer_start_time is not None:
                optimizer_time_ms = (now - self._optimizer_start_time) * 1000
                self._record_timing("optimizer_and_sync", optimizer_time_ms)

            # Record total step time
            if self._forward_start_time is not None:
                step_time_ms = (now - self._forward_start_time) * 1000
                self._record_timing("step_total", step_time_ms)

        # Record end time for next data_loading measurement
        self._last_batch_end_time = now

        # Increment step and print summary if needed
        self._step_count += 1

        # Print at intervals, but only after warmup
        if self._step_count > self._warmup_steps and \
           (self._step_count - self._warmup_steps) % self._print_interval == 0:
            self._print_summary(self._step_count)

        # Clear profiler for next step
        profiler().current_step_timings.clear()

        # Reset timing states
        self._forward_start_time = None
        self._backward_start_time = None
        self._optimizer_start_time = None


def get_profiling_callback():
    """Get profiling callback if enabled."""
    if os.environ.get("PROFILE_SAM3_3D", "0") == "1":
        return EnhancedProfilingCallback()
    return None
