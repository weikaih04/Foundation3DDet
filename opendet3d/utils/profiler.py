"""Training profiler for SAM3_3D performance analysis.

Usage:
    Set environment variable PROFILE_SAM3_3D=1 to enable profiling.
    Timing results are printed every N iterations.
"""

import os
import time
from collections import defaultdict
from typing import Dict, List, Optional

import torch
import torch.distributed as dist


class TrainingProfiler:
    """Profiler for measuring training component timings.

    Tracks timing for:
    - Data loading (collator)
    - Model forward pass components
    - Loss computation
    - Backward pass
    """

    _instance: Optional["TrainingProfiler"] = None

    def __init__(self, print_interval: int = 10, enabled: bool = True):
        self.print_interval = print_interval
        self.enabled = enabled
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.step_count = 0
        self.current_step_timings: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}

    @classmethod
    def get_instance(cls) -> "TrainingProfiler":
        """Get singleton instance."""
        if cls._instance is None:
            enabled = os.environ.get("PROFILE_SAM3_3D", "0") == "1"
            print_interval = int(os.environ.get("PROFILE_INTERVAL", "10"))
            cls._instance = cls(print_interval=print_interval, enabled=enabled)
            if enabled:
                print(f"[TrainingProfiler] Enabled, printing every {print_interval} steps")
        return cls._instance

    def _is_main_process(self) -> bool:
        """Check if running in main process (not DataLoader worker)."""
        import multiprocessing
        # DataLoader workers have different process names
        current = multiprocessing.current_process()
        return current.name == "MainProcess"

    def _safe_cuda_sync(self) -> None:
        """Synchronize CUDA only in main process."""
        if self._is_main_process() and torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self, name: str) -> None:
        """Start timing a component."""
        if not self.enabled:
            return
        # Skip profiling in DataLoader workers
        if not self._is_main_process():
            return
        self._safe_cuda_sync()
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """Stop timing and record."""
        if not self.enabled:
            return 0.0
        # Skip profiling in DataLoader workers
        if not self._is_main_process():
            return 0.0
        self._safe_cuda_sync()
        elapsed = time.perf_counter() - self._start_times.get(name, time.perf_counter())
        self.current_step_timings[name] = elapsed
        return elapsed

    def step(self) -> None:
        """Called at end of each training step.

        NOTE: Does NOT clear current_step_timings here.
        The EnhancedProfilingCallback reads current_step_timings
        in on_before_backward and clears them in on_train_batch_end.
        """
        if not self.enabled:
            return

        # Record all timings from this step into history
        for name, elapsed in self.current_step_timings.items():
            self.timings[name].append(elapsed)

        self.step_count += 1
        # Do NOT clear current_step_timings or print here.
        # The callback handles both.

    def _is_rank_zero(self) -> bool:
        """Check if current process is rank 0."""
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def _print_summary(self) -> None:
        """Print timing summary (rank 0 only)."""
        if not self._is_rank_zero():
            return
        print("\n" + "=" * 80)
        print(f"[TrainingProfiler] Step {self.step_count} - Timing Summary (last {self.print_interval} steps)")
        print("=" * 80)

        # Calculate stats
        stats = {}
        total_time = 0.0
        for name, times in self.timings.items():
            recent = times[-self.print_interval:] if len(times) >= self.print_interval else times
            if recent:
                avg = sum(recent) / len(recent)
                stats[name] = {
                    "avg_ms": avg * 1000,
                    "min_ms": min(recent) * 1000,
                    "max_ms": max(recent) * 1000,
                }
                # Accumulate for total (exclude sub-timings)
                if not name.startswith("  "):
                    total_time += avg

        # Print in order
        order = [
            "data_loading",
            "  collator_total",
            "  collator_image_stack",
            "  collator_category_group",
            "  collator_coord_convert",
            "  collator_tensor_stack",
            "forward_total",
            "  backbone",
            "  geometry_backend",
            "  sam3_grounding",
            "  3d_head",
            "loss_compute",
            "backward",
        ]

        # Print known keys first, then any unknown
        printed = set()
        for name in order:
            if name in stats:
                s = stats[name]
                prefix = "" if not name.startswith("  ") else "  "
                clean_name = name.strip()
                print(f"{prefix}{clean_name:30s}: avg={s['avg_ms']:8.2f}ms  min={s['min_ms']:8.2f}ms  max={s['max_ms']:8.2f}ms")
                printed.add(name)

        # Print any remaining
        for name in sorted(stats.keys()):
            if name not in printed:
                s = stats[name]
                print(f"{name:30s}: avg={s['avg_ms']:8.2f}ms  min={s['min_ms']:8.2f}ms  max={s['max_ms']:8.2f}ms")

        print("-" * 80)
        print(f"{'Total (non-nested)':30s}: {total_time * 1000:8.2f}ms")
        print("=" * 80 + "\n")


# Convenience functions
def profiler() -> TrainingProfiler:
    """Get the global profiler instance."""
    return TrainingProfiler.get_instance()


def profile_start(name: str) -> None:
    """Start profiling a named section."""
    TrainingProfiler.get_instance().start(name)


def profile_stop(name: str) -> float:
    """Stop profiling a named section and return elapsed time."""
    return TrainingProfiler.get_instance().stop(name)


def profile_step() -> None:
    """Mark end of training step."""
    TrainingProfiler.get_instance().step()
