from __future__ import annotations

import time
from dataclasses import dataclass

import psutil
import torch


@dataclass
class StepProfile:
    dataloader_ms: float
    step_ms: float
    cpu_mem_mb: float
    gpu_mem_mb: float | None


def measure_cpu_mem_mb() -> float:
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 * 1024)


def measure_gpu_mem_mb() -> float | None:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return None


class StepTimer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._t1 = self._t0

    def mark_loader_done(self) -> None:
        self._t1 = time.perf_counter()

    def mark_step_done(self) -> StepProfile:
        t2 = time.perf_counter()
        dataloader_ms = (self._t1 - self._t0) * 1000
        step_ms = (t2 - self._t1) * 1000
        return StepProfile(
            dataloader_ms=dataloader_ms,
            step_ms=step_ms,
            cpu_mem_mb=measure_cpu_mem_mb(),
            gpu_mem_mb=measure_gpu_mem_mb(),
        )
