from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import torch
import logging
from contextlib import contextmanager
import time

from typing import Any, List, Dict, Optional
import asyncio
logger = logging.getLogger(__name__)

@contextmanager
def profile_section(section_name, profiler=None) -> Any:
    """Context manager for profiling a code section."""
    start_time = time.time()
    if profiler is not None:
        with profiler.record_function(section_name):
            yield
    else:
        yield
    elapsed = time.time() - start_time
    logger.info(f"[PROFILE] {section_name} took {elapsed:.4f} seconds")

class ProfilerManager:
    """Manages PyTorch profiler for training, data loading, and preprocessing."""
    def __init__(self, enabled=True, profile_memory=True, export_trace=False, trace_file="profile_trace.json") -> Any:
        self.enabled = enabled
        self.profile_memory = profile_memory
        self.export_trace = export_trace
        self.trace_file = trace_file
        self.profiler = None

    def __enter__(self) -> Any:
        if self.enabled:
            self.profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=self._trace_handler if self.export_trace else None,
                record_shapes=True,
                profile_memory=self.profile_memory,
                with_stack=True
            )
            self.profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
        if self.profiler is not None:
            self.profiler.__exit__(exc_type, exc_val, exc_tb)
            self.profiler = None

    def step(self) -> Any:
        if self.profiler is not None:
            self.profiler.step()

    def _trace_handler(self, p) -> Any:
        logger.info(f"[PROFILE] Exporting trace to {self.trace_file}")
        p.export_chrome_trace(self.trace_file)

    def summary(self) -> Any:
        if self.profiler is not None:
            logger.info("[PROFILE] Key Events:")
            logger.info(self.profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)) 