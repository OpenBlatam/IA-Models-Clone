from __future__ import annotations

import cProfile
import pstats
import time
from contextlib import contextmanager
from typing import Callable, Generator, Optional


class Timer:
    def __init__(self, name: str = "timer", logger=None) -> None:
        self.name = name
        self.logger = logger
        self._start = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        if self.logger is not None:
            self.logger.info("%s %.2f ms", self.name, self.elapsed_ms)


@contextmanager
def cprofile_to_file(output_path: str, sort_by: str = "cumtime") -> Generator[None, None, None]:
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield
    finally:
        profiler.disable()
        with open(output_path, "w", encoding="utf-8") as f:
            ps = pstats.Stats(profiler, stream=f).strip_dirs().sort_stats(sort_by)
            ps.print_stats()


def profile_function(output_path: Optional[str] = None) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003
            if output_path:
                with cprofile_to_file(output_path):
                    return func(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator


