"""
Benchmarking Module - Performance Benchmarking
==============================================

Utilities for performance benchmarking:
- Model benchmarking
- Training benchmarking
- Inference benchmarking
- Comparison benchmarking
"""

from typing import Optional, Dict, Any

from .benchmark_utils import (
    benchmark_inference,
    benchmark_training,
    compare_models,
    BenchmarkSuite
)

__all__ = [
    "benchmark_inference",
    "benchmark_training",
    "compare_models",
    "BenchmarkSuite",
]

