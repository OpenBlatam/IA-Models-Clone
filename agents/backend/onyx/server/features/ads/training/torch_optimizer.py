"""
Unified Torch Optimization utilities for the ads training system.

This module adapts and re-exports the legacy torch optimization utilities under the new
training package namespace to support gradual migration while maintaining backward compatibility.
"""

from typing import Any, Dict, Tuple
import os

# Prefer PyTorch compile for speed if user opts in
ENABLE_TORCH_COMPILE = os.getenv("ADS_ENABLE_TORCH_COMPILE", "0").lower() in ("1", "true", "yes", "y")

# Re-export legacy implementation for now to preserve behavior
from ..torch_optimizer import (  # type: ignore
    TorchOptimizationConfig as _LegacyTorchOptimizationConfig,
    TorchMemoryOptimizer as _LegacyTorchMemoryOptimizer,
    TorchPerformanceOptimizer as _LegacyTorchPerformanceOptimizer,
    TorchMixedPrecisionTrainer as _LegacyTorchMixedPrecisionTrainer,
    TorchModelOptimizer as _LegacyTorchModelOptimizer,
    OptimizedTorchDataset as _LegacyOptimizedTorchDataset,
    TorchOptimizedTrainer as _LegacyTorchOptimizedTrainer,
    TorchGradientAccumulator as _LegacyTorchGradientAccumulator,
    TorchProfiler as _LegacyTorchProfiler,
    TorchBenchmarker as _LegacyTorchBenchmarker,
    setup_torch_optimization as _legacy_setup_torch_optimization,
    create_optimized_dataloader as _legacy_create_optimized_dataloader,
    optimize_model_for_training as _legacy_optimize_model_for_training,
)

# Public API (stable names under training namespace)
TorchOptimizationConfig = _LegacyTorchOptimizationConfig
TorchMemoryOptimizer = _LegacyTorchMemoryOptimizer
TorchPerformanceOptimizer = _LegacyTorchPerformanceOptimizer
TorchMixedPrecisionTrainer = _LegacyTorchMixedPrecisionTrainer
TorchModelOptimizer = _LegacyTorchModelOptimizer
OptimizedTorchDataset = _LegacyOptimizedTorchDataset
TorchOptimizedTrainer = _LegacyTorchOptimizedTrainer
TorchGradientAccumulator = _LegacyTorchGradientAccumulator
TorchProfiler = _LegacyTorchProfiler
TorchBenchmarker = _LegacyTorchBenchmarker

setup_torch_optimization = _legacy_setup_torch_optimization
create_optimized_dataloader = _legacy_create_optimized_dataloader
optimize_model_for_training = _legacy_optimize_model_for_training

# Small adapter to enable torch.compile based on env flag without touching call sites
def optimize_model_for_training_fast(model, config=None):  # type: ignore
    model = optimize_model_for_training(model, config)
    if ENABLE_TORCH_COMPILE:
        try:
            import torch  # type: ignore
            if hasattr(torch, "compile"):
                # Use better mode when available, fall back otherwise
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                except Exception:
                    model = torch.compile(model, mode="max-autotune")
        except Exception:
            pass
    return model

__all__ = [
    "TorchOptimizationConfig",
    "TorchMemoryOptimizer",
    "TorchPerformanceOptimizer",
    "TorchMixedPrecisionTrainer",
    "TorchModelOptimizer",
    "OptimizedTorchDataset",
    "TorchOptimizedTrainer",
    "TorchGradientAccumulator",
    "TorchProfiler",
    "TorchBenchmarker",
    "setup_torch_optimization",
    "create_optimized_dataloader",
    "optimize_model_for_training",
    "optimize_model_for_training_fast",
]
