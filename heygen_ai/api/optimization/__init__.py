from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .quantum_model_optimizer import QuantumModelOptimizer, create_quantum_model_optimizer
from .advanced_gpu_optimizer import AdvancedGPUOptimizer
from .model_quantization import ModelQuantizationSystem
from .model_distillation import ModelDistillationSystem
from .model_pruning import ModelPruningSystem
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Optimization package for HeyGen AI.

Advanced model optimization with GPU utilization and mixed precision training
following PEP 8 style guidelines.
"""


__all__ = [
    "QuantumModelOptimizer",
    "AdvancedGPUOptimizer",
    "ModelQuantizationSystem",
    "ModelDistillationSystem",
    "ModelPruningSystem",
    "create_quantum_model_optimizer"
] 