"""
Adapters Module
Adapter pattern for model integration.
"""

from .base_adapter import (
    ModelAdapter,
    HuggingFaceAdapter,
    ONNXAdapter,
    TensorFlowAdapter,
    AdapterRegistry,
)

__all__ = [
    "ModelAdapter",
    "HuggingFaceAdapter",
    "ONNXAdapter",
    "TensorFlowAdapter",
    "AdapterRegistry",
]



