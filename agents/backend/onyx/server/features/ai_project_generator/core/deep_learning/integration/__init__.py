"""
Integration Module - Third-party Integrations
=============================================

Provides integrations with external tools and services:
- Hugging Face Hub
- Weights & Biases
- TensorBoard
- MLflow
- ONNX Runtime
"""

from typing import Optional, Dict, Any

# Try to import integrations
try:
    from .huggingface_hub import HuggingFaceHubIntegration
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    HuggingFaceHubIntegration = None

try:
    from .mlflow import MLflowIntegration
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    MLflowIntegration = None

__all__ = []

if HF_HUB_AVAILABLE:
    __all__.append("HuggingFaceHubIntegration")

if MLFLOW_AVAILABLE:
    __all__.append("MLflowIntegration")

