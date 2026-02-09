"""
Inference Module - Model Inference and Gradio Integration
==========================================================

This module provides inference functionality:
- Model inference utilities
- Gradio app creation
- Interactive demos
- Batch inference
"""

from typing import Optional, Dict, Any, Callable, List
import torch
import gradio as gr

from .inference_engine import InferenceEngine, batch_inference
from .gradio_apps import create_gradio_app

# Try to import advanced Gradio apps
try:
    from .gradio_advanced import (
        create_model_comparison_app,
        create_interactive_training_app,
        create_batch_inference_app
    )
    ADVANCED_GRADIO_AVAILABLE = True
except ImportError:
    ADVANCED_GRADIO_AVAILABLE = False
    create_model_comparison_app = None
    create_interactive_training_app = None
    create_batch_inference_app = None

__all__ = [
    "InferenceEngine",
    "create_gradio_app",
    "batch_inference",
]

if ADVANCED_GRADIO_AVAILABLE:
    __all__.extend([
        "create_model_comparison_app",
        "create_interactive_training_app",
        "create_batch_inference_app"
    ])

