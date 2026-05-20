"""Gradio interfaces for deep learning models."""

from .model_demo import create_model_demo, ModelDemo

# Optional imports
try:
    from .transformers_demo import create_transformers_demo
    TRANSFORMERS_DEMO_AVAILABLE = True
except ImportError:
    TRANSFORMERS_DEMO_AVAILABLE = False
    create_transformers_demo = None

try:
    from .diffusion_demo import create_diffusion_demo
    DIFFUSION_DEMO_AVAILABLE = True
except ImportError:
    DIFFUSION_DEMO_AVAILABLE = False
    create_diffusion_demo = None

__all__ = [
    "create_model_demo",
    "ModelDemo",
]

if TRANSFORMERS_DEMO_AVAILABLE:
    __all__.append("create_transformers_demo")

if DIFFUSION_DEMO_AVAILABLE:
    __all__.append("create_diffusion_demo")

