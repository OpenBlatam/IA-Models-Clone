"""
Templates Module - Code Templates and Snippets
==============================================

Provides code templates for common deep learning tasks:
- Training scripts
- Inference scripts
- Model definitions
- Configuration files
"""

from typing import Dict, Any, Optional

from .templates import (
    get_training_template,
    get_inference_template,
    get_config_template,
    generate_project_structure
)

__all__ = [
    "get_training_template",
    "get_inference_template",
    "get_config_template",
    "generate_project_structure",
]

