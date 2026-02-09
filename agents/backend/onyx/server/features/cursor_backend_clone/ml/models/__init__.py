"""
Models - Modelos de Deep Learning
===================================

Modelos modulares para diferentes tareas.
"""

from .base import BaseModel
from .code_completion import CodeCompletionModel
from .code_explanation import CodeExplanationModel

__all__ = [
    "BaseModel",
    "CodeCompletionModel",
    "CodeExplanationModel",
]


