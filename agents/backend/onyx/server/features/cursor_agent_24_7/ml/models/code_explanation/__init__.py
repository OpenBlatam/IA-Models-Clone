"""
Code Explanation Model - Implementación modular
================================================

Módulo completamente modular para explicación de código con componentes separados:
- CodeExplanationModel: Modelo principal
- ExplanationCache: Gestión de caché
- ModelStats: Estadísticas
- InputValidator: Validación de inputs
- PromptBuilder: Construcción de prompts
- BatchProcessor: Procesamiento en batch
- ModelLoader: Carga de modelos
"""

from .model import CodeExplanationModel
from .cache import ExplanationCache
from .stats import ModelStats
from .validator import InputValidator
from .prompt_builder import PromptBuilder
from .batch_processor import BatchProcessor
from .model_loader import ModelLoader

__all__ = [
    "CodeExplanationModel",
    "ExplanationCache",
    "ModelStats",
    "InputValidator",
    "PromptBuilder",
    "BatchProcessor",
    "ModelLoader",
]

