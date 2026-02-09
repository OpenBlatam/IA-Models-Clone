"""
Research Paper Code Improver
============================

Sistema de IA que entrena modelos basados en papers de investigación (PDFs/links)
y utiliza ese conocimiento para mejorar código de GitHub.
"""

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__license__ = "Proprietary"
__description__ = "AI system that trains models from research papers to improve GitHub code"

# Core modules - main entry points with error handling
try:
    from .core.paper_extractor import PaperExtractor
except ImportError:
    PaperExtractor = None

try:
    from .core.model_trainer import ModelTrainer
except ImportError:
    ModelTrainer = None

try:
    from .core.code_improver import CodeImprover
except ImportError:
    CodeImprover = None

__all__ = [
    "PaperExtractor",
    "ModelTrainer",
    "CodeImprover",
]

