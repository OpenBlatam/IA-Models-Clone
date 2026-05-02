"""
Multimodal Interactive Agent Implementation

Refactored architecture with separated concerns:
- models.py: Data models (enums, dataclasses)
- modality_processors.py: Input processing logic
- modality_generators.py: Output generation logic
- context_analyzer.py: Context analysis and reasoning
- classifiers.py: Interaction classification
- multimodal_interactive.py: Main agent class
"""

from .multimodal_interactive import MultimodalInteractiveAgent
from .models import (
    ModalityType,
    InteractionType,
    MultimodalInput,
    MultimodalOutput,
    Interaction
)

__all__ = [
    "MultimodalInteractiveAgent",
    "ModalityType",
    "InteractionType",
    "MultimodalInput",
    "MultimodalOutput",
    "Interaction",
]



