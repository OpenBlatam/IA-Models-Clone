"""
Composition Submodule
Aggregates model composition components.
"""

from .composer import ModelComposer
from .models import ComposedModel, ParallelModel
from .sequential import SequentialComposer
from .parallel import ParallelComposer

__all__ = [
    "ModelComposer",
    "ComposedModel",
    "SequentialComposer",
    "ParallelComposer",
    "ParallelModel",
]



