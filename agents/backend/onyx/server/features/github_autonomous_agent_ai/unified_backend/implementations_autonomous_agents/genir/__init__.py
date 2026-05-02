"""
GenIR Framework
===============

Framework for Generative Information Retrieval.
"""

from .genir import (
    GenIRAgent,
    GenIRTaskType,
    RetrievalStrategy,
    Query,
    GeneratedDocument,
    RetrievalResult
)

__all__ = [
    "GenIRAgent",
    "GenIRTaskType",
    "RetrievalStrategy",
    "Query",
    "GeneratedDocument",
    "RetrievalResult"
]


