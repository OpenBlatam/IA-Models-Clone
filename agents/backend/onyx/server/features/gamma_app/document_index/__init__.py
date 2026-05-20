"""
Document Index Module
Document indexing and search
"""

from .base import (
    Document,
    DocumentIndex,
    IndexBase
)
from .service import DocumentIndexService

__all__ = [
    "Document",
    "DocumentIndex",
    "IndexBase",
    "DocumentIndexService",
]

