"""
Embedding Services Module

Semantic embedding services for search and similarity.
"""

import sys
from pathlib import Path

# Import from parent directory
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from embedding_service import EmbeddingService
from embedding_service_refactored import EmbeddingService as EmbeddingServiceRefactored

__all__ = [
    "EmbeddingService",
    "EmbeddingServiceRefactored",
]

