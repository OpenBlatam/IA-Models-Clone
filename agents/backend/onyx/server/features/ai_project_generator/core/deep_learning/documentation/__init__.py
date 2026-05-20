"""
Documentation Module - Auto-Documentation Utilities
===================================================

Utilities for automatic documentation generation:
- Model documentation
- API documentation
- Training documentation
- Project documentation
"""

from typing import Optional, Dict, Any

from .doc_generator import (
    generate_model_docs,
    generate_training_docs,
    generate_api_docs,
    create_project_readme
)

__all__ = [
    "generate_model_docs",
    "generate_training_docs",
    "generate_api_docs",
    "create_project_readme",
]

