"""
Core AI Infrastructure

Base classes and utilities for all AI services.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from base_service import BaseAIService

__all__ = ["BaseAIService"]

