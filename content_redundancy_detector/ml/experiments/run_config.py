"""
Run Configuration
Configuration for experiment runs
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

# Re-export from experiment_manager for convenience
from .experiment_manager import RunConfig

__all__ = ["RunConfig"]



