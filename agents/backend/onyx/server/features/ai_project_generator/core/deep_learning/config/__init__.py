"""
Configuration Module - YAML Configuration Management
======================================================

This module provides configuration management:
- YAML config loading
- Hyperparameter management
- Model configuration
- Training configuration
"""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml

__all__ = [
    "ConfigManager",
    "load_config",
    "save_config",
    "merge_configs",
]



