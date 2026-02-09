"""
Experiments Module
Experiment tracking, configuration, and management
"""

import sys
from pathlib import Path

# Add parent to path for imports
_parent = Path(__file__).parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

try:
    from core.experiment_tracker import (
        ExperimentTracker,
        ExperimentConfig,
        ExperimentMetrics,
        ExperimentStatus
    )
    EXPERIMENT_TRACKING_AVAILABLE = True
except ImportError:
    EXPERIMENT_TRACKING_AVAILABLE = False

try:
    from config import load_config, save_config, get_config_value
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

__all__ = []

if EXPERIMENT_TRACKING_AVAILABLE:
    __all__.extend([
        'ExperimentTracker',
        'ExperimentConfig',
        'ExperimentMetrics',
        'ExperimentStatus'
    ])

if CONFIG_AVAILABLE:
    __all__.extend([
        'load_config',
        'save_config',
        'get_config_value'
    ])

