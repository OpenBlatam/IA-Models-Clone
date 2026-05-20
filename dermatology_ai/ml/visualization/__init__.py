"""
Visualization Module
Gradio demos and visualization utilities
"""

import sys
from pathlib import Path

# Add parent to path for imports
_parent = Path(__file__).parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

__all__ = []

try:
    from core.gradio_integration import GradioDemo, ModelComparisonDemo
    __all__.extend(['GradioDemo', 'ModelComparisonDemo'])
except ImportError:
    pass

