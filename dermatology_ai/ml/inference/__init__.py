"""
Inference Module
Model inference engines and utilities
"""

import sys
from pathlib import Path

# Add parent to path for imports
_parent = Path(__file__).parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

__all__ = []

try:
    from utils.optimization import FastInferenceEngine
    __all__.append('FastInferenceEngine')
except ImportError:
    pass

try:
    from utils.async_inference import AsyncInferenceEngine, BatchInferenceEngine
    __all__.extend(['AsyncInferenceEngine', 'BatchInferenceEngine'])
except ImportError:
    pass

try:
    from utils.advanced_optimization import SmartBatchProcessor
    __all__.append('SmartBatchProcessor')
except ImportError:
    pass

try:
    from core.ml_model_manager import MLModelManager, ModelConfig, ModelType, InferenceResult
    __all__.extend(['MLModelManager', 'ModelConfig', 'ModelType', 'InferenceResult'])
except ImportError:
    pass

