from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .ultra_performance_boost import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Ultra Performance Boost - Advanced Optimization Package
🚀 Next-generation performance optimizations for NotebookLM AI
"""

    UltraPerformanceBoost,
    UltraBoostConfig,
    GPUMemoryManager,
    ModelQuantizer,
    AsyncBatchProcessor,
    IntelligentCache,
    get_ultra_boost,
    cleanup_ultra_boost,
    ultra_boost_monitor,
    ultra_boost_cache
)

__version__ = "1.0.0"
__author__ = "NotebookLM AI Team"
__description__ = "Ultra Performance Boost - Advanced optimization engine for AI systems"

__all__ = [
    "UltraPerformanceBoost",
    "UltraBoostConfig", 
    "GPUMemoryManager",
    "ModelQuantizer",
    "AsyncBatchProcessor",
    "IntelligentCache",
    "get_ultra_boost",
    "cleanup_ultra_boost",
    "ultra_boost_monitor",
    "ultra_boost_cache"
]

# Package metadata
PACKAGE_INFO = {
    "name": "ultra-performance-boost",
    "version": __version__,
    "description": __description__,
    "features": [
        "Intelligent caching with adaptive TTL",
        "Async batch processing for maximum throughput", 
        "GPU/CPU memory optimization",
        "Model quantization capabilities",
        "Comprehensive health monitoring",
        "Prometheus metrics integration",
        "Global instance management"
    ],
    "components": [
        "UltraPerformanceBoost - Main optimization engine",
        "GPUMemoryManager - GPU memory management",
        "ModelQuantizer - Model optimization",
        "AsyncBatchProcessor - Batch processing",
        "IntelligentCache - Smart caching system"
    ]
} 