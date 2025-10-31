from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .core.config import EnterpriseConfig, get_config
from .core.models import (
from .core.service import EnterpriseCopywritingService
from .core.optimization import OptimizationEngine
from .core.cache import EnterpriseCache
from .core.monitoring import MetricsEngine
from .core.security import SecurityManager
from .api.application import create_enterprise_app
from .deployment.manager import DeploymentManager
from .utils.performance import PerformanceBenchmark
from .utils.health import HealthChecker
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Enterprise Copywriting Service
==============================

Production-ready, enterprise-grade copywriting service with:
- 50+ optimization libraries with intelligent detection
- Multi-AI provider support (OpenRouter, OpenAI, Anthropic, Google)
- Advanced caching with compression (L1/L2/L3)
- JIT compilation and GPU acceleration
- Comprehensive monitoring and metrics
- Production deployment with Docker/Kubernetes
- Enterprise security and authentication
- Real-time performance optimization

Architecture:
- Clean modular design with separation of concerns
- Intelligent optimization detection with graceful fallbacks
- Multi-level caching for maximum performance
- Comprehensive monitoring and health checks
- Production-ready deployment infrastructure
"""

__version__ = "3.0.0"
__author__ = "Blatam Academy"
__description__ = "Enterprise-grade copywriting service with ultra-high performance"

# Core components
    CopywritingRequest,
    CopywritingResponse, 
    GenerationMetrics,
    BatchRequest,
    BatchResponse
)

# API and deployment

# Utilities

__all__ = [
    # Core
    "EnterpriseConfig",
    "get_config",
    "CopywritingRequest",
    "CopywritingResponse", 
    "GenerationMetrics",
    "BatchRequest",
    "BatchResponse",
    "EnterpriseCopywritingService",
    "OptimizationEngine",
    "EnterpriseCache",
    "MetricsEngine",
    "SecurityManager",
    
    # API & Deployment
    "create_enterprise_app",
    "DeploymentManager",
    
    # Utilities
    "PerformanceBenchmark",
    "HealthChecker",
    
    # Metadata
    "__version__",
    "__author__",
    "__description__"
]

# Package-level configuration
DEFAULT_CONFIG = {
    "optimization_level": "auto",  # auto, basic, optimized, ultra, maximum
    "cache_strategy": "multi_level",  # memory, redis, multi_level
    "monitoring_enabled": True,
    "security_enabled": True,
    "performance_tracking": True
}

def get_version_info():
    """Get detailed version information"""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "python_version": "3.11+",
        "optimization_libraries": "50+",
        "performance_gain": "up to 50x",
        "deployment": "Docker, Kubernetes, Bare Metal"
    } 