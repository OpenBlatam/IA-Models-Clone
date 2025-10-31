from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .cache_interface import ICacheService
from .metrics_interface import IMetricsService
from .health_interface import IHealthService
from .rate_limit_interface import IRateLimitService
from .circuit_breaker_interface import ICircuitBreaker
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Domain Interfaces
=================

Abstract interfaces defining contracts for the enterprise API.
These interfaces follow the Dependency Inversion Principle.
"""


__all__ = [
    "ICacheService",
    "IMetricsService", 
    "IHealthService",
    "IRateLimitService",
    "ICircuitBreaker",
] 