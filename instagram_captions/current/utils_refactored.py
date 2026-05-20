"""
Instagram Captions API v10.0 - Refactored Utilities Module

This is the new, modular version that imports from focused modules.
"""

# Import from new modular structure
from .security import SecurityUtils
from .monitoring import PerformanceMonitor
from .resilience import CircuitBreaker, ErrorHandler
from .core import (
    setup_logging, get_logger, CacheManager, RateLimiter,
    rate_limit_middleware, logging_middleware, security_middleware
)

# Export all utilities
__all__ = [
    # Security
    'SecurityUtils',
    
    # Monitoring
    'PerformanceMonitor',
    
    # Resilience
    'CircuitBreaker',
    'ErrorHandler',
    
    # Core
    'setup_logging',
    'get_logger',
    'CacheManager',
    'RateLimiter',
    'rate_limit_middleware',
    'logging_middleware',
    'security_middleware'
]

# Backward compatibility - import everything for existing code
# This allows gradual migration to the new modular structure






