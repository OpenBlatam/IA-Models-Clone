"""
Perplexity Query Processing System
===================================

A modular system for processing queries in Perplexity style with proper
formatting, citations, and query type detection.
"""

from .types import QueryType, SearchResult, ProcessedQuery
from .detector import QueryTypeDetector
from .formatter import ResponseFormatter
from .citations import CitationManager
from .prompt_builder import PromptBuilder
from .processor import PerplexityProcessor
from .validator import PerplexityValidator, ValidationIssue, ValidationLevel
from .cache import PerplexityCache, CacheEntry
from .metrics import PerplexityMetrics, QueryMetrics
from .service import PerplexityService
from .config import PerplexityConfig
from .exceptions import (
    PerplexityError,
    QueryProcessingError,
    LLMProviderError,
    ValidationError,
    CacheError,
    FormattingError,
    CitationError
)
try:
    from .middleware import (
        timing_middleware,
        error_handling_middleware,
        metrics_middleware,
        RateLimiter
    )
except ImportError:
    timing_middleware = None
    error_handling_middleware = None
    metrics_middleware = None
    RateLimiter = None

try:
    from .rate_limiter import (
        TokenBucketRateLimiter,
        SlidingWindowRateLimiter
    )
except ImportError:
    TokenBucketRateLimiter = None
    SlidingWindowRateLimiter = None
from . import utils
from . import helpers
from . import constants
from .version import __version__, __version_info__, __author__, __description__

__all__ = [

__all__ = [
    # Core types
    "QueryType",
    "SearchResult",
    "ProcessedQuery",
    
    # Core components
    "QueryTypeDetector",
    "ResponseFormatter",
    "CitationManager",
    "PromptBuilder",
    "PerplexityProcessor",
    "PerplexityService",
    "PerplexityValidator",
    "ValidationIssue",
    "ValidationLevel",
    
    # Infrastructure
    "PerplexityCache",
    "CacheEntry",
    "PerplexityMetrics",
    "QueryMetrics",
    "PerplexityConfig",
    
    # Exceptions
    "PerplexityError",
    "QueryProcessingError",
    "LLMProviderError",
    "ValidationError",
    "CacheError",
    "FormattingError",
    "CitationError",
    
    # Middleware
    "timing_middleware",
    "error_handling_middleware",
    "metrics_middleware",
    "RateLimiter",
    "TokenBucketRateLimiter",
    "SlidingWindowRateLimiter",
    
    # Utilities and helpers
    "utils",
    "helpers",
    "constants",
    
    # Version info
    "__version__",
    "__version_info__",
    "__author__",
    "__description__",
]
