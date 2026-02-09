from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from .interfaces import HTMLParser, HTTPClient, CacheManager, SEOAnalyzer
from .parsers import SelectolaxUltraParser, LXMLFallbackParser
from .http_client import UltraFastHTTPClient
from .cache_manager import UltraOptimizedCacheManager
from .analyzer import UltraFastSEOAnalyzer
from .metrics import SEOMetrics, PerformanceTracker
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Core module for ultra-optimized SEO service.
Contains the fundamental components and interfaces.
"""


__all__ = [
    'HTMLParser',
    'HTTPClient', 
    'CacheManager',
    'SEOAnalyzer',
    'SelectolaxUltraParser',
    'LXMLFallbackParser',
    'UltraFastHTTPClient',
    'UltraOptimizedCacheManager',
    'UltraFastSEOAnalyzer',
    'SEOMetrics',
    'PerformanceTracker'
] 