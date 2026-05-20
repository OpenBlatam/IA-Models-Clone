from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Enterprise Constants
===================

System-wide constants for the enterprise API.
"""

# API Constants
API_PREFIX = "/api/v1"
HEALTH_CHECK_PATH = "/health"
METRICS_PATH = "/metrics"
DOCS_PATH = "/docs"

# Cache Keys
CACHE_KEY_PREFIX = "enterprise:"
CACHE_KEY_METRICS = f"{CACHE_KEY_PREFIX}metrics"
CACHE_KEY_HEALTH = f"{CACHE_KEY_PREFIX}health"

# Rate Limiting
RATE_LIMIT_KEY_PREFIX = "rate_limit:"
RATE_LIMIT_DEFAULT_WINDOW = 3600
RATE_LIMIT_DEFAULT_REQUESTS = 1000

# Circuit Breaker
CIRCUIT_BREAKER_DEFAULT_THRESHOLD = 5
CIRCUIT_BREAKER_DEFAULT_TIMEOUT = 60
CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS = 5

# Metrics
METRICS_COLLECTION_INTERVAL = 60
METRICS_RETENTION_PERIOD = 86400  # 24 hours

# Health Checks
HEALTH_CHECK_CACHE_TTL = 30
HEALTH_CHECK_TIMEOUT = 5

# HTTP Status Codes
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVICE_UNAVAILABLE = 503
HTTP_INTERNAL_SERVER_ERROR = 500

# Logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Performance
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
MAX_RESPONSE_SIZE = 50 * 1024 * 1024  # 50MB
DEFAULT_TIMEOUT = 30 