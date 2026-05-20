"""
Infrastructure Module
====================

Consolidated infrastructure exports.
"""

# Base client
from .base_client import BaseHTTPClient

# Clients
from .openrouter_client import OpenRouterClient
from .truthgpt_client import TruthGPTClient

# Error handling
from .error_handlers import handle_openrouter_error

# Response parsing
from .response_parser import OpenRouterResponseParser

# Retry helpers
from .retry_helpers import (
    retry_with_exponential_backoff,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY
)

# TruthGPT helpers
from .truthgpt_helpers import (
    check_truthgpt_ready,
    safe_truthgpt_call
)

# TruthGPT status
from .truthgpt_status import TruthGPTStatus

# Client base
from .client_base import BaseAPIClient, RetryableClient, ClientConfig

# Response handler
from .response_handler import ResponseHandler, ResponseProcessor

__all__ = [
    # Base client
    "BaseHTTPClient",
    # Clients
    "OpenRouterClient",
    "TruthGPTClient",
    # Error handling
    "handle_openrouter_error",
    # Response parsing
    "OpenRouterResponseParser",
    # Retry helpers
    "retry_with_exponential_backoff",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_DELAY",
    # TruthGPT helpers
    "check_truthgpt_ready",
    "safe_truthgpt_call",
    # TruthGPT status
    "TruthGPTStatus",
    # Client base
    "BaseAPIClient",
    "RetryableClient",
    "ClientConfig",
    # Response handler
    "ResponseHandler",
    "ResponseProcessor",
]
