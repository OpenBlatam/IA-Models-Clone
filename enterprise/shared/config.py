from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
from dataclasses import dataclass, field
from typing import List
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Enterprise Configuration
=======================

Configuration management for the enterprise API.
"""



@dataclass
class EnterpriseConfig:
    """Enterprise API configuration with all advanced features."""
    
    # Application
    app_name: str = "Enterprise API"
    app_version: str = "2.0.0"
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "production"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    
    # Security
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "change-me-in-production"))
    allowed_origins: List[str] = field(default_factory=lambda: os.getenv("ALLOWED_ORIGINS", "*").split(","))
    trusted_hosts: List[str] = field(default_factory=lambda: os.getenv("TRUSTED_HOSTS", "*").split(","))
    oauth2_scheme: str = "Bearer"
    
    # Performance
    max_workers: int = field(default_factory=lambda: int(os.getenv("MAX_WORKERS", "10")))
    connection_pool_size: int = field(default_factory=lambda: int(os.getenv("CONNECTION_POOL_SIZE", "50")))
    request_timeout: int = field(default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "30")))
    
    # Caching
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
    cache_ttl: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL", "3600")))
    cache_max_size: int = field(default_factory=lambda: int(os.getenv("CACHE_MAX_SIZE", "10000")))
    
    # Rate Limiting
    rate_limit_requests: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_REQUESTS", "1000")))
    rate_limit_window: int = field(default_factory=lambda: int(os.getenv("RATE_LIMIT_WINDOW", "3600")))
    
    # Circuit Breaker
    circuit_breaker_threshold: int = field(default_factory=lambda: int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")))
    circuit_breaker_timeout: int = field(default_factory=lambda: int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "60")))
    
    # Monitoring
    enable_metrics: bool = field(default_factory=lambda: os.getenv("ENABLE_METRICS", "true").lower() == "true")
    enable_tracing: bool = field(default_factory=lambda: os.getenv("ENABLE_TRACING", "true").lower() == "true")
    enable_health_checks: bool = field(default_factory=lambda: os.getenv("ENABLE_HEALTH_CHECKS", "true").lower() == "true")
    
    # Serverless Optimization
    cold_start_optimization: bool = field(default_factory=lambda: os.getenv("COLD_START_OPTIMIZATION", "true").lower() == "true")
    preload_dependencies: bool = field(default_factory=lambda: os.getenv("PRELOAD_DEPENDENCIES", "true").lower() == "true")
    lazy_loading: bool = field(default_factory=lambda: os.getenv("LAZY_LOADING", "true").lower() == "true")
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() in ["development", "dev"]
    
    def get_cors_config(self) -> dict:
        """Get CORS configuration."""
        return {
            "allow_origins": self.allowed_origins,
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
    
    def get_security_headers(self) -> dict:
        """Get security headers configuration."""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        } 