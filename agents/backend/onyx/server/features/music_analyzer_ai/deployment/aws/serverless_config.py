"""
Serverless configuration and optimizations for AWS Lambda
"""

import os
import logging
from typing import Dict, Any, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


class ServerlessConfig:
    """Configuration for serverless environments"""
    
    # Lambda-specific settings
    LAMBDA_TIMEOUT = int(os.getenv("LAMBDA_TIMEOUT", "300"))  # 5 minutes max
    LAMBDA_MEMORY = int(os.getenv("LAMBDA_MEMORY", "1024"))  # 1GB default
    LAMBDA_HANDLER = "deployment.aws.lambda_handler.lambda_handler"
    
    # Cold start optimization
    ENABLE_LAZY_LOADING = os.getenv("ENABLE_LAZY_LOADING", "true").lower() == "true"
    PREWARM_ENABLED = os.getenv("PREWARM_ENABLED", "false").lower() == "true"
    
    # Connection pooling
    MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "10"))
    CONNECTION_TIMEOUT = int(os.getenv("CONNECTION_TIMEOUT", "5"))
    
    # Caching
    CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
    
    # Monitoring
    CLOUDWATCH_ENABLED = os.getenv("CLOUDWATCH_ENABLED", "true").lower() == "true"
    XRAY_TRACING = os.getenv("XRAY_TRACING", "false").lower() == "true"
    
    @classmethod
    @lru_cache(maxsize=1)
    def get_config(cls) -> Dict[str, Any]:
        """Get serverless configuration"""
        return {
            "lambda": {
                "timeout": cls.LAMBDA_TIMEOUT,
                "memory": cls.LAMBDA_MEMORY,
                "handler": cls.LAMBDA_HANDLER
            },
            "optimization": {
                "lazy_loading": cls.ENABLE_LAZY_LOADING,
                "prewarm": cls.PREWARM_ENABLED
            },
            "connections": {
                "max": cls.MAX_CONNECTIONS,
                "timeout": cls.CONNECTION_TIMEOUT
            },
            "cache": {
                "enabled": cls.CACHE_ENABLED,
                "ttl": cls.CACHE_TTL
            },
            "monitoring": {
                "cloudwatch": cls.CLOUDWATCH_ENABLED,
                "xray": cls.XRAY_TRACING
            }
        }


def optimize_for_lambda():
    """Apply Lambda-specific optimizations"""
    # Set environment variables for optimization
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    
    # Reduce logging verbosity in production
    if os.getenv("ENVIRONMENT") == "production":
        logging.getLogger().setLevel(logging.WARNING)
    
    logger.info("Lambda optimizations applied")


# Initialize on import
optimize_for_lambda()




