"""
Serverless configuration and optimizations for Azure Functions
"""

import os
import logging
from typing import Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


class AzureServerlessConfig:
    """Configuration for Azure Functions serverless environments"""
    
    # Function-specific settings
    FUNCTION_TIMEOUT = int(os.getenv("FUNCTION_TIMEOUT", "600"))  # 10 minutes max
    FUNCTION_MEMORY = int(os.getenv("FUNCTION_MEMORY", "1536"))  # 1.5GB default
    
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
    APPLICATION_INSIGHTS_ENABLED = os.getenv("APPLICATION_INSIGHTS_ENABLED", "true").lower() == "true"
    ENABLE_DISTRIBUTED_TRACING = os.getenv("ENABLE_DISTRIBUTED_TRACING", "true").lower() == "true"
    
    # Azure-specific
    AZURE_STORAGE_CONNECTION = os.getenv("AzureWebJobsStorage", "")
    AZURE_FUNCTIONS_ENVIRONMENT = os.getenv("AZURE_FUNCTIONS_ENVIRONMENT", "production")
    
    @classmethod
    @lru_cache(maxsize=1)
    def get_config(cls) -> Dict[str, Any]:
        """Get Azure serverless configuration"""
        return {
            "function": {
                "timeout": cls.FUNCTION_TIMEOUT,
                "memory": cls.FUNCTION_MEMORY
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
                "application_insights": cls.APPLICATION_INSIGHTS_ENABLED,
                "distributed_tracing": cls.ENABLE_DISTRIBUTED_TRACING
            },
            "azure": {
                "storage_connection": bool(cls.AZURE_STORAGE_CONNECTION),
                "environment": cls.AZURE_FUNCTIONS_ENVIRONMENT
            }
        }


def optimize_for_azure_functions():
    """Apply Azure Functions-specific optimizations"""
    # Set environment variables for optimization
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    
    # Azure Functions specific
    os.environ["FUNCTIONS_WORKER_RUNTIME"] = "python"
    os.environ["FUNCTIONS_EXTENSION_VERSION"] = "~4"
    
    # Reduce logging verbosity in production
    if os.getenv("AZURE_FUNCTIONS_ENVIRONMENT") == "production":
        logging.getLogger().setLevel(logging.WARNING)
    
    logger.info("Azure Functions optimizations applied")


# Initialize on import
optimize_for_azure_functions()




