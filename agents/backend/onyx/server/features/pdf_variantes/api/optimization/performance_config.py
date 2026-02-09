"""
Performance Configuration
Centralized performance tuning
"""

from typing import Dict, Any
import os


class PerformanceConfig:
    """Performance configuration"""
    
    # Cache settings
    CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    CACHE_DEFAULT_TTL = int(os.getenv("CACHE_DEFAULT_TTL", "60"))
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
    
    # Compression
    COMPRESSION_ENABLED = os.getenv("COMPRESSION_ENABLED", "true").lower() == "true"
    COMPRESSION_THRESHOLD = int(os.getenv("COMPRESSION_THRESHOLD", "1024"))
    COMPRESSION_LEVEL = int(os.getenv("COMPRESSION_LEVEL", "6"))
    
    # Async settings
    MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "100"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))
    
    # Database
    DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "20"))
    DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "40"))
    DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    
    # Query optimization
    ENABLE_QUERY_CACHE = os.getenv("ENABLE_QUERY_CACHE", "true").lower() == "true"
    QUERY_CACHE_TTL = int(os.getenv("QUERY_CACHE_TTL", "300"))
    
    # Response optimization
    MINIFY_RESPONSES = os.getenv("MINIFY_RESPONSES", "false").lower() == "true"
    ENABLE_ETAGS = os.getenv("ENABLE_ETAGS", "true").lower() == "true"
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get all configuration"""
        return {
            "cache": {
                "enabled": cls.CACHE_ENABLED,
                "default_ttl": cls.CACHE_DEFAULT_TTL,
                "max_size": cls.CACHE_MAX_SIZE
            },
            "compression": {
                "enabled": cls.COMPRESSION_ENABLED,
                "threshold": cls.COMPRESSION_THRESHOLD,
                "level": cls.COMPRESSION_LEVEL
            },
            "async": {
                "max_concurrent": cls.MAX_CONCURRENT,
                "batch_size": cls.BATCH_SIZE
            },
            "database": {
                "pool_size": cls.DB_POOL_SIZE,
                "max_overflow": cls.DB_MAX_OVERFLOW,
                "pool_timeout": cls.DB_POOL_TIMEOUT
            },
            "query": {
                "cache_enabled": cls.ENABLE_QUERY_CACHE,
                "cache_ttl": cls.QUERY_CACHE_TTL
            },
            "response": {
                "minify": cls.MINIFY_RESPONSES,
                "enable_etags": cls.ENABLE_ETAGS
            }
        }






