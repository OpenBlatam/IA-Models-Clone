"""
CDN Integration Optimizer
Hints and headers for CDN optimization
"""

import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """CDN cache levels"""
    NO_CACHE = "no-cache"
    PRIVATE = "private"
    PUBLIC = "public"
    IMMUTABLE = "immutable"


@dataclass
class CDNConfig:
    """CDN configuration"""
    cache_level: CacheLevel = CacheLevel.PUBLIC
    max_age: int = 3600  # seconds
    stale_while_revalidate: int = 86400  # seconds
    stale_if_error: int = 604800  # seconds
    must_revalidate: bool = False
    no_transform: bool = True


class CDNOptimizer:
    """
    CDN integration optimizer
    
    Features:
    - Cache control headers
    - Surrogate control headers
    - Cache tags
    - Purge hints
    - Edge optimization
    """
    
    def __init__(self):
        self._cache_configs: Dict[str, CDNConfig] = {}
        self._cache_tags: Dict[str, List[str]] = {}
        logger.info("✅ CDN optimizer initialized")
    
    def configure_endpoint(
        self,
        endpoint: str,
        config: CDNConfig
    ):
        """
        Configure CDN settings for endpoint
        
        Args:
            endpoint: API endpoint path
            config: CDN configuration
        """
        self._cache_configs[endpoint] = config
        logger.debug(f"Configured CDN for endpoint: {endpoint}")
    
    def get_cache_headers(
        self,
        endpoint: str,
        custom_config: Optional[CDNConfig] = None
    ) -> Dict[str, str]:
        """
        Get CDN cache headers for endpoint
        
        Args:
            endpoint: API endpoint path
            custom_config: Optional custom configuration
            
        Returns:
            Dictionary of cache headers
        """
        config = custom_config or self._cache_configs.get(endpoint)
        
        if not config:
            # Default configuration
            config = CDNConfig()
        
        headers = {}
        
        # Cache-Control
        cache_control_parts = []
        
        if config.cache_level == CacheLevel.NO_CACHE:
            cache_control_parts.append("no-cache")
        elif config.cache_level == CacheLevel.PRIVATE:
            cache_control_parts.append("private")
        elif config.cache_level == CacheLevel.PUBLIC:
            cache_control_parts.append("public")
        elif config.cache_level == CacheLevel.IMMUTABLE:
            cache_control_parts.append("public")
            cache_control_parts.append("immutable")
        
        cache_control_parts.append(f"max-age={config.max_age}")
        
        if config.stale_while_revalidate:
            cache_control_parts.append(f"stale-while-revalidate={config.stale_while_revalidate}")
        
        if config.stale_if_error:
            cache_control_parts.append(f"stale-if-error={config.stale_if_error}")
        
        if config.must_revalidate:
            cache_control_parts.append("must-revalidate")
        
        if config.no_transform:
            cache_control_parts.append("no-transform")
        
        headers["Cache-Control"] = ", ".join(cache_control_parts)
        
        # Surrogate-Control (for CDN-specific control)
        surrogate_parts = [f"max-age={config.max_age}"]
        if config.stale_while_revalidate:
            surrogate_parts.append(f"stale-while-revalidate={config.stale_while_revalidate}")
        
        headers["Surrogate-Control"] = ", ".join(surrogate_parts)
        
        # Cache tags
        if endpoint in self._cache_tags:
            tags = self._cache_tags[endpoint]
            headers["Cache-Tag"] = ", ".join(tags)
        
        return headers
    
    def add_cache_tag(self, endpoint: str, tag: str):
        """
        Add cache tag for endpoint
        
        Args:
            endpoint: API endpoint path
            tag: Cache tag
        """
        if endpoint not in self._cache_tags:
            self._cache_tags[endpoint] = []
        
        if tag not in self._cache_tags[endpoint]:
            self._cache_tags[endpoint].append(tag)
    
    def get_purge_hints(self, endpoint: str) -> List[str]:
        """
        Get cache purge hints for endpoint
        
        Args:
            endpoint: API endpoint path
            
        Returns:
            List of cache tags to purge
        """
        return self._cache_tags.get(endpoint, [])


# Global optimizer instance
_cdn_optimizer: Optional[CDNOptimizer] = None


def get_cdn_optimizer() -> CDNOptimizer:
    """Get global CDN optimizer instance"""
    global _cdn_optimizer
    if _cdn_optimizer is None:
        _cdn_optimizer = CDNOptimizer()
    return _cdn_optimizer















