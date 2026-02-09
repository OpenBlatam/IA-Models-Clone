"""
Cold Start Optimization for Serverless
Techniques to minimize Lambda cold start times
"""

import logging
import sys
from typing import Any, Dict
from functools import lru_cache

logger = logging.getLogger(__name__)


class ColdStartOptimizer:
    """
    Optimize cold starts for serverless deployment
    
    Techniques:
    - Lazy imports
    - Pre-warming
    - Connection pooling
    - Model preloading
    """
    
    def __init__(self):
        self._initialized = False
        self._warm_connections: Dict[str, Any] = {}
    
    def optimize_imports(self) -> None:
        """Optimize imports for faster cold starts"""
        # Use lazy imports for heavy libraries
        # Import only when needed
        pass
    
    @lru_cache(maxsize=1)
    def get_app(self):
        """Get FastAPI app with caching"""
        # Lazy import main app
        from main import app
        return app
    
    def preload_models(self) -> None:
        """Preload AI models on cold start"""
        try:
            from config.aws_settings import get_aws_settings
            settings = get_aws_settings()
            
            if settings.preload_models:
                logger.info("Preloading AI models...")
                # Lazy import models
                from core.ultra_fast_engine import create_ultra_fast_engine
                engine = create_ultra_fast_engine()
                logger.info("AI models preloaded")
        except Exception as e:
            logger.warning(f"Failed to preload models: {str(e)}")
    
    def warm_connections(self) -> None:
        """Warm up connections to external services"""
        try:
            from core.service_container import get_container
            
            container = get_container()
            
            # Warm storage connection
            try:
                storage = container.get_storage_service()
                self._warm_connections["storage"] = storage
            except Exception as e:
                logger.warning(f"Failed to warm storage: {str(e)}")
            
            # Warm cache connection
            try:
                cache = container.get_cache_service()
                self._warm_connections["cache"] = cache
            except Exception as e:
                logger.warning(f"Failed to warm cache: {str(e)}")
            
            logger.info("Connections warmed up")
        except Exception as e:
            logger.warning(f"Failed to warm connections: {str(e)}")
    
    def initialize(self) -> None:
        """Initialize optimizer"""
        if self._initialized:
            return
        
        logger.info("Initializing cold start optimizer...")
        
        # Preload models
        self.preload_models()
        
        # Warm connections
        self.warm_connections()
        
        self._initialized = True
        logger.info("Cold start optimizer initialized")


# Global optimizer instance
_optimizer: ColdStartOptimizer = None


def get_optimizer() -> ColdStartOptimizer:
    """Get global cold start optimizer"""
    global _optimizer
    if _optimizer is None:
        _optimizer = ColdStartOptimizer()
    return _optimizer


# Initialize on module import (for Lambda)
def init_cold_start() -> None:
    """Initialize cold start optimizations"""
    optimizer = get_optimizer()
    optimizer.initialize()


# Auto-initialize if in Lambda
if sys.platform != "win32":  # Lambda runs on Linux
    try:
        import os
        if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
            init_cold_start()
    except Exception:
        pass















