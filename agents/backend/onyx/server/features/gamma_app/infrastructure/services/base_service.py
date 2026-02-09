"""
Base Service
Base class for services with common functionality
"""

import logging
from typing import Optional, Dict, Any
from abc import ABC

from ...infrastructure.database.session import DatabaseSessionManager, get_db_manager
from ...infrastructure.redis.connection import RedisConnectionManager, get_redis_manager

logger = logging.getLogger(__name__)

class BaseService(ABC):
    """Base class for services with Redis and database support"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize base service"""
        self.config = config or {}
        self.redis_manager: Optional[RedisConnectionManager] = None
        self.db_manager: Optional[DatabaseSessionManager] = None
        self._initialized = False
    
    def _init_redis(self, decode_responses: bool = True, redis_db: Optional[int] = None) -> bool:
        """Initialize Redis connection"""
        try:
            redis_url = self.config.get('redis_url')
            if redis_db is not None and redis_url:
                redis_url = redis_url.rsplit('/', 1)[0] + f'/{redis_db}'
            
            self.redis_manager = get_redis_manager(redis_url, decode_responses)
            if self.redis_manager.client:
                logger.info(f"Redis connection established for {self.__class__.__name__}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Redis connection failed for {self.__class__.__name__}: {e}")
            return False
    
    def _init_database(self) -> bool:
        """Initialize database connection"""
        try:
            self.db_manager = get_db_manager()
            logger.info(f"Database connection established for {self.__class__.__name__}")
            return True
        except Exception as e:
            logger.error(f"Database connection failed for {self.__class__.__name__}: {e}")
            return False
    
    @property
    def redis_client(self) -> Optional[Any]:
        """Get Redis client"""
        if self.redis_manager:
            return self.redis_manager.get_client()
        return None
    
    async def close(self):
        """Close service connections"""
        if self.redis_manager:
            await self.redis_manager.close()







