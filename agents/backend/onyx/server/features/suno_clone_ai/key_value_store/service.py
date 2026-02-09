"""Key Value Store Service - Servicio de KV store"""
from typing import Optional
from .base import BaseKeyValueStore
from redis.service import RedisService
from configs.settings import Settings

class KeyValueStoreService:
    def __init__(self, redis_service: Optional[RedisService] = None, settings: Optional[Settings] = None):
        self.redis_service = redis_service
        self.settings = settings or Settings()

