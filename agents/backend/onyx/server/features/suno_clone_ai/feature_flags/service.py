"""Feature Flag Service - Servicio de feature flags"""
from typing import Optional
from .base import BaseFeatureFlag
from redis.service import RedisService
from db.service import DatabaseService
from configs.settings import Settings

class FeatureFlagService:
    def __init__(self, redis_service: Optional[RedisService] = None, db_service: Optional[DatabaseService] = None, settings: Optional[Settings] = None):
        self.redis_service = redis_service
        self.db_service = db_service
        self.settings = settings or Settings()

