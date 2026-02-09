"""Seeding Service - Servicio de seeding"""
from typing import Optional
from .base import BaseSeeder
from db.service import DatabaseService
from configs.settings import Settings

class SeedingService:
    def __init__(self, db_service: Optional[DatabaseService] = None, settings: Optional[Settings] = None):
        self.db_service = db_service
        self.settings = settings or Settings()

