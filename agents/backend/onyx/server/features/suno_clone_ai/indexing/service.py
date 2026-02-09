"""Indexing Service - Servicio de indexación"""
from typing import Optional
from .base import BaseIndexer
from db.service import DatabaseService
from key_value_store.service import KeyValueStoreService

class IndexingService:
    def __init__(self, db_service: Optional[DatabaseService] = None, kv_store: Optional[KeyValueStoreService] = None):
        self.db_service = db_service
        self.kv_store = kv_store

