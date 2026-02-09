"""Document Index Service - Servicio de indexación"""
from typing import Optional
from .base import BaseDocumentIndex
from llm.service import LLMService
from indexing.service import IndexingService
from db.service import DatabaseService

class DocumentIndexService:
    def __init__(self, llm_service: Optional[LLMService] = None, indexing_service: Optional[IndexingService] = None, db_service: Optional[DatabaseService] = None):
        self.llm_service = llm_service
        self.indexing_service = indexing_service
        self.db_service = db_service

