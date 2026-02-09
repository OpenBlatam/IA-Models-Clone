"""Knowledge Graph Service - Servicio de knowledge graph"""
from typing import Optional
from .base import BaseKnowledgeGraph
from db.service import DatabaseService
from indexing.service import IndexingService
from llm.service import LLMService

class KnowledgeGraphService:
    def __init__(self, db_service: Optional[DatabaseService] = None, indexing_service: Optional[IndexingService] = None, llm_service: Optional[LLMService] = None):
        self.db_service = db_service
        self.indexing_service = indexing_service
        self.llm_service = llm_service

