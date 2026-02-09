"""Eval Service - Servicio de evaluación"""
from typing import Optional
from .base import BaseEvaluator
from llm.service import LLMService
from db.service import DatabaseService
from tracing.service import TracingService

class EvalService:
    def __init__(self, llm_service: Optional[LLMService] = None, db_service: Optional[DatabaseService] = None, tracing_service: Optional[TracingService] = None):
        self.llm_service = llm_service
        self.db_service = db_service
        self.tracing_service = tracing_service

