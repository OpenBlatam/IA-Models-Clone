"""NLP Service - Servicio de NLP"""
from typing import Optional
from .base import BaseNLP
from llm.service import LLMService
from utils.service import UtilService

class NLPService:
    def __init__(self, llm_service: Optional[LLMService] = None, util_service: Optional[UtilService] = None):
        self.llm_service = llm_service
        self.util_service = util_service

