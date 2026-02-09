"""Secondary LLM Flow Service - Servicio de flujos secundarios"""
from typing import Optional
from .base import BaseSecondaryFlow
from llm.service import LLMService
from prompts.service import PromptService
from tracing.service import TracingService

class SecondaryLLMFlowService:
    def __init__(self, llm_service: Optional[LLMService] = None, prompt_service: Optional[PromptService] = None, tracing_service: Optional[TracingService] = None):
        self.llm_service = llm_service
        self.prompt_service = prompt_service
        self.tracing_service = tracing_service

