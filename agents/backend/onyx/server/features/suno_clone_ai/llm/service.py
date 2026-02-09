"""
LLM Service - Servicio de LLM
"""

from typing import List, Dict, Any, Optional
from .base import BaseLLM
from .provider import LLMProvider
from .generator import TextGenerator
from prompts.service import PromptService
from tracing.service import TracingService


class LLMService:
    """Servicio para gestionar LLMs"""

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        prompt_service: Optional[PromptService] = None,
        tracing_service: Optional[TracingService] = None
    ):
        """Inicializa el servicio de LLM"""
        self.provider = provider or LLMProvider()
        self.prompt_service = prompt_service
        self.tracing_service = tracing_service
        self.generator = TextGenerator(self.provider)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Genera texto a partir de un prompt"""
        if self.tracing_service:
            self.tracing_service.trace("llm.generate", {"prompt_length": len(prompt)})
        return await self.generator.generate(prompt, **kwargs)

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Genera respuesta en formato chat"""
        if self.tracing_service:
            self.tracing_service.trace("llm.chat", {"message_count": len(messages)})
        return await self.generator.chat(messages, **kwargs)

