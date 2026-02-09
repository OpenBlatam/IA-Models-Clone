"""
Text Generator - Generador de texto
"""

from typing import List, Dict, Any, Optional
from .base import BaseLLM
from .provider import LLMProvider


class TextGenerator:
    """Generador de texto usando LLMs"""

    def __init__(self, provider: LLMProvider):
        """Inicializa el generador de texto"""
        self.provider = provider

    async def generate(self, prompt: str, **kwargs) -> str:
        """Genera texto a partir de un prompt"""
        llm = self.provider.get_provider()
        if not llm:
            raise ValueError("No LLM provider available")
        return await llm.generate(prompt, **kwargs)

    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Genera respuesta en formato chat"""
        llm = self.provider.get_provider()
        if not llm:
            raise ValueError("No LLM provider available")
        return await llm.chat(messages, **kwargs)

