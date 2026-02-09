"""
Base LLM - Clase base para LLMs
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseLLM(ABC):
    """Clase base abstracta para modelos de lenguaje"""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Genera texto a partir de un prompt"""
        pass

    @abstractmethod
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Genera texto para múltiples prompts"""
        pass

    @abstractmethod
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Genera respuesta en formato chat"""
        pass

