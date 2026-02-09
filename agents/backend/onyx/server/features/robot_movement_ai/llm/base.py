"""
Base LLM - Clase base para modelos de lenguaje
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseLLM(ABC):
    """Clase base abstracta para modelos de lenguaje"""
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Genera texto a partir de un prompt"""
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Genera embeddings para un texto"""
        pass
    
    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Genera respuesta en formato chat"""
        pass

