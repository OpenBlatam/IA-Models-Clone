"""
Base Prompt - Clase base para prompts
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BasePrompt(ABC):
    """Clase base abstracta para prompts"""

    @abstractmethod
    def build(self, **kwargs) -> str:
        """Construye el prompt con los parámetros dados"""
        pass

    @abstractmethod
    def validate(self, **kwargs) -> bool:
        """Valida los parámetros del prompt"""
        pass

