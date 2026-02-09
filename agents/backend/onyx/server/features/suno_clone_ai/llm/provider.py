"""
LLM Provider - Proveedores de LLM
"""

from typing import Dict, Any, Optional
from .base import BaseLLM


class LLMProvider:
    """Proveedor de modelos de lenguaje"""

    def __init__(self, provider_type: str = "openai"):
        """Inicializa el proveedor de LLM"""
        self.provider_type = provider_type
        self._providers: Dict[str, BaseLLM] = {}

    def register_provider(self, name: str, provider: BaseLLM) -> None:
        """Registra un proveedor de LLM"""
        self._providers[name] = provider

    def get_provider(self, name: Optional[str] = None) -> Optional[BaseLLM]:
        """Obtiene un proveedor por nombre"""
        provider_name = name or self.provider_type
        return self._providers.get(provider_name)

