"""
LLM Factory - Factory para clientes LLM
========================================
"""

import os
import logging
from typing import Optional, Dict, Any
from enum import Enum

from .core_utils import get_logger
from .optional_imports import get_openai, get_anthropic

logger = get_logger(__name__)


class LLMProvider(Enum):
    """Proveedores de LLM disponibles"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    NONE = "none"


class LLMFactory:
    """Factory para crear clientes LLM"""
    
    @staticmethod
    def create_client(provider: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Crea un cliente LLM basado en configuración.
        
        Args:
            provider: Proveedor específico (opcional, auto-detecta)
            
        Returns:
            Dict con información del cliente o None
        """
        if provider:
            return LLMFactory._create_specific_client(provider)
        
        # Auto-detect
        # Try OpenAI first
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and get_openai():
            return LLMFactory._create_openai_client(openai_key)
        
        # Try Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key and get_anthropic():
            return LLMFactory._create_anthropic_client(anthropic_key)
        
        logger.warning("No se encontró cliente LLM configurado")
        return None
    
    @staticmethod
    def _create_specific_client(provider: str) -> Optional[Dict[str, Any]]:
        """Crea cliente específico"""
        provider_lower = provider.lower()
        
        if provider_lower == "openai":
            key = os.getenv("OPENAI_API_KEY")
            if key:
                return LLMFactory._create_openai_client(key)
        
        elif provider_lower == "anthropic":
            key = os.getenv("ANTHROPIC_API_KEY")
            if key:
                return LLMFactory._create_anthropic_client(key)
        
        return None
    
    @staticmethod
    def _create_openai_client(api_key: str) -> Dict[str, Any]:
        """Crea cliente OpenAI"""
        try:
            openai = get_openai()
            if not openai:
                return None
            
            return {
                "provider": LLMProvider.OPENAI,
                "client": "openai",
                "api_key": api_key,
                "available": True
            }
        except Exception as e:
            logger.error(f"Error creando cliente OpenAI: {e}")
            return None
    
    @staticmethod
    def _create_anthropic_client(api_key: str) -> Dict[str, Any]:
        """Crea cliente Anthropic"""
        try:
            anthropic = get_anthropic()
            if not anthropic:
                return None
            
            return {
                "provider": LLMProvider.ANTHROPIC,
                "client": "anthropic",
                "api_key": api_key,
                "available": True
            }
        except Exception as e:
            logger.error(f"Error creando cliente Anthropic: {e}")
            return None
    
    @staticmethod
    def is_available(provider: str) -> bool:
        """Verifica si un proveedor está disponible"""
        provider_lower = provider.lower()
        
        if provider_lower == "openai":
            return get_openai() is not None and os.getenv("OPENAI_API_KEY") is not None
        
        if provider_lower == "anthropic":
            return get_anthropic() is not None and os.getenv("ANTHROPIC_API_KEY") is not None
        
        return False

