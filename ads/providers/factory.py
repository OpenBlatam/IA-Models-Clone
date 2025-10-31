from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import Dict, Type, Optional
from .base import BaseProvider
from .openai_provider import OpenAIProvider
from .cohere_provider import CohereProvider
from ..config.providers import ProvidersConfig, OpenAIConfig, CohereConfig
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Factory for creating AI providers.
"""

class ProviderFactory:
    """Factory for creating AI providers."""
    
    _providers: Dict[str, Type[BaseProvider]] = {
        "openai": OpenAIProvider,
        "cohere": CohereProvider
    }
    
    _instances: Dict[str, BaseProvider] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseProvider]) -> None:
        """Register a new provider."""
        cls._providers[name] = provider_class
    
    @classmethod
    def get_provider(cls, name: str, config: Optional[ProvidersConfig] = None) -> BaseProvider:
        """Get a provider instance."""
        if name not in cls._providers:
            raise ValueError(f"Unknown provider: {name}")
        
        if name not in cls._instances:
            provider_class = cls._providers[name]
            if config is None:
                config = ProvidersConfig()
            
            if name == "openai":
                provider_config = config.openai
            elif name == "cohere":
                provider_config = config.cohere
            else:
                raise ValueError(f"Unknown provider config: {name}")
            
            cls._instances[name] = provider_class(provider_config)
        
        return cls._instances[name]
    
    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available providers."""
        return list(cls._providers.keys())
    
    @classmethod
    def clear_instances(cls) -> None:
        """Clear all provider instances."""
        cls._instances.clear() 