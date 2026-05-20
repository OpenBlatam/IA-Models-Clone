"""
Service Providers
=================

System for providing external services and dependencies.
"""

import logging
from typing import Dict, Any, Optional, Type, Callable
from abc import ABC, abstractmethod

from ..infrastructure import OpenRouterClient, TruthGPTClient
from ..config.enhancer_config import EnhancerConfig

logger = logging.getLogger(__name__)


class ServiceProvider(ABC):
    """Base class for service providers."""
    
    @abstractmethod
    def provide(self, config: Any) -> Any:
        """Provide service instance."""
        pass


class OpenRouterProvider(ServiceProvider):
    """Provider for OpenRouter client."""
    
    def provide(self, config: EnhancerConfig) -> OpenRouterClient:
        """Provide OpenRouter client."""
        return OpenRouterClient(api_key=config.openrouter.api_key)


class TruthGPTProvider(ServiceProvider):
    """Provider for TruthGPT client."""
    
    def provide(self, config: EnhancerConfig) -> TruthGPTClient:
        """Provide TruthGPT client."""
        return TruthGPTClient(
            config=config.truthgpt.to_dict() if config.truthgpt else {}
        )


class ProviderRegistry:
    """Registry for service providers."""
    
    def __init__(self):
        """Initialize provider registry."""
        self.providers: Dict[str, ServiceProvider] = {}
        self.instances: Dict[str, Any] = {}
    
    def register(
        self,
        name: str,
        provider: ServiceProvider,
        singleton: bool = True
    ):
        """
        Register a service provider.
        
        Args:
            name: Service name
            provider: Provider instance
            singleton: Whether to cache instance
        """
        self.providers[name] = provider
        logger.info(f"Registered provider: {name}")
    
    def get(
        self,
        name: str,
        config: Any,
        force_new: bool = False
    ) -> Any:
        """
        Get service instance from provider.
        
        Args:
            name: Service name
            config: Configuration for provider
            force_new: Force new instance (ignore singleton)
            
        Returns:
            Service instance
        """
        if name not in self.providers:
            raise ValueError(f"Provider not found: {name}")
        
        # Check singleton cache
        if not force_new and name in self.instances:
            return self.instances[name]
        
        # Get from provider
        provider = self.providers[name]
        instance = provider.provide(config)
        
        # Cache if singleton
        if not force_new:
            self.instances[name] = instance
        
        return instance
    
    def clear_cache(self, name: Optional[str] = None):
        """
        Clear provider cache.
        
        Args:
            name: Optional service name (clears all if not provided)
        """
        if name:
            self.instances.pop(name, None)
        else:
            self.instances.clear()




