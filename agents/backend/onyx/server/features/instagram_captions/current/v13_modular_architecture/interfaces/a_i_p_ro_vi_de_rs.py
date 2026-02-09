from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from ..domain.entities import CaptionRequest, CaptionResponse, CaptionStyle
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Instagram Captions API v13.0 - AI Provider Interfaces

Interfaces for AI service providers following dependency inversion principle.
Domain defines contracts, infrastructure implements them.
"""



class IAIProvider(ABC):
    """Interface for AI caption generation providers."""
    
    @abstractmethod
    async def generate_caption(
        self, 
        content_description: str,
        style: CaptionStyle,
        custom_instructions: Optional[str] = None
    ) -> str:
        """Generate a caption using AI."""
        pass
    
    @abstractmethod
    async def generate_hashtags(
        self,
        content_description: str,
        caption: str,
        count: int = 20
    ) -> List[str]:
        """Generate hashtags using AI."""
        pass
    
    @abstractmethod
    async def analyze_sentiment(self, caption: str) -> Dict[str, Any]:
        """Analyze sentiment of generated caption."""
        pass
    
    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the AI provider."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if AI provider is healthy and available."""
        pass


class ITransformersProvider(IAIProvider):
    """Interface for Transformers-based AI providers."""
    
    @abstractmethod
    async def load_model(self, model_name: str) -> None:
        """Load specific transformer model."""
        pass
    
    @abstractmethod
    async def unload_model(self) -> None:
        """Unload current model to free memory."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model."""
        pass


class IOpenAIProvider(IAIProvider):
    """Interface for OpenAI-based providers."""
    
    @abstractmethod
    async async def set_api_key(self, api_key: str) -> None:
        """Set OpenAI API key."""
        pass
    
    @abstractmethod
    async def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        pass


class IClaudeProvider(IAIProvider):
    """Interface for Claude AI provider."""
    
    @abstractmethod
    async def set_anthropic_key(self, api_key: str) -> None:
        """Set Anthropic API key."""
        pass


class IAIProviderFactory(ABC):
    """Factory interface for creating AI providers."""
    
    @abstractmethod
    def create_transformers_provider(
        self, 
        model_name: str = "distilgpt2"
    ) -> ITransformersProvider:
        """Create a Transformers-based provider."""
        pass
    
    @abstractmethod
    def create_openai_provider(self, api_key: str) -> IOpenAIProvider:
        """Create an OpenAI provider."""
        pass
    
    @abstractmethod
    def create_claude_provider(self, api_key: str) -> IClaudeProvider:
        """Create a Claude provider."""
        pass
    
    @abstractmethod
    def create_fallback_provider(self) -> IAIProvider:
        """Create a fallback provider for when others fail."""
        pass
    
    @abstractmethod
    def get_available_providers(self) -> List[str]:
        """Get list of available provider types."""
        pass


class IAIProviderRegistry(ABC):
    """Registry interface for managing multiple AI providers."""
    
    @abstractmethod
    async def register_provider(self, name: str, provider: IAIProvider) -> None:
        """Register an AI provider."""
        pass
    
    @abstractmethod
    async def unregister_provider(self, name: str) -> None:
        """Unregister an AI provider."""
        pass
    
    @abstractmethod
    async def get_provider(self, name: str) -> Optional[IAIProvider]:
        """Get provider by name."""
        pass
    
    @abstractmethod
    async def get_best_available_provider(self) -> IAIProvider:
        """Get the best available provider based on health and performance."""
        pass
    
    @abstractmethod
    async def health_check_all_providers(self) -> Dict[str, bool]:
        """Check health of all registered providers."""
        pass


class IAILoadBalancer(ABC):
    """Interface for load balancing between multiple AI providers."""
    
    @abstractmethod
    async def select_provider(self, request: CaptionRequest) -> IAIProvider:
        """Select the best provider for the given request."""
        pass
    
    @abstractmethod
    async def record_provider_performance(
        self, 
        provider_name: str,
        response_time: float,
        success: bool
    ) -> None:
        """Record performance metrics for a provider."""
        pass
    
    @abstractmethod
    async def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all providers."""
        pass


class IAIFallbackChain(ABC):
    """Interface for AI provider fallback chain."""
    
    @abstractmethod
    async def execute_with_fallback(
        self, 
        request: CaptionRequest
    ) -> CaptionResponse:
        """Execute request with automatic fallback on failure."""
        pass
    
    @abstractmethod
    async def add_fallback_provider(self, provider: IAIProvider, priority: int) -> None:
        """Add provider to fallback chain with priority."""
        pass
    
    @abstractmethod
    async def remove_fallback_provider(self, provider: IAIProvider) -> None:
        """Remove provider from fallback chain."""
        pass
    
    @abstractmethod
    async def get_fallback_chain_status(self) -> List[Dict[str, Any]]:
        """Get status of all providers in fallback chain."""
        pass 