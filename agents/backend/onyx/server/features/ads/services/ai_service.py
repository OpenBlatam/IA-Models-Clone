from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import List, Dict, Any, Optional
import logging
from ..providers.factory import ProviderFactory
from ..providers.base import BaseProvider
from ..config.providers import ProvidersConfig
from typing import Any, List, Dict, Optional
import asyncio
"""
AI service for handling AI operations.
"""

logger = logging.getLogger(__name__)

class AIService:
    """Service for handling AI operations."""
    
    def __init__(self, config: Optional[ProvidersConfig] = None):
        
    """__init__ function."""
self.config = config or ProvidersConfig()
        self.primary_provider = self.config.default_provider
        self.fallback_provider = self.config.fallback_provider
        self.logger = logger
    
    async def _get_provider(self, provider_name: str) -> BaseProvider:
        """Get a provider instance."""
        try:
            provider = ProviderFactory.get_provider(provider_name, self.config)
            await provider.initialize()
            return provider
        except Exception as e:
            self.logger.error(f"Failed to initialize provider {provider_name}: {e}")
            raise
    
    async def _try_with_fallback(self, operation: str, *args, **kwargs) -> Any:
        """Try operation with primary provider, fallback to secondary if needed."""
        try:
            provider = await self._get_provider(self.primary_provider)
            method = getattr(provider, operation)
            return await method(*args, **kwargs)
        except Exception as e:
            self.logger.warning(f"Primary provider failed: {e}")
            try:
                provider = await self._get_provider(self.fallback_provider)
                method = getattr(provider, operation)
                return await method(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Fallback provider failed: {e}")
                raise
    
    async def generate_ads(self, content: str, num_ads: int = 3) -> List[str]:
        """Generate ads using AI."""
        return await self._try_with_fallback(
            "generate_variations",
            content,
            num_ads,
            system_prompt="You are an expert copywriter for social media ads. Generate concise, engaging ads based on the content."
        )
    
    async def analyze_brand_voice(self, content: str) -> Dict[str, Any]:
        """Analyze brand voice using AI."""
        return await self._try_with_fallback(
            "analyze_text",
            content,
            system_prompt="You are an expert in brand voice analysis. Analyze the content and provide insights about the brand voice."
        )
    
    async def optimize_content(self, content: str, target_audience: str) -> str:
        """Optimize content for target audience using AI."""
        return await self._try_with_fallback(
            "optimize_text",
            content,
            target_audience,
            system_prompt="You are an expert in content optimization. Optimize the content for the target audience."
        )
    
    async def generate_content_variations(self, content: str, num_variations: int = 3) -> List[str]:
        """Generate content variations using AI."""
        return await self._try_with_fallback(
            "generate_variations",
            content,
            num_variations,
            system_prompt="You are an expert in content creation. Generate variations while maintaining the core message."
        )
    
    async def analyze_audience(self, content: str) -> Dict[str, Any]:
        """Analyze audience from content using AI."""
        return await self._try_with_fallback(
            "analyze_text",
            content,
            system_prompt="You are an expert in audience analysis. Analyze the content and provide insights about the target audience."
        )
    
    async def generate_recommendations(self, content: str, context: Dict[str, Any]) -> List[str]:
        """Generate recommendations using AI."""
        return await self._try_with_fallback(
            "generate_variations",
            f"Content: {content}\nContext: {context}",
            3,
            system_prompt="You are an expert in marketing strategy. Generate recommendations based on the content and context."
        )
    
    async def analyze_competitor_content(self, content: str, competitor_urls: List[str]) -> Dict[str, Any]:
        """Analyze competitor content using AI."""
        return await self._try_with_fallback(
            "analyze_text",
            f"Content: {content}\nCompetitor URLs: {competitor_urls}",
            system_prompt="You are an expert in competitive analysis. Analyze the content and provide insights about competitors."
        )
    
    async def track_content_performance(self, content_id: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Track content performance using AI."""
        return await self._try_with_fallback(
            "analyze_metrics",
            metrics,
            system_prompt="You are an expert in content performance analysis. Analyze the metrics and provide insights."
        ) 