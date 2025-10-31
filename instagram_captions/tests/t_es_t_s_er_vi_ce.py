from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime
from ..models import (
from ..service import InstagramCaptionsService, OpenAIProvider, LangChainProvider
from typing import Any, List, Dict, Optional
import logging
"""
Tests for Instagram Captions Service.
"""


    InstagramCaptionRequest,
    PostContent,
    CaptionStyle,
    TimeZone,
    InstagramTarget
)


@pytest.fixture
def sample_request():
    """Create a sample Instagram caption request."""
    return InstagramCaptionRequest(
        content=PostContent(
            description="Beautiful sunset at the beach",
            image_description="Golden sunset over ocean waves"
        ),
        style=CaptionStyle.INSPIRATIONAL,
        target_timezone=TimeZone.EST,
        include_hashtags=True,
        hashtag_count=10
    )


@pytest.fixture
def mock_service():
    """Create a mock Instagram captions service."""
    service = InstagramCaptionsService()
    # Mock providers for testing
    service.providers = {
        'openai': Mock(spec=OpenAIProvider),
        'langchain': Mock(spec=LangChainProvider)
    }
    service.providers['openai'].is_available.return_value = True
    service.providers['openai'].name = 'openai'
    service.providers['langchain'].is_available.return_value = True
    service.providers['langchain'].name = 'langchain'
    return service


class TestInstagramCaptionsService:
    """Test Instagram Captions Service functionality."""
    
    def test_service_initialization(self) -> Any:
        """Test service initializes correctly."""
        service = InstagramCaptionsService()
        assert isinstance(service.providers, dict)
    
    def test_get_best_provider_langchain_preference(self, mock_service, sample_request) -> Optional[Dict[str, Any]]:
        """Test provider selection with LangChain preference."""
        sample_request.use_langchain = True
        sample_request.use_openai = False
        
        provider = mock_service.get_best_provider(sample_request)
        assert provider.name == 'langchain'
    
    def test_get_best_provider_openai_preference(self, mock_service, sample_request) -> Optional[Dict[str, Any]]:
        """Test provider selection with OpenAI preference."""
        sample_request.use_langchain = False
        sample_request.use_openai = True
        
        provider = mock_service.get_best_provider(sample_request)
        assert provider.name == 'openai'
    
    def test_get_best_provider_no_available_providers(self, mock_service, sample_request) -> Optional[Dict[str, Any]]:
        """Test error when no providers are available."""
        mock_service.providers['openai'].is_available.return_value = False
        mock_service.providers['langchain'].is_available.return_value = False
        
        with pytest.raises(ValueError, match="No AI providers available"):
            mock_service.get_best_provider(sample_request)
    
    @pytest.mark.asyncio
    async def test_generate_captions_success(self, mock_service, sample_request) -> Any:
        """Test successful caption generation."""
        # Mock provider response
        mock_provider = mock_service.providers['openai']
        mock_provider.generate_caption.return_value = "Beautiful sunset vibes! 🌅 #sunset #inspiration"
        
        with patch.object(mock_service, 'get_best_provider', return_value=mock_provider):
            response = await mock_service.generate_captions(sample_request)
        
        assert response is not None
        assert len(response.variations) == 1
        assert response.variations[0].caption == "Beautiful sunset vibes! 🌅 #sunset #inspiration"
        assert response.generation_metrics.provider_used == 'openai'
    
    @pytest.mark.asyncio
    async def test_generate_captions_provider_failure(self, mock_service, sample_request) -> Any:
        """Test caption generation when provider fails."""
        mock_provider = mock_service.providers['openai']
        mock_provider.generate_caption.side_effect = Exception("Provider error")
        
        with patch.object(mock_service, 'get_best_provider', return_value=mock_provider):
            with pytest.raises(Exception, match="Provider error"):
                await mock_service.generate_captions(sample_request)


class TestAIProviders:
    """Test AI provider classes."""
    
    def test_openai_provider_initialization(self) -> Any:
        """Test OpenAI provider initialization."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            with patch('agents.backend.onyx.server.features.instagram_captions.service.OPENAI_AVAILABLE', True):
                provider = OpenAIProvider()
                assert provider.name == 'openai'
                # Note: actual availability depends on import success
    
    def test_langchain_provider_initialization(self) -> Any:
        """Test LangChain provider initialization."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            with patch('agents.backend.onyx.server.features.instagram_captions.service.LANGCHAIN_AVAILABLE', True):
                provider = LangChainProvider()
                assert provider.name == 'langchain'
    
    async def test_provider_unavailable_without_api_key(self) -> Any:
        """Test provider is unavailable without API key."""
        with patch.dict('os.environ', {}, clear=True):
            provider = OpenAIProvider()
            assert not provider.is_available()


match __name__:
    case "__main__":
    pytest.main([__file__]) 