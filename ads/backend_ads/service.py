from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from typing import Dict, Any, AsyncGenerator, List, Optional
from onyx.utils.logger import setup_logger
from onyx.core.functions import format_response
from onyx.server.features.ads.db_service import AdsDBService
from onyx.server.features.ads.service import AdsService
from onyx.server.features.ads.advanced.service import AdvancedAdsService
from onyx.server.features.ads.langchain.service import LangchainService
from onyx.server.features.ads.models import (
from onyx.server.features.ads.advanced.models import AdvancedAdsRequest
from onyx.server.features.ads.langchain.models import LangchainRequest
from backend_ads.llm_interface import (
from backend_ads.remove_bg_api import remove_background
from backend_ads.ads_api import get_ads_history, get_brand_kit_history
from backend_ads.scraper import get_website_text
from backend_ads.email_sequence import generate_email_sequence
from backend_ads.model_mapping import ModelAdapter
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Backend Ads Service - Enhanced Onyx Integration
Complete integration with advanced Onyx capabilities and model adaptation.
"""
    AdsGenerationRequest,
    BrandKitRequest,
    EmailSequenceRequest,
    BackgroundRemovalRequest
)

# Direct imports from backend_ads
    generate_ads_lcel,
    generate_brand_kit_lcel,
    generate_custom_content_lcel,
    generate_ads_lcel_streaming
)

logger = setup_logger()

class BackendAdsService:
    """Enhanced service with advanced Onyx capabilities."""
    
    def __init__(self) -> Any:
        """Initialize all services."""
        self.ads_service = AdsService()
        self.advanced_ads_service = AdvancedAdsService()
        self.langchain_service = LangchainService()
        self.db_service = AdsDBService()
        self.model_adapter = ModelAdapter()

    async def generate_ads(
        self,
        url: str,
        prompt: Optional[str] = None,
        type: str = "ads",
        user_id: Optional[str] = None,
        brand_voice: Optional[Dict[str, Any]] = None,
        audience_profile: Optional[Dict[str, Any]] = None,
        project_context: Optional[Dict[str, Any]] = None,
        advanced: bool = False,
        use_langchain: bool = False,
        model_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate ads with enhanced capabilities."""
        try:
            # Get website content
            website_text = await get_website_text(url)
            if not website_text:
                website_text = ""

            # Convert to Onyx request format
            request = self.model_adapter.to_onyx_ads_request(
                url=url,
                prompt=prompt,
                type=type,
                brand_voice=brand_voice,
                audience_profile=audience_profile,
                project_context=project_context,
                model_config=model_config
            )

            # Choose generation method
            if advanced:
                result = await self.advanced_ads_service.generate_ads(
                    request=AdvancedAdsRequest(**request)
                )
            elif use_langchain:
                result = await self.langchain_service.generate_ads(
                    request=LangchainRequest(**request)
                )
            else:
                result = await self.ads_service.generate_ads(
                    request=AdsGenerationRequest(**request)
                )

            # Convert result to backend_ads format
            backend_result = self.model_adapter.to_backend_ads_response(result)

            # Store in database
            if user_id:
                await self.db_service.create_ads_generation(
                    user_id=user_id,
                    url=url,
                    type=type,
                    content=backend_result,
                    prompt=prompt,
                    metadata={
                        "brand_voice": brand_voice,
                        "audience_profile": audience_profile,
                        "project_context": project_context,
                        "advanced": advanced,
                        "use_langchain": use_langchain,
                        "model_config": model_config
                    }
                )

            return format_response(backend_result)
        except Exception as e:
            logger.error(f"Error generating ads: {e}")
            raise

    async def generate_brand_kit(
        self,
        url: str,
        prompt: Optional[str] = None,
        user_id: Optional[str] = None,
        brand_voice: Optional[Dict[str, Any]] = None,
        audience_profile: Optional[Dict[str, Any]] = None,
        project_context: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate brand kit with enhanced capabilities."""
        try:
            # Convert to Onyx request format
            request = self.model_adapter.to_onyx_brand_kit_request(
                url=url,
                prompt=prompt,
                brand_voice=brand_voice,
                audience_profile=audience_profile,
                project_context=project_context,
                model_config=model_config
            )

            result = await self.ads_service.generate_brand_kit(
                request=BrandKitRequest(**request)
            )

            # Convert result to backend_ads format
            backend_result = self.model_adapter.to_backend_brand_kit_response(result)

            if user_id:
                await self.db_service.create_brand_kit_generation(
                    user_id=user_id,
                    url=url,
                    content=backend_result,
                    prompt=prompt,
                    metadata={
                        "brand_voice": brand_voice,
                        "audience_profile": audience_profile,
                        "project_context": project_context,
                        "model_config": model_config
                    }
                )

            return format_response(backend_result)
        except Exception as e:
            logger.error(f"Error generating brand kit: {e}")
            raise

    async def generate_email_sequence(
        self,
        url: str,
        prompt: Optional[str] = None,
        user_id: Optional[str] = None,
        brand_voice: Optional[Dict[str, Any]] = None,
        audience_profile: Optional[Dict[str, Any]] = None,
        project_context: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate email sequence with enhanced capabilities."""
        try:
            # Convert to Onyx request format
            request = self.model_adapter.to_onyx_email_sequence_request(
                url=url,
                prompt=prompt,
                brand_voice=brand_voice,
                audience_profile=audience_profile,
                project_context=project_context,
                model_config=model_config
            )

            result = await self.ads_service.generate_email_sequence(
                request=EmailSequenceRequest(**request)
            )

            # Convert result to backend_ads format
            backend_result = self.model_adapter.to_backend_email_sequence_response(result)

            if user_id:
                await self.db_service.create_email_sequence_generation(
                    user_id=user_id,
                    url=url,
                    content=backend_result,
                    prompt=prompt,
                    metadata={
                        "brand_voice": brand_voice,
                        "audience_profile": audience_profile,
                        "project_context": project_context,
                        "model_config": model_config
                    }
                )

            return format_response(backend_result)
        except Exception as e:
            logger.error(f"Error generating email sequence: {e}")
            raise

    async def remove_background(
        self,
        image_url: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Remove background with enhanced capabilities."""
        try:
            # Convert to Onyx request format
            request = self.model_adapter.to_onyx_background_removal_request(
                image_url=image_url,
                metadata=metadata,
                model_config=model_config
            )

            result = await self.ads_service.remove_background(
                request=BackgroundRemovalRequest(**request)
            )

            # Convert result to backend_ads format
            backend_result = self.model_adapter.to_backend_background_removal_response(result)

            if user_id:
                await self.db_service.create_background_removal(
                    user_id=user_id,
                    processed_image_url=backend_result["processed_url"],
                    original_image_url=image_url,
                    metadata={
                        "metadata": metadata,
                        "model_config": model_config
                    }
                )

            return format_response(backend_result)
        except Exception as e:
            logger.error(f"Error removing background: {e}")
            raise

    async def get_ads_history(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get ads history with enhanced capabilities."""
        try:
            result = await self.db_service.get_ads_history(
                user_id=user_id,
                limit=limit,
                offset=offset
            )
            return format_response(result)
        except Exception as e:
            logger.error(f"Error getting ads history: {e}")
            raise

    async def get_brand_kit_history(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get brand kit history with enhanced capabilities."""
        try:
            result = await self.db_service.get_brand_kit_history(
                user_id=user_id,
                limit=limit,
                offset=offset
            )
            return format_response(result)
        except Exception as e:
            logger.error(f"Error getting brand kit history: {e}")
            raise

    async def stream_ads(
        self,
        url: str,
        prompt: Optional[str] = None,
        type: str = "ads",
        advanced: bool = False,
        use_langchain: bool = False,
        model_config: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream ads generation with enhanced capabilities."""
        try:
            # Convert to Onyx request format
            request = self.model_adapter.to_onyx_ads_request(
                url=url,
                prompt=prompt,
                type=type,
                model_config=model_config
            )

            if advanced:
                async for chunk in self.advanced_ads_service.stream_ads(
                    request=AdvancedAdsRequest(**request)
                ):
                    yield self.model_adapter.to_backend_stream_chunk(chunk)
            elif use_langchain:
                async for chunk in self.langchain_service.stream_ads(
                    request=LangchainRequest(**request)
                ):
                    yield self.model_adapter.to_backend_stream_chunk(chunk)
            else:
                async for chunk in self.ads_service.stream_ads(
                    request=AdsGenerationRequest(**request)
                ):
                    yield self.model_adapter.to_backend_stream_chunk(chunk)
        except Exception as e:
            logger.error(f"Error streaming ads: {e}")
            raise 