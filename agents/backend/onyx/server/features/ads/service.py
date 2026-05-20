from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

from typing import Dict, Any, Optional, List
import httpx
import base64
import io
import numpy as np
from PIL import Image
import pyvips
import cv2
from rembg import remove, new_session
from uuid import UUID
from onyx.llm.interface import (
from onyx.utils.logger import setup_logger
from onyx.server.features.ads.storage import StorageService
from onyx.server.features.ads.config import settings
from .models import Ad
from .schemas import AdCreate
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Service for handling ads-related business logic.
"""

    generate_ads_lcel,
    generate_brand_kit_lcel,
    generate_custom_content_lcel
)

logger = setup_logger()

class AdsService:
    """Service for handling ads-related business logic."""
    
    def __init__(self) -> Any:
        """Initialize the service."""
        self.rembg_session = new_session()
        self.storage_service = StorageService()
    
    async def generate_ads(
        self,
        prompt: str,
        type: str
    ) -> Dict[str, Any]:
        """
        Generate ads based on prompt using if-return pattern.
        
        Args:
            prompt: The prompt to generate ads from
            type: The type of content to generate (ads, brand-kit, custom)
            
        Returns:
            Dict containing the generated content
        """
        try:
            if type == "ads":
                content = await generate_ads_lcel(
                    prompt,
                    model=settings.LLM_MODEL,
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS
                )
                return {
                    "type": "ads",
                    "content": content,
                    "url": None  # TODO: Add URL generation
                }
            
            if type == "brand-kit":
                content = await generate_brand_kit_lcel(
                    prompt,
                    model=settings.LLM_MODEL,
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS
                )
                return {
                    "type": "brand_kit",
                    "content": content,
                    "url": None  # TODO: Add URL generation
                }
            
            if type == "custom":
                content = await generate_custom_content_lcel(
                    prompt,
                    model=settings.LLM_MODEL,
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS
                )
                return {
                    "type": "custom_content",
                    "content": content,
                    "url": None  # TODO: Add URL generation
                }
            
            # Default case for invalid type
            raise ValueError(f"Invalid content type: {type}")
            
        except Exception as e:
            logger.exception("Error generating ads")
            raise
    
    async def remove_background(
        self,
        image_url: str
    ) -> Dict[str, Any]:
        """
        Remove background from an image.
        
        Args:
            image_url: URL of the image to process
            
        Returns:
            Dict containing the processed image URL and metadata
        """
        try:
            # Download image
            async with httpx.AsyncClient() as client:
                resp = await client.get(image_url)
                if resp.status_code != 200:
                    raise ValueError(f"Failed to download image: {resp.status_code}")
                image_bytes = resp.content
            
            # Process image
            try:
                img_vips = pyvips.Image.new_from_buffer(image_bytes, "", access="sequential")
                max_size = settings.MAX_IMAGE_SIZE
                scale = min(max_size / img_vips.width, max_size / img_vips.height, 1.0)
                if scale < 1.0:
                    img_vips = img_vips.resize(scale)
                png_bytes = img_vips.write_to_buffer(".png")
                input_image = Image.open(io.BytesIO(png_bytes))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            except Exception as e:
                logger.warning(f"pyvips failed: {e}, falling back to OpenCV")
                arr = np.frombuffer(image_bytes, np.uint8)
                img_cv = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if img_cv is None:
                    raise ValueError("Failed to decode image")
                max_size = settings.MAX_IMAGE_SIZE
                h, w = img_cv.shape[:2]
                scale = min(max_size / h, max_size / w, 1.0)
                if scale < 1.0:
                    img_cv = cv2.resize(img_cv, (int(w * scale), int(h * scale)))
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                input_image = Image.fromarray(img_rgb)
            
            # Remove background
            output_image = remove(input_image, session=self.rembg_session)
            has_alpha = output_image.mode == "RGBA" and np.array(output_image)[..., 3].max() > 0
            
            # Save processed image
            buffered = io.BytesIO()
            
            if has_alpha:
                output_image.save(buffered, format="PNG")
                mime = "image/png"
                ext = ".png"
            else:
                output_image = output_image.convert("RGB")
                output_image.save(buffered, format="JPEG", quality=settings.JPEG_QUALITY)
                mime = "image/jpeg"
                ext = ".jpg"
            
            # Save to storage
            buffered.seek(0)
            filename = await self.storage_service.save_file(
                file=buffered,
                original_filename=f"processed{ext}"
            )
            processed_url = self.storage_service.get_file_url(filename)
            
            return {
                "type": "background_removal",
                "original_url": image_url,
                "processed_url": processed_url,
                "mime": mime
            }
        except Exception as e:
            logger.exception("Error removing background")
            raise

class AdService:
    """Service layer for Ad business logic and persistence."""

    async def create_ad(self, data: AdCreate) -> Ad:
        """Create a new Ad."""
        # TODO: Implement DB insert
        raise NotImplementedError

    async def get_ad(self, ad_id: UUID) -> Optional[Ad]:
        """Retrieve an Ad by ID."""
        # TODO: Implement DB fetch
        raise NotImplementedError

    async def list_ads(self, skip: int = 0, limit: int = 100) -> List[Ad]:
        """List Ads with pagination."""
        # TODO: Implement DB query
        raise NotImplementedError

    async def update_ad(self, ad_id: UUID, data: AdCreate) -> Optional[Ad]:
        """Update an existing Ad."""
        # TODO: Implement DB update
        raise NotImplementedError

    async def delete_ad(self, ad_id: UUID) -> bool:
        """Delete an Ad by ID (soft delete if supported)."""
        # TODO: Implement DB delete
        raise NotImplementedError 