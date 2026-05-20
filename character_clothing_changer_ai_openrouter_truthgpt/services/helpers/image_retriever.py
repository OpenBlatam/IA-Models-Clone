"""
Image Retriever
===============
Retrieves output images from ComfyUI
"""

import logging
import httpx
from typing import List, Dict, Any, Optional
from .http_client_manager import HTTPClientManager

logger = logging.getLogger(__name__)


class ImageRetriever:
    """
    Retrieves output images from ComfyUI.
    """
    
    def __init__(self, api_url: str, http_client_manager: HTTPClientManager):
        """
        Initialize image retriever.
        
        Args:
            api_url: ComfyUI API URL
            http_client_manager: HTTP client manager instance
        """
        self.api_url = api_url.rstrip('/')
        self.http_client_manager = http_client_manager
    
    async def get_output_images(self, prompt_id: str) -> List[Dict[str, Any]]:
        """
        Get output images for a completed prompt.
        
        Args:
            prompt_id: Prompt ID to get images for
            
        Returns:
            List of image dictionaries with filename and URL
        """
        client = await self.http_client_manager.get_client()
        
        # Get history to find the prompt
        response = await client.get(f"{self.api_url}/history")
        response.raise_for_status()
        history = response.json()
        
        # Find prompt in history
        prompt_data = history.get(prompt_id)
        if not prompt_data:
            logger.warning(f"Prompt {prompt_id} not found in history")
            return []
        
        # Extract output images
        outputs = prompt_data.get("outputs", {})
        images = []
        
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for image_info in node_output["images"]:
                    filename = image_info.get("filename")
                    if filename:
                        image_url = f"{self.api_url}/view?filename={filename}"
                        images.append({
                            "filename": filename,
                            "url": image_url,
                            "node_id": node_id,
                            "type": image_info.get("type", "output"),
                            "subfolder": image_info.get("subfolder", ""),
                        })
        
        logger.info(f"Retrieved {len(images)} images for prompt {prompt_id}")
        return images
    
    def get_image_url(self, filename: str, subfolder: str = "") -> str:
        """
        Get URL for an image file.
        
        Args:
            filename: Image filename
            subfolder: Optional subfolder path
            
        Returns:
            Full URL to image
        """
        if subfolder:
            return f"{self.api_url}/view?filename={filename}&subfolder={subfolder}"
        return f"{self.api_url}/view?filename={filename}"

