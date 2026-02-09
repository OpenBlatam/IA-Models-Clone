"""
Message preparation utilities for OpenRouter API.

Refactored to consolidate message preparation logic into a dedicated class.
"""

import os
import base64
from typing import List, Dict, Any
from pathlib import Path


class MessagePreparer:
    """
    Message preparation utilities for OpenRouter API.
    
    Responsibilities:
    - Prepare messages for OpenRouter API format
    - Convert images to base64
    - Handle message content formatting
    
    Single Responsibility: Handle all message preparation operations.
    """
    
    def __init__(self):
        """Initialize message preparer."""
        self._mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
    
    def prepare_messages_for_openrouter(self, messages: List[Dict]) -> List[Dict]:
        """
        Prepare messages for OpenRouter API format.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Processed messages in OpenRouter format
        """
        processed = []
        for msg in messages:
            if msg["role"] == "user" and isinstance(msg.get("content"), list):
                content = []
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image":
                        # Convert to base64 format for OpenRouter
                        image_path = item["image"]
                        base64_image, mime_type = self.get_image_base64(image_path)
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}",
                                "detail": "high"
                            }
                        })
                    else:
                        content.append(item)
                processed.append({"role": msg["role"], "content": content})
            else:
                processed.append(msg)
        return processed
    
    def get_image_base64(self, image_path: str) -> tuple[str, str]:
        """
        Convert image to base64 and return with MIME type.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (base64_data, mime_type)
        """
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = self._mime_types.get(ext, "image/jpeg")
        
        with open(image_path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode("utf-8")
        
        return base64_data, mime_type

