"""
OpenRouter API service for handling LLM API calls.
Separated from maintenance_tutor for better separation of concerns.
"""

import logging
import httpx
from typing import Dict, Any, Optional, List
from datetime import datetime

from ...config.maintenance_config import OpenRouterConfig
from ...utils.retry_handler import retry_with_backoff

logger = logging.getLogger(__name__)


class OpenRouterService:
    """Service for interacting with OpenRouter API."""
    
    def __init__(self, config: OpenRouterConfig):
        """
        Initialize OpenRouter service.
        
        Args:
            config: OpenRouter configuration
        """
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=config.timeout,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "HTTP-Referer": "https://blatam-academy.com",
                "X-Title": "Robot Maintenance AI"
            }
        )
        logger.info("OpenRouter service initialized")
    
    async def chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Make a chat completion request to OpenRouter.
        
        Args:
            system_prompt: System prompt for the conversation
            user_prompt: User prompt/message
            model: Model to use (defaults to config default)
            temperature: Temperature setting (defaults to config)
            max_tokens: Max tokens (defaults to config)
        
        Returns:
            API response dictionary
        
        Raises:
            ValueError: If API request fails
            TimeoutError: If request times out
            ConnectionError: If network error occurs
        """
        model = model or self.config.default_model
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        async def _make_api_call():
            response = await self.client.post(
                f"{self.config.base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            response.raise_for_status()
            return response.json()
        
        try:
            data = await retry_with_backoff(
                _make_api_call,
                max_retries=self.config.max_retries,
                initial_delay=1.0
            )
            
            return {
                "content": data["choices"][0]["message"]["content"],
                "model": data["model"],
                "usage": data.get("usage", {}),
                "raw_response": data
            }
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code}: {e.response.text}"
            logger.error(f"Error calling OpenRouter API: {error_msg}")
            raise ValueError(f"API request failed: {error_msg}") from e
        except httpx.TimeoutException as e:
            error_msg = "Request timeout - API took too long to respond"
            logger.error(f"Timeout calling OpenRouter API: {e}")
            raise TimeoutError(error_msg) from e
        except httpx.NetworkError as e:
            error_msg = "Network error - unable to reach API"
            logger.error(f"Network error calling OpenRouter API: {e}")
            raise ConnectionError(error_msg) from e
        except Exception as e:
            logger.error(f"Unexpected error in OpenRouter service: {e}", exc_info=True)
            raise
    
    async def close(self):
        """Close the HTTP client and cleanup resources."""
        await self.client.aclose()
        logger.debug("OpenRouter service closed")






