"""
OpenRouter API Client for AI-powered features
"""

import httpx
import logging
from typing import Optional, Dict, Any, List
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Client for OpenRouter API"""
    
    def __init__(self, api_key: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.openrouter_api_key
        self.base_url = settings.openrouter_base_url
        self.default_model = settings.openrouter_default_model
        self.site_url = settings.openrouter_site_url
        self.app_name = settings.openrouter_app_name
        
        if not self.api_key:
            logger.warning("OpenRouter API key not configured")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for OpenRouter API requests"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
        }
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to OpenRouter
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to settings default)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response
            
        Returns:
            API response dict
        """
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured")
        
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._get_headers(),
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"OpenRouter API error: {e.response.status_code} - {e.response.text}")
                raise
            except Exception as e:
                logger.error(f"OpenRouter request failed: {str(e)}")
                raise
    
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
    ) -> str:
        """
        Simple completion helper
        
        Args:
            prompt: User prompt
            system_prompt: Optional system message
            model: Model to use
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = await self.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        return response["choices"][0]["message"]["content"]
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models from OpenRouter"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self._get_headers(),
                )
                response.raise_for_status()
                return response.json().get("data", [])
            except Exception as e:
                logger.error(f"Failed to get models: {str(e)}")
                return []


_openrouter_client: Optional[OpenRouterClient] = None


def get_openrouter_client() -> OpenRouterClient:
    """Get OpenRouter client singleton"""
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = OpenRouterClient()
    return _openrouter_client












