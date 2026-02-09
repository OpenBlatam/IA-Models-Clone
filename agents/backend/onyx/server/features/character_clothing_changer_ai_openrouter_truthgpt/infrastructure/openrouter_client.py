"""
OpenRouter Client for Character Clothing Changer AI
===================================================

Client for OpenRouter API integration for intelligent prompt processing.
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, List
import httpx

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    """Client for OpenRouter API - optimized with connection pooling"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = OPENROUTER_API_URL
        self.timeout = 60.0
        self._client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling"""
        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    limits = httpx.Limits(
                        max_connections=100,
                        max_keepalive_connections=20,
                        keepalive_expiry=30.0
                    )
                    timeout = httpx.Timeout(self.timeout, connect=10.0)
                    self._client = httpx.AsyncClient(
                        limits=limits,
                        timeout=timeout,
                        http2=True
                    )
        return self._client
    
    async def close(self) -> None:
        """Close HTTP client and cleanup resources"""
        if self._client:
            try:
                await self._client.aclose()
            except Exception as e:
                logger.warning(f"Error closing OpenRouter client: {e}")
            finally:
                self._client = None
    
    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion using OpenRouter
        
        Args:
            model: Model identifier (e.g., 'openai/gpt-4', 'anthropic/claude-3-opus')
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            Dict with 'response' and 'tokens_used'
        """
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable.")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://blatam-academy.com",
            "X-Title": "Character Clothing Changer AI"
        }
        
        payload = {
            "model": model,
            "messages": messages,
        }
        
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        payload.update(kwargs)
        
        client = await self._get_client()
        try:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            
            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)
            
            return {
                "response": content,
                "tokens_used": tokens_used,
                "model": data.get("model", model),
                "finish_reason": choice.get("finish_reason"),
                "usage": usage,
                "id": data.get("id"),
                "created": data.get("created")
            }
        except httpx.HTTPStatusError as e:
            error_msg = f"OpenRouter API error: {e.response.status_code}"
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {})
                error_msg = error_detail.get("message", error_msg)
                error_type = error_detail.get("type", "api_error")
            except Exception:
                error_type = "http_error"
            
            logger.error(f"OpenRouter API error ({error_type}): {error_msg}")
            raise Exception(f"{error_type}: {error_msg}")
        except httpx.TimeoutException:
            logger.error(f"OpenRouter API timeout after {self.timeout}s")
            raise Exception(f"Request timeout after {self.timeout}s")
        except Exception as e:
            logger.error(f"Error calling OpenRouter: {e}")
            raise
    
    async def optimize_prompt(
        self,
        original_prompt: str,
        context: Optional[Dict[str, Any]] = None,
        model: str = "openai/gpt-4"
    ) -> str:
        """
        Optimize a prompt using OpenRouter for better results.
        
        Args:
            original_prompt: Original prompt to optimize
            context: Additional context (image description, clothing type, etc.)
            model: OpenRouter model to use
            
        Returns:
            Optimized prompt
        """
        system_prompt = """You are an expert at creating optimized prompts for AI image generation, specifically for character clothing changes using inpainting techniques.

Your task is to optimize user prompts to:
1. Be clear and specific about the clothing item
2. Include relevant details (color, style, material, fit)
3. Maintain character consistency
4. Work well with Flux Fill inpainting models
5. Follow best practices for inpainting prompts

Return only the optimized prompt, nothing else."""

        user_prompt = f"Original prompt: {original_prompt}\n\n"
        if context:
            user_prompt += f"Context: {context}\n\n"
        user_prompt += "Please optimize this prompt for character clothing inpainting:"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        result = await self.chat_completion(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return result["response"].strip()


# Global client instance
_openrouter_client: Optional[OpenRouterClient] = None


def get_openrouter_client() -> OpenRouterClient:
    """Get or create OpenRouter client instance"""
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = OpenRouterClient()
    return _openrouter_client

