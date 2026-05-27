from tenacity import retry, stop_after_attempt, wait_exponential
from agents.exceptions import InferenceError
import asyncio
import aiohttp
import json
import os
from typing import Dict, Any, Optional
import httpx
try:
    from agents.ssl_context import httpx_verify_setting
except ImportError:
    from ..ssl_context import httpx_verify_setting
from .base import BaseProvider
from agents.utils.config_utils import _resolve_api_key
import logging
logger = logging.getLogger(__name__)

class AnthropicProvider(BaseProvider):
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None):
        super().__init__(model, api_key, env_var="ANTHROPIC_API_KEY")
        self.url = "https://api.anthropic.com/v1/messages"
        model_lower = str(self.model).lower().strip()
        if model_lower in ("opus", "claude-3-opus", "claude-3-opus-20240229"):
            self.model = "claude-3-opus-20240229"
        elif model_lower in ("sonnet", "claude-3-5-sonnet", "claude-3.5-sonnet", "claude-3-5-sonnet-latest"):
            self.model = "claude-3-5-sonnet-20241022"
        elif model_lower in ("claude-3-7-sonnet", "claude-3.7-sonnet", "claude-3-7-sonnet-latest", "1", ""):
            self.model = "claude-sonnet-4-20250514"
        else:
            self.model = "claude-sonnet-4-20250514"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            or_key = _resolve_api_key("OPENROUTER_API_KEY")
            if or_key:
                logger.warning("Anthropic API Key missing. Trying fallback to OpenRouter.")
                try:
                    fallback_provider = OpenRouterProvider(api_key=or_key)
                    return await fallback_provider.generate(prompt, **kwargs)
                except Exception as fallback_exc:
                    logger.error(f"Fallback to OpenRouter failed: {fallback_exc}")
            return self._safe_fallback("Anthropic API Key missing.", "Configura ANTHROPIC_API_KEY.")
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.1)
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout, verify=httpx_verify_setting()) as client:
                resp = await client.post(self.url, headers=headers, json=data)
                if resp.status_code == 200:
                    return resp.json()["content"][0]["text"]
                
                logger.warning(f"Anthropic API Error {resp.status_code}: {resp.text}")
                raise InferenceError(f"Anthropic API Error {resp.status_code}")
        except Exception as e:
            or_key = _resolve_api_key("OPENROUTER_API_KEY")
            if or_key:
                logger.warning(f"Anthropic direct call failed: {e}. Trying fallback to OpenRouter.")
                try:
                    fallback_provider = OpenRouterProvider(api_key=or_key)
                    return await fallback_provider.generate(prompt, **kwargs)
                except Exception as fallback_exc:
                    logger.error(f"Fallback to OpenRouter failed: {fallback_exc}")
                    raise e
            else:
                raise e