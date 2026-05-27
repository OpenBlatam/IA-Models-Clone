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

class OpenRouterProvider(BaseProvider):
    def __init__(self, model: str = "~anthropic/claude-sonnet-latest", api_key: Optional[str] = None):
        super().__init__(model, api_key, env_var="OPENROUTER_API_KEY")
        model_lower = str(self.model).lower().strip()
        # Map retired model IDs to current ones
        _retired_models = {
            "anthropic/claude-sonnet-latest": "~anthropic/claude-sonnet-latest",
            "anthropic/claude-3.7-sonnet": "~anthropic/claude-sonnet-latest",
            "anthropic/claude-3-7-sonnet": "~anthropic/claude-sonnet-latest",
            "anthropic/claude-3.5-sonnet": "~anthropic/claude-sonnet-latest",
            "anthropic/claude-sonnet-4-20250514": "~anthropic/claude-sonnet-latest",
            "anthropic/claude-sonnet-4-0": "~anthropic/claude-sonnet-latest",
            "anthropic/claude-3.7-sonnet:beta": "~anthropic/claude-sonnet-latest",
            "anthropic/claude-3.5-sonnet:beta": "~anthropic/claude-sonnet-latest",
        }
        if model_lower in ("1", ""):
            self.model = "~anthropic/claude-sonnet-latest"
        elif model_lower in _retired_models:
            self.model = _retired_models[model_lower]
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            return self._safe_fallback("OpenRouter API Key missing.", "Configura OPENROUTER_API_KEY.")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://truthgpt.ai",
            "X-Title": "TruthGPT OS",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 4096)
        }
        
        async with httpx.AsyncClient(timeout=self.timeout, verify=httpx_verify_setting()) as client:
            resp = await client.post(self.url, headers=headers, json=data)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            
            logger.warning(f"OpenRouter API Error {resp.status_code}: {resp.text}")
            raise InferenceError(f"OpenRouter API Error {resp.status_code}")