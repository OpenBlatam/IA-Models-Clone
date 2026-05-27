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

class OpenAIProvider(BaseProvider):
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        super().__init__(model, api_key, env_var="OPENAI_API_KEY")
        self.url = "https://api.openai.com/v1/chat/completions"
        model_lower = str(self.model).lower().strip()
        if model_lower in ("si", "gpt4", "gpt-4", "gpt-4o", "1", ""):
            self.model = "gpt-4o"
        else:
            if not model_lower.startswith("gpt-"):
                self.model = "gpt-4o"
            else:
                self.model = self.model.strip()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            return self._safe_fallback("OpenAI API Key missing.", "Configura OPENAI_API_KEY.")
        
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
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
            
            logger.warning(f"OpenAI API Error {resp.status_code}: {resp.text}")
            raise InferenceError(f"OpenAI API Error {resp.status_code}")