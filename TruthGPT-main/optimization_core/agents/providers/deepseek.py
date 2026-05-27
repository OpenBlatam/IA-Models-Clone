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

class DeepSeekProvider(BaseProvider):
    def __init__(self, model: str = "deepseek-reasoner", api_key: Optional[str] = None):
        super().__init__(model, api_key, env_var="DEEPSEEK_API_KEY")
        self.url = "https://api.deepseek.com/chat/completions"
        model_lower = str(self.model).lower().strip()
        if model_lower in ("v4-flash", "flash", "chat", "v3", "v4", "deepseek-chat", "deepseek-v4-flash"):
            self.model = "deepseek-v4-flash"
        elif model_lower in ("v4-pro", "pro", "reasoner", "r1", "deepseek-reasoner", "deepseek-v4-pro", "1", ""):
            self.model = "deepseek-v4-pro"
        else:
            self.model = "deepseek-v4-pro"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            return self._safe_fallback("DeepSeek API Key missing.", "Configura DEEPSEEK_API_KEY.")
        
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 8192)
        }
        if "chat" in self.model:
            data["temperature"] = kwargs.get("temperature", 0.1)
        elif "temperature" in kwargs:
            data["temperature"] = kwargs["temperature"]
        
        async with httpx.AsyncClient(timeout=180.0, verify=httpx_verify_setting()) as client:
            resp = await client.post(self.url, headers=headers, json=data)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            
            logger.warning(f"DeepSeek API Error {resp.status_code}: {resp.text}")
            raise InferenceError(f"DeepSeek API Error {resp.status_code}")