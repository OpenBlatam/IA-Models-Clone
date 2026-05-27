from tenacity import retry, stop_after_attempt, wait_exponential
from agents.exceptions import InferenceError
try:
    import google.generativeai as genai
except ImportError:
    genai = None
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

class GoogleGeminiProvider(BaseProvider):
    def __init__(self, model: str = "gemini-2.0-flash-exp", api_key: Optional[str] = None):
        super().__init__(model, api_key, env_var="GOOGLE_API_KEY")
        model_lower = str(self.model).lower().strip()
        if model_lower in ("1", "", "flash", "gemini-2.0-flash-exp"):
            self.model = "gemini-2.0-flash-exp"
        else:
            if not model_lower.startswith("gemini-"):
                self.model = "gemini-2.0-flash-exp"
            else:
                self.model = self.model.strip()
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.api_key:
            return self._safe_fallback("Google API Key missing.", "Google API Key missing.")
        
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.1),
                "maxOutputTokens": kwargs.get("max_tokens", 8192)
            }
        }
        
        async with httpx.AsyncClient(timeout=self.timeout, verify=httpx_verify_setting()) as client:
            resp = await client.post(self.url, json=data)
            if resp.status_code == 200:
                return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            
            logger.warning(f"Google API Error {resp.status_code}: {resp.text}")
            raise InferenceError(f"Google API Error {resp.status_code}")