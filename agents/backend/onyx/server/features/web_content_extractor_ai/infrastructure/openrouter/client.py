"""
Cliente OpenRouter para procesamiento de contenido web

Refactored to:
- Use centralized error handling
- Use prompt builder for prompt construction
- Eliminate duplicate error handling logic
"""

import logging
from typing import Dict, Any, Optional, Union
import httpx
from config import settings
from .error_handlers import handle_openrouter_error
from .prompt_builder import build_extraction_prompt, truncate_content

# Importar constantes desde schemas
try:
    from ...api.v1.schemas.constants import DEFAULT_MODEL, DEFAULT_MAX_TOKENS
except ImportError:
    # Fallback si no se puede importar
    DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"
    DEFAULT_MAX_TOKENS = 4000

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient:
    """Cliente para OpenRouter API con connection pooling"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.openrouter_api_key
        self.base_url = OPENROUTER_API_URL
        self.timeout = 60.0
        self.http_referer = settings.openrouter_http_referer
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Obtener o crear cliente HTTP con connection pooling"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                    keepalive_expiry=30.0
                ),
                http2=True
            )
        return self._client
    
    async def close(self):
        """Cerrar cliente HTTP"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def extract_content(
        self,
        web_content: str,
        url: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extrae información estructurada del contenido web usando OpenRouter
        
        Args:
            web_content: Contenido HTML/texto de la página web
            url: URL de la página
            model: Modelo de OpenRouter a usar
            max_tokens: Máximo de tokens en la respuesta
            
        Returns:
            Dict con información extraída
        """
        if not self.api_key:
            raise ValueError("OpenRouter API key no configurada. Establece OPENROUTER_API_KEY")
        
        # Usar valores por defecto si no se proporcionan
        model = model or DEFAULT_MODEL
        max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        
        # Truncar contenido si es muy largo
        content_preview = truncate_content(web_content)
        
        # Construir prompt usando helper
        prompt = build_extraction_prompt(url, content_preview)
        messages = [{"role": "user", "content": prompt}]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.http_referer,
            "X-Title": "Web Content Extractor AI"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3
        }
        
        client = await self._get_client()
        try:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            # Validar respuesta
            if not data.get("choices"):
                raise ValueError("Respuesta de OpenRouter sin choices")
            
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            
            if not content:
                logger.warning("OpenRouter retornó contenido vacío")
            
            usage = data.get("usage", {})
            
            return {
                "extracted_content": content,
                "tokens_used": usage.get("total_tokens", 0),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "model": data.get("model", model),
                "url": url,
                "finish_reason": choice.get("finish_reason")
            }
        except (httpx.HTTPStatusError, httpx.TimeoutException) as e:
            raise handle_openrouter_error(e, "extract content", timeout=self.timeout)
        except Exception as e:
            raise handle_openrouter_error(e, "extract content", timeout=self.timeout)

