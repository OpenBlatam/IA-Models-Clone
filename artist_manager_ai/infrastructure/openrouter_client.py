"""
OpenRouter Client
=================

Cliente para integración con OpenRouter API.
"""

import os
import logging
from typing import Dict, Any, Optional, List
import httpx
from datetime import datetime

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Cliente para OpenRouter API."""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializar cliente OpenRouter.
        
        Args:
            api_key: API key de OpenRouter. Si no se proporciona, se busca en env.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.warning("OpenRouter API key not configured")
        
        self.http_client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
                "HTTP-Referer": "https://blatam-academy.com",
                "X-Title": "Artist Manager AI"
            }
        )
        self._logger = logger
    
    async def generate_text(
        self,
        prompt: str,
        model: str = "anthropic/claude-3-haiku",
        max_tokens: int = 2000,
        temperature: float = 0.7,
        messages: Optional[List[Dict[str, str]]] = None,
        retry_count: int = 3
    ) -> Dict[str, Any]:
        """
        Generar texto usando OpenRouter API.
        
        Args:
            prompt: Prompt del usuario
            model: Modelo a usar (default: claude-3-haiku)
            max_tokens: Máximo de tokens
            temperature: Temperatura para generación
            messages: Lista de mensajes para conversación (opcional)
        
        Returns:
            Respuesta de la API con el texto generado
        """
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured")
        
        # Si se proporcionan mensajes, usarlos; si no, crear mensaje simple
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        
        last_error = None
        for attempt in range(retry_count):
            try:
                response = await self.http_client.post(
                    f"{self.BASE_URL}/chat/completions",
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                return response.json()
            
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in [429, 500, 502, 503, 504] and attempt < retry_count - 1:
                    import asyncio
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    self._logger.warning(f"Retrying after {wait_time}s (attempt {attempt + 1}/{retry_count})")
                    await asyncio.sleep(wait_time)
                    continue
                self._logger.error(f"OpenRouter API HTTP error: {e.response.status_code} - {e.response.text}")
                raise
            except httpx.TimeoutException as e:
                last_error = e
                if attempt < retry_count - 1:
                    import asyncio
                    wait_time = (attempt + 1) * 2
                    self._logger.warning(f"Timeout, retrying after {wait_time}s")
                    await asyncio.sleep(wait_time)
                    continue
                self._logger.error(f"OpenRouter API timeout: {str(e)}")
                raise
            except Exception as e:
                last_error = e
                if attempt < retry_count - 1:
                    import asyncio
                    await asyncio.sleep((attempt + 1) * 2)
                    continue
                self._logger.error(f"OpenRouter API call failed: {str(e)}")
                raise
        
        if last_error:
            raise last_error
    
    async def generate_stream(
        self,
        prompt: str,
        model: str = "anthropic/claude-3-haiku",
        max_tokens: int = 2000,
        temperature: float = 0.7
    ):
        """
        Generar texto en streaming usando OpenRouter API.
        
        Args:
            prompt: Prompt del usuario
            model: Modelo a usar
            max_tokens: Máximo de tokens
            temperature: Temperatura para generación
        
        Yields:
            Chunks de texto generado
        """
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured")
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "https://blatam-academy.com",
                        "X-Title": "Artist Manager AI"
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "stream": True
                    }
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                import json
                                chunk = json.loads(data)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        yield delta["content"]
                            except json.JSONDecodeError:
                                continue
        
        except Exception as e:
            self._logger.error(f"OpenRouter streaming failed: {str(e)}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Verificar salud de la API de OpenRouter.
        
        Returns:
            Estado de salud de la API
        """
        if not self.api_key:
            return {"status": "no_api_key", "healthy": False}
        
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            return {"status": "healthy", "healthy": True, "timestamp": datetime.now().isoformat()}
        
        except Exception as e:
            return {
                "status": "unhealthy",
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def close(self):
        """Cerrar cliente HTTP."""
        await self.http_client.aclose()
    
    async def __aenter__(self):
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()

