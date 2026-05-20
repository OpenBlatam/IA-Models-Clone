from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import Optional, Dict, Any
import httpx
import structlog
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
External API Manager

Manages connections and clients for external APIs like OpenRouter, OpenAI, etc.
"""


logger = structlog.get_logger()


class ExternalAPIManager:
    """
    Manages external API clients and connections.
    
    Features:
    - HTTP client management with connection pooling
    - API key management
    - Rate limiting and retry logic
    - Health monitoring
    """
    
    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        huggingface_token: Optional[str] = None
    ):
        
    """__init__ function."""
self.openrouter_api_key = openrouter_api_key
        self.openai_api_key = openai_api_key
        self.huggingface_token = huggingface_token
        
        self.http_client: Optional[httpx.AsyncClient] = None
        self._is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize external API clients."""
        if self._is_initialized:
            return
        
        logger.info("Initializing external API clients")
        
        # Create HTTP client with connection pooling
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(
                max_keepalive_connections=10,
                max_connections=20,
                keepalive_expiry=30.0
            ),
            headers={
                "User-Agent": "HeyGen-AI-FastAPI/2.0.0"
            }
        )
        
        # Test API connections
        await self._test_connections()
        
        self._is_initialized = True
        logger.info("External API clients initialized successfully")
    
    async def shutdown(self) -> None:
        """Shutdown external API clients."""
        if not self._is_initialized:
            return
        
        logger.info("Shutting down external API clients")
        
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
        
        self._is_initialized = False
        logger.info("External API clients shutdown completed")
    
    def get_client(self, service: str) -> Optional[Dict[str, Any]]:
        """Get API client for specific service."""
        if not self._is_initialized:
            raise RuntimeError("External API manager not initialized")
        
        if service == "http":
            return self.http_client
        elif service == "openrouter":
            return OpenRouterClient(self.http_client, self.openrouter_api_key)
        elif service == "openai":
            return OpenAIClient(self.http_client, self.openai_api_key)
        elif service == "huggingface":
            return HuggingFaceClient(self.http_client, self.huggingface_token)
        else:
            raise ValueError(f"Unknown service: {service}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check for external APIs."""
        if not self._is_initialized:
            return {"status": "not_initialized", "healthy": False}
        
        health_status = {
            "status": "healthy",
            "healthy": True,
            "services": {}
        }
        
        # Check OpenRouter
        if self.openrouter_api_key:
            try:
                client = self.get_client("openrouter")
                await client.health_check()
                health_status["services"]["openrouter"] = {"status": "healthy", "healthy": True}
            except Exception as e:
                health_status["services"]["openrouter"] = {"status": "unhealthy", "healthy": False, "error": str(e)}
                health_status["healthy"] = False
        
        # Check OpenAI
        if self.openai_api_key:
            try:
                client = self.get_client("openai")
                await client.health_check()
                health_status["services"]["openai"] = {"status": "healthy", "healthy": True}
            except Exception as e:
                health_status["services"]["openai"] = {"status": "unhealthy", "healthy": False, "error": str(e)}
                health_status["healthy"] = False
        
        return health_status
    
    async def _test_connections(self) -> None:
        """Test API connections."""
        try:
            # Test basic HTTP connectivity
            response = await self.http_client.get("https://httpbin.org/status/200")
            response.raise_for_status()
            logger.info("HTTP client test successful")
        except Exception as e:
            logger.warning("HTTP client test failed", error=str(e))


class BaseAPIClient:
    """Base class for API clients."""
    
    def __init__(self, http_client: httpx.AsyncClient, api_key: Optional[str]):
        
    """__init__ function."""
self.http_client = http_client
        self.api_key = api_key
        self._logger = structlog.get_logger(self.__class__.__name__)
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the API."""
        return {"status": "healthy", "healthy": True}


class OpenRouterClient(BaseAPIClient):
    """OpenRouter API client."""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    async def generate_text(
        self,
        prompt: str,
        model: str = "anthropic/claude-3-haiku",
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate text using OpenRouter API."""
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured")
        
        try:
            response = await self.http_client.post(
                f"{self.BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://heygen-ai.example.com",
                    "X-Title": "HeyGen AI"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            self._logger.error("OpenRouter API call failed", error=str(e))
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for OpenRouter API."""
        if not self.api_key:
            return {"status": "no_api_key", "healthy": False}
        
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            return {"status": "healthy", "healthy": True}
        
        except Exception as e:
            return {"status": "unhealthy", "healthy": False, "error": str(e)}


class OpenAIClient(BaseAPIClient):
    """OpenAI API client."""
    
    BASE_URL = "https://api.openai.com/v1"
    
    async def generate_text(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generate text using OpenAI API."""
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")
        
        try:
            response = await self.http_client.post(
                f"{self.BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            self._logger.error("OpenAI API call failed", error=str(e))
            raise
    
    async def generate_speech(
        self,
        text: str,
        voice: str = "alloy",
        model: str = "tts-1"
    ) -> bytes:
        """Generate speech using OpenAI TTS."""
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")
        
        try:
            response = await self.http_client.post(
                f"{self.BASE_URL}/audio/speech",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model,
                    "input": text,
                    "voice": voice
                }
            )
            response.raise_for_status()
            return response.content
        
        except Exception as e:
            self._logger.error("OpenAI TTS call failed", error=str(e))
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for OpenAI API."""
        if not self.api_key:
            return {"status": "no_api_key", "healthy": False}
        
        try:
            response = await self.http_client.get(
                f"{self.BASE_URL}/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            return {"status": "healthy", "healthy": True}
        
        except Exception as e:
            return {"status": "unhealthy", "healthy": False, "error": str(e)}


class HuggingFaceClient(BaseAPIClient):
    """Hugging Face API client."""
    
    BASE_URL = "https://api-inference.huggingface.co"
    
    async def generate_text(
        self,
        prompt: str,
        model: str = "microsoft/DialoGPT-medium"
    ) -> Dict[str, Any]:
        """Generate text using Hugging Face API."""
        if not self.api_key:
            raise ValueError("Hugging Face token not configured")
        
        try:
            response = await self.http_client.post(
                f"{self.BASE_URL}/models/{model}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"inputs": prompt}
            )
            response.raise_for_status()
            return response.json()
        
        except Exception as e:
            self._logger.error("Hugging Face API call failed", error=str(e))
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for Hugging Face API."""
        if not self.api_key:
            return {"status": "no_api_key", "healthy": False}
        
        try:
            # Use a simple model for health check
            response = await self.http_client.post(
                f"{self.BASE_URL}/models/bert-base-uncased",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"inputs": "test"}
            )
            # Note: HF API might return 503 when model is loading, which is normal
            if response.status_code in [200, 503]:
                return {"status": "healthy", "healthy": True}
            else:
                response.raise_for_status()
        
        except Exception as e:
            return {"status": "unhealthy", "healthy": False, "error": str(e)}


# Convenience functions for dependency injection
_external_api_manager: Optional[ExternalAPIManager] = None


async def get_external_api_manager() -> ExternalAPIManager:
    """Get global external API manager instance."""
    global _external_api_manager
    if _external_api_manager is None:
        raise RuntimeError("External API manager not initialized")
    return _external_api_manager


async def set_external_api_manager(manager: ExternalAPIManager) -> None:
    """Set global external API manager instance."""
    global _external_api_manager
    _external_api_manager = manager


async def get_external_api_client(service: str) -> Optional[Dict[str, Any]]:
    """FastAPI dependency for external API client."""
    api_manager = get_external_api_manager()
    return api_manager.get_client(service) 