"""
OpenRouter Checker
==================

Checker especializado para OpenRouter API.
"""

import httpx
from typing import Dict, Any
from ...core.base.service_base import BaseService
from ...config.settings import get_settings


class OpenRouterChecker(BaseService):
    """Checker de OpenRouter API."""
    
    def __init__(self):
        """Inicializar checker."""
        super().__init__(logger_name=__name__)
        self.settings = get_settings()
    
    async def check(self) -> Dict[str, Any]:
        """
        Verificar conectividad de OpenRouter API.
        
        Returns:
            Diccionario con status y message
        """
        if not self.settings.openrouter_api_key:
            return {
                "status": "degraded",
                "message": "OpenRouter API key not configured",
            }
        
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(5.0, connect=2.0)
            ) as client:
                response = await client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {self.settings.openrouter_api_key}"},
                )
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "message": "OpenRouter API is accessible",
                    }
                else:
                    return {
                        "status": "degraded",
                        "message": f"OpenRouter API returned {response.status_code}",
                    }
        except httpx.TimeoutException:
            return {
                "status": "degraded",
                "message": "OpenRouter API timeout",
            }
        except httpx.ConnectError:
            return {
                "status": "degraded",
                "message": "OpenRouter API connection failed",
            }
        except Exception as e:
            self.log_warning(f"OpenRouter health check failed: {e}")
            return {
                "status": "degraded",
                "message": f"OpenRouter API error: {str(e)}",
            }

