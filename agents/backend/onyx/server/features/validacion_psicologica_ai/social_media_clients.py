"""
Clientes para APIs de Redes Sociales
=====================================
Integración con APIs reales de diferentes plataformas
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import structlog
import asyncio
import aiohttp
from abc import ABC, abstractmethod

from .models import SocialMediaConnection, SocialMediaPlatform
from .config import config
from .exceptions import (
    SocialMediaAPIError,
    TokenExpiredError,
    TokenInvalidError,
    RateLimitExceededError,
)

logger = structlog.get_logger()


class BaseSocialMediaClient(ABC):
    """Cliente base para APIs de redes sociales"""
    
    def __init__(self, connection: SocialMediaConnection):
        """Inicializar cliente"""
        self.connection = connection
        self.platform = connection.platform
        self.access_token = connection.access_token
        self.timeout = aiohttp.ClientTimeout(total=config.social_media_timeout)
    
    async def _make_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Realizar petición HTTP
        
        Args:
            method: Método HTTP
            url: URL de la petición
            headers: Headers adicionales
            params: Parámetros de query
            json_data: Datos JSON para el body
            
        Returns:
            Respuesta de la API
        """
        default_headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        if headers:
            default_headers.update(headers)
        
        for attempt in range(config.social_media_retry_attempts):
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.request(
                        method,
                        url,
                        headers=default_headers,
                        params=params,
                        json=json_data
                    ) as response:
                        if response.status == 429:
                            raise RateLimitExceededError(
                                "Rate limit exceeded",
                                error_code="RATE_LIMIT",
                                details={"platform": self.platform.value}
                            )
                        
                        if response.status == 401:
                            raise TokenInvalidError(
                                "Invalid or expired token",
                                error_code="TOKEN_INVALID",
                                details={"platform": self.platform.value}
                            )
                        
                        if response.status >= 400:
                            error_text = await response.text()
                            raise SocialMediaAPIError(
                                f"API error: {response.status}",
                                error_code=f"API_ERROR_{response.status}",
                                details={"platform": self.platform.value, "error": error_text}
                            )
                        
                        return await response.json()
            
            except asyncio.TimeoutError:
                if attempt == config.social_media_retry_attempts - 1:
                    raise SocialMediaAPIError(
                        "Request timeout",
                        error_code="TIMEOUT",
                        details={"platform": self.platform.value}
                    )
                await asyncio.sleep(config.social_media_retry_delay * (attempt + 1))
            
            except (TokenInvalidError, RateLimitExceededError):
                raise
            
            except Exception as e:
                if attempt == config.social_media_retry_attempts - 1:
                    raise SocialMediaAPIError(
                        f"Request failed: {str(e)}",
                        error_code="REQUEST_FAILED",
                        details={"platform": self.platform.value, "error": str(e)}
                    )
                await asyncio.sleep(config.social_media_retry_delay * (attempt + 1))
        
        raise SocialMediaAPIError(
            "Max retries exceeded",
            error_code="MAX_RETRIES",
            details={"platform": self.platform.value}
        )
    
    @abstractmethod
    async def get_profile(self) -> Dict[str, Any]:
        """Obtener perfil del usuario"""
        pass
    
    @abstractmethod
    async def get_posts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener posts del usuario"""
        pass


class InstagramClient(BaseSocialMediaClient):
    """Cliente para Instagram Graph API"""
    
    BASE_URL = "https://graph.instagram.com"
    
    async def get_profile(self) -> Dict[str, Any]:
        """Obtener perfil de Instagram"""
        url = f"{self.BASE_URL}/me"
        params = {
            "fields": "id,username,account_type,media_count",
            "access_token": self.access_token
        }
        
        try:
            data = await self._make_request("GET", url, params=params)
            return {
                "username": data.get("username", ""),
                "display_name": data.get("username", ""),
                "follower_count": 0,  # Requiere permisos adicionales
                "following_count": 0,
                "post_count": data.get("media_count", 0),
                "verified": False,
                "created_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error("Error fetching Instagram profile", error=str(e))
            raise
    
    async def get_posts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener posts de Instagram"""
        url = f"{self.BASE_URL}/me/media"
        params = {
            "fields": "id,caption,like_count,comments_count,timestamp",
            "limit": min(limit, 100),
            "access_token": self.access_token
        }
        
        try:
            data = await self._make_request("GET", url, params=params)
            posts = []
            
            for item in data.get("data", [])[:limit]:
                posts.append({
                    "id": item.get("id", ""),
                    "text": item.get("caption", ""),
                    "created_at": item.get("timestamp", datetime.utcnow().isoformat()),
                    "likes": item.get("like_count", 0),
                    "comments": item.get("comments_count", 0),
                    "shares": 0
                })
            
            return posts
        except Exception as e:
            logger.error("Error fetching Instagram posts", error=str(e))
            raise


class TwitterClient(BaseSocialMediaClient):
    """Cliente para Twitter API v2"""
    
    BASE_URL = "https://api.twitter.com/2"
    
    async def get_profile(self) -> Dict[str, Any]:
        """Obtener perfil de Twitter"""
        url = f"{self.BASE_URL}/users/me"
        params = {
            "user.fields": "name,username,public_metrics,verified"
        }
        
        try:
            data = await self._make_request("GET", url, params=params)
            user_data = data.get("data", {})
            metrics = user_data.get("public_metrics", {})
            
            return {
                "username": user_data.get("username", ""),
                "display_name": user_data.get("name", ""),
                "follower_count": metrics.get("followers_count", 0),
                "following_count": metrics.get("following_count", 0),
                "post_count": metrics.get("tweet_count", 0),
                "verified": user_data.get("verified", False),
                "created_at": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error("Error fetching Twitter profile", error=str(e))
            raise
    
    async def get_posts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener tweets del usuario"""
        url = f"{self.BASE_URL}/users/me/tweets"
        params = {
            "max_results": min(limit, 100),
            "tweet.fields": "created_at,public_metrics,text"
        }
        
        try:
            data = await self._make_request("GET", url, params=params)
            posts = []
            
            for item in data.get("data", [])[:limit]:
                metrics = item.get("public_metrics", {})
                posts.append({
                    "id": item.get("id", ""),
                    "text": item.get("text", ""),
                    "created_at": item.get("created_at", datetime.utcnow().isoformat()),
                    "likes": metrics.get("like_count", 0),
                    "comments": metrics.get("reply_count", 0),
                    "shares": metrics.get("retweet_count", 0)
                })
            
            return posts
        except Exception as e:
            logger.error("Error fetching Twitter posts", error=str(e))
            raise


class SocialMediaClientFactory:
    """Factory para crear clientes de redes sociales"""
    
    _clients = {
        SocialMediaPlatform.INSTAGRAM: InstagramClient,
        SocialMediaPlatform.TWITTER: TwitterClient,
        # Agregar más clientes aquí
    }
    
    @classmethod
    def create_client(
        cls,
        connection: SocialMediaConnection
    ) -> BaseSocialMediaClient:
        """
        Crear cliente para una conexión
        
        Args:
            connection: Conexión a red social
            
        Returns:
            Cliente configurado
        """
        client_class = cls._clients.get(connection.platform)
        
        if not client_class:
            # Cliente mock para plataformas no implementadas
            return MockSocialMediaClient(connection)
        
        return client_class(connection)


class MockSocialMediaClient(BaseSocialMediaClient):
    """Cliente mock para desarrollo y testing"""
    
    async def get_profile(self) -> Dict[str, Any]:
        """Obtener perfil mock"""
        await asyncio.sleep(0.1)  # Simular latencia
        return {
            "username": f"user_{self.platform.value}",
            "display_name": f"User on {self.platform.value}",
            "follower_count": 100,
            "following_count": 50,
            "post_count": 25,
            "verified": False,
            "created_at": datetime.utcnow().isoformat()
        }
    
    async def get_posts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener posts mock"""
        await asyncio.sleep(0.2)  # Simular latencia
        
        posts = []
        for i in range(min(limit, 10)):
            posts.append({
                "id": f"post_{i}",
                "text": f"Sample post {i} from {self.platform.value}",
                "created_at": (datetime.utcnow() - timedelta(days=i)).isoformat(),
                "likes": 10 + i,
                "comments": 2 + i,
                "shares": 1
            })
        
        return posts




