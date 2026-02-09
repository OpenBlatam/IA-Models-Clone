"""
Conector para API de Instagram

Refactorizado con:
- Soporte para instaloader (scraping de Instagram)
- Mejor manejo de errores
- Type hints completos
- Logging estructurado
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    import instaloader
    INSTALOADER_AVAILABLE = True
except ImportError:
    INSTALOADER_AVAILABLE = False

from .base_connector import BaseConnector
from ..core.exceptions import ConnectorError

logger = logging.getLogger(__name__)


class InstagramConnector(BaseConnector):
    """
    Conector para extraer datos de Instagram
    
    Refactorizado para usar BaseConnector, eliminando código duplicado
    
    Soporta:
    - Instagram Graph API (oficial, requiere OAuth2)
    - Instaloader (scraping, no oficial pero funcional)
    """
    
    def __init__(self, api_key: Optional[str] = None, use_scraping: bool = False):
        """
        Inicializa el conector de Instagram
        
        Args:
            api_key: API key para Instagram Graph API (opcional)
            use_scraping: Si usar instaloader para scraping (default: False)
        """
        super().__init__(api_key=api_key)
        self.use_scraping = use_scraping
        self._loader = None
        
        # Inicializar instaloader si está disponible y se solicita
        if use_scraping and INSTALOADER_AVAILABLE:
            try:
                self._loader = instaloader.Instaloader(
                    download_videos=False,
                    download_video_thumbnails=False,
                    download_geotags=False,
                    download_comments=False,
                    save_metadata=False,
                    compress_json=False
                )
                logger.info("Instaloader inicializado para scraping")
            except Exception as e:
                logger.warning(f"Error inicializando Instaloader: {e}")
    
    async def get_profile(self, username: str) -> Dict[str, Any]:
        """
        Obtiene información básica del perfil de Instagram
        
        Args:
            username: Nombre de usuario de Instagram
            
        Returns:
            Diccionario con información del perfil
            
        Raises:
            ConnectorError: Si hay error obteniendo el perfil
        """
        # Intentar con instaloader si está disponible
        if self.use_scraping and self._loader:
            try:
                return await self._get_profile_with_instaloader(username)
            except Exception as e:
                logger.warning(f"Error con instaloader, usando fallback: {e}")
        
        # Fallback a API oficial o estructura de ejemplo
        async def _fetch_profile():
            if self.api_key:
                # TODO: Implementar llamada real a Instagram Graph API
                # from facebook_business.api import FacebookAdsApi
                # from facebook_business.adobjects.instagramuser import InstagramUser
                pass
            
            # Estructura de ejemplo
            return {
                "username": username,
                "display_name": username,
                "bio": "Perfil de Instagram",
                "profile_image_url": None,
                "followers_count": 0,
                "following_count": 0,
                "posts_count": 0,
            }
        
        return await self._execute_with_retry(
            "get_profile",
            _fetch_profile,
            username=username
        )
    
    async def _get_profile_with_instaloader(self, username: str) -> Dict[str, Any]:
        """
        Obtiene perfil usando instaloader (scraping)
        
        Args:
            username: Nombre de usuario
            
        Returns:
            Diccionario con información del perfil
        """
        if not self._loader:
            raise ValueError("Instaloader no inicializado")
        
        try:
            profile = instaloader.Profile.from_username(self._loader.context, username)
            
            return {
                "username": profile.username,
                "display_name": profile.full_name or profile.username,
                "bio": profile.biography or "",
                "profile_image_url": profile.profile_pic_url,
                "followers_count": profile.followers,
                "following_count": profile.followees,
                "posts_count": profile.mediacount,
                "is_verified": profile.is_verified,
                "is_private": profile.is_private,
            }
        except instaloader.exceptions.ProfileNotExistsException:
            logger.error(f"Perfil de Instagram no existe: {username}")
            raise
        except instaloader.exceptions.PrivateProfileNotFollowedException:
            logger.error(f"Perfil de Instagram es privado: {username}")
            raise
        except Exception as e:
            logger.error(f"Error obteniendo perfil con instaloader: {e}")
            raise
    
    async def get_posts(self, username: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Obtiene posts del perfil de Instagram
        
        Args:
            username: Nombre de usuario
            limit: Límite de posts a obtener
            
        Returns:
            Lista de diccionarios con información de posts
        """
        logger.info(f"Obteniendo posts de Instagram: {username}, límite: {limit}")
        
        # Intentar con instaloader si está disponible
        if self.use_scraping and self._loader:
            try:
                return await self._get_posts_with_instaloader(username, limit)
            except Exception as e:
                logger.warning(f"Error con instaloader, usando fallback: {e}")
        
        # Fallback
        async def _fetch_posts():
            if self.api_key:
                # TODO: Implementar llamada real a Instagram Graph API
                pass
            return []
        
        try:
            return await self.retry_handler.execute_async(
                lambda: self.circuit_breaker.call_async(_fetch_posts)
            )
        except Exception as e:
            logger.error(f"Error obteniendo posts de Instagram: {e}")
            return []
    
    async def _get_posts_with_instaloader(self, username: str, limit: int) -> List[Dict[str, Any]]:
        """Obtiene posts usando instaloader"""
        if not self._loader:
            return []
        
        try:
            profile = instaloader.Profile.from_username(self._loader.context, username)
            posts = []
            
            for post in profile.get_posts():
                if len(posts) >= limit:
                    break
                
                posts.append({
                    "post_id": post.shortcode,
                    "url": f"https://www.instagram.com/p/{post.shortcode}/",
                    "caption": post.caption or "",
                    "image_urls": [post.url] if post.is_video else [post.url],
                    "likes": post.likes,
                    "comments": post.comments,
                    "created_at": post.date_utc,
                    "is_video": post.is_video,
                })
            
            return posts
        except Exception as e:
            logger.error(f"Error obteniendo posts con instaloader: {e}")
            return []
    
    async def get_post_comments(self, post_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Obtiene comentarios de un post de Instagram
        
        Args:
            post_id: ID del post (shortcode)
            limit: Límite de comentarios
            
        Returns:
            Lista de diccionarios con información de comentarios
        """
        logger.info(f"Obteniendo comentarios de post Instagram: {post_id}")
        
        if self.use_scraping and self._loader:
            try:
                return await self._get_comments_with_instaloader(post_id, limit)
            except Exception as e:
                logger.warning(f"Error con instaloader: {e}")
        
        # Fallback
        async def _fetch_comments():
            if self.api_key:
                # TODO: Implementar llamada real a Instagram Graph API
                pass
            return []
        
        try:
            return await self.retry_handler.execute_async(
                lambda: self.circuit_breaker.call_async(_fetch_comments)
            )
        except Exception as e:
            logger.error(f"Error obteniendo comentarios: {e}")
            return []
    
    async def _get_comments_with_instaloader(self, post_id: str, limit: int) -> List[Dict[str, Any]]:
        """Obtiene comentarios usando instaloader"""
        if not self._loader:
            return []
        
        try:
            post = instaloader.Post.from_shortcode(self._loader.context, post_id)
            comments = []
            
            for comment in post.get_comments():
                if len(comments) >= limit:
                    break
                
                comments.append({
                    "comment_id": str(comment.id),
                    "text": comment.text,
                    "author": comment.owner.username,
                    "likes": comment.likes_count,
                    "created_at": comment.created_at_utc,
                })
            
            return comments
        except Exception as e:
            logger.error(f"Error obteniendo comentarios con instaloader: {e}")
            return []
