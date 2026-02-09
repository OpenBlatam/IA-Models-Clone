"""
Community Manager - Gestor Principal
=====================================

Clase principal que coordina todas las funcionalidades del sistema.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from .scheduler import PostScheduler
from .calendar import ContentCalendar
from ..services.meme_manager import MemeManager
from ..services.social_media_connector import SocialMediaConnector
from ..services.analytics_service import AnalyticsService

logger = logging.getLogger(__name__)


class CommunityManager:
    """Gestor principal del sistema de community management"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar el Community Manager
        
        Args:
            config: Configuración opcional del sistema
        """
        self.config = config or {}
        self.scheduler = PostScheduler()
        self.calendar = ContentCalendar()
        self.meme_manager = MemeManager()
        self.social_connector = SocialMediaConnector()
        self.analytics_service = AnalyticsService()
        
        logger.info("Community Manager inicializado")
    
    def schedule_post(
        self,
        content: str,
        platforms: List[str],
        scheduled_time: Optional[datetime] = None,
        media_paths: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Programar una publicación en múltiples plataformas
        
        Args:
            content: Contenido del post
            platforms: Lista de plataformas (facebook, instagram, twitter, etc.)
            scheduled_time: Fecha y hora programada
            media_paths: Rutas a archivos multimedia
            tags: Tags para categorización
            
        Returns:
            Dict con información de la publicación programada
        """
        # Validaciones
        from ..utils.validators import (
            validate_platforms,
            validate_content_length,
            validate_scheduled_time,
            validate_media_paths
        )
        
        # Validar plataformas
        is_valid, error = validate_platforms(platforms)
        if not is_valid:
            raise ValueError(error)
        
        # Validar contenido para cada plataforma
        for platform in platforms:
            is_valid, error = validate_content_length(content, platform)
            if not is_valid:
                raise ValueError(error)
        
        # Validar fecha programada
        if scheduled_time:
            is_valid, error = validate_scheduled_time(scheduled_time)
            if not is_valid:
                raise ValueError(error)
        
        # Validar medios
        if media_paths:
            for platform in platforms:
                is_valid, error = validate_media_paths(media_paths, platform)
                if not is_valid:
                    raise ValueError(error)
        
        # Sanitizar contenido
        from ..utils.helpers import sanitize_content
        sanitized_content = sanitize_content(content)
        
        post_data = {
            "content": sanitized_content,
            "platforms": platforms,
            "scheduled_time": scheduled_time or datetime.now(),
            "media_paths": media_paths or [],
            "tags": tags or [],
            "created_at": datetime.now()
        }
        
        post_id = self.scheduler.add_post(post_data)
        self.calendar.add_event(post_id, post_data["scheduled_time"], post_data)
        
        logger.info(f"Post programado: {post_id} para {platforms}")
        
        return {
            "post_id": post_id,
            "status": "scheduled",
            "scheduled_time": post_data["scheduled_time"].isoformat(),
            "platforms": platforms
        }
    
    def publish_now(
        self,
        content: str,
        platforms: List[str],
        media_paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Publicar inmediatamente en las plataformas especificadas
        
        Args:
            content: Contenido del post
            platforms: Lista de plataformas
            media_paths: Rutas a archivos multimedia
            
        Returns:
            Dict con resultados de la publicación
        """
        results = {}
        
        for platform in platforms:
            try:
                result = self.social_connector.publish(
                    platform=platform,
                    content=content,
                    media_paths=media_paths or []
                )
                results[platform] = {
                    "status": "success",
                    "post_id": result.get("post_id"),
                    "url": result.get("url")
                }
            except Exception as e:
                logger.error(f"Error publicando en {platform}: {e}")
                results[platform] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results
    
    def add_meme(
        self,
        image_path: str,
        caption: Optional[str] = None,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None
    ) -> str:
        """
        Agregar un meme al sistema
        
        Args:
            image_path: Ruta a la imagen del meme
            caption: Caption opcional
            tags: Tags para categorización
            category: Categoría del meme
            
        Returns:
            ID del meme agregado
        """
        meme_id = self.meme_manager.add_meme(
            image_path=image_path,
            caption=caption,
            tags=tags or [],
            category=category
        )
        
        logger.info(f"Meme agregado: {meme_id}")
        return meme_id
    
    def get_meme_for_post(self, category: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Obtener un meme aleatorio para usar en un post
        
        Args:
            category: Categoría específica (opcional)
            
        Returns:
            Dict con información del meme o None
        """
        return self.meme_manager.get_random_meme(category=category)
    
    def connect_platform(
        self,
        platform: str,
        credentials: Dict[str, str]
    ) -> bool:
        """
        Conectar una plataforma social
        
        Args:
            platform: Nombre de la plataforma
            credentials: Credenciales de autenticación
            
        Returns:
            True si la conexión fue exitosa
        """
        try:
            self.social_connector.connect(platform, credentials)
            logger.info(f"Plataforma {platform} conectada exitosamente")
            return True
        except Exception as e:
            logger.error(f"Error conectando {platform}: {e}")
            return False
    
    def get_calendar_view(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtener vista del calendario de publicaciones
        
        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Lista de eventos programados
        """
        return self.calendar.get_events(start_date, end_date)
    
    def get_analytics(
        self,
        platform: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Obtener analytics de las publicaciones
        
        Args:
            platform: Plataforma específica (opcional)
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Dict con métricas de analytics
        """
        from datetime import timedelta
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        days = (end_date - start_date).days
        if days <= 0:
            days = 30
        
        all_posts = self.scheduler.get_all_posts()
        
        published_posts = [
            p for p in all_posts
            if p.get("status") == "published"
            and p.get("published_at")
        ]
        
        if start_date or end_date:
            filtered_posts = []
            for post in published_posts:
                published_at_str = post.get("published_at")
                if published_at_str:
                    if isinstance(published_at_str, str):
                        published_at = datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
                    else:
                        published_at = published_at_str
                    
                    if start_date and published_at < start_date:
                        continue
                    if end_date and published_at > end_date:
                        continue
                    
                    filtered_posts.append(post)
            published_posts = filtered_posts
        
        total_posts = len(published_posts)
        
        if platform:
            platform_analytics = self.analytics_service.get_platform_analytics(platform, days)
            return {
                "platform": platform,
                "total_posts": total_posts,
                "period": {
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                    "days": days
                },
                **platform_analytics
            }
        
        platforms_data = {}
        all_platforms = set()
        
        for post in published_posts:
            post_platforms = post.get("platforms", [])
            all_platforms.update(post_platforms)
        
        total_engagement = 0
        for plat in all_platforms:
            plat_analytics = self.analytics_service.get_platform_analytics(plat, days)
            platforms_data[plat] = plat_analytics
            total_engagement += plat_analytics.get("total_engagement", 0)
        
        best_posts = self.analytics_service.get_best_performing_posts(limit=10)
        
        return {
            "total_posts": total_posts,
            "total_engagement": total_engagement,
            "period": {
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "days": days
            },
            "platforms": platforms_data,
            "best_performing_posts": best_posts[:5]
        }

