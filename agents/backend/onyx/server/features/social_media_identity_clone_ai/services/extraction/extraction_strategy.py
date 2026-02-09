"""
Strategy pattern para diferentes estrategias de extracción
Mejora modularidad y extensibilidad
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from ...core.models import SocialProfile, Platform
from ...core.interfaces import IConnector


class ExtractionStrategy(ABC):
    """Estrategia base para extracción de perfiles"""
    
    def __init__(self, connector: IConnector):
        """
        Inicializa la estrategia
        
        Args:
            connector: Conector para la plataforma
        """
        self.connector = connector
    
    @abstractmethod
    async def extract_profile(
        self,
        identifier: str,
        use_cache: bool = True
    ) -> SocialProfile:
        """
        Extrae perfil completo
        
        Args:
            identifier: Username o ID del perfil
            use_cache: Si usar caché
            
        Returns:
            SocialProfile completo
        """
        pass
    
    @abstractmethod
    def get_platform(self) -> Platform:
        """Retorna la plataforma de esta estrategia"""
        pass


class TikTokExtractionStrategy(ExtractionStrategy):
    """Estrategia para extracción de perfiles de TikTok"""
    
    async def extract_profile(
        self,
        identifier: str,
        use_cache: bool = True
    ) -> SocialProfile:
        """Extrae perfil de TikTok"""
        from ...core.models import VideoContent, PostContent, CommentContent
        from datetime import datetime
        
        # Obtener datos del conector
        profile_data = await self.connector.get_profile(identifier)
        videos_data = await self.connector.get_videos(identifier)
        
        # Convertir a modelos
        videos = [
            VideoContent(
                video_id=video.get("id", ""),
                url=video.get("url", ""),
                title=video.get("title", ""),
                description=video.get("description", ""),
                transcript=video.get("transcript"),
                views=video.get("views", 0),
                likes=video.get("likes", 0),
                comments_count=video.get("comments_count", 0),
                created_at=video.get("created_at", datetime.now())
            )
            for video in videos_data
        ]
        
        return SocialProfile(
            platform=Platform.TIKTOK,
            username=profile_data.get("username", identifier),
            display_name=profile_data.get("display_name", identifier),
            bio=profile_data.get("bio", ""),
            profile_image_url=profile_data.get("profile_image_url"),
            followers_count=profile_data.get("followers_count", 0),
            following_count=profile_data.get("following_count", 0),
            posts_count=profile_data.get("posts_count", 0),
            videos=videos,
            posts=[],
            comments=[]
        )
    
    def get_platform(self) -> Platform:
        return Platform.TIKTOK


class InstagramExtractionStrategy(ExtractionStrategy):
    """Estrategia para extracción de perfiles de Instagram"""
    
    async def extract_profile(
        self,
        identifier: str,
        use_cache: bool = True
    ) -> SocialProfile:
        """Extrae perfil de Instagram"""
        from ...core.models import PostContent, CommentContent
        from datetime import datetime
        
        profile_data = await self.connector.get_profile(identifier)
        posts_data = await self.connector.get_videos(identifier)  # Reutiliza método
        
        posts = [
            PostContent(
                post_id=post.get("id", ""),
                url=post.get("url", ""),
                caption=post.get("caption", ""),
                image_url=post.get("image_url"),
                likes=post.get("likes", 0),
                comments_count=post.get("comments_count", 0),
                created_at=post.get("created_at", datetime.now())
            )
            for post in posts_data
        ]
        
        return SocialProfile(
            platform=Platform.INSTAGRAM,
            username=profile_data.get("username", identifier),
            display_name=profile_data.get("display_name", identifier),
            bio=profile_data.get("bio", ""),
            profile_image_url=profile_data.get("profile_image_url"),
            followers_count=profile_data.get("followers_count", 0),
            following_count=profile_data.get("following_count", 0),
            posts_count=profile_data.get("posts_count", 0),
            videos=[],
            posts=posts,
            comments=[]
        )
    
    def get_platform(self) -> Platform:
        return Platform.INSTAGRAM


class YouTubeExtractionStrategy(ExtractionStrategy):
    """Estrategia para extracción de perfiles de YouTube"""
    
    async def extract_profile(
        self,
        identifier: str,
        use_cache: bool = True
    ) -> SocialProfile:
        """Extrae perfil de YouTube"""
        from ...core.models import VideoContent
        from datetime import datetime
        
        profile_data = await self.connector.get_profile(identifier)
        videos_data = await self.connector.get_videos(identifier)
        
        videos = [
            VideoContent(
                video_id=video.get("id", ""),
                url=video.get("url", ""),
                title=video.get("title", ""),
                description=video.get("description", ""),
                transcript=video.get("transcript"),
                views=video.get("views", 0),
                likes=video.get("likes", 0),
                comments_count=video.get("comments_count", 0),
                created_at=video.get("created_at", datetime.now())
            )
            for video in videos_data
        ]
        
        return SocialProfile(
            platform=Platform.YOUTUBE,
            username=profile_data.get("username", identifier),
            display_name=profile_data.get("display_name", identifier),
            bio=profile_data.get("bio", ""),
            profile_image_url=profile_data.get("profile_image_url"),
            followers_count=profile_data.get("followers_count", 0),
            following_count=profile_data.get("following_count", 0),
            posts_count=profile_data.get("posts_count", 0),
            videos=videos,
            posts=[],
            comments=[]
        )
    
    def get_platform(self) -> Platform:
        return Platform.YOUTUBE

