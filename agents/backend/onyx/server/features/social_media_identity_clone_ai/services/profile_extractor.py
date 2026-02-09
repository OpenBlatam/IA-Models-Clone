"""
Servicio para extraer perfiles de redes sociales

Refactorizado con:
- Mejor manejo de errores
- Type hints completos
- Logging estructurado
- Optimización de imports
"""

import logging
import asyncio
from typing import Optional, List, Dict, Any
from ..core.models import (
    SocialProfile,
    Platform,
    VideoContent,
    PostContent,
    CommentContent
)
from ..core.base_service import BaseService
from ..core.exceptions import (
    ProfileExtractionError,
    ConnectorError,
    CacheError
)
from ..connectors.tiktok_connector import TikTokConnector
from ..connectors.instagram_connector import InstagramConnector
from ..connectors.youtube_connector import YouTubeConnector
from ..config import get_settings

# Importación de helpers
from ..utils.cache_manager import get_cache
from ..utils.datetime_helpers import now
from ..utils.dict_helpers import extract_fields
from ..utils.collection_helpers import safe_map
from ..utils.serialization_helpers import serialize_model
from ..utils.logging_helpers import log_operation, log_error
from ..utils.error_handling_helpers import handle_errors
from ..utils.async_helpers import safe_gather

logger = logging.getLogger(__name__)


class ProfileExtractor(BaseService):
    """
    Extrae perfiles completos de redes sociales con caché
    
    Mejoras:
    - Herencia de BaseService para funcionalidad común
    - Manejo de errores robusto
    - Type hints completos
    - Logging estructurado
    """
    
    def __init__(self):
        super().__init__()
        self.cache = get_cache("profile_cache", max_size=100)
        self._initialize_connectors()
    
    def _initialize_connectors(self) -> None:
        """Inicializa conectores con manejo de errores"""
        try:
            self.tiktok_connector = TikTokConnector(
                api_key=self.settings.tiktok_api_key
            )
            self.instagram_connector = InstagramConnector(
                api_key=self.settings.instagram_api_key
            )
            self.youtube_connector = YouTubeConnector(
                api_key=self.settings.youtube_api_key
            )
        except Exception as e:
            self._handle_error(
                e,
                "connector_initialization",
                {"component": "ProfileExtractor"}
            )
    
    async def extract_tiktok_profile(
        self,
        username: str,
        use_cache: bool = True
    ) -> SocialProfile:
        """
        Extrae perfil completo de TikTok
        
        Args:
            username: Nombre de usuario de TikTok
            use_cache: Si usar caché (default: True)
            
        Returns:
            SocialProfile con todos los datos extraídos
            
        Raises:
            ProfileExtractionError: Si hay error en la extracción
            ValidationError: Si el username es inválido
        """
        self._validate_input(username, "username")
        self._log_operation("extract_tiktok_profile", username=username)
        
        # Verificar caché usando helper
        from ..utils.cache_helpers import generate_cache_key
        
        cache_key = generate_cache_key("tiktok_profile", username)
        if use_cache and self.cache.has(cache_key):
            logger.info(f"Perfil de TikTok obtenido de caché: {username}")
            return self.cache.get(cache_key)
        
        try:
            # Obtener información básica del perfil
            profile_data = await self.tiktok_connector.get_profile(username)
            
            # Extraer videos
            videos = await self._extract_tiktok_videos(username)
            
            # Construir perfil usando helpers
            from ..utils.dict_helpers import extract_fields
            from ..utils.datetime_helpers import now
            
            fields = extract_fields(
                profile_data,
                ["display_name", "bio", "profile_image_url", 
                 "followers_count", "following_count", "posts_count"],
                defaults={"followers_count": 0, "following_count": 0, "posts_count": 0}
            )
            
            profile = SocialProfile(
                platform=Platform.TIKTOK,
                username=username,
                **fields,
                videos=videos,
                extracted_at=now(),
                metadata=profile_data
            )
            
            # Guardar en caché
            if use_cache:
                try:
                    cache_key = generate_cache_key("tiktok_profile", username)
                    self.cache.set(cache_key, profile)
                except Exception as e:
                    self.logger.warning(f"Error guardando en caché: {e}")
            
            self.logger.info(
                f"Perfil de TikTok extraído exitosamente: "
                f"{len(videos)} videos"
            )
            return profile
            
        except ConnectorError as e:
            self._handle_error(
                e,
                "extract_tiktok_profile",
                {"username": username, "platform": "TikTok"}
            )
        except Exception as e:
            self._handle_error(
                e,
                "extract_tiktok_profile",
                {"username": username, "platform": "TikTok"}
            )
    
    @log_operation(logger, "extract_instagram_profile")
    @handle_errors(
        "extract_instagram_profile",
        error_types=(ConnectorError, CacheError),
        default_error=ProfileExtractionError
    )
    async def extract_instagram_profile(self, username: str, use_cache: bool = True) -> SocialProfile:
        """
        Extrae perfil completo de Instagram
        
        Args:
            username: Nombre de usuario de Instagram
            
        Returns:
            SocialProfile con todos los datos extraídos
        """
        from ..utils.cache_helpers import generate_cache_key
        
        # Verificar caché usando helper
        cache_key = generate_cache_key("instagram_profile", username)
        if use_cache and self.cache.has(cache_key):
            logger.info(f"Perfil de Instagram obtenido de caché: {username}")
            return self.cache.get(cache_key)
        
        # Obtener información básica del perfil
        profile_data = await self.instagram_connector.get_profile(username)
        
        # Extraer posts
        posts = await self._extract_instagram_posts(username)
        
        # Construir perfil usando extract_fields helper
        profile_fields = extract_fields(
            profile_data,
            [
                "display_name", "bio", "profile_image_url",
                "followers_count", "following_count", "posts_count"
            ],
            defaults={
                "display_name": username, "bio": "", "profile_image_url": None,
                "followers_count": 0, "following_count": 0, "posts_count": 0
            }
        )
        
        profile = SocialProfile(
            platform=Platform.INSTAGRAM,
            username=username,
            posts=posts,
            extracted_at=now(),
            metadata=profile_data,
            **profile_fields
        )
        
        # Guardar en caché usando helper
        if use_cache:
            self.cache.set(cache_key, profile)
        
        logger.info(f"Perfil de Instagram extraído: {len(posts)} posts")
        return profile
    
    @log_operation(logger, "extract_youtube_profile")
    @handle_errors(
        "extract_youtube_profile",
        error_types=(ConnectorError, CacheError),
        default_error=ProfileExtractionError
    )
    async def extract_youtube_profile(self, channel_id: str, use_cache: bool = True) -> SocialProfile:
        """
        Extrae perfil completo de YouTube
        
        Args:
            channel_id: ID del canal de YouTube
            
        Returns:
            SocialProfile con todos los datos extraídos
        """
        from ..utils.cache_helpers import generate_cache_key
        from ..utils.dict_helpers import safe_get
        
        # Verificar caché usando helper
        cache_key = generate_cache_key("youtube_profile", channel_id)
        if use_cache and self.cache.has(cache_key):
            logger.info(f"Perfil de YouTube obtenido de caché: {channel_id}")
            return self.cache.get(cache_key)
        
        # Obtener información básica del canal
        channel_data = await self.youtube_connector.get_channel(channel_id)
        
        # Extraer videos
        videos = await self._extract_youtube_videos(channel_id)
        
        # Construir perfil usando extract_fields helper
        profile_fields = extract_fields(
            channel_data,
            [
                "display_name", "description", "profile_image_url",
                "subscribers_count"
            ],
            defaults={
                "display_name": channel_id, "description": "", "profile_image_url": None,
                "subscribers_count": 0
            }
        )
        
        profile = SocialProfile(
            platform=Platform.YOUTUBE,
            username=safe_get(channel_data, "username", channel_id),
            bio=profile_fields.pop("description", ""),
            profile_image_url=profile_fields.pop("profile_image_url"),
            followers_count=profile_fields.pop("subscribers_count", 0),
            videos=videos,
            extracted_at=now(),
            metadata=channel_data,
            **profile_fields
        )
        
        # Guardar en caché usando helper
        if use_cache:
            self.cache.set(cache_key, profile)
        
        logger.info(f"Perfil de YouTube extraído: {len(videos)} videos")
        return profile
    
    async def _extract_tiktok_videos(
        self,
        username: str
    ) -> List[VideoContent]:
        """
        Extrae videos de TikTok
        
        Args:
            username: Nombre de usuario
            
        Returns:
            Lista de VideoContent
        """
        videos: List[VideoContent] = []
        max_videos = self.settings.max_videos_per_profile
        
        try:
            video_list = await self.tiktok_connector.get_videos(
                username,
                limit=max_videos
            )
            
            # Process videos safely using helper
            from ..utils.collection_helpers import safe_map
            from ..utils.dict_helpers import extract_fields
            
            videos = safe_map(
                video_list,
                lambda video_data: VideoContent(
                    **extract_fields(
                        video_data,
                        [
                            "video_id", "url", "title", "description", "duration",
                            "views", "likes", "comments", "created_at"
                        ],
                        defaults={"hashtags": [], "metadata": video_data}
                    )
                ),
                operation="process_tiktok_video"
            )
                    
        except ConnectorError as e:
            self.logger.error(
                f"Error extrayendo videos de TikTok: {e}",
                exc_info=True
            )
            raise ProfileExtractionError(
                f"Error extrayendo videos de TikTok: {str(e)}",
                error_code="VIDEO_EXTRACTION_ERROR",
                details={"username": username}
            ) from e
        except Exception as e:
            self.logger.error(
                f"Error inesperado extrayendo videos: {e}",
                exc_info=True
            )
            raise ProfileExtractionError(
                f"Error inesperado: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                details={"username": username}
            ) from e
        
        return videos
    
    @handle_errors(
        "extract_instagram_posts",
        error_types=(ConnectorError,),
        default_error=ProfileExtractionError
    )
    async def _extract_instagram_posts(self, username: str) -> List[PostContent]:
        """Extrae posts de Instagram usando safe_map helper"""
        max_posts = self.settings.max_posts_per_profile
        post_list = await self.instagram_connector.get_posts(username, limit=max_posts)
        
        posts = safe_map(
            post_list,
            lambda post_data: PostContent(
                **extract_fields(
                    post_data,
                    [
                        "post_id", "url", "caption", "image_urls",
                        "likes", "comments", "created_at", "hashtags", "mentions"
                    ],
                    defaults={
                        "image_urls": [], "hashtags": [], "mentions": [],
                        "metadata": post_data
                    }
                )
            ),
            error_logger=logger,
            operation_name="process_instagram_post",
            default_value=None
        )
        
        return [p for p in posts if p is not None]
    
    @handle_errors(
        "extract_youtube_videos",
        error_types=(ConnectorError,),
        default_error=ProfileExtractionError
    )
    async def _extract_youtube_videos(self, channel_id: str) -> List[VideoContent]:
        """Extrae videos de YouTube usando safe_map helper"""
        max_videos = self.settings.max_videos_per_profile
        video_list = await self.youtube_connector.get_videos(channel_id, limit=max_videos)
        
        videos = safe_map(
            video_list,
            lambda video_data: VideoContent(
                **extract_fields(
                    video_data,
                    [
                        "video_id", "url", "title", "description", "duration",
                        "views", "likes", "comments", "created_at", "hashtags"
                    ],
                    defaults={"hashtags": [], "metadata": video_data}
                )
            ),
            error_logger=logger,
            operation_name="process_youtube_video",
            default_value=None
        )
        
        return [v for v in videos if v is not None]
    
    async def extract_multiple_profiles(
        self,
        tiktok_usernames: Optional[List[str]] = None,
        instagram_usernames: Optional[List[str]] = None,
        youtube_channels: Optional[List[str]] = None,
        use_cache: bool = True,
        max_concurrent: int = 5
    ) -> Dict[str, List[SocialProfile]]:
        """
        Extrae múltiples perfiles en paralelo (optimizado para velocidad)
        
        Args:
            tiktok_usernames: Lista de usernames de TikTok
            instagram_usernames: Lista de usernames de Instagram
            youtube_channels: Lista de channel IDs de YouTube
            use_cache: Si usar caché
            max_concurrent: Máximo de extracciones concurrentes
            
        Returns:
            Diccionario con perfiles extraídos por plataforma
        """
        results = {
            "tiktok": [],
            "instagram": [],
            "youtube": []
        }
        
        tasks = []
        
        # TikTok profiles
        if tiktok_usernames:
            for username in tiktok_usernames:
                tasks.append(self.extract_tiktok_profile(username, use_cache))
        
        # Instagram profiles
        if instagram_usernames:
            for username in instagram_usernames:
                tasks.append(self.extract_instagram_profile(username, use_cache))
        
        # YouTube channels
        if youtube_channels:
            for channel_id in youtube_channels:
                tasks.append(self.extract_youtube_profile(channel_id, use_cache))
        
        # Procesar en paralelo usando safe_gather helper
        if tasks:
            # Aplicar límite de concurrencia
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def bounded_extract(task):
                async with semaphore:
                    return await task
            
            bounded_tasks = [bounded_extract(task) for task in tasks]
            completed = await safe_gather(*bounded_tasks, return_exceptions=True)
            
            # Organizar resultados por plataforma
            idx = 0
            if tiktok_usernames:
                results["tiktok"] = [
                    p for p in completed[idx:idx+len(tiktok_usernames)]
                    if p and not isinstance(p, Exception)
                ]
                idx += len(tiktok_usernames)
            if instagram_usernames:
                results["instagram"] = [
                    p for p in completed[idx:idx+len(instagram_usernames)]
                    if p and not isinstance(p, Exception)
                ]
                idx += len(instagram_usernames)
            if youtube_channels:
                results["youtube"] = [
                    p for p in completed[idx:idx+len(youtube_channels)]
                    if p and not isinstance(p, Exception)
                ]
        
        self.logger.info(
            f"Extracción paralela completada: "
            f"{len(results['tiktok'])} TikTok, "
            f"{len(results['instagram'])} Instagram, "
            f"{len(results['youtube'])} YouTube"
        )
        
        return results

