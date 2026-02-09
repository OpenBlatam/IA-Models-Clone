"""
Servicio de extracción de perfiles usando Strategy Pattern
Mejora modularidad y separación de responsabilidades
"""

import logging
import asyncio
from typing import Optional, List, Dict
from collections import defaultdict

from ...core.models import SocialProfile, Platform
from ...core.interfaces import IProfileExtractor
from ...core.base_service import BaseService
from ...core.exceptions import ProfileExtractionError
from ...connectors.tiktok_connector import TikTokConnector
from ...connectors.instagram_connector import InstagramConnector
from ...connectors.youtube_connector import YouTubeConnector
from ...utils.cache import CacheManager
from .extraction_strategy import (
    TikTokExtractionStrategy,
    InstagramExtractionStrategy,
    YouTubeExtractionStrategy
)

logger = logging.getLogger(__name__)


class ProfileExtractorService(BaseService, IProfileExtractor):
    """
    Servicio de extracción de perfiles usando Strategy Pattern
    
    Mejoras:
    - Usa estrategias para cada plataforma
    - Mejor separación de responsabilidades
    - Fácil agregar nuevas plataformas
    """
    
    def __init__(self):
        super().__init__()
        self.cache = CacheManager()
        self._strategies = {}
        self._initialize_strategies()
    
    def _initialize_strategies(self) -> None:
        """Inicializa estrategias para cada plataforma"""
        try:
            # Crear conectores
            tiktok_connector = TikTokConnector(
                api_key=self.settings.tiktok_api_key
            )
            instagram_connector = InstagramConnector(
                api_key=self.settings.instagram_api_key
            )
            youtube_connector = YouTubeConnector(
                api_key=self.settings.youtube_api_key
            )
            
            # Crear estrategias
            self._strategies[Platform.TIKTOK] = TikTokExtractionStrategy(
                tiktok_connector
            )
            self._strategies[Platform.INSTAGRAM] = InstagramExtractionStrategy(
                instagram_connector
            )
            self._strategies[Platform.YOUTUBE] = YouTubeExtractionStrategy(
                youtube_connector
            )
            
            self.logger.info("Estrategias de extracción inicializadas")
        except Exception as e:
            self._handle_error(
                e,
                "strategy_initialization",
                {"component": "ProfileExtractorService"}
            )
    
    async def extract_tiktok_profile(
        self,
        username: str,
        use_cache: bool = True
    ) -> SocialProfile:
        """Extrae perfil de TikTok usando estrategia"""
        self._validate_input(username, "username")
        self._log_operation("extract_tiktok_profile", username=username)
        
        # Verificar caché
        if use_cache:
            try:
                cached = self.cache.get("tiktok", username)
                if cached:
                    self.logger.debug(f"Perfil de TikTok obtenido de caché: {username}")
                    return cached
            except Exception as e:
                self.logger.warning(f"Error accediendo caché: {e}")
        
        # Extraer usando estrategia
        strategy = self._strategies.get(Platform.TIKTOK)
        if not strategy:
            raise ProfileExtractionError(
                f"Estrategia para TikTok no disponible",
                details={"platform": "tiktok", "username": username}
            )
        
        try:
            profile = await strategy.extract_profile(username, use_cache)
            
            # Guardar en caché
            if use_cache:
                try:
                    self.cache.set("tiktok", username, profile)
                except Exception as e:
                    self.logger.warning(f"Error guardando en caché: {e}")
            
            return profile
        except Exception as e:
            self._handle_error(
                e,
                "extract_tiktok_profile",
                {"username": username}
            )
    
    async def extract_instagram_profile(
        self,
        username: str,
        use_cache: bool = True
    ) -> SocialProfile:
        """Extrae perfil de Instagram usando estrategia"""
        self._validate_input(username, "username")
        self._log_operation("extract_instagram_profile", username=username)
        
        if use_cache:
            try:
                cached = self.cache.get("instagram", username)
                if cached:
                    return cached
            except Exception:
                pass
        
        strategy = self._strategies.get(Platform.INSTAGRAM)
        if not strategy:
            raise ProfileExtractionError(
                f"Estrategia para Instagram no disponible",
                details={"platform": "instagram", "username": username}
            )
        
        try:
            profile = await strategy.extract_profile(username, use_cache)
            if use_cache:
                try:
                    self.cache.set("instagram", username, profile)
                except Exception:
                    pass
            return profile
        except Exception as e:
            self._handle_error(e, "extract_instagram_profile", {"username": username})
    
    async def extract_youtube_profile(
        self,
        channel_id: str,
        use_cache: bool = True
    ) -> SocialProfile:
        """Extrae perfil de YouTube usando estrategia"""
        self._validate_input(channel_id, "channel_id")
        self._log_operation("extract_youtube_profile", channel_id=channel_id)
        
        if use_cache:
            try:
                cached = self.cache.get("youtube", channel_id)
                if cached:
                    return cached
            except Exception:
                pass
        
        strategy = self._strategies.get(Platform.YOUTUBE)
        if not strategy:
            raise ProfileExtractionError(
                f"Estrategia para YouTube no disponible",
                details={"platform": "youtube", "channel_id": channel_id}
            )
        
        try:
            profile = await strategy.extract_profile(channel_id, use_cache)
            if use_cache:
                try:
                    self.cache.set("youtube", channel_id, profile)
                except Exception:
                    pass
            return profile
        except Exception as e:
            self._handle_error(e, "extract_youtube_profile", {"channel_id": channel_id})
    
    async def extract_multiple_profiles(
        self,
        tiktok_usernames: Optional[List[str]] = None,
        instagram_usernames: Optional[List[str]] = None,
        youtube_channel_ids: Optional[List[str]] = None,
        max_concurrent: int = 5
    ) -> Dict[str, SocialProfile]:
        """
        Extrae múltiples perfiles en paralelo usando estrategias
        
        Args:
            tiktok_usernames: Lista de usernames de TikTok
            instagram_usernames: Lista de usernames de Instagram
            youtube_channel_ids: Lista de IDs de canales de YouTube
            max_concurrent: Máximo de extracciones concurrentes
            
        Returns:
            Diccionario con perfiles extraídos (key: "platform:identifier")
        """
        self._log_operation(
            "extract_multiple_profiles",
            tiktok_count=len(tiktok_usernames or []),
            instagram_count=len(instagram_usernames or []),
            youtube_count=len(youtube_channel_ids or [])
        )
        
        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}
        
        async def extract_with_semaphore(platform: Platform, identifier: str, extract_func):
            async with semaphore:
                try:
                    profile = await extract_func(identifier)
                    key = f"{platform.value}:{identifier}"
                    results[key] = profile
                except Exception as e:
                    self.logger.error(
                        f"Error extrayendo {platform.value}/{identifier}: {e}",
                        exc_info=True
                    )
        
        tasks = []
        
        # TikTok
        if tiktok_usernames:
            for username in tiktok_usernames:
                tasks.append(
                    extract_with_semaphore(
                        Platform.TIKTOK,
                        username,
                        lambda u=username: self.extract_tiktok_profile(u)
                    )
                )
        
        # Instagram
        if instagram_usernames:
            for username in instagram_usernames:
                tasks.append(
                    extract_with_semaphore(
                        Platform.INSTAGRAM,
                        username,
                        lambda u=username: self.extract_instagram_profile(u)
                    )
                )
        
        # YouTube
        if youtube_channel_ids:
            for channel_id in youtube_channel_ids:
                tasks.append(
                    extract_with_semaphore(
                        Platform.YOUTUBE,
                        channel_id,
                        lambda c=channel_id: self.extract_youtube_profile(c)
                    )
                )
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return results

