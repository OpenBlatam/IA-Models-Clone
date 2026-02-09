"""
Interfaces y contratos para servicios principales
Define contratos claros para mejor modularidad
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from ..core.models import SocialProfile, IdentityProfile, GeneratedContent, Platform, ContentType


class IProfileExtractor(ABC):
    """Interfaz para extractores de perfiles"""
    
    @abstractmethod
    async def extract_tiktok_profile(
        self,
        username: str,
        use_cache: bool = True
    ) -> SocialProfile:
        """Extrae perfil de TikTok"""
        pass
    
    @abstractmethod
    async def extract_instagram_profile(
        self,
        username: str,
        use_cache: bool = True
    ) -> SocialProfile:
        """Extrae perfil de Instagram"""
        pass
    
    @abstractmethod
    async def extract_youtube_profile(
        self,
        channel_id: str,
        use_cache: bool = True
    ) -> SocialProfile:
        """Extrae perfil de YouTube"""
        pass
    
    @abstractmethod
    async def extract_multiple_profiles(
        self,
        tiktok_usernames: Optional[List[str]] = None,
        instagram_usernames: Optional[List[str]] = None,
        youtube_channel_ids: Optional[List[str]] = None,
        max_concurrent: int = 5
    ) -> Dict[str, SocialProfile]:
        """Extrae múltiples perfiles en paralelo"""
        pass


class IIdentityAnalyzer(ABC):
    """Interfaz para analizadores de identidad"""
    
    @abstractmethod
    async def build_identity(
        self,
        tiktok_profile: Optional[SocialProfile] = None,
        instagram_profile: Optional[SocialProfile] = None,
        youtube_profile: Optional[SocialProfile] = None
    ) -> IdentityProfile:
        """Construye perfil de identidad desde perfiles sociales"""
        pass
    
    @abstractmethod
    async def analyze_content(
        self,
        content: str,
        platform: Platform
    ) -> Dict[str, Any]:
        """Analiza contenido individual"""
        pass


class IContentGenerator(ABC):
    """Interfaz para generadores de contenido"""
    
    @abstractmethod
    async def generate_instagram_post(
        self,
        topic: Optional[str] = None,
        style: Optional[str] = None,
        use_lora: bool = False
    ) -> GeneratedContent:
        """Genera post para Instagram"""
        pass
    
    @abstractmethod
    async def generate_tiktok_script(
        self,
        topic: Optional[str] = None,
        duration: int = 60
    ) -> GeneratedContent:
        """Genera script para TikTok"""
        pass
    
    @abstractmethod
    async def generate_youtube_description(
        self,
        video_title: str,
        tags: Optional[List[str]] = None
    ) -> GeneratedContent:
        """Genera descripción para YouTube"""
        pass


class IConnector(ABC):
    """Interfaz base para conectores de redes sociales"""
    
    @abstractmethod
    async def get_profile(self, username: str) -> Dict[str, Any]:
        """Obtiene información del perfil"""
        pass
    
    @abstractmethod
    async def get_videos(
        self,
        username: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Obtiene videos del perfil"""
        pass


class IStorageService(ABC):
    """Interfaz para servicios de almacenamiento"""
    
    @abstractmethod
    def save_identity(self, identity: IdentityProfile) -> None:
        """Guarda identidad en almacenamiento"""
        pass
    
    @abstractmethod
    def get_identity(self, identity_id: str) -> Optional[IdentityProfile]:
        """Obtiene identidad por ID"""
        pass
    
    @abstractmethod
    def save_generated_content(self, content: GeneratedContent) -> None:
        """Guarda contenido generado"""
        pass


class ICacheManager(ABC):
    """Interfaz para gestores de caché"""
    
    @abstractmethod
    def get(self, platform: str, identifier: str) -> Optional[Any]:
        """Obtiene valor del caché"""
        pass
    
    @abstractmethod
    def set(
        self,
        platform: str,
        identifier: str,
        data: Any,
        ttl: Optional[int] = None
    ) -> None:
        """Guarda valor en caché"""
        pass
    
    @abstractmethod
    def clear(self, platform: Optional[str] = None) -> None:
        """Limpia caché"""
        pass

