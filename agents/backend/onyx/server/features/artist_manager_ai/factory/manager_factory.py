"""
Manager Factory
===============

Factory para crear managers con dependency injection.
"""

import logging
from typing import Optional, Any
from ..core.artist_manager import ArtistManager
from ..config.config_loader import get_config, AppConfig
from .service_factory import ServiceFactory
from ..infrastructure.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


class ManagerFactory:
    """Factory de managers."""
    
    def __init__(self, config: Optional[AppConfig] = None, service_factory: Optional[ServiceFactory] = None):
        """
        Inicializar factory.
        
        Args:
            config: Configuración
            service_factory: Factory de servicios
        """
        self.config = config or get_config()
        self.service_factory = service_factory or ServiceFactory(self.config)
        self._logger = logger
    
    def create_artist_manager(
        self,
        artist_id: str,
        openrouter_api_key: Optional[str] = None
    ) -> ArtistManager:
        """
        Crear ArtistManager con todas las dependencias.
        
        Args:
            artist_id: ID del artista
            openrouter_api_key: API key de OpenRouter
        
        Returns:
            ArtistManager configurado
        """
        # Obtener API key de configuración si no se proporciona
        if not openrouter_api_key:
            openrouter_api_key = self.config.openrouter.api_key
        
        # Crear OpenRouter client
        openrouter_client = OpenRouterClient(openrouter_api_key) if openrouter_api_key else None
        
        # Crear manager con servicios
        manager = ArtistManager(
            artist_id=artist_id,
            openrouter_api_key=openrouter_api_key,
            enable_persistence=self.config.services.persistence,
            enable_notifications=self.config.services.notifications,
            enable_analytics=self.config.services.analytics
        )
        
        # Inyectar servicios adicionales si es necesario
        # (Los servicios ya se crean dentro de ArtistManager)
        
        self._logger.info(f"Created ArtistManager for artist {artist_id}")
        return manager

