"""
Base Platform - Plataforma Base
================================

Clase base para todas las integraciones de plataformas sociales.
"""

import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class SocialPlatform(ABC):
    """Clase base abstracta para plataformas sociales"""
    
    def __init__(self, name: str):
        """
        Inicializar plataforma
        
        Args:
            name: Nombre de la plataforma
        """
        self.name = name
        self.connected = False
        self.credentials: Dict[str, str] = {}
    
    @abstractmethod
    def connect(self, credentials: Dict[str, str]) -> bool:
        """
        Conectar con la plataforma
        
        Args:
            credentials: Credenciales de autenticación
            
        Returns:
            True si la conexión fue exitosa
        """
        pass
    
    @abstractmethod
    def publish(
        self,
        content: str,
        media_paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Publicar contenido
        
        Args:
            content: Contenido del post
            media_paths: Rutas a archivos multimedia
            
        Returns:
            Dict con información del post publicado
        """
        pass
    
    @abstractmethod
    def get_analytics(self, post_id: str) -> Dict[str, Any]:
        """
        Obtener analytics de un post
        
        Args:
            post_id: ID del post
            
        Returns:
            Dict con métricas
        """
        pass
    
    def disconnect(self):
        """Desconectar de la plataforma"""
        self.connected = False
        self.credentials = {}
        logger.info(f"Desconectado de {self.name}")




