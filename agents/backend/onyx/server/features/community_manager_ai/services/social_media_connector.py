"""
Social Media Connector - Conector de Redes Sociales
====================================================

Sistema unificado para conectar y publicar en todas las plataformas sociales.
"""

import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)


class SocialPlatform(ABC):
    """Interfaz base para plataformas sociales"""
    
    @abstractmethod
    def connect(self, credentials: Dict[str, str]) -> bool:
        """Conectar con la plataforma"""
        pass
    
    @abstractmethod
    def publish(
        self,
        content: str,
        media_paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Publicar contenido"""
        pass
    
    @abstractmethod
    def get_analytics(self, post_id: str) -> Dict[str, Any]:
        """Obtener analytics de un post"""
        pass


class SocialMediaConnector:
    """Conector unificado para todas las plataformas sociales"""
    
    def __init__(self):
        """Inicializar el conector"""
        self.platforms: Dict[str, SocialPlatform] = {}
        self.connections: Dict[str, Dict[str, Any]] = {}
        logger.info("Social Media Connector inicializado")
    
    def connect(
        self,
        platform: str,
        credentials: Dict[str, str]
    ) -> bool:
        """
        Conectar una plataforma
        
        Args:
            platform: Nombre de la plataforma
            credentials: Credenciales de autenticación
            
        Returns:
            True si la conexión fue exitosa
        """
        try:
            platform_handler = self._get_platform_handler(platform)
            
            if platform_handler.connect(credentials):
                self.connections[platform] = {
                    "connected": True,
                    "connected_at": datetime.now().isoformat(),
                    "credentials": {k: "***" for k in credentials.keys()}  # Ocultar credenciales
                }
                logger.info(f"Plataforma {platform} conectada exitosamente")
                return True
            else:
                logger.error(f"Error conectando {platform}: credenciales inválidas")
                return False
                
        except Exception as e:
            logger.error(f"Error conectando {platform}: {e}")
            return False
    
    def publish(
        self,
        platform: str,
        content: str,
        media_paths: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Publicar en una plataforma
        
        Args:
            platform: Nombre de la plataforma
            content: Contenido del post
            media_paths: Rutas a archivos multimedia
            
        Returns:
            Dict con resultados de la publicación
        """
        if platform not in self.connections:
            raise ValueError(f"Plataforma {platform} no está conectada")
        
        if not self.connections[platform].get("connected"):
            raise ValueError(f"Plataforma {platform} no está conectada")
        
        try:
            platform_handler = self._get_platform_handler(platform)
            result = platform_handler.publish(content, media_paths)
            
            logger.info(f"Publicación exitosa en {platform}")
            return result
            
        except Exception as e:
            logger.error(f"Error publicando en {platform}: {e}")
            raise
    
    def publish_multiple(
        self,
        platforms: List[str],
        content: str,
        media_paths: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Publicar en múltiples plataformas
        
        Args:
            platforms: Lista de plataformas
            content: Contenido del post
            media_paths: Rutas a archivos multimedia
            
        Returns:
            Dict con resultados por plataforma
        """
        results = {}
        
        for platform in platforms:
            try:
                result = self.publish(platform, content, media_paths)
                results[platform] = {
                    "status": "success",
                    **result
                }
            except Exception as e:
                results[platform] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results
    
    def get_analytics(
        self,
        platform: str,
        post_id: str
    ) -> Dict[str, Any]:
        """
        Obtener analytics de un post
        
        Args:
            platform: Nombre de la plataforma
            post_id: ID del post
            
        Returns:
            Dict con métricas de analytics
        """
        if platform not in self.connections:
            raise ValueError(f"Plataforma {platform} no está conectada")
        
        try:
            platform_handler = self._get_platform_handler(platform)
            return platform_handler.get_analytics(post_id)
        except Exception as e:
            logger.error(f"Error obteniendo analytics de {platform}: {e}")
            raise
    
    def disconnect(self, platform: str) -> bool:
        """
        Desconectar una plataforma
        
        Args:
            platform: Nombre de la plataforma
            
        Returns:
            True si se desconectó exitosamente
        """
        if platform in self.connections:
            del self.connections[platform]
            logger.info(f"Plataforma {platform} desconectada")
            return True
        return False
    
    def get_connected_platforms(self) -> List[str]:
        """
        Obtener lista de plataformas conectadas
        
        Returns:
            Lista de nombres de plataformas
        """
        return [
            platform for platform, conn in self.connections.items()
            if conn.get("connected")
        ]
    
    def _get_platform_handler(self, platform: str) -> SocialPlatform:
        """
        Obtener el handler para una plataforma
        
        Args:
            platform: Nombre de la plataforma
            
        Returns:
            Handler de la plataforma
        """
        if platform not in self.platforms:
            # Lazy import de handlers
            from ..integrations import get_platform_handler
            self.platforms[platform] = get_platform_handler(platform)
        
        return self.platforms[platform]



