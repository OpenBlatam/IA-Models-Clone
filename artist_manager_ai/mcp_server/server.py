"""
MCP Server Implementation
==========================

Implementación del servidor MCP para Artist Manager AI.
"""

import logging
import os
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ArtistManagerMCPServer:
    """Servidor MCP para Artist Manager AI."""
    
    def __init__(self, openrouter_api_key: Optional[str] = None):
        """
        Inicializar servidor MCP.
        
        Args:
            openrouter_api_key: API key de OpenRouter
        """
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self._logger = logger
    
    async def get_artist_dashboard(self, artist_id: str) -> Dict[str, Any]:
        """
        Obtener dashboard del artista.
        
        Args:
            artist_id: ID del artista
        
        Returns:
            Datos del dashboard
        """
        try:
            from ..core.artist_manager import ArtistManager
            
            manager = ArtistManager(
                artist_id=artist_id,
                openrouter_api_key=self.openrouter_api_key
            )
            return manager.get_dashboard_data()
        except Exception as e:
            self._logger.error(f"Error getting dashboard: {str(e)}")
            raise
    
    async def generate_daily_summary(self, artist_id: str) -> Dict[str, Any]:
        """
        Generar resumen diario.
        
        Args:
            artist_id: ID del artista
        
        Returns:
            Resumen diario generado por IA
        """
        try:
            from ..core.artist_manager import ArtistManager
            
            manager = ArtistManager(
                artist_id=artist_id,
                openrouter_api_key=self.openrouter_api_key
            )
            return await manager.generate_daily_summary()
        except Exception as e:
            self._logger.error(f"Error generating daily summary: {str(e)}")
            raise
    
    async def get_wardrobe_recommendation(self, artist_id: str, event_id: str) -> Dict[str, Any]:
        """
        Obtener recomendación de vestimenta.
        
        Args:
            artist_id: ID del artista
            event_id: ID del evento
        
        Returns:
            Recomendación de vestimenta
        """
        try:
            from ..core.artist_manager import ArtistManager
            
            manager = ArtistManager(
                artist_id=artist_id,
                openrouter_api_key=self.openrouter_api_key
            )
            recommendation = await manager.generate_wardrobe_recommendation(event_id)
            return recommendation.to_dict()
        except Exception as e:
            self._logger.error(f"Error getting wardrobe recommendation: {str(e)}")
            raise
    
    async def check_protocol_compliance(self, artist_id: str, event_id: str) -> Dict[str, Any]:
        """
        Verificar cumplimiento de protocolos.
        
        Args:
            artist_id: ID del artista
            event_id: ID del evento
        
        Returns:
            Reporte de cumplimiento
        """
        try:
            from ..core.artist_manager import ArtistManager
            
            manager = ArtistManager(
                artist_id=artist_id,
                openrouter_api_key=self.openrouter_api_key
            )
            return await manager.check_protocol_compliance(event_id)
        except Exception as e:
            self._logger.error(f"Error checking protocol compliance: {str(e)}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Verificar salud del servidor.
        
        Returns:
            Estado de salud
        """
        try:
            from ..infrastructure.openrouter_client import OpenRouterClient
            
            if not self.openrouter_api_key:
                return {
                    "status": "degraded",
                    "openrouter": "not_configured",
                    "timestamp": datetime.now().isoformat()
                }
            
            client = OpenRouterClient(self.openrouter_api_key)
            health = await client.health_check()
            await client.close()
            
            return {
                "status": "healthy" if health.get("healthy") else "degraded",
                "openrouter": health,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




