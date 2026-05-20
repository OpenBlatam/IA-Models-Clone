"""
Analytics Service - Servicio de analytics
==========================================

Servicio independiente para analytics y métricas.
"""

import logging
from typing import Dict, Any

from ..utils.analytics_engine import AnalyticsEngine

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Servicio para analytics"""
    
    def __init__(self, analytics_engine: AnalyticsEngine = None):
        self.analytics_engine = analytics_engine or AnalyticsEngine()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas"""
        try:
            return self.analytics_engine.get_statistics()
        except Exception as e:
            logger.error(f"Error getting analytics stats: {e}", exc_info=True)
            return {}















