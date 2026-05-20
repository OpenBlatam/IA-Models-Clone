"""
History endpoints for analysis history
"""

from fastapi import Query
from typing import Optional
import logging

from ..base_router import BaseRouter

logger = logging.getLogger(__name__)


class HistoryRouter(BaseRouter):
    """Router for history endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/history", tags=["History"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all history routes"""
        
        @self.router.get("", response_model=dict)
        @self.handle_exceptions
        async def get_history(
            user_id: Optional[str] = Query(None),
            limit: int = Query(50, ge=1, le=100)
        ):
            """Obtiene el historial de análisis"""
            history_service = self.get_service("history_service")
            history = history_service.get_history(user_id, limit)
            return self.list_response(history, key="history")
        
        @self.router.get("/stats", response_model=dict)
        @self.handle_exceptions
        async def get_history_stats(user_id: Optional[str] = Query(None)):
            """Obtiene estadísticas del historial"""
            history_service = self.get_service("history_service")
            stats = history_service.get_stats(user_id)
            return self.success_response({"stats": stats})
        
        @self.router.delete("/{analysis_id}", response_model=dict)
        @self.handle_exceptions
        async def delete_history_entry(
            analysis_id: str,
            user_id: Optional[str] = Query(None)
        ):
            """Elimina una entrada del historial"""
            history_service = self.get_service("history_service")
            success = history_service.delete_analysis(analysis_id, user_id)
            self.require_success(success, "Análisis no encontrado", status_code=404)
            return self.success_response(None, message="Análisis eliminado del historial")


def get_history_router() -> HistoryRouter:
    """Factory function to get history router"""
    return HistoryRouter()

