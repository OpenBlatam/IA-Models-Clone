"""
Export endpoints for analysis data
"""

from fastapi import Query
import logging
import time

from ..base_router import BaseRouter
from ..utils.analysis_helpers import perform_track_analysis, add_coaching_to_analysis, trigger_webhook_safe
from ..utils.export_helpers import export_analysis
from ...services.webhook_service import WebhookEvent

logger = logging.getLogger(__name__)


class ExportRouter(BaseRouter):
    """Router for export endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/export", tags=["Export"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all export routes"""
        
        @self.router.post("/{track_id}", response_model=dict)
        @self.handle_exceptions
        async def export_analysis(
            track_id: str,
            format: str = Query("json", regex="^(json|text|markdown)$"),
            include_coaching: bool = Query(True)
        ):
            """
            Exporta un análisis a diferentes formatos
            
            - **track_id**: ID de la canción
            - **format**: Formato de exportación (json, text, markdown)
            - **include_coaching**: Incluir análisis de coaching
            """
            start_time = time.time()
            
            spotify_service, music_analyzer, music_coach, export_service, webhook_service = \
                self.get_services(
                    "spotify_service",
                    "music_analyzer",
                    "music_coach",
                    "export_service",
                    "webhook_service"
                )
            
            analysis = await perform_track_analysis(spotify_service, music_analyzer, track_id)
            
            if include_coaching:
                analysis = add_coaching_to_analysis(analysis, music_coach)
            
            try:
                exported = export_analysis(export_service, analysis, format)
            except ValueError as e:
                raise self.error_response(str(e), status_code=400)
            
            await trigger_webhook_safe(
                webhook_service,
                WebhookEvent.EXPORT_COMPLETED,
                {
                    "track_id": track_id,
                    "format": format
                }
            )
            
            return self.success_response({
                "track_id": track_id,
                "format": format,
                "content": exported,
                "export_time": time.time() - start_time
            })


def get_export_router() -> ExportRouter:
    """Factory function to get export router"""
    return ExportRouter()

