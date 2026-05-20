"""
Predictions endpoints for success and pattern prediction
"""

from fastapi import Query
import logging

from ..base_router import BaseRouter

logger = logging.getLogger(__name__)


class PredictionsRouter(BaseRouter):
    """Router for predictions endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/predict", tags=["Predictions"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all predictions routes"""
        
        @self.router.post("/success", response_model=dict)
        @self.handle_exceptions
        async def predict_commercial_success(track_id: str = Query(...)):
            """Predice el éxito comercial de un track"""
            trends_analyzer = self.get_service("trends_analyzer")
            prediction = trends_analyzer.predict_commercial_success(track_id)
            return self.success_response({"prediction": prediction})
        
        @self.router.get("/rhythmic/patterns", response_model=dict)
        @self.handle_exceptions
        async def get_rhythmic_patterns(track_id: str = Query(...)):
            """Analiza patrones rítmicos avanzados de un track"""
            spotify_service, trends_analyzer = self.get_services(
                "spotify_service",
                "trends_analyzer"
            )
            
            audio_analysis = spotify_service.get_track_audio_analysis(track_id)
            patterns = trends_analyzer.analyze_rhythmic_patterns(audio_analysis)
            
            return self.success_response({
                "track_id": track_id,
                "rhythmic_patterns": patterns
            })


def get_predictions_router() -> PredictionsRouter:
    """Factory function to get predictions router"""
    return PredictionsRouter()

