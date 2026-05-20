"""
Trends endpoints for popularity and prediction analysis
"""

from fastapi import Query
from typing import List
import logging

from ..base_router import BaseRouter
from ..constants import MAX_ARTISTS_FOR_NETWORK
from ..utils.router_helpers import validate_track_ids_count

logger = logging.getLogger(__name__)


class TrendsRouter(BaseRouter):
    """Router for trends endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/trends", tags=["Trends"])
        self._trends_analyzer = None
        self._trend_predictor = None
        self._register_routes()
    
    def _get_trends_analyzer(self):
        """Get or cache trends analyzer"""
        if self._trends_analyzer is None:
            self._trends_analyzer = self.get_service("trends_analyzer")
        return self._trends_analyzer
    
    def _get_trend_predictor(self):
        """Get or cache trend predictor"""
        if self._trend_predictor is None:
            self._trend_predictor = self.get_service("trend_predictor")
        return self._trend_predictor
    
    def _register_routes(self):
        """Register all trends routes"""
        
        @self.router.get("/popularity", response_model=dict)
        @self.handle_exceptions
        async def get_popularity_trends(track_id: str = Query(...)):
            """Analiza tendencias de popularidad de un track"""
            trends_analyzer = self._get_trends_analyzer()
            trends = trends_analyzer.analyze_popularity_trends(track_id)
            return self.success_response({
                "track_id": track_id,
                "trends": trends
            })
        
        @self.router.post("/artists", response_model=dict)
        @self.handle_exceptions
        async def analyze_artist_trends(artist_ids: List[str]):
            """Analiza tendencias de múltiples artistas"""
            try:
                validate_track_ids_count(
                    artist_ids,
                    1,
                    MAX_ARTISTS_FOR_NETWORK,
                    "artistas"
                )
            except ValueError as e:
                raise self.error_response(str(e), status_code=400)
            
            trends_analyzer = self._get_trends_analyzer()
            trends = trends_analyzer.analyze_artist_trends(artist_ids)
            return self.success_response({"trends": trends})
        
        @self.router.get("/predict/genre", response_model=dict)
        @self.handle_exceptions
        async def predict_genre_trends(
            horizon_days: int = Query(30, ge=1, le=365)
        ):
            """Predice tendencias de géneros"""
            trend_predictor = self._get_trend_predictor()
            predictions = trend_predictor.predict_genre_trends(horizon_days)
            return self.success_response({"predictions": predictions})
        
        @self.router.get("/predict/emotion", response_model=dict)
        @self.handle_exceptions
        async def predict_emotion_trends(
            horizon_days: int = Query(30, ge=1, le=365)
        ):
            """Predice tendencias de emociones"""
            trend_predictor = self._get_trend_predictor()
            predictions = trend_predictor.predict_emotion_trends(horizon_days)
            return self.success_response({"predictions": predictions})
        
        @self.router.get("/predict/features", response_model=dict)
        @self.handle_exceptions
        async def predict_feature_trends(
            horizon_days: int = Query(30, ge=1, le=365)
        ):
            """Predice tendencias de características musicales"""
            trend_predictor = self._get_trend_predictor()
            predictions = trend_predictor.predict_feature_trends(horizon_days)
            return self.success_response({"predictions": predictions})


def get_trends_router() -> TrendsRouter:
    """Factory function to get trends router"""
    return TrendsRouter()

