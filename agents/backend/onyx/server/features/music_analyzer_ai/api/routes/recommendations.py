"""
Recommendations endpoints
"""

from fastapi import Query
from typing import List, Optional, Dict, Any
import logging

from ..base_router import BaseRouter

logger = logging.getLogger(__name__)


class RecommendationsRouter(BaseRouter):
    """Router for recommendations endpoints"""
    
    def __init__(self):
        super().__init__(prefix="/recommendations", tags=["Recommendations"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all recommendations routes"""
        
        @self.router.post("/intelligent", response_model=dict)
        @self.handle_exceptions
        async def intelligent_recommendations(
            track_id: str = Query(...),
            limit: int = Query(10, ge=1, le=50),
            method: str = Query("similarity", regex="^(similarity|mood|genre)$")
        ):
            """Recomendaciones inteligentes basadas en similitud, mood o género"""
            intelligent_recommender = self.get_service("intelligent_recommender")
            recommendations = intelligent_recommender.recommend(
                track_id=track_id,
                limit=limit,
                method=method
            )
            return self.list_response(
                recommendations,
                key="recommendations",
                method=method
            )
        
        @self.router.post("/playlist", response_model=dict)
        @self.handle_exceptions
        async def generate_playlist(
            playlist_length: int = Query(20, ge=1, le=100),
            genres: Optional[List[str]] = Query(None),
            moods: Optional[List[str]] = Query(None),
            energy_range: Optional[List[float]] = Query(None),
            tempo_range: Optional[List[int]] = Query(None),
            seed_track_id: Optional[str] = Query(None)
        ):
            """Genera una playlist recomendada"""
            intelligent_recommender = self.get_service("intelligent_recommender")
            playlist = intelligent_recommender.generate_playlist(
                playlist_length=playlist_length,
                genres=genres,
                moods=moods,
                energy_range=energy_range,
                tempo_range=tempo_range,
                seed_track_id=seed_track_id
            )
            return self.list_response(playlist, key="playlist")
        
        @self.router.post("/contextual", response_model=dict)
        @self.handle_exceptions
        async def contextual_recommendations(
            track_id: str = Query(...),
            context: Optional[Dict[str, Any]] = None,
            limit: int = Query(10, ge=1, le=50)
        ):
            """Recomendaciones basadas en contexto personalizado"""
            if context is None:
                context = {}
            
            contextual_recommender = self.get_service("contextual_recommender")
            recommendations = contextual_recommender.recommend_by_context(track_id, context, limit)
            return self.list_response(recommendations, key="recommendations")
        
        @self.router.post("/time-of-day", response_model=dict)
        @self.handle_exceptions
        async def time_of_day_recommendations(
            track_id: str = Query(...),
            time_of_day: str = Query(..., regex="^(morning|afternoon|evening|night)$"),
            limit: int = Query(10, ge=1, le=50)
        ):
            """Recomendaciones basadas en hora del día"""
            contextual_recommender = self.get_service("contextual_recommender")
            recommendations = contextual_recommender.recommend_by_time_of_day(track_id, time_of_day, limit)
            return self.list_response(recommendations, key="recommendations")
        
        @self.router.post("/activity", response_model=dict)
        @self.handle_exceptions
        async def activity_recommendations(
            track_id: str = Query(...),
            activity: str = Query(..., regex="^(workout|study|party|relax|drive)$"),
            limit: int = Query(10, ge=1, le=50)
        ):
            """Recomendaciones basadas en actividad"""
            contextual_recommender = self.get_service("contextual_recommender")
            recommendations = contextual_recommender.recommend_by_activity(track_id, activity, limit)
            return self.list_response(recommendations, key="recommendations")
        
        @self.router.post("/mood", response_model=dict)
        @self.handle_exceptions
        async def mood_recommendations(
            track_id: str = Query(...),
            target_mood: str = Query(..., regex="^(happy|sad|energetic|calm|romantic)$"),
            limit: int = Query(10, ge=1, le=50)
        ):
            """Recomendaciones basadas en mood objetivo"""
            contextual_recommender = self.get_service("contextual_recommender")
            recommendations = contextual_recommender.recommend_by_mood(track_id, target_mood, limit)
            return self.list_response(recommendations, key="recommendations")


def get_recommendations_router() -> RecommendationsRouter:
    """Factory function to get recommendations router"""
    return RecommendationsRouter()

