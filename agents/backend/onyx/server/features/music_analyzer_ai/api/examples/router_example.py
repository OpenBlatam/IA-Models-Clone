"""
Example of how to create a new router using all the refactored infrastructure
"""

from fastapi import Query
from typing import List, Optional
import logging

from ..base_router import BaseRouter
from ..validators import validate_track_id, validate_limit
from ..utils import format_tracks_response, create_error_response
from ..schemas import SuccessResponse, TrackResponse
from ..decorators import log_request

logger = logging.getLogger(__name__)


class ExampleRouter(BaseRouter):
    """Example router demonstrating best practices"""
    
    def __init__(self):
        super().__init__(prefix="/example", tags=["Example"])
        self._register_routes()
    
    def _register_routes(self):
        """Register all example routes"""
        
        @self.router.get("/track/{track_id}", response_model=SuccessResponse)
        @self.handle_exceptions
        @log_request
        async def get_example_track(track_id: str):
            """
            Example endpoint showing best practices
            
            - Uses validation
            - Uses error handling
            - Uses response schemas
            - Uses decorators
            """
            # Validate input
            validate_track_id(track_id)
            
            # Get service
            spotify_service = self.get_service("spotify_service")
            
            try:
                # Get data
                track = spotify_service.get_track(track_id)
                
                # Format response
                formatted = format_tracks_response([track])[0]
                
                # Return standardized response
                return self.success_response(
                    data=formatted,
                    message="Track retrieved successfully"
                )
            except Exception as e:
                logger.error(f"Error getting track: {e}")
                raise self.track_not_found(track_id)
        
        @self.router.get("/search", response_model=SuccessResponse)
        @self.handle_exceptions
        @log_request
        async def search_example(
            query: str = Query(...),
            limit: int = Query(10, ge=1, le=50)
        ):
            """Example search endpoint"""
            # Validate
            validate_limit(limit, min_val=1, max_val=50)
            
            # Get service
            spotify_service = self.get_service("spotify_service")
            
            # Get data
            tracks = spotify_service.search_track(query, limit)
            
            # Format
            formatted = format_tracks_response(tracks)
            
            # Return with pagination
            return self.paginated_response(
                items=formatted,
                page=1,
                limit=limit,
                total=len(formatted)
            )


def get_example_router() -> ExampleRouter:
    """Factory function to get example router"""
    return ExampleRouter()

