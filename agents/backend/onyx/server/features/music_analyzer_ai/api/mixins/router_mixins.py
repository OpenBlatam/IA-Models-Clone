"""
Mixins for router functionality
"""

from typing import Optional, List, Dict, Any
from fastapi import Query


class TrackOperationsMixin:
    """Mixin for common track operations"""
    
    def get_track_id_from_request(
        self,
        spotify_service,
        track_id: Optional[str] = None,
        track_name: Optional[str] = None
    ) -> str:
        """Get track ID from request parameters"""
        from ...utils.service_helpers import get_track_or_search
        return get_track_or_search(spotify_service, track_id, track_name)
    
    def get_track_data(self, spotify_service, track_id: str) -> Dict[str, Any]:
        """Get complete track data"""
        from ...utils.service_helpers import get_track_full_data
        return get_track_full_data(spotify_service, track_id)


class ValidationMixin:
    """Mixin for validation operations"""
    
    def validate_track_ids_list(
        self,
        track_ids: List[str],
        min_count: int = 1,
        max_count: int = 100
    ) -> None:
        """Validate track IDs list"""
        self.validate_track_ids(track_ids, min_count, max_count)
    
    def validate_pagination(
        self,
        page: int = 1,
        limit: int = 20,
        max_limit: int = 100
    ) -> None:
        """Validate pagination parameters"""
        if page < 1:
            raise self.error_response("Page must be >= 1", status_code=400)
        self.validate_limit(limit, min_val=1, max_val=max_limit)


class ResponseMixin:
    """Mixin for response operations"""
    
    def format_track_list_response(
        self,
        tracks: List[Dict[str, Any]],
        include_metadata: bool = True
    ) -> dict:
        """Format track list response"""
        from ..utils.response_formatters import format_tracks_response
        formatted = format_tracks_response(tracks)
        
        response = {"tracks": formatted, "total": len(formatted)}
        
        if include_metadata:
            response["metadata"] = {
                "count": len(formatted),
                "formatted": True
            }
        
        return response

