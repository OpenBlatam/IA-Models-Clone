"""
Base router class with common utilities and error handling
"""

from fastapi import APIRouter, HTTPException, Request
from typing import Any, Optional
import logging
from functools import wraps

from ...utils.exceptions import (
    SpotifyAPIException,
    TrackNotFoundException,
    InvalidTrackIDException,
    AnalysisException
)
from ...core.di import get_container
from ..utils.di_helpers import get_service_from_di, get_service_optional

logger = logging.getLogger(__name__)


class BaseRouter:
    """Base router class with common utilities"""
    
    def __init__(self, prefix: str, tags: list):
        self.router = APIRouter(prefix=prefix, tags=tags)
        self.prefix = prefix
        self.tags = tags
    
    def get_service(self, service_name: str) -> Any:
        """
        Get a service from the DI container.
        
        Uses the new DI system instead of the legacy service registry.
        
        Args:
            service_name: Name of the service to retrieve
        
        Returns:
            Service instance
        
        Raises:
            Exception: If service is not found
        """
        return get_service_from_di(service_name)
    
    def get_services(self, *service_names: str) -> tuple:
        """
        Get multiple services from the DI container.
        
        Uses the new DI system instead of the legacy service registry.
        
        Args:
            *service_names: Variable number of service names
        
        Returns:
            Tuple of services in the same order as service_names
        
        Example:
            spotify, analyzer = router.get_services("spotify_service", "music_analyzer")
        """
        from ..utils.di_helpers import get_multiple_services
        return get_multiple_services(*service_names)
    
    def get_service_optional(self, service_name: str) -> Optional[Any]:
        """
        Get an optional service from the DI container.
        
        Returns None if service is not available instead of raising an exception.
        
        Args:
            service_name: Name of the service to retrieve
        
        Returns:
            Service instance or None if not available
        """
        return get_service_optional(service_name)
    
    @staticmethod
    def handle_exceptions(func):
        """Decorator to handle common exceptions"""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                raise
            except TrackNotFoundException as e:
                raise HTTPException(status_code=404, detail=str(e))
            except InvalidTrackIDException as e:
                raise HTTPException(status_code=400, detail=str(e))
            except SpotifyAPIException as e:
                raise HTTPException(
                    status_code=e.status_code or 500,
                    detail=str(e)
                )
            except AnalysisException as e:
                raise HTTPException(status_code=500, detail=str(e))
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal server error: {str(e)}"
                )
        return wrapper
    
    def success_response(self, data: Any, message: Optional[str] = None) -> dict:
        """Create a standardized success response"""
        response = {
            "success": True,
            "data": data
        }
        if message:
            response["message"] = message
        return response
    
    def error_response(self, message: str, status_code: int = 500) -> HTTPException:
        """Create a standardized error response"""
        return HTTPException(status_code=status_code, detail=message)
    
    def validate_track_ids(self, track_ids: list, min_count: int = 1, max_count: int = 100) -> None:
        """Validate track IDs list"""
        if not track_ids:
            raise self.error_response("Lista de track IDs no puede estar vacía", status_code=400)
        if len(track_ids) < min_count:
            raise self.error_response(f"Se necesitan al menos {min_count} track(s)", status_code=400)
        if len(track_ids) > max_count:
            raise self.error_response(f"Máximo {max_count} tracks", status_code=400)
    
    def validate_limit(self, limit: int, min_val: int = 1, max_val: int = 100) -> None:
        """Validate limit parameter"""
        if limit < min_val or limit > max_val:
            raise self.error_response(
                f"Limit debe estar entre {min_val} y {max_val}",
                status_code=400
            )
    
    def paginated_response(self, items: list, page: int = 1, limit: int = 20, total: Optional[int] = None) -> dict:
        """Create a paginated response"""
        if total is None:
            total = len(items)
        
        return self.success_response({
            "items": items,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total,
                "pages": (total + limit - 1) // limit if limit > 0 else 0
            }
        })
    
    def track_not_found(self, track_id: str) -> HTTPException:
        """Create a track not found error"""
        return self.error_response(f"Track {track_id} no encontrado", status_code=404)
    
    def invalid_request(self, message: str) -> HTTPException:
        """Create an invalid request error"""
        return self.error_response(message, status_code=400)
    
    def service_unavailable(self, service_name: str) -> HTTPException:
        """Create a service unavailable error"""
        return self.error_response(
            f"Servicio {service_name} no disponible",
            status_code=503
        )
    
    def require_success(self, result: Any, error_message: str, status_code: int = 400) -> None:
        """
        Require that a service result indicates success
        
        Args:
            result: Service result
            error_message: Error message if not successful
            status_code: HTTP status code
        """
        from ..utils.service_result_helpers import require_success as _require_success
        _require_success(result, error_message, status_code)
    
    def require_not_none(self, value: Any, error_message: str, status_code: int = 404) -> None:
        """
        Require that a value is not None
        
        Args:
            value: Value to check
            error_message: Error message if None
            status_code: HTTP status code
        """
        from ..utils.service_result_helpers import require_not_none as _require_not_none
        _require_not_none(value, error_message, status_code)
    
    def extract_bearer_token(self, authorization: Optional[str]) -> str:
        """
        Extract Bearer token from authorization header
        
        Args:
            authorization: Authorization header value
        
        Returns:
            Extracted token
        """
        from ..utils.service_result_helpers import extract_bearer_token as _extract_bearer_token
        return _extract_bearer_token(authorization)
    
    def list_response(self, items: list, key: str = "items", include_total: bool = True, **kwargs) -> dict:
        """
        Create a standardized list response
        
        Args:
            items: List of items
            key: Key name for items in response
            include_total: Whether to include total count
            **kwargs: Additional fields to include in response
        
        Returns:
            Standardized list response
        """
        response = {key: items}
        if include_total:
            response["total"] = len(items)
        response.update(kwargs)
        return self.success_response(response)
    
    def count_response(self, items: list, count_key: str = "count", **kwargs) -> dict:
        """
        Create a response with count
        
        Args:
            items: List of items
            count_key: Key name for count
            **kwargs: Additional fields to include
        
        Returns:
            Response with count
        """
        response = {count_key: len(items)}
        response.update(kwargs)
        return self.success_response(response)

