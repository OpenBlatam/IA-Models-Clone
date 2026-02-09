"""
Utility functions for route handlers
"""

from typing import Dict, Any, Optional, List
from fastapi import Request, Query
from .metrics import get_metrics_collector, increment_counter
from .logging_config import get_logger

logger = get_logger(__name__)


def get_client_info(request: Request) -> Dict[str, Any]:
    """Extract client information from request"""
    client_host = request.client.host if request.client else None
    forwarded = request.headers.get("X-Forwarded-For")
    real_ip = forwarded.split(",")[0].strip() if forwarded else client_host
    
    return {
        "ip": real_ip,
        "user_agent": request.headers.get("User-Agent"),
        "referer": request.headers.get("Referer"),
        "origin": request.headers.get("Origin")
    }


def track_request_metrics(request: Request, route_name: str):
    """Track request metrics"""
    metrics = get_metrics_collector()
    client_info = get_client_info(request)
    
    metrics.increment("http.requests", tags={
        "method": request.method,
        "route": route_name,
        "path": request.url.path
    })
    
    return client_info


def log_request_start(request: Request, route_name: str, **extra_context):
    """Log request start with context"""
    client_info = get_client_info(request)
    logger.info(
        f"Request started: {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "route": route_name,
            "query_params": dict(request.query_params),
            **client_info,
            **extra_context
        }
    )


def log_request_end(request: Request, route_name: str, duration_ms: float, status_code: int, **extra_context):
    """Log request end with metrics"""
    logger.info(
        f"Request completed: {request.method} {request.url.path} - {status_code}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "route": route_name,
            "status_code": status_code,
            "duration_ms": duration_ms,
            **extra_context
        }
    )
    
    metrics = get_metrics_collector()
    metrics.increment("http.responses", tags={
        "method": request.method,
        "status": str(status_code),
        "route": route_name
    })


def get_query_params(request: Request) -> Dict[str, Any]:
    """Extract all query parameters from request"""
    return dict(request.query_params)


def get_path_params(request: Request) -> Dict[str, Any]:
    """Extract path parameters from request"""
    return dict(request.path_params)


def get_request_body_size(request: Request) -> Optional[int]:
    """Get request body size in bytes"""
    content_length = request.headers.get("Content-Length")
    if content_length:
        try:
            return int(content_length)
        except ValueError:
            return None
    return None


def is_json_request(request: Request) -> bool:
    """Check if request has JSON content type"""
    content_type = request.headers.get("Content-Type", "")
    return "application/json" in content_type.lower()


def get_request_id(request: Request) -> Optional[str]:
    """Get request ID from headers or generate one"""
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        from ..core.utils import generate_id
        request_id = generate_id("req")
    return request_id


def extract_user_agent_info(user_agent: Optional[str]) -> Dict[str, Any]:
    """Extract information from User-Agent string"""
    if not user_agent:
        return {}
    
    info = {
        "raw": user_agent,
        "is_bot": False,
        "is_mobile": False
    }
    
    user_agent_lower = user_agent.lower()
    
    # Detect bots
    bot_keywords = ["bot", "crawler", "spider", "scraper"]
    info["is_bot"] = any(keyword in user_agent_lower for keyword in bot_keywords)
    
    # Detect mobile
    mobile_keywords = ["mobile", "android", "iphone", "ipad", "ipod"]
    info["is_mobile"] = any(keyword in user_agent_lower for keyword in mobile_keywords)
    
    return info


def build_pagination_response(
    items: List[Any],
    total: int,
    page: int,
    page_size: int,
    **extra_metadata
) -> Dict[str, Any]:
    """Build paginated response with metadata"""
    total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0
    
    return {
        "items": items,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_previous": page > 1
        },
        **extra_metadata
    }

