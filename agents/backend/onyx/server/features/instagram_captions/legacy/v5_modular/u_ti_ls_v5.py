from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import hashlib
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
    from .config_v5 import config
    from config_v5 import config
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Instagram Captions API v5.0 - Utils Module

Ultra-fast utility functions for common operations and optimizations.
"""

try:
except ImportError:


class UltraFastUtils:
    """Ultra-fast utility functions for common operations."""
    
    @staticmethod
    async def generate_request_id(prefix: str = "ultra") -> str:
        """Generate unique request ID for tracking."""
        timestamp = int(time.time() * 1000000)  # Microsecond precision
        return f"{prefix}-{timestamp % 1000000:06d}"
    
    @staticmethod
    def generate_batch_id(client_id: str) -> str:
        """Generate unique batch ID."""
        timestamp = int(time.time())
        return f"ultra-fast-batch-{timestamp}"
    
    @staticmethod
    def create_cache_key(data: Dict[str, Any], prefix: str = "v5") -> str:
        """Create optimized cache key from request data."""
        # Create deterministic hash from request data
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        hash_obj = hashlib.md5(json_str.encode())
        return f"{prefix}:{hash_obj.hexdigest()[:16]}"
    
    @staticmethod
    def get_current_timestamp() -> datetime:
        """Get current UTC timestamp."""
        return datetime.now(timezone.utc)
    
    @staticmethod
    def calculate_processing_time(start_time: float) -> float:
        """Calculate processing time in milliseconds."""
        return round((time.time() - start_time) * 1000, 3)
    
    @staticmethod
    def sanitize_content(content: str) -> str:
        """Sanitize content for safe processing."""
        if not content:
            return ""
        
        # Remove dangerous patterns
        dangerous_patterns = [
            '<script>', '</script>', '<iframe>', '</iframe>',
            'javascript:', 'onload=', 'onerror=', 'onclick='
        ]
        
        sanitized = content
        for pattern in dangerous_patterns:
            sanitized = sanitized.replace(pattern, '')
        
        return sanitized.strip()
    
    @staticmethod
    def validate_client_id(client_id: str) -> bool:
        """Validate client ID format."""
        if not client_id or len(client_id) > 50:
            return False
        
        # Allow only alphanumeric, hyphens, and underscores
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_')
        return all(c in allowed_chars for c in client_id)
    
    @staticmethod
    def calculate_quality_bonus(style: str, audience: str, priority: str) -> float:
        """Calculate quality bonus based on request parameters."""
        bonus = 0.0
        
        # Style bonus
        if style in ["professional", "inspirational"]:
            bonus += config.STYLE_BONUS
        
        # Audience bonus
        if audience != "general":
            bonus += config.AUDIENCE_BONUS
        
        # Priority bonus
        if priority in ["high", "urgent"]:
            bonus += config.PRIORITY_BONUS
        
        return bonus
    
    @staticmethod
    def format_response_time(time_ms: float) -> str:
        """Format response time for display."""
        if time_ms < 1:
            return f"{time_ms:.3f}ms"
        elif time_ms < 1000:
            return f"{time_ms:.1f}ms"
        else:
            return f"{time_ms/1000:.2f}s"
    
    @staticmethod
    def calculate_throughput(count: int, time_seconds: float) -> float:
        """Calculate throughput (items per second)."""
        if time_seconds <= 0:
            return 0.0
        return round(count / time_seconds, 2)


class ResponseBuilder:
    """Builder for creating standardized API responses."""
    
    @staticmethod
    def build_success_response(
        data: Any,
        message: str = "Success",
        request_id: str = None,
        processing_time_ms: float = None
    ) -> Dict[str, Any]:
        """Build standardized success response."""
        response = {
            "status": "success",
            "message": message,
            "data": data,
            "timestamp": UltraFastUtils.get_current_timestamp().isoformat(),
            "api_version": config.API_VERSION
        }
        
        if request_id:
            response["request_id"] = request_id
        
        if processing_time_ms is not None:
            response["processing_time_ms"] = processing_time_ms
        
        return response
    
    @staticmethod
    def build_error_response(
        message: str,
        status_code: int = 500,
        request_id: str = None,
        details: Any = None
    ) -> Dict[str, Any]:
        """Build standardized error response."""
        response = {
            "status": "error",
            "error": {
                "message": message,
                "status_code": status_code,
                "timestamp": UltraFastUtils.get_current_timestamp().isoformat(),
                "api_version": config.API_VERSION
            }
        }
        
        if request_id:
            response["error"]["request_id"] = request_id
        
        if details:
            response["error"]["details"] = details
        
        return response
    
    @staticmethod
    def build_batch_response(
        batch_id: str,
        results: List[Dict[str, Any]],
        total_time_ms: float,
        cache_hits: int = 0
    ) -> Dict[str, Any]:
        """Build standardized batch response."""
        total_processed = len(results)
        successful_results = [r for r in results if r.get("status") == "success"]
        
        # Calculate average quality score
        quality_scores = [r.get("quality_score", 0) for r in successful_results]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            "batch_id": batch_id,
            "status": "completed",
            "results": results,
            "summary": {
                "total_processed": total_processed,
                "successful": len(successful_results),
                "failed": total_processed - len(successful_results),
                "total_time_ms": total_time_ms,
                "avg_time_per_item_ms": round(total_time_ms / max(1, total_processed), 3),
                "avg_quality_score": round(avg_quality, 2),
                "cache_hits": cache_hits,
                "throughput_per_second": UltraFastUtils.calculate_throughput(
                    total_processed, total_time_ms / 1000
                )
            },
            "timestamp": UltraFastUtils.get_current_timestamp().isoformat(),
            "api_version": config.API_VERSION
        }


class CacheKeyGenerator:
    """Specialized cache key generation for different types of requests."""
    
    @staticmethod
    def generate_caption_key(request_data: Dict[str, Any]) -> str:
        """Generate cache key for single caption requests."""
        relevant_fields = {
            "content_description": request_data.get("content_description", ""),
            "style": request_data.get("style", "casual"),
            "audience": request_data.get("audience", "general"),
            "content_type": request_data.get("content_type", "post"),
            "hashtag_count": request_data.get("hashtag_count", 10)
        }
        return UltraFastUtils.create_cache_key(relevant_fields, "caption")
    
    @staticmethod
    def generate_batch_key(batch_data: Dict[str, Any]) -> str:
        """Generate cache key for batch requests."""
        # Create hash of all individual requests
        request_hashes = []
        for req in batch_data.get("requests", []):
            req_key = CacheKeyGenerator.generate_caption_key(req)
            request_hashes.append(req_key)
        
        batch_signature = {
            "request_hashes": sorted(request_hashes),  # Sort for consistency
            "batch_size": len(request_hashes)
        }
        
        return UltraFastUtils.create_cache_key(batch_signature, "batch")
    
    @staticmethod
    def generate_health_key() -> str:
        """Generate cache key for health checks (time-based)."""
        # Cache health checks for 1 minute intervals
        minute_timestamp = int(time.time()) // 60
        return f"health:{minute_timestamp}"


class PerformanceTracker:
    """Track and analyze performance patterns."""
    
    def __init__(self) -> Any:
        self.operation_times: Dict[str, List[float]] = {}
    
    def track_operation(self, operation_name: str, duration_ms: float) -> None:
        """Track operation performance."""
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
        
        self.operation_times[operation_name].append(duration_ms)
        
        # Keep only last 1000 measurements for memory efficiency
        if len(self.operation_times[operation_name]) > 1000:
            self.operation_times[operation_name] = self.operation_times[operation_name][-1000:]
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for a specific operation."""
        times = self.operation_times.get(operation_name, [])
        if not times:
            return {"count": 0, "avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
        
        return {
            "count": len(times),
            "avg_ms": round(sum(times) / len(times), 3),
            "min_ms": round(min(times), 3),
            "max_ms": round(max(times), 3),
            "last_ms": round(times[-1], 3)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all tracked operations."""
        return {
            operation: self.get_operation_stats(operation)
            for operation in self.operation_times.keys()
        }


# Global performance tracker instance
performance_tracker = PerformanceTracker()


# Export public interface
__all__ = [
    'UltraFastUtils',
    'ResponseBuilder',
    'CacheKeyGenerator',
    'PerformanceTracker',
    'performance_tracker'
] 