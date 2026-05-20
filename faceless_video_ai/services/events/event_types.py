"""
Event Types
Event type definitions
"""

from enum import Enum
from typing import Dict, Any
from uuid import UUID


class EventType(str, Enum):
    """Event types"""
    VIDEO_GENERATION_STARTED = "video.generation.started"
    VIDEO_GENERATION_COMPLETED = "video.generation.completed"
    VIDEO_GENERATION_FAILED = "video.generation.failed"
    VIDEO_DELETED = "video.deleted"
    VIDEO_SHARED = "video.shared"
    USER_REGISTERED = "user.registered"
    USER_LOGIN = "user.login"
    SYSTEM_ERROR = "system.error"
    SYSTEM_WARNING = "system.warning"
    CACHE_HIT = "cache.hit"
    CACHE_MISS = "cache.miss"
    API_REQUEST = "api.request"
    ALERT_TRIGGERED = "alert.triggered"


class VideoEvent:
    """Video-related events"""
    
    @staticmethod
    def generation_started(video_id: UUID, user_id: str) -> Dict[str, Any]:
        return {
            "event_type": EventType.VIDEO_GENERATION_STARTED,
            "video_id": str(video_id),
            "user_id": user_id,
        }
    
    @staticmethod
    def generation_completed(video_id: UUID, duration: float, file_size: int) -> Dict[str, Any]:
        return {
            "event_type": EventType.VIDEO_GENERATION_COMPLETED,
            "video_id": str(video_id),
            "duration": duration,
            "file_size": file_size,
        }
    
    @staticmethod
    def generation_failed(video_id: UUID, error: str) -> Dict[str, Any]:
        return {
            "event_type": EventType.VIDEO_GENERATION_FAILED,
            "video_id": str(video_id),
            "error": error,
        }


class SystemEvent:
    """System-related events"""
    
    @staticmethod
    def error(service: str, error: str) -> Dict[str, Any]:
        return {
            "event_type": EventType.SYSTEM_ERROR,
            "service": service,
            "error": error,
        }
    
    @staticmethod
    def warning(service: str, message: str) -> Dict[str, Any]:
        return {
            "event_type": EventType.SYSTEM_WARNING,
            "service": service,
            "message": message,
        }

