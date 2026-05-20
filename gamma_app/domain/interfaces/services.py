"""
Service Interfaces
Abstract base classes for service layer
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

class IContentGenerator(ABC):
    """Content generator service interface"""
    
    @abstractmethod
    async def generate_content(self, request: Any) -> Any:
        """Generate content based on request"""
        pass
    
    @abstractmethod
    async def enhance_content(self, content_id: str, enhancements: Dict[str, Any]) -> Any:
        """Enhance existing content"""
        pass

class ICollaborationService(ABC):
    """Collaboration service interface"""
    
    @abstractmethod
    async def create_session(self, project_id: str, session_name: str, creator_id: str) -> Any:
        """Create a new collaboration session"""
        pass
    
    @abstractmethod
    async def join_session(self, session_id: str, user_id: str) -> bool:
        """Join a collaboration session"""
        pass
    
    @abstractmethod
    async def leave_session(self, session_id: str, user_id: str) -> bool:
        """Leave a collaboration session"""
        pass

class IAnalyticsService(ABC):
    """Analytics service interface"""
    
    @abstractmethod
    async def track_content_creation(self, user_id: str, content_id: str, 
                                    content_type: str, processing_time: float,
                                    quality_score: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Track content creation event"""
        pass
    
    @abstractmethod
    async def get_user_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get user analytics"""
        pass
    
    @abstractmethod
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics"""
        pass

class ICacheService(ABC):
    """Cache service interface"""
    
    @abstractmethod
    async def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, namespace: str = "default") -> bool:
        """Set value in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    async def clear_namespace(self, namespace: str) -> bool:
        """Clear all keys in a namespace"""
        pass







