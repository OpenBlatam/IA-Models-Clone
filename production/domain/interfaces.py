from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Protocol
from datetime import datetime
from .entities import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Domain Interfaces
=================

Core interfaces for the copywriting system following clean architecture principles.
"""


    CopywritingRequest,
    CopywritingResponse,
    PerformanceMetrics,
    User,
    RequestStatus
)


class CopywritingRepository(ABC):
    """Repository interface for copywriting data persistence."""
    
    @abstractmethod
    async async def save_request(self, request: CopywritingRequest) -> CopywritingRequest:
        """Save a copywriting request."""
        pass
    
    @abstractmethod
    async def save_response(self, response: CopywritingResponse) -> CopywritingResponse:
        """Save a copywriting response."""
        pass
    
    @abstractmethod
    async async def get_request_by_id(self, request_id: str) -> Optional[CopywritingRequest]:
        """Get request by ID."""
        pass
    
    @abstractmethod
    async def get_response_by_id(self, response_id: str) -> Optional[CopywritingResponse]:
        """Get response by ID."""
        pass
    
    @abstractmethod
    async async def get_responses_by_request_id(self, request_id: str) -> List[CopywritingResponse]:
        """Get all responses for a request."""
        pass
    
    @abstractmethod
    async async def update_request_status(self, request_id: str, status: RequestStatus) -> bool:
        """Update request status."""
        pass
    
    @abstractmethod
    async def get_user_history(self, user_id: str, limit: int = 50) -> List[CopywritingRequest]:
        """Get user's copywriting history."""
        pass
    
    @abstractmethod
    async async def get_requests_by_status(self, status: RequestStatus) -> List[CopywritingRequest]:
        """Get requests by status."""
        pass
    
    @abstractmethod
    async async def delete_request(self, request_id: str) -> bool:
        """Delete a request."""
        pass
    
    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        pass


class CacheService(ABC):
    """Cache service interface for performance optimization."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter in cache."""
        pass
    
    @abstractmethod
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        pass
    
    @abstractmethod
    async def set_many(self, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in cache."""
        pass
    
    @abstractmethod
    async def clear_pattern(self, pattern: str) -> int:
        """Clear cache entries matching pattern."""
        pass
    
    @abstractmethod
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class AIService(ABC):
    """AI service interface for copywriting generation."""
    
    @abstractmethod
    async def generate_copywriting(self, request: CopywritingRequest) -> CopywritingResponse:
        """Generate copywriting based on request."""
        pass
    
    @abstractmethod
    async def improve_copywriting(self, text: str, suggestions: List[str]) -> str:
        """Improve existing copywriting text."""
        pass
    
    @abstractmethod
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for various metrics."""
        pass
    
    @abstractmethod
    async def get_suggestions(self, text: str, context: Dict[str, Any]) -> List[str]:
        """Get improvement suggestions for text."""
        pass
    
    @abstractmethod
    async def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """Validate and analyze prompt."""
        pass
    
    @abstractmethod
    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about the AI model."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if AI service is available."""
        pass


class EventPublisher(ABC):
    """Event publisher interface for event-driven architecture."""
    
    @abstractmethod
    async def publish(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Publish an event."""
        pass
    
    @abstractmethod
    async def publish_batch(self, events: List[Dict[str, Any]]) -> bool:
        """Publish multiple events."""
        pass
    
    @abstractmethod
    async def subscribe(self, event_type: str, handler: callable) -> str:
        """Subscribe to an event type."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        pass
    
    @abstractmethod
    async def get_event_stats(self) -> Dict[str, Any]:
        """Get event publishing statistics."""
        pass


class MonitoringService(ABC):
    """Monitoring service interface for system observability."""
    
    @abstractmethod
    async def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric."""
        pass
    
    @abstractmethod
    async def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter."""
        pass
    
    @abstractmethod
    async def record_timing(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record timing information."""
        pass
    
    @abstractmethod
    async def record_event(self, name: str, data: Dict[str, Any]) -> None:
        """Record an event."""
        pass
    
    @abstractmethod
    async def get_metrics(self, metric_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get current metrics."""
        pass
    
    @abstractmethod
    async def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        pass


class NotificationService(ABC):
    """Notification service interface for user communications."""
    
    @abstractmethod
    async def send_email(self, to_email: str, subject: str, body: str, template: Optional[str] = None) -> bool:
        """Send email notification."""
        pass
    
    @abstractmethod
    async def send_push_notification(self, user_id: str, title: str, message: str, data: Optional[Dict[str, Any]] = None) -> bool:
        """Send push notification."""
        pass
    
    @abstractmethod
    async def send_sms(self, phone_number: str, message: str) -> bool:
        """Send SMS notification."""
        pass
    
    @abstractmethod
    async def get_notification_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get notification history for user."""
        pass


class SecurityService(ABC):
    """Security service interface for authentication and authorization."""
    
    @abstractmethod
    async def authenticate_user(self, credentials: Dict[str, str]) -> Optional[User]:
        """Authenticate user with credentials."""
        pass
    
    @abstractmethod
    async async def authorize_request(self, user: User, resource: str, action: str) -> bool:
        """Authorize user action on resource."""
        pass
    
    @abstractmethod
    async def generate_token(self, user: User) -> str:
        """Generate authentication token for user."""
        pass
    
    @abstractmethod
    async def validate_token(self, token: str) -> Optional[User]:
        """Validate authentication token."""
        pass
    
    @abstractmethod
    async def rate_limit_check(self, user_id: str, action: str) -> bool:
        """Check if user is rate limited for action."""
        pass


class FileStorageService(ABC):
    """File storage service interface for document management."""
    
    @abstractmethod
    async async def upload_file(self, file_data: bytes, filename: str, content_type: str) -> str:
        """Upload file and return file ID."""
        pass
    
    @abstractmethod
    async async def download_file(self, file_id: str) -> Optional[bytes]:
        """Download file by ID."""
        pass
    
    @abstractmethod
    async def delete_file(self, file_id: str) -> bool:
        """Delete file by ID."""
        pass
    
    @abstractmethod
    async def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get file information."""
        pass
    
    @abstractmethod
    async def list_files(self, user_id: str) -> List[Dict[str, Any]]:
        """List user's files."""
        pass


class AnalyticsService(ABC):
    """Analytics service interface for data analysis."""
    
    @abstractmethod
    async def track_event(self, event_name: str, user_id: str, properties: Dict[str, Any]) -> None:
        """Track user event."""
        pass
    
    @abstractmethod
    async def get_user_analytics(self, user_id: str, time_range: str) -> Dict[str, Any]:
        """Get analytics for specific user."""
        pass
    
    @abstractmethod
    async def get_system_analytics(self, time_range: str) -> Dict[str, Any]:
        """Get system-wide analytics."""
        pass
    
    @abstractmethod
    async def generate_report(self, report_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analytics report."""
        pass


# Protocol interfaces for better type checking
class CopywritingGenerator(Protocol):
    """Protocol for copywriting generation."""
    
    async def generate(self, request: CopywritingRequest) -> CopywritingResponse:
        """Generate copywriting."""
        ...


class TextAnalyzer(Protocol):
    """Protocol for text analysis."""
    
    async def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze text."""
        ...


class QualityChecker(Protocol):
    """Protocol for quality checking."""
    
    async def check_quality(self, text: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Check text quality."""
        ...


class ContentOptimizer(Protocol):
    """Protocol for content optimization."""
    
    async def optimize(self, content: str, target_metrics: Dict[str, Any]) -> str:
        """Optimize content."""
        ... 