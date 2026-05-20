"""
Core Interfaces and Abstractions
Defines contracts for services and components
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Protocol
from datetime import datetime


class IStorageService(ABC):
    """Interface for storage services (DynamoDB, PostgreSQL, etc.)"""
    
    @abstractmethod
    async def get(self, key: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get item by key"""
        pass
    
    @abstractmethod
    async def put(self, item: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Put/update item"""
        pass
    
    @abstractmethod
    async def delete(self, key: str, **kwargs) -> None:
        """Delete item by key"""
        pass
    
    @abstractmethod
    async def query(self, **kwargs) -> List[Dict[str, Any]]:
        """Query items"""
        pass


class ICacheService(ABC):
    """Interface for cache services (Redis, Memcached, etc.)"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete key from cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass


class IFileStorageService(ABC):
    """Interface for file storage services (S3, Azure Blob, etc.)"""
    
    @abstractmethod
    async def upload(self, key: str, content: bytes, **kwargs) -> str:
        """Upload file"""
        pass
    
    @abstractmethod
    async def download(self, key: str) -> bytes:
        """Download file"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete file"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if file exists"""
        pass


class IMessageQueueService(ABC):
    """Interface for message queue services (SQS, RabbitMQ, etc.)"""
    
    @abstractmethod
    async def send(self, message: Dict[str, Any], **kwargs) -> str:
        """Send message to queue"""
        pass
    
    @abstractmethod
    async def receive(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        """Receive messages from queue"""
        pass
    
    @abstractmethod
    async def delete(self, receipt_handle: str) -> None:
        """Delete message from queue"""
        pass


class INotificationService(ABC):
    """Interface for notification services (SNS, Email, etc.)"""
    
    @abstractmethod
    async def send(self, recipient: str, message: str, **kwargs) -> str:
        """Send notification"""
        pass


class IMetricsService(ABC):
    """Interface for metrics services (CloudWatch, Prometheus, etc.)"""
    
    @abstractmethod
    async def record(self, metric_name: str, value: float, **kwargs) -> None:
        """Record metric"""
        pass
    
    @abstractmethod
    async def increment(self, metric_name: str, **kwargs) -> None:
        """Increment counter"""
        pass


class ITracingService(ABC):
    """Interface for tracing services (X-Ray, Jaeger, etc.)"""
    
    @abstractmethod
    def start_span(self, name: str, **kwargs):
        """Start a new span"""
        pass
    
    @abstractmethod
    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute"""
        pass
    
    @abstractmethod
    def record_exception(self, exception: Exception) -> None:
        """Record exception in span"""
        pass


class IAuthenticationService(ABC):
    """Interface for authentication services"""
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate user"""
        pass
    
    @abstractmethod
    async def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate authentication token"""
        pass
    
    @abstractmethod
    async def refresh_token(self, token: str) -> str:
        """Refresh authentication token"""
        pass


class IEventPublisher(Protocol):
    """Protocol for event publishers"""
    
    async def publish(self, event_type: str, event_data: Dict[str, Any]) -> str:
        """Publish event"""
        ...


class IEventHandler(Protocol):
    """Protocol for event handlers"""
    
    async def handle(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle event"""
        ...


class IBackgroundTask(Protocol):
    """Protocol for background tasks"""
    
    async def execute(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute background task"""
        ...


class IServiceFactory(ABC):
    """Interface for service factories"""
    
    @abstractmethod
    def create_storage_service(self) -> IStorageService:
        """Create storage service instance"""
        pass
    
    @abstractmethod
    def create_cache_service(self) -> ICacheService:
        """Create cache service instance"""
        pass
    
    @abstractmethod
    def create_file_storage_service(self) -> IFileStorageService:
        """Create file storage service instance"""
        pass
    
    @abstractmethod
    def create_message_queue_service(self) -> IMessageQueueService:
        """Create message queue service instance"""
        pass
    
    @abstractmethod
    def create_notification_service(self) -> INotificationService:
        """Create notification service instance"""
        pass
    
    @abstractmethod
    def create_metrics_service(self) -> IMetricsService:
        """Create metrics service instance"""
        pass
    
    @abstractmethod
    def create_tracing_service(self) -> ITracingService:
        """Create tracing service instance"""
        pass
    
    @abstractmethod
    def create_authentication_service(self) -> IAuthenticationService:
        """Create authentication service instance"""
        pass















