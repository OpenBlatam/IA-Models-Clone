"""
Ports (interfaces) for Hexagonal Architecture.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..domains.export.entities import ExportRequest, ExportTask, ExportResult
from ..domains.export.value_objects import ExportConfig, QualityMetrics


class ExportPort(ABC):
    """Port for export operations."""
    
    @abstractmethod
    async def export_document(
        self,
        request: ExportRequest,
        config: ExportConfig
    ) -> ExportResult:
        """Export a document."""
        pass
    
    @abstractmethod
    async def get_export_status(self, task_id: str) -> Optional[ExportTask]:
        """Get export task status."""
        pass
    
    @abstractmethod
    async def cancel_export(self, task_id: str) -> bool:
        """Cancel an export task."""
        pass
    
    @abstractmethod
    async def list_supported_formats(self) -> List[str]:
        """List supported export formats."""
        pass


class QualityPort(ABC):
    """Port for quality operations."""
    
    @abstractmethod
    async def validate_content(
        self,
        content: Dict[str, Any],
        config: ExportConfig
    ) -> QualityMetrics:
        """Validate document content."""
        pass
    
    @abstractmethod
    async def enhance_content(
        self,
        content: Dict[str, Any],
        config: ExportConfig
    ) -> Dict[str, Any]:
        """Enhance document content."""
        pass
    
    @abstractmethod
    async def get_quality_metrics(
        self,
        content: Dict[str, Any],
        config: ExportConfig
    ) -> QualityMetrics:
        """Get quality metrics for content."""
        pass
    
    @abstractmethod
    async def apply_quality_improvements(
        self,
        content: Dict[str, Any],
        metrics: QualityMetrics
    ) -> Dict[str, Any]:
        """Apply quality improvements to content."""
        pass


class TaskPort(ABC):
    """Port for task management operations."""
    
    @abstractmethod
    async def create_task(
        self,
        task_type: str,
        data: Dict[str, Any],
        priority: int = 0
    ) -> str:
        """Create a new task."""
        pass
    
    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID."""
        pass
    
    @abstractmethod
    async def update_task_status(
        self,
        task_id: str,
        status: str,
        progress: float = 0.0
    ) -> bool:
        """Update task status."""
        pass
    
    @abstractmethod
    async def complete_task(
        self,
        task_id: str,
        result: Dict[str, Any]
    ) -> bool:
        """Complete a task."""
        pass
    
    @abstractmethod
    async def fail_task(
        self,
        task_id: str,
        error_message: str
    ) -> bool:
        """Mark task as failed."""
        pass
    
    @abstractmethod
    async def list_tasks(
        self,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List tasks with optional filtering."""
        pass


class UserPort(ABC):
    """Port for user management operations."""
    
    @abstractmethod
    async def authenticate_user(
        self,
        username: str,
        password: str
    ) -> Optional[Dict[str, Any]]:
        """Authenticate user."""
        pass
    
    @abstractmethod
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        pass
    
    @abstractmethod
    async def create_user(
        self,
        user_data: Dict[str, Any]
    ) -> str:
        """Create a new user."""
        pass
    
    @abstractmethod
    async def update_user(
        self,
        user_id: str,
        user_data: Dict[str, Any]
    ) -> bool:
        """Update user data."""
        pass
    
    @abstractmethod
    async def delete_user(self, user_id: str) -> bool:
        """Delete user."""
        pass
    
    @abstractmethod
    async def check_permission(
        self,
        user_id: str,
        permission: str
    ) -> bool:
        """Check if user has permission."""
        pass


class StoragePort(ABC):
    """Port for storage operations."""
    
    @abstractmethod
    async def store_file(
        self,
        file_path: str,
        content: bytes,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a file."""
        pass
    
    @abstractmethod
    async def retrieve_file(self, file_id: str) -> Optional[bytes]:
        """Retrieve a file."""
        pass
    
    @abstractmethod
    async def delete_file(self, file_id: str) -> bool:
        """Delete a file."""
        pass
    
    @abstractmethod
    async def get_file_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get file metadata."""
        pass
    
    @abstractmethod
    async def list_files(
        self,
        prefix: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List files with optional filtering."""
        pass
    
    @abstractmethod
    async def copy_file(
        self,
        source_id: str,
        destination_id: str
    ) -> bool:
        """Copy a file."""
        pass
    
    @abstractmethod
    async def move_file(
        self,
        source_id: str,
        destination_id: str
    ) -> bool:
        """Move a file."""
        pass


class NotificationPort(ABC):
    """Port for notification operations."""
    
    @abstractmethod
    async def send_notification(
        self,
        recipient: str,
        message: str,
        notification_type: str = "info"
    ) -> bool:
        """Send a notification."""
        pass
    
    @abstractmethod
    async def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        attachments: Optional[List[str]] = None
    ) -> bool:
        """Send an email."""
        pass
    
    @abstractmethod
    async def send_sms(
        self,
        phone_number: str,
        message: str
    ) -> bool:
        """Send an SMS."""
        pass
    
    @abstractmethod
    async def send_webhook(
        self,
        url: str,
        payload: Dict[str, Any]
    ) -> bool:
        """Send a webhook."""
        pass
    
    @abstractmethod
    async def schedule_notification(
        self,
        recipient: str,
        message: str,
        scheduled_time: datetime,
        notification_type: str = "info"
    ) -> str:
        """Schedule a notification."""
        pass
    
    @abstractmethod
    async def cancel_scheduled_notification(self, notification_id: str) -> bool:
        """Cancel a scheduled notification."""
        pass


class CachePort(ABC):
    """Port for caching operations."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
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
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MessageQueuePort(ABC):
    """Port for message queue operations."""
    
    @abstractmethod
    async def publish_message(
        self,
        topic: str,
        message: Dict[str, Any]
    ) -> bool:
        """Publish a message to a topic."""
        pass
    
    @abstractmethod
    async def subscribe_to_topic(
        self,
        topic: str,
        handler: callable
    ) -> str:
        """Subscribe to a topic."""
        pass
    
    @abstractmethod
    async def unsubscribe_from_topic(
        self,
        topic: str,
        subscription_id: str
    ) -> bool:
        """Unsubscribe from a topic."""
        pass
    
    @abstractmethod
    async def send_request(
        self,
        service: str,
        endpoint: str,
        data: Dict[str, Any],
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Send a request to a service."""
        pass
    
    @abstractmethod
    async def register_request_handler(
        self,
        endpoint: str,
        handler: callable
    ) -> bool:
        """Register a request handler."""
        pass


class DatabasePort(ABC):
    """Port for database operations."""
    
    @abstractmethod
    async def create_record(
        self,
        table: str,
        data: Dict[str, Any]
    ) -> str:
        """Create a database record."""
        pass
    
    @abstractmethod
    async def get_record(
        self,
        table: str,
        record_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a database record."""
        pass
    
    @abstractmethod
    async def update_record(
        self,
        table: str,
        record_id: str,
        data: Dict[str, Any]
    ) -> bool:
        """Update a database record."""
        pass
    
    @abstractmethod
    async def delete_record(
        self,
        table: str,
        record_id: str
    ) -> bool:
        """Delete a database record."""
        pass
    
    @abstractmethod
    async def query_records(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Query database records."""
        pass
    
    @abstractmethod
    async def execute_transaction(
        self,
        operations: List[Dict[str, Any]]
    ) -> bool:
        """Execute database transaction."""
        pass




