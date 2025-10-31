"""
Domain Repositories - Data Access Interfaces
==========================================

Repository interfaces defining data access contracts for domain entities.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from .entities import Document, ProcessingResult, User, Organization
from .value_objects import DocumentId, UserId, ProcessingStatus, DocumentType, DocumentStatus


class DocumentRepository(ABC):
    """Document repository interface."""
    
    @abstractmethod
    async def save(self, document: Document) -> Document:
        """Save document."""
        pass
    
    @abstractmethod
    async def find_by_id(self, document_id: DocumentId) -> Optional[Document]:
        """Find document by ID."""
        pass
    
    @abstractmethod
    async def find_by_user(self, user_id: UserId, limit: int = 100, offset: int = 0) -> List[Document]:
        """Find documents by user."""
        pass
    
    @abstractmethod
    async def find_by_status(self, status: DocumentStatus, limit: int = 100, offset: int = 0) -> List[Document]:
        """Find documents by status."""
        pass
    
    @abstractmethod
    async def find_by_type(self, document_type: DocumentType, limit: int = 100, offset: int = 0) -> List[Document]:
        """Find documents by type."""
        pass
    
    @abstractmethod
    async def find_by_tag(self, tag: str, limit: int = 100, offset: int = 0) -> List[Document]:
        """Find documents by tag."""
        pass
    
    @abstractmethod
    async def search(self, query: str, limit: int = 100, offset: int = 0) -> List[Document]:
        """Search documents by content."""
        pass
    
    @abstractmethod
    async def delete(self, document_id: DocumentId) -> bool:
        """Delete document."""
        pass
    
    @abstractmethod
    async def count_by_user(self, user_id: UserId) -> int:
        """Count documents by user."""
        pass
    
    @abstractmethod
    async def count_by_status(self, status: DocumentStatus) -> int:
        """Count documents by status."""
        pass
    
    @abstractmethod
    async def exists(self, document_id: DocumentId) -> bool:
        """Check if document exists."""
        pass


class ProcessingResultRepository(ABC):
    """Processing result repository interface."""
    
    @abstractmethod
    async def save(self, result: ProcessingResult) -> ProcessingResult:
        """Save processing result."""
        pass
    
    @abstractmethod
    async def find_by_id(self, result_id: str) -> Optional[ProcessingResult]:
        """Find processing result by ID."""
        pass
    
    @abstractmethod
    async def find_by_document_id(self, document_id: DocumentId, limit: int = 100, offset: int = 0) -> List[ProcessingResult]:
        """Find processing results by document ID."""
        pass
    
    @abstractmethod
    async def find_by_status(self, status: ProcessingStatus, limit: int = 100, offset: int = 0) -> List[ProcessingResult]:
        """Find processing results by status."""
        pass
    
    @abstractmethod
    async def find_recent(self, limit: int = 100, offset: int = 0) -> List[ProcessingResult]:
        """Find recent processing results."""
        pass
    
    @abstractmethod
    async def find_failed(self, limit: int = 100, offset: int = 0) -> List[ProcessingResult]:
        """Find failed processing results."""
        pass
    
    @abstractmethod
    async def delete(self, result_id: str) -> bool:
        """Delete processing result."""
        pass
    
    @abstractmethod
    async def count_by_document_id(self, document_id: DocumentId) -> int:
        """Count processing results by document ID."""
        pass
    
    @abstractmethod
    async def count_by_status(self, status: ProcessingStatus) -> int:
        """Count processing results by status."""
        pass
    
    @abstractmethod
    async def exists(self, result_id: str) -> bool:
        """Check if processing result exists."""
        pass


class UserRepository(ABC):
    """User repository interface."""
    
    @abstractmethod
    async def save(self, user: User) -> User:
        """Save user."""
        pass
    
    @abstractmethod
    async def find_by_id(self, user_id: UserId) -> Optional[User]:
        """Find user by ID."""
        pass
    
    @abstractmethod
    async def find_by_username(self, username: str) -> Optional[User]:
        """Find user by username."""
        pass
    
    @abstractmethod
    async def find_by_email(self, email: str) -> Optional[User]:
        """Find user by email."""
        pass
    
    @abstractmethod
    async def find_active_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """Find active users."""
        pass
    
    @abstractmethod
    async def find_by_role(self, role: str, limit: int = 100, offset: int = 0) -> List[User]:
        """Find users by role."""
        pass
    
    @abstractmethod
    async def search(self, query: str, limit: int = 100, offset: int = 0) -> List[User]:
        """Search users."""
        pass
    
    @abstractmethod
    async def delete(self, user_id: UserId) -> bool:
        """Delete user."""
        pass
    
    @abstractmethod
    async def count_active_users(self) -> int:
        """Count active users."""
        pass
    
    @abstractmethod
    async def count_by_role(self, role: str) -> int:
        """Count users by role."""
        pass
    
    @abstractmethod
    async def exists(self, user_id: UserId) -> bool:
        """Check if user exists."""
        pass
    
    @abstractmethod
    async def exists_by_username(self, username: str) -> bool:
        """Check if username exists."""
        pass
    
    @abstractmethod
    async def exists_by_email(self, email: str) -> bool:
        """Check if email exists."""
        pass


class OrganizationRepository(ABC):
    """Organization repository interface."""
    
    @abstractmethod
    async def save(self, organization: Organization) -> Organization:
        """Save organization."""
        pass
    
    @abstractmethod
    async def find_by_id(self, org_id: str) -> Optional[Organization]:
        """Find organization by ID."""
        pass
    
    @abstractmethod
    async def find_by_name(self, name: str) -> Optional[Organization]:
        """Find organization by name."""
        pass
    
    @abstractmethod
    async def find_by_member(self, user_id: UserId, limit: int = 100, offset: int = 0) -> List[Organization]:
        """Find organizations by member."""
        pass
    
    @abstractmethod
    async def find_active_organizations(self, limit: int = 100, offset: int = 0) -> List[Organization]:
        """Find active organizations."""
        pass
    
    @abstractmethod
    async def search(self, query: str, limit: int = 100, offset: int = 0) -> List[Organization]:
        """Search organizations."""
        pass
    
    @abstractmethod
    async def delete(self, org_id: str) -> bool:
        """Delete organization."""
        pass
    
    @abstractmethod
    async def count_active_organizations(self) -> int:
        """Count active organizations."""
        pass
    
    @abstractmethod
    async def count_by_member(self, user_id: UserId) -> int:
        """Count organizations by member."""
        pass
    
    @abstractmethod
    async def exists(self, org_id: str) -> bool:
        """Check if organization exists."""
        pass
    
    @abstractmethod
    async def exists_by_name(self, name: str) -> bool:
        """Check if organization name exists."""
        pass


class AuditLogRepository(ABC):
    """Audit log repository interface."""
    
    @abstractmethod
    async def log_document_action(self, document_id: DocumentId, action: str, user_id: Optional[UserId], details: Optional[Dict[str, Any]] = None) -> None:
        """Log document action."""
        pass
    
    @abstractmethod
    async def log_processing_action(self, processing_id: str, action: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log processing action."""
        pass
    
    @abstractmethod
    async def log_user_action(self, user_id: UserId, action: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Log user action."""
        pass
    
    @abstractmethod
    async def find_document_audit_log(self, document_id: DocumentId, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Find document audit log."""
        pass
    
    @abstractmethod
    async def find_user_audit_log(self, user_id: UserId, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Find user audit log."""
        pass
    
    @abstractmethod
    async def find_audit_log_by_date_range(self, start_date: datetime, end_date: datetime, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Find audit log by date range."""
        pass


class CacheRepository(ABC):
    """Cache repository interface."""
    
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
    async def clear(self) -> bool:
        """Clear all cache."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MetricsRepository(ABC):
    """Metrics repository interface."""
    
    @abstractmethod
    async def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric."""
        pass
    
    @abstractmethod
    async def record_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        pass
    
    @abstractmethod
    async def record_timing(self, name: str, duration_seconds: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric."""
        pass
    
    @abstractmethod
    async def get_metric(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get metric value."""
        pass
    
    @abstractmethod
    async def get_metrics_by_name(self, name: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get metrics by name."""
        pass
    
    @abstractmethod
    async def get_all_metrics(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get all metrics."""
        pass

















