from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Protocol
from uuid import UUID
from .entities import (
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🔌 Domain Interfaces
===================

Abstract interfaces defining contracts for the domain layer
in the clean architecture pattern.
"""


    User, ContentRequest, GeneratedContent, ContentTemplate, UsageMetrics,
    Status, ContentType, Language, Tone
)


class UserRepository(ABC):
    """Abstract interface for user data access"""
    
    @abstractmethod
    async def create(self, user: User) -> User:
        """Create a new user"""
        pass
    
    @abstractmethod
    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID"""
        pass
    
    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        pass
    
    @abstractmethod
    async def update(self, user: User) -> User:
        """Update user information"""
        pass
    
    @abstractmethod
    async def delete(self, user_id: UUID) -> bool:
        """Delete user"""
        pass
    
    @abstractmethod
    async def list_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """List users with pagination"""
        pass
    
    @abstractmethod
    async def update_credits(self, user_id: UUID, credits: int) -> bool:
        """Update user credits"""
        pass
    
    @abstractmethod
    async def get_usage_metrics(self, user_id: UUID, start_date: str, end_date: str) -> List[UsageMetrics]:
        """Get user usage metrics for a date range"""
        pass


class ContentRepository(ABC):
    """Abstract interface for content data access"""
    
    @abstractmethod
    async async def create_request(self, request: ContentRequest) -> ContentRequest:
        """Create a new content request"""
        pass
    
    @abstractmethod
    async async def get_request_by_id(self, request_id: UUID) -> Optional[ContentRequest]:
        """Get content request by ID"""
        pass
    
    @abstractmethod
    async async def update_request(self, request: ContentRequest) -> ContentRequest:
        """Update content request"""
        pass
    
    @abstractmethod
    async async def list_user_requests(self, user_id: UUID, skip: int = 0, limit: int = 100) -> List[ContentRequest]:
        """List user's content requests"""
        pass
    
    @abstractmethod
    async def create_content(self, content: GeneratedContent) -> GeneratedContent:
        """Create generated content"""
        pass
    
    @abstractmethod
    async def get_content_by_id(self, content_id: UUID) -> Optional[GeneratedContent]:
        """Get generated content by ID"""
        pass
    
    @abstractmethod
    async async def get_content_by_request(self, request_id: UUID) -> Optional[GeneratedContent]:
        """Get content by request ID"""
        pass
    
    @abstractmethod
    async def update_content(self, content: GeneratedContent) -> GeneratedContent:
        """Update generated content"""
        pass
    
    @abstractmethod
    async def list_user_content(self, user_id: UUID, skip: int = 0, limit: int = 100) -> List[GeneratedContent]:
        """List user's generated content"""
        pass
    
    @abstractmethod
    async def delete_content(self, content_id: UUID) -> bool:
        """Delete generated content"""
        pass
    
    @abstractmethod
    async def search_content(self, user_id: UUID, query: str, 
                           content_type: Optional[ContentType] = None,
                           skip: int = 0, limit: int = 100) -> List[GeneratedContent]:
        """Search user's content"""
        pass


class TemplateRepository(ABC):
    """Abstract interface for template data access"""
    
    @abstractmethod
    async def create_template(self, template: ContentTemplate) -> ContentTemplate:
        """Create a new template"""
        pass
    
    @abstractmethod
    async def get_template_by_id(self, template_id: UUID) -> Optional[ContentTemplate]:
        """Get template by ID"""
        pass
    
    @abstractmethod
    async def update_template(self, template: ContentTemplate) -> ContentTemplate:
        """Update template"""
        pass
    
    @abstractmethod
    async def delete_template(self, template_id: UUID) -> bool:
        """Delete template"""
        pass
    
    @abstractmethod
    async def list_user_templates(self, user_id: UUID, skip: int = 0, limit: int = 100) -> List[ContentTemplate]:
        """List user's templates"""
        pass
    
    @abstractmethod
    async def list_public_templates(self, skip: int = 0, limit: int = 100) -> List[ContentTemplate]:
        """List public templates"""
        pass
    
    @abstractmethod
    async def search_templates(self, query: str, user_id: Optional[UUID] = None,
                             content_type: Optional[ContentType] = None,
                             skip: int = 0, limit: int = 100) -> List[ContentTemplate]:
        """Search templates"""
        pass


class MetricsRepository(ABC):
    """Abstract interface for metrics data access"""
    
    @abstractmethod
    async def create_metrics(self, metrics: UsageMetrics) -> UsageMetrics:
        """Create usage metrics"""
        pass
    
    @abstractmethod
    async def get_user_metrics(self, user_id: UUID, date: str) -> Optional[UsageMetrics]:
        """Get user metrics for a specific date"""
        pass
    
    @abstractmethod
    async def update_metrics(self, metrics: UsageMetrics) -> UsageMetrics:
        """Update usage metrics"""
        pass
    
    @abstractmethod
    async def get_user_metrics_range(self, user_id: UUID, start_date: str, end_date: str) -> List[UsageMetrics]:
        """Get user metrics for a date range"""
        pass
    
    @abstractmethod
    async def get_aggregated_metrics(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get aggregated metrics for all users"""
        pass


class CacheService(ABC):
    """Abstract interface for caching service"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    @abstractmethod
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment counter in cache"""
        pass
    
    @abstractmethod
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for key"""
        pass
    
    @abstractmethod
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        pass


class AIService(ABC):
    """Abstract interface for AI service"""
    
    @abstractmethod
    async def generate_content(self, prompt: str, **kwargs) -> str:
        """Generate content using AI"""
        pass
    
    @abstractmethod
    async def analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze content quality"""
        pass
    
    @abstractmethod
    async def optimize_seo(self, content: str, keywords: List[str]) -> str:
        """Optimize content for SEO"""
        pass
    
    @abstractmethod
    async def translate_content(self, content: str, target_language: Language) -> str:
        """Translate content to target language"""
        pass
    
    @abstractmethod
    async def summarize_content(self, content: str, max_length: int = 200) -> str:
        """Summarize content"""
        pass
    
    @abstractmethod
    async def check_plagiarism(self, content: str) -> float:
        """Check content for plagiarism"""
        pass
    
    @abstractmethod
    async def get_embeddings(self, text: str) -> List[float]:
        """Get text embeddings"""
        pass
    
    @abstractmethod
    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate content for multiple prompts"""
        pass


class EventPublisher(ABC):
    """Abstract interface for event publishing"""
    
    @abstractmethod
    async def publish(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Publish an event"""
        pass
    
    @abstractmethod
    async def publish_batch(self, events: List[Dict[str, Any]]) -> bool:
        """Publish multiple events"""
        pass
    
    @abstractmethod
    async def subscribe(self, event_type: str, handler: callable) -> bool:
        """Subscribe to events"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, event_type: str, handler: callable) -> bool:
        """Unsubscribe from events"""
        pass


class NotificationService(ABC):
    """Abstract interface for notification service"""
    
    @abstractmethod
    async def send_email(self, to_email: str, subject: str, content: str) -> bool:
        """Send email notification"""
        pass
    
    @abstractmethod
    async def send_push_notification(self, user_id: UUID, title: str, message: str) -> bool:
        """Send push notification"""
        pass
    
    @abstractmethod
    async def send_sms(self, phone_number: str, message: str) -> bool:
        """Send SMS notification"""
        pass


class FileStorageService(ABC):
    """Abstract interface for file storage"""
    
    @abstractmethod
    async async def upload_file(self, file_data: bytes, filename: str, content_type: str) -> str:
        """Upload file and return URL"""
        pass
    
    @abstractmethod
    async async def download_file(self, file_url: str) -> bytes:
        """Download file content"""
        pass
    
    @abstractmethod
    async def delete_file(self, file_url: str) -> bool:
        """Delete file"""
        pass
    
    @abstractmethod
    async def get_file_info(self, file_url: str) -> Dict[str, Any]:
        """Get file information"""
        pass


class SearchService(ABC):
    """Abstract interface for search functionality"""
    
    @abstractmethod
    async def index_content(self, content: GeneratedContent) -> bool:
        """Index content for search"""
        pass
    
    @abstractmethod
    async def search_content(self, query: str, user_id: UUID, 
                           filters: Optional[Dict[str, Any]] = None,
                           skip: int = 0, limit: int = 100) -> List[GeneratedContent]:
        """Search indexed content"""
        pass
    
    @abstractmethod
    async def delete_from_index(self, content_id: UUID) -> bool:
        """Delete content from search index"""
        pass
    
    @abstractmethod
    async def update_index(self, content: GeneratedContent) -> bool:
        """Update content in search index"""
        pass


class RateLimiter(ABC):
    """Abstract interface for rate limiting"""
    
    @abstractmethod
    async def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """Check if request is allowed"""
        pass
    
    @abstractmethod
    async def increment(self, key: str, window: int) -> int:
        """Increment request counter"""
        pass
    
    @abstractmethod
    async def get_remaining(self, key: str, limit: int, window: int) -> int:
        """Get remaining requests allowed"""
        pass
    
    @abstractmethod
    async def reset(self, key: str) -> bool:
        """Reset rate limit for key"""
        pass


class MonitoringService(ABC):
    """Abstract interface for monitoring and observability"""
    
    @abstractmethod
    async def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric"""
        pass
    
    @abstractmethod
    async def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter"""
        pass
    
    @abstractmethod
    async def record_timing(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record timing information"""
        pass
    
    @abstractmethod
    async def record_event(self, name: str, data: Dict[str, Any]) -> None:
        """Record an event"""
        pass
    
    @abstractmethod
    async def set_health_status(self, service: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Set health status for a service"""
        pass


class SecurityService(ABC):
    """Abstract interface for security operations"""
    
    @abstractmethod
    async def hash_password(self, password: str) -> str:
        """Hash a password"""
        pass
    
    @abstractmethod
    async def verify_password(self, password: str, hashed: str) -> bool:
        """Verify a password against hash"""
        pass
    
    @abstractmethod
    async def generate_token(self, user_id: UUID, expires_in: int = 3600) -> str:
        """Generate JWT token"""
        pass
    
    @abstractmethod
    async def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        pass
    
    @abstractmethod
    async def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        pass
    
    @abstractmethod
    async def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        pass


# Protocol-based interfaces for better type checking
class ContentGenerator(Protocol):
    """Protocol for content generation"""
    
    async def generate(self, request: ContentRequest) -> GeneratedContent:
        """Generate content from request"""
        ...
    
    async async def validate_request(self, request: ContentRequest) -> bool:
        """Validate content request"""
        ...
    
    async def estimate_cost(self, request: ContentRequest) -> int:
        """Estimate cost for request"""
        ...


class ContentAnalyzer(Protocol):
    """Protocol for content analysis"""
    
    async def analyze_quality(self, content: str) -> Dict[str, float]:
        """Analyze content quality"""
        ...
    
    async def check_plagiarism(self, content: str) -> float:
        """Check for plagiarism"""
        ...
    
    async def suggest_improvements(self, content: str) -> List[str]:
        """Suggest content improvements"""
        ...


class ContentOptimizer(Protocol):
    """Protocol for content optimization"""
    
    async def optimize_seo(self, content: str, keywords: List[str]) -> str:
        """Optimize for SEO"""
        ...
    
    async def improve_readability(self, content: str) -> str:
        """Improve readability"""
        ...
    
    async def adjust_tone(self, content: str, target_tone: Tone) -> str:
        """Adjust content tone"""
        ... 