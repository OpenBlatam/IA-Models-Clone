"""
Domain Interfaces (Ports) - Define contracts for adapters
Hexagonal Architecture: Domain defines ports, infrastructure implements adapters
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List
from ..domain.entities import ContentAnalysis, SimilarityAnalysis, QualityAnalysis


class IAnalysisRepository(ABC):
    """
    Port: Repository interface for persistence
    Infrastructure layer implements this
    """
    
    @abstractmethod
    async def save_analysis(self, analysis: ContentAnalysis) -> None:
        """Save analysis to persistence"""
        pass
    
    @abstractmethod
    async def get_analysis(self, content_hash: str) -> Optional[ContentAnalysis]:
        """Retrieve analysis by content hash"""
        pass
    
    @abstractmethod
    async def get_recent_analyses(self, limit: int = 10) -> List[ContentAnalysis]:
        """Get recent analyses"""
        pass


class ICacheService(ABC):
    """
    Port: Cache service interface
    Infrastructure layer can implement Redis, Memcached, etc.
    """
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass


class IMLService(ABC):
    """
    Port: Machine Learning service interface
    Infrastructure layer implements ML model integration
    """
    
    @abstractmethod
    async def analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze sentiment of content"""
        pass
    
    @abstractmethod
    async def extract_topics(self, content: str) -> List[str]:
        """Extract topics from content"""
        pass
    
    @abstractmethod
    async def detect_language(self, content: str) -> str:
        """Detect language of content"""
        pass
    
    @abstractmethod
    async def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        pass
    
    @abstractmethod
    async def generate_summary(self, content: str, max_length: int = 200) -> str:
        """Generate summary of content"""
        pass


class IEventBus(ABC):
    """
    Port: Event bus interface for event-driven architecture
    Infrastructure layer implements RabbitMQ, Kafka, etc.
    """
    
    @abstractmethod
    async def publish(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Publish event to bus"""
        pass
    
    @abstractmethod
    async def subscribe(self, event_type: str, handler: callable) -> None:
        """Subscribe to event type"""
        pass


class IExportService(ABC):
    """
    Port: Export service interface
    """
    
    @abstractmethod
    async def export_to_json(self, data: Any) -> bytes:
        """Export data to JSON"""
        pass
    
    @abstractmethod
    async def export_to_csv(self, data: Any) -> bytes:
        """Export data to CSV"""
        pass
    
    @abstractmethod
    async def export_to_pdf(self, data: Any) -> bytes:
        """Export data to PDF"""
        pass


class IMetricsService(ABC):
    """
    Port: Metrics service interface for observability
    """
    
    @abstractmethod
    async def record_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None) -> None:
        """Record counter metric"""
        pass
    
    @abstractmethod
    async def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record gauge metric"""
        pass
    
    @abstractmethod
    async def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record histogram metric"""
        pass


class ITracingService(ABC):
    """
    Port: Distributed tracing service interface
    """
    
    @abstractmethod
    def start_span(self, name: str, **kwargs) -> Any:
        """Start a new trace span"""
        pass
    
    @abstractmethod
    def set_tag(self, key: str, value: str) -> None:
        """Set span tag"""
        pass
    
    @abstractmethod
    def add_event(self, name: str, **kwargs) -> None:
        """Add event to current span"""
        pass
