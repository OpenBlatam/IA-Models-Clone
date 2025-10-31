"""
Infrastructure Adapters - Concrete implementations of domain interfaces (Ports)
Hexagonal Architecture: These are the "adapters" that implement "ports"
"""

import json
import hashlib
from typing import Optional, Any, List, Dict
from datetime import datetime

from ..domain.interfaces import (
    IAnalysisRepository,
    ICacheService,
    IMLService,
    IEventBus,
    IExportService
)
from ..domain.entities import ContentAnalysis
from ..core.config import get_settings
from ..core.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Repository Adapters
# ============================================================================

class InMemoryAnalysisRepository(IAnalysisRepository):
    """
    Adapter: In-memory repository implementation
    For production, implement with PostgreSQL, MongoDB, etc.
    """
    
    def __init__(self):
        self._storage: Dict[str, ContentAnalysis] = {}
        logger.debug("InMemoryAnalysisRepository initialized")
    
    async def save_analysis(self, analysis: ContentAnalysis) -> None:
        """Save analysis to in-memory storage"""
        self._storage[analysis.content_hash] = analysis
        logger.debug(f"Saved analysis: {analysis.content_hash}")
    
    async def get_analysis(self, content_hash: str) -> Optional[ContentAnalysis]:
        """Retrieve analysis by content hash"""
        return self._storage.get(content_hash)
    
    async def get_recent_analyses(self, limit: int = 10) -> List[ContentAnalysis]:
        """Get recent analyses"""
        analyses = list(self._storage.values())
        # Sort by timestamp (newest first)
        analyses.sort(key=lambda x: x.timestamp, reverse=True)
        return analyses[:limit]


# ============================================================================
# Cache Adapters
# ============================================================================

class RedisCacheAdapter(ICacheService):
    """
    Adapter: Redis cache implementation with fallback to memory
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.redis_client = None
        self.memory_cache: Dict[str, Any] = {}
        self._initialized = False
    
    async def _init_redis(self):
        """Initialize Redis connection"""
        try:
            import redis.asyncio as redis
            self.redis_client = await redis.from_url(
                self.settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5
            )
            await self.redis_client.ping()
            logger.info("Redis cache connected")
            return True
        except Exception as e:
            logger.warning(f"Redis not available: {e} - Using in-memory cache")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.redis_client:
                value = await self.redis_client.get(key)
                if value:
                    return json.loads(value)
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
        
        # Fallback to memory
        return self.memory_cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        ttl = ttl or self.settings.cache_ttl
        
        try:
            if self.redis_client:
                await self.redis_client.setex(
                    key,
                    ttl,
                    json.dumps(value, default=str)
                )
                return True
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
        
        # Fallback to memory
        if len(self.memory_cache) >= self.settings.max_cache_size:
            # Simple FIFO eviction
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = value
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            if self.redis_client:
                await self.redis_client.delete(key)
            self.memory_cache.pop(key, None)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            if self.redis_client:
                return await self.redis_client.exists(key) > 0
            return key in self.memory_cache
        except Exception as e:
            logger.warning(f"Cache exists error: {e}")
            return key in self.memory_cache


# ============================================================================
# ML Service Adapters
# ============================================================================

class AIMLServiceAdapter(IMLService):
    """
    Adapter: ML service implementation
    Can integrate with OpenAI, HuggingFace, custom models, etc.
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.models = {}
        logger.debug("AIMLServiceAdapter initialized")
    
    async def analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze sentiment (placeholder - integrate with real ML)"""
        # TODO: Integrate with actual ML model
        return {
            "score": 0.5,  # Neutral
            "label": "neutral",
            "confidence": 0.7
        }
    
    async def extract_topics(self, content: str) -> List[str]:
        """Extract topics (placeholder)"""
        # TODO: Integrate with topic modeling
        words = content.lower().split()
        # Simple implementation: return unique words
        return list(set(words[:10]))
    
    async def detect_language(self, content: str) -> str:
        """Detect language (placeholder)"""
        # TODO: Integrate with langdetect or similar
        # Simple heuristic
        if any(word in content.lower() for word in ['the', 'and', 'is', 'are']):
            return "en"
        return "unknown"
    
    async def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity (placeholder)"""
        # TODO: Integrate with sentence transformers
        # Simple word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        common = words1.intersection(words2)
        all_words = words1.union(words2)
        return len(common) / len(all_words) if all_words else 0.0
    
    async def generate_summary(self, content: str, max_length: int = 200) -> str:
        """Generate summary (placeholder)"""
        # TODO: Integrate with summarization model
        sentences = content.split('.')
        summary = '. '.join(sentences[:3])
        return summary[:max_length] if len(summary) > max_length else summary


# ============================================================================
# Event Bus Adapters
# ============================================================================

class InMemoryEventBus(IEventBus):
    """
    Adapter: In-memory event bus
    For production, implement with RabbitMQ, Kafka, etc.
    """
    
    def __init__(self):
        self._subscribers: Dict[str, List[callable]] = {}
        self._events: List[Dict[str, Any]] = []
        logger.debug("InMemoryEventBus initialized")
    
    async def publish(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Publish event"""
        event = {
            "type": event_type,
            "data": event_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        self._events.append(event)
        
        # Notify subscribers
        handlers = self._subscribers.get(event_type, [])
        for handler in handlers:
            try:
                if callable(handler):
                    await handler(event_data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
        
        logger.debug(f"Published event: {event_type}")
    
    async def subscribe(self, event_type: str, handler: callable) -> None:
        """Subscribe to event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed to: {event_type}")


# ============================================================================
# Export Service Adapters
# ============================================================================

class ExportServiceAdapter(IExportService):
    """
    Adapter: Export service implementation
    """
    
    def __init__(self):
        logger.debug("ExportServiceAdapter initialized")
    
    async def export_to_json(self, data: Any) -> bytes:
        """Export to JSON"""
        json_str = json.dumps(data, indent=2, default=str)
        return json_str.encode('utf-8')
    
    async def export_to_csv(self, data: Any) -> bytes:
        """Export to CSV"""
        import csv
        import io
        
        output = io.StringIO()
        if isinstance(data, list) and data:
            writer = csv.DictWriter(output, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        else:
            writer = csv.writer(output)
            writer.writerow(["data"])
            writer.writerow([str(data)])
        
        return output.getvalue().encode('utf-8')
    
    async def export_to_pdf(self, data: Any) -> bytes:
        """Export to PDF (placeholder)"""
        # TODO: Integrate with reportlab or similar
        return json.dumps(data).encode('utf-8')
