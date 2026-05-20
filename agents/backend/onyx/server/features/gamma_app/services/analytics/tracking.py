"""
Analytics Tracking Helpers
Shared utilities for tracking analytics events
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class AnalyticsTracker:
    """Helper class for tracking analytics events"""
    
    def __init__(self, redis_client: Optional[Any] = None, db_session: Optional[Any] = None):
        """Initialize analytics tracker"""
        self.redis_client = redis_client
        self.db_session = db_session
    
    async def track_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        content_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a standardized analytics event"""
        event = {
            'event_type': event_type,
            'timestamp': datetime.now(timezone.utc),
            'metadata': metadata or {}
        }
        
        if user_id:
            event['user_id'] = user_id
        if content_id:
            event['content_id'] = content_id
        
        return event
    
    async def save_to_database(self, event: Dict[str, Any], table_class: type) -> bool:
        """Save event to database"""
        if not self.db_session:
            return False
        
        try:
            db_event = table_class(**event)
            self.db_session.add(db_event)
            self.db_session.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving event to database: {e}")
            if self.db_session:
                self.db_session.rollback()
            return False
    
    async def update_redis_metrics(self, event_type: str, event: Dict[str, Any]) -> bool:
        """Update Redis metrics"""
        if not self.redis_client:
            return False
        
        try:
            key = f"analytics:{event_type}:{event.get('timestamp', datetime.now(timezone.utc)).strftime('%Y-%m-%d')}"
            await self.redis_client.incr(key)
            await self.redis_client.expire(key, 86400 * 7)
            return True
        except Exception as e:
            logger.error(f"Error updating Redis metrics: {e}")
            return False
    
    async def update_real_time_metrics(self, event_type: str, event: Dict[str, Any], metrics_store: Dict) -> None:
        """Update real-time metrics store"""
        if event_type not in metrics_store:
            metrics_store[event_type] = []
        
        metrics_store[event_type].append({
            'value': event.get('metadata', {}).get('value', 1),
            'timestamp': event.get('timestamp', datetime.now(timezone.utc))
        })
        
        if len(metrics_store[event_type]) > 1000:
            metrics_store[event_type] = metrics_store[event_type][-1000:]







