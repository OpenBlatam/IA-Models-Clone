"""
Analytics Service
=================

Advanced analytics service for tracking and analyzing system usage.
"""

from __future__ import annotations
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json

from ...shared.events.event_bus import get_event_bus, DomainEvent, EventMetadata


logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Analytics event types"""
    WORKFLOW_CREATED = "workflow.created"
    WORKFLOW_UPDATED = "workflow.updated"
    WORKFLOW_DELETED = "workflow.deleted"
    WORKFLOW_STATUS_CHANGED = "workflow.status_changed"
    NODE_CREATED = "node.created"
    NODE_UPDATED = "node.updated"
    NODE_DELETED = "node.deleted"
    NODE_CONTENT_UPDATED = "node.content_updated"
    NODE_PRIORITY_CHANGED = "node.priority_changed"
    NODE_TAG_ADDED = "node.tag_added"
    NODE_TAG_REMOVED = "node.tag_removed"
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    API_REQUEST = "api.request"
    ERROR_OCCURRED = "error.occurred"


@dataclass
class AnalyticsEvent:
    """Analytics event data"""
    event_type: EventType
    user_id: Optional[str]
    session_id: Optional[str]
    timestamp: datetime
    properties: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class AnalyticsConfig:
    """Analytics service configuration"""
    enabled: bool = True
    batch_size: int = 100
    flush_interval: int = 60  # seconds
    retention_days: int = 90
    anonymize_data: bool = False
    
    # Storage settings
    storage_type: str = "memory"  # memory, database, file
    database_url: Optional[str] = None
    file_path: Optional[str] = None
    
    # Real-time settings
    real_time_enabled: bool = True
    real_time_events: List[EventType] = None


class AnalyticsStorage(ABC):
    """Abstract analytics storage interface"""
    
    @abstractmethod
    async def store_event(self, event: AnalyticsEvent) -> None:
        """Store analytics event"""
        pass
    
    @abstractmethod
    async def store_events(self, events: List[AnalyticsEvent]) -> None:
        """Store multiple analytics events"""
        pass
    
    @abstractmethod
    async def get_events(
        self,
        event_type: Optional[EventType] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[AnalyticsEvent]:
        """Get analytics events"""
        pass
    
    @abstractmethod
    async def get_aggregated_data(
        self,
        event_type: EventType,
        start_date: datetime,
        end_date: datetime,
        group_by: str = "day"
    ) -> Dict[str, Any]:
        """Get aggregated analytics data"""
        pass


class MemoryAnalyticsStorage(AnalyticsStorage):
    """In-memory analytics storage"""
    
    def __init__(self, max_events: int = 10000):
        self._events: List[AnalyticsEvent] = []
        self._max_events = max_events
        self._lock = asyncio.Lock()
    
    async def store_event(self, event: AnalyticsEvent) -> None:
        """Store analytics event"""
        async with self._lock:
            self._events.append(event)
            
            # Maintain max events limit
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]
    
    async def store_events(self, events: List[AnalyticsEvent]) -> None:
        """Store multiple analytics events"""
        async with self._lock:
            self._events.extend(events)
            
            # Maintain max events limit
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]
    
    async def get_events(
        self,
        event_type: Optional[EventType] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[AnalyticsEvent]:
        """Get analytics events"""
        async with self._lock:
            filtered_events = self._events
            
            # Filter by event type
            if event_type:
                filtered_events = [e for e in filtered_events if e.event_type == event_type]
            
            # Filter by user ID
            if user_id:
                filtered_events = [e for e in filtered_events if e.user_id == user_id]
            
            # Filter by date range
            if start_date:
                filtered_events = [e for e in filtered_events if e.timestamp >= start_date]
            
            if end_date:
                filtered_events = [e for e in filtered_events if e.timestamp <= end_date]
            
            # Sort by timestamp (newest first)
            filtered_events.sort(key=lambda e: e.timestamp, reverse=True)
            
            return filtered_events[:limit]
    
    async def get_aggregated_data(
        self,
        event_type: EventType,
        start_date: datetime,
        end_date: datetime,
        group_by: str = "day"
    ) -> Dict[str, Any]:
        """Get aggregated analytics data"""
        async with self._lock:
            # Filter events
            filtered_events = [
                e for e in self._events
                if e.event_type == event_type
                and start_date <= e.timestamp <= end_date
            ]
            
            # Group by time period
            groups = {}
            for event in filtered_events:
                if group_by == "day":
                    key = event.timestamp.date().isoformat()
                elif group_by == "hour":
                    key = event.timestamp.strftime("%Y-%m-%d %H:00")
                elif group_by == "week":
                    week_start = event.timestamp - timedelta(days=event.timestamp.weekday())
                    key = week_start.date().isoformat()
                else:
                    key = event.timestamp.date().isoformat()
                
                if key not in groups:
                    groups[key] = 0
                groups[key] += 1
            
            return {
                "event_type": event_type.value,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "group_by": group_by,
                "data": groups,
                "total_events": len(filtered_events)
            }


class AnalyticsService:
    """
    Advanced analytics service
    
    Provides comprehensive analytics tracking with real-time processing,
    aggregation, and reporting capabilities.
    """
    
    def __init__(self, config: Optional[AnalyticsConfig] = None):
        self.config = config or AnalyticsConfig()
        self._storage: Optional[AnalyticsStorage] = None
        self._event_bus = get_event_bus()
        self._event_buffer: List[AnalyticsEvent] = []
        self._statistics = {
            "events_tracked": 0,
            "events_processed": 0,
            "events_failed": 0,
            "by_event_type": {event_type.value: 0 for event_type in EventType}
        }
        self._flush_task: Optional[asyncio.Task] = None
        self._initialize_storage()
        self._start_flush_task()
    
    def _initialize_storage(self):
        """Initialize analytics storage"""
        if self.config.storage_type == "memory":
            self._storage = MemoryAnalyticsStorage()
        else:
            logger.warning(f"Storage type {self.config.storage_type} not implemented")
            self._storage = MemoryAnalyticsStorage()
    
    def _start_flush_task(self):
        """Start background flush task"""
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_events_periodically())
    
    async def track_event(
        self,
        event_type: EventType,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track analytics event"""
        try:
            if not self.config.enabled:
                return
            
            # Create analytics event
            event = AnalyticsEvent(
                event_type=event_type,
                user_id=user_id,
                session_id=session_id,
                timestamp=datetime.utcnow(),
                properties=properties or {},
                metadata=metadata or {}
            )
            
            # Anonymize data if configured
            if self.config.anonymize_data:
                event.user_id = self._anonymize_user_id(user_id) if user_id else None
            
            # Add to buffer
            self._event_buffer.append(event)
            
            # Update statistics
            self._statistics["events_tracked"] += 1
            self._statistics["by_event_type"][event_type.value] += 1
            
            # Process real-time if enabled
            if self.config.real_time_enabled and event_type in (self.config.real_time_events or []):
                await self._process_real_time_event(event)
            
            # Flush if buffer is full
            if len(self._event_buffer) >= self.config.batch_size:
                await self._flush_events()
            
        except Exception as e:
            logger.error(f"Failed to track event {event_type}: {e}")
            self._statistics["events_failed"] += 1
    
    async def track_workflow_created(
        self,
        workflow_id: str,
        name: str,
        created_at: str
    ) -> None:
        """Track workflow created event"""
        await self.track_event(
            event_type=EventType.WORKFLOW_CREATED,
            properties={
                "workflow_id": workflow_id,
                "workflow_name": name,
                "created_at": created_at
            }
        )
    
    async def track_workflow_updated(
        self,
        workflow_id: str,
        field: str,
        old_value: Any,
        new_value: Any,
        updated_at: str
    ) -> None:
        """Track workflow updated event"""
        await self.track_event(
            event_type=EventType.WORKFLOW_UPDATED,
            properties={
                "workflow_id": workflow_id,
                "field": field,
                "old_value": str(old_value),
                "new_value": str(new_value),
                "updated_at": updated_at
            }
        )
    
    async def track_workflow_deleted(
        self,
        workflow_id: str,
        deleted_at: str
    ) -> None:
        """Track workflow deleted event"""
        await self.track_event(
            event_type=EventType.WORKFLOW_DELETED,
            properties={
                "workflow_id": workflow_id,
                "deleted_at": deleted_at
            }
        )
    
    async def track_workflow_status_changed(
        self,
        workflow_id: str,
        old_status: str,
        new_status: str,
        changed_at: str
    ) -> None:
        """Track workflow status changed event"""
        await self.track_event(
            event_type=EventType.WORKFLOW_STATUS_CHANGED,
            properties={
                "workflow_id": workflow_id,
                "old_status": old_status,
                "new_status": new_status,
                "changed_at": changed_at
            }
        )
    
    async def track_node_created(
        self,
        node_id: str,
        title: str,
        created_at: str
    ) -> None:
        """Track node created event"""
        await self.track_event(
            event_type=EventType.NODE_CREATED,
            properties={
                "node_id": node_id,
                "title": title,
                "created_at": created_at
            }
        )
    
    async def track_node_updated(
        self,
        node_id: str,
        field: str,
        old_value: Any,
        new_value: Any,
        updated_at: str
    ) -> None:
        """Track node updated event"""
        await self.track_event(
            event_type=EventType.NODE_UPDATED,
            properties={
                "node_id": node_id,
                "field": field,
                "old_value": str(old_value),
                "new_value": str(new_value),
                "updated_at": updated_at
            }
        )
    
    async def track_node_deleted(
        self,
        node_id: str,
        deleted_at: str
    ) -> None:
        """Track node deleted event"""
        await self.track_event(
            event_type=EventType.NODE_DELETED,
            properties={
                "node_id": node_id,
                "deleted_at": deleted_at
            }
        )
    
    async def track_content_updated(
        self,
        node_id: str,
        old_content_length: int,
        new_content_length: int,
        updated_at: str
    ) -> None:
        """Track content updated event"""
        await self.track_event(
            event_type=EventType.NODE_CONTENT_UPDATED,
            properties={
                "node_id": node_id,
                "old_content_length": old_content_length,
                "new_content_length": new_content_length,
                "content_length_change": new_content_length - old_content_length,
                "updated_at": updated_at
            }
        )
    
    async def track_priority_changed(
        self,
        node_id: str,
        old_priority: int,
        new_priority: int,
        changed_at: str
    ) -> None:
        """Track priority changed event"""
        await self.track_event(
            event_type=EventType.NODE_PRIORITY_CHANGED,
            properties={
                "node_id": node_id,
                "old_priority": old_priority,
                "new_priority": new_priority,
                "priority_change": new_priority - old_priority,
                "changed_at": changed_at
            }
        )
    
    async def track_tag_added(
        self,
        node_id: str,
        tag: str,
        added_at: str
    ) -> None:
        """Track tag added event"""
        await self.track_event(
            event_type=EventType.NODE_TAG_ADDED,
            properties={
                "node_id": node_id,
                "tag": tag,
                "added_at": added_at
            }
        )
    
    async def track_tag_removed(
        self,
        node_id: str,
        tag: str,
        removed_at: str
    ) -> None:
        """Track tag removed event"""
        await self.track_event(
            event_type=EventType.NODE_TAG_REMOVED,
            properties={
                "node_id": node_id,
                "tag": tag,
                "removed_at": removed_at
            }
        )
    
    async def get_analytics_data(
        self,
        event_type: Optional[EventType] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[AnalyticsEvent]:
        """Get analytics data"""
        try:
            return await self._storage.get_events(
                event_type=event_type,
                user_id=user_id,
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Failed to get analytics data: {e}")
            return []
    
    async def get_aggregated_analytics(
        self,
        event_type: EventType,
        start_date: datetime,
        end_date: datetime,
        group_by: str = "day"
    ) -> Dict[str, Any]:
        """Get aggregated analytics data"""
        try:
            return await self._storage.get_aggregated_data(
                event_type=event_type,
                start_date=start_date,
                end_date=end_date,
                group_by=group_by
            )
        except Exception as e:
            logger.error(f"Failed to get aggregated analytics: {e}")
            return {}
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard analytics data"""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)
            
            # Get aggregated data for different event types
            workflow_created = await self.get_aggregated_analytics(
                EventType.WORKFLOW_CREATED, start_date, end_date
            )
            workflow_updated = await self.get_aggregated_analytics(
                EventType.WORKFLOW_UPDATED, start_date, end_date
            )
            node_created = await self.get_aggregated_analytics(
                EventType.NODE_CREATED, start_date, end_date
            )
            node_updated = await self.get_aggregated_analytics(
                EventType.NODE_UPDATED, start_date, end_date
            )
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": 30
                },
                "workflow_created": workflow_created,
                "workflow_updated": workflow_updated,
                "node_created": node_created,
                "node_updated": node_updated,
                "statistics": self._statistics
            }
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {}
    
    async def _flush_events(self) -> None:
        """Flush buffered events to storage"""
        if not self._event_buffer:
            return
        
        try:
            events_to_flush = self._event_buffer.copy()
            self._event_buffer.clear()
            
            await self._storage.store_events(events_to_flush)
            self._statistics["events_processed"] += len(events_to_flush)
            
            logger.debug(f"Flushed {len(events_to_flush)} analytics events")
            
        except Exception as e:
            logger.error(f"Failed to flush analytics events: {e}")
            self._statistics["events_failed"] += len(self._event_buffer)
    
    async def _flush_events_periodically(self) -> None:
        """Periodically flush events to storage"""
        while True:
            try:
                await asyncio.sleep(self.config.flush_interval)
                await self._flush_events()
            except Exception as e:
                logger.error(f"Error in periodic flush task: {e}")
    
    async def _process_real_time_event(self, event: AnalyticsEvent) -> None:
        """Process real-time analytics event"""
        try:
            # Publish real-time event
            domain_event = DomainEvent(
                event_type="analytics.real_time",
                data={
                    "event_type": event.event_type.value,
                    "user_id": event.user_id,
                    "timestamp": event.timestamp.isoformat(),
                    "properties": event.properties
                },
                metadata=EventMetadata(
                    source="analytics_service",
                    priority=4  # LOW
                )
            )
            
            await self._event_bus.publish(domain_event)
            
        except Exception as e:
            logger.error(f"Failed to process real-time event: {e}")
    
    def _anonymize_user_id(self, user_id: str) -> str:
        """Anonymize user ID for privacy"""
        import hashlib
        return hashlib.sha256(user_id.encode()).hexdigest()[:8]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get analytics service statistics"""
        return {
            **self._statistics,
            "buffer_size": len(self._event_buffer),
            "config": {
                "enabled": self.config.enabled,
                "batch_size": self.config.batch_size,
                "flush_interval": self.config.flush_interval,
                "retention_days": self.config.retention_days,
                "anonymize_data": self.config.anonymize_data,
                "storage_type": self.config.storage_type,
                "real_time_enabled": self.config.real_time_enabled
            }
        }




