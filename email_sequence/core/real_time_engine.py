"""
Real-Time Processing Engine for Email Sequence System

This module provides real-time event processing, WebSocket connections,
and live analytics for the email sequence system.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from uuid import UUID
import time
from dataclasses import dataclass, field
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .exceptions import RealTimeProcessingError
from .cache import cache_manager

logger = logging.getLogger(__name__)
settings = get_settings()


class EventType(str, Enum):
    """Types of real-time events"""
    EMAIL_SENT = "email_sent"
    EMAIL_OPENED = "email_opened"
    EMAIL_CLICKED = "email_clicked"
    EMAIL_BOUNCED = "email_bounced"
    EMAIL_UNSUBSCRIBED = "email_unsubscribed"
    SEQUENCE_STARTED = "sequence_started"
    SEQUENCE_COMPLETED = "sequence_completed"
    SUBSCRIBER_ADDED = "subscriber_added"
    SUBSCRIBER_REMOVED = "subscriber_removed"
    CAMPAIGN_LAUNCHED = "campaign_launched"
    CAMPAIGN_PAUSED = "campaign_paused"
    ANALYTICS_UPDATE = "analytics_update"
    PERFORMANCE_ALERT = "performance_alert"


class ConnectionType(str, Enum):
    """Types of WebSocket connections"""
    DASHBOARD = "dashboard"
    ANALYTICS = "analytics"
    MONITORING = "monitoring"
    ADMIN = "admin"


@dataclass
class RealTimeEvent:
    """Real-time event data structure"""
    event_type: EventType
    sequence_id: UUID
    subscriber_id: Optional[UUID] = None
    campaign_id: Optional[UUID] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None


@dataclass
class WebSocketConnection:
    """WebSocket connection information"""
    websocket: WebSocket
    connection_id: str
    connection_type: ConnectionType
    user_id: str
    subscriptions: Set[str] = field(default_factory=set)
    last_ping: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)


class RealTimeEngine:
    """Real-time processing engine for email sequences"""
    
    def __init__(self):
        """Initialize real-time engine"""
        self.active_connections: Dict[str, WebSocketConnection] = {}
        self.event_handlers: Dict[EventType, List[Callable]] = {}
        self.redis_client: Optional[redis.Redis] = None
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.is_running = False
        
        # Performance metrics
        self.events_processed = 0
        self.connections_active = 0
        self.last_cleanup = datetime.utcnow()
        
        logger.info("Real-Time Engine initialized")
    
    async def initialize(self) -> None:
        """Initialize the real-time engine"""
        try:
            # Initialize Redis client for pub/sub
            self.redis_client = redis.from_url(settings.redis_url)
            await self.redis_client.ping()
            
            # Start background tasks
            self.is_running = True
            asyncio.create_task(self._process_event_queue())
            asyncio.create_task(self._cleanup_connections())
            asyncio.create_task(self._publish_analytics_updates())
            
            logger.info("Real-Time Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing real-time engine: {e}")
            raise RealTimeProcessingError(f"Failed to initialize real-time engine: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the real-time engine"""
        try:
            self.is_running = False
            
            # Close all connections
            for connection in self.active_connections.values():
                try:
                    await connection.websocket.close()
                except Exception as e:
                    logger.warning(f"Error closing connection {connection.connection_id}: {e}")
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Real-Time Engine shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during real-time engine shutdown: {e}")
    
    async def connect_websocket(
        self,
        websocket: WebSocket,
        connection_type: ConnectionType,
        user_id: str
    ) -> str:
        """
        Connect a new WebSocket client.
        
        Args:
            websocket: WebSocket connection
            connection_type: Type of connection
            user_id: User ID
            
        Returns:
            Connection ID
        """
        try:
            await websocket.accept()
            
            connection_id = f"{user_id}_{int(time.time())}"
            connection = WebSocketConnection(
                websocket=websocket,
                connection_id=connection_id,
                connection_type=connection_type,
                user_id=user_id
            )
            
            self.active_connections[connection_id] = connection
            self.connections_active = len(self.active_connections)
            
            # Send welcome message
            await self._send_to_connection(connection_id, {
                "type": "connection_established",
                "connection_id": connection_id,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info(f"WebSocket connected: {connection_id} ({connection_type})")
            return connection_id
            
        except Exception as e:
            logger.error(f"Error connecting WebSocket: {e}")
            raise RealTimeProcessingError(f"Failed to connect WebSocket: {e}")
    
    async def disconnect_websocket(self, connection_id: str) -> None:
        """
        Disconnect a WebSocket client.
        
        Args:
            connection_id: Connection ID to disconnect
        """
        try:
            if connection_id in self.active_connections:
                connection = self.active_connections[connection_id]
                
                try:
                    await connection.websocket.close()
                except Exception as e:
                    logger.warning(f"Error closing WebSocket {connection_id}: {e}")
                
                del self.active_connections[connection_id]
                self.connections_active = len(self.active_connections)
                
                logger.info(f"WebSocket disconnected: {connection_id}")
                
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket {connection_id}: {e}")
    
    async def subscribe_to_events(
        self,
        connection_id: str,
        event_types: List[EventType],
        filters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Subscribe a connection to specific event types.
        
        Args:
            connection_id: Connection ID
            event_types: List of event types to subscribe to
            filters: Optional filters for events
        """
        try:
            if connection_id not in self.active_connections:
                raise RealTimeProcessingError(f"Connection {connection_id} not found")
            
            connection = self.active_connections[connection_id]
            
            # Add event types to subscriptions
            for event_type in event_types:
                connection.subscriptions.add(event_type.value)
            
            # Store filters if provided
            if filters:
                connection.data = filters
            
            # Send subscription confirmation
            await self._send_to_connection(connection_id, {
                "type": "subscription_confirmed",
                "event_types": [et.value for et in event_types],
                "filters": filters,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info(f"Connection {connection_id} subscribed to {len(event_types)} event types")
            
        except Exception as e:
            logger.error(f"Error subscribing to events: {e}")
            raise RealTimeProcessingError(f"Failed to subscribe to events: {e}")
    
    async def publish_event(self, event: RealTimeEvent) -> None:
        """
        Publish a real-time event.
        
        Args:
            event: Event to publish
        """
        try:
            # Add to event queue for processing
            await self.event_queue.put(event)
            
            # Publish to Redis for distributed processing
            if self.redis_client:
                await self.redis_client.publish(
                    "email_sequence_events",
                    json.dumps({
                        "event_type": event.event_type.value,
                        "sequence_id": str(event.sequence_id),
                        "subscriber_id": str(event.subscriber_id) if event.subscriber_id else None,
                        "campaign_id": str(event.campaign_id) if event.campaign_id else None,
                        "data": event.data,
                        "timestamp": event.timestamp.isoformat(),
                        "user_id": event.user_id
                    })
                )
            
            logger.debug(f"Event published: {event.event_type.value} for sequence {event.sequence_id}")
            
        except Exception as e:
            logger.error(f"Error publishing event: {e}")
            raise RealTimeProcessingError(f"Failed to publish event: {e}")
    
    async def send_analytics_update(
        self,
        sequence_id: UUID,
        analytics_data: Dict[str, Any]
    ) -> None:
        """
        Send real-time analytics update.
        
        Args:
            sequence_id: Sequence ID
            analytics_data: Analytics data to send
        """
        try:
            event = RealTimeEvent(
                event_type=EventType.ANALYTICS_UPDATE,
                sequence_id=sequence_id,
                data=analytics_data
            )
            
            await self.publish_event(event)
            
        except Exception as e:
            logger.error(f"Error sending analytics update: {e}")
    
    async def send_performance_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "warning",
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Send performance alert.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity
            data: Additional alert data
        """
        try:
            event = RealTimeEvent(
                event_type=EventType.PERFORMANCE_ALERT,
                sequence_id=UUID("00000000-0000-0000-0000-000000000000"),  # System-wide alert
                data={
                    "alert_type": alert_type,
                    "message": message,
                    "severity": severity,
                    "data": data or {}
                }
            )
            
            await self.publish_event(event)
            
        except Exception as e:
            logger.error(f"Error sending performance alert: {e}")
    
    async def broadcast_to_connections(
        self,
        message: Dict[str, Any],
        connection_type: Optional[ConnectionType] = None,
        user_id: Optional[str] = None
    ) -> None:
        """
        Broadcast message to connections.
        
        Args:
            message: Message to broadcast
            connection_type: Optional connection type filter
            user_id: Optional user ID filter
        """
        try:
            sent_count = 0
            
            for connection in self.active_connections.values():
                # Apply filters
                if connection_type and connection.connection_type != connection_type:
                    continue
                
                if user_id and connection.user_id != user_id:
                    continue
                
                try:
                    await self._send_to_connection(connection.connection_id, message)
                    sent_count += 1
                except Exception as e:
                    logger.warning(f"Error sending to connection {connection.connection_id}: {e}")
            
            logger.debug(f"Broadcast sent to {sent_count} connections")
            
        except Exception as e:
            logger.error(f"Error broadcasting message: {e}")
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get real-time connection statistics.
        
        Returns:
            Dictionary with connection statistics
        """
        try:
            stats = {
                "total_connections": len(self.active_connections),
                "connections_by_type": {},
                "events_processed": self.events_processed,
                "uptime_seconds": (datetime.utcnow() - self.last_cleanup).total_seconds(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Count connections by type
            for connection in self.active_connections.values():
                conn_type = connection.connection_type.value
                stats["connections_by_type"][conn_type] = stats["connections_by_type"].get(conn_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting connection stats: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    async def _process_event_queue(self) -> None:
        """Process events from the queue"""
        while self.is_running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                # Process event
                await self._process_event(event)
                self.events_processed += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _process_event(self, event: RealTimeEvent) -> None:
        """Process a single event"""
        try:
            # Send to subscribed connections
            await self._distribute_event(event)
            
            # Call event handlers
            if event.event_type in self.event_handlers:
                for handler in self.event_handlers[event.event_type]:
                    try:
                        await handler(event)
                    except Exception as e:
                        logger.error(f"Error in event handler: {e}")
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_type.value}: {e}")
    
    async def _distribute_event(self, event: RealTimeEvent) -> None:
        """Distribute event to subscribed connections"""
        try:
            message = {
                "type": "event",
                "event_type": event.event_type.value,
                "sequence_id": str(event.sequence_id),
                "subscriber_id": str(event.subscriber_id) if event.subscriber_id else None,
                "campaign_id": str(event.campaign_id) if event.campaign_id else None,
                "data": event.data,
                "timestamp": event.timestamp.isoformat()
            }
            
            # Send to connections subscribed to this event type
            for connection in self.active_connections.values():
                if event.event_type.value in connection.subscriptions:
                    try:
                        # Apply filters if any
                        if await self._should_send_event(connection, event):
                            await self._send_to_connection(connection.connection_id, message)
                    except Exception as e:
                        logger.warning(f"Error sending event to connection {connection.connection_id}: {e}")
                        
        except Exception as e:
            logger.error(f"Error distributing event: {e}")
    
    async def _should_send_event(self, connection: WebSocketConnection, event: RealTimeEvent) -> bool:
        """Check if event should be sent to connection based on filters"""
        try:
            # Apply basic filters
            if connection.data:
                filters = connection.data
                
                # Filter by sequence ID
                if "sequence_ids" in filters:
                    if str(event.sequence_id) not in filters["sequence_ids"]:
                        return False
                
                # Filter by user ID
                if "user_id" in filters and filters["user_id"] != event.user_id:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking event filters: {e}")
            return True
    
    async def _send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> None:
        """Send message to specific connection"""
        try:
            if connection_id in self.active_connections:
                connection = self.active_connections[connection_id]
                await connection.websocket.send_text(json.dumps(message))
                connection.last_ping = datetime.utcnow()
                
        except WebSocketDisconnect:
            await self.disconnect_websocket(connection_id)
        except Exception as e:
            logger.warning(f"Error sending to connection {connection_id}: {e}")
            await self.disconnect_websocket(connection_id)
    
    async def _cleanup_connections(self) -> None:
        """Clean up inactive connections"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds
                
                current_time = datetime.utcnow()
                inactive_connections = []
                
                for connection_id, connection in self.active_connections.items():
                    # Check if connection is inactive (no ping for 5 minutes)
                    if (current_time - connection.last_ping).total_seconds() > 300:
                        inactive_connections.append(connection_id)
                
                # Remove inactive connections
                for connection_id in inactive_connections:
                    await self.disconnect_websocket(connection_id)
                
                if inactive_connections:
                    logger.info(f"Cleaned up {len(inactive_connections)} inactive connections")
                
                self.last_cleanup = current_time
                
            except Exception as e:
                logger.error(f"Error during connection cleanup: {e}")
    
    async def _publish_analytics_updates(self) -> None:
        """Publish periodic analytics updates"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                # Get system statistics
                stats = await self.get_connection_stats()
                
                # Send analytics update
                await self.broadcast_to_connections({
                    "type": "analytics_update",
                    "data": {
                        "system_stats": stats,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }, connection_type=ConnectionType.ANALYTICS)
                
            except Exception as e:
                logger.error(f"Error publishing analytics updates: {e}")
    
    def register_event_handler(self, event_type: EventType, handler: Callable) -> None:
        """Register an event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)


# Global real-time engine instance
real_time_engine = RealTimeEngine()






























