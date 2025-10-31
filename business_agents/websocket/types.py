"""
WebSocket Types
===============

Type definitions for WebSocket communication.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

class MessageType(Enum):
    """WebSocket message types."""
    # Connection management
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    PING = "ping"
    PONG = "pong"
    
    # Agent operations
    AGENT_EXECUTION_START = "agent_execution_start"
    AGENT_EXECUTION_PROGRESS = "agent_execution_progress"
    AGENT_EXECUTION_COMPLETE = "agent_execution_complete"
    AGENT_EXECUTION_ERROR = "agent_execution_error"
    
    # Workflow operations
    WORKFLOW_EXECUTION_START = "workflow_execution_start"
    WORKFLOW_EXECUTION_PROGRESS = "workflow_execution_progress"
    WORKFLOW_EXECUTION_COMPLETE = "workflow_execution_complete"
    WORKFLOW_EXECUTION_ERROR = "workflow_execution_error"
    
    # Document operations
    DOCUMENT_GENERATION_START = "document_generation_start"
    DOCUMENT_GENERATION_PROGRESS = "document_generation_progress"
    DOCUMENT_GENERATION_COMPLETE = "document_generation_complete"
    DOCUMENT_GENERATION_ERROR = "document_generation_error"
    
    # System events
    SYSTEM_ALERT = "system_alert"
    SYSTEM_STATUS_UPDATE = "system_status_update"
    METRICS_UPDATE = "metrics_update"
    
    # Custom events
    CUSTOM_EVENT = "custom_event"

@dataclass
class WebSocketMessage:
    """WebSocket message structure."""
    type: MessageType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: Optional[str] = None
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps({
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "user_id": self.user_id,
            "session_id": self.session_id
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'WebSocketMessage':
        """Create message from JSON string."""
        data = json.loads(json_str)
        return cls(
            type=MessageType(data["type"]),
            data=data["data"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            message_id=data.get("message_id"),
            correlation_id=data.get("correlation_id"),
            user_id=data.get("user_id"),
            session_id=data.get("session_id")
        )

@dataclass
class ConnectionInfo:
    """WebSocket connection information."""
    connection_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    connected_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    subscriptions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_subscribed_to(self, topic: str) -> bool:
        """Check if connection is subscribed to a topic."""
        return topic in self.subscriptions
    
    def add_subscription(self, topic: str):
        """Add a subscription."""
        if topic not in self.subscriptions:
            self.subscriptions.append(topic)
    
    def remove_subscription(self, topic: str):
        """Remove a subscription."""
        if topic in self.subscriptions:
            self.subscriptions.remove(topic)

@dataclass
class BroadcastMessage:
    """Message for broadcasting to multiple connections."""
    message: WebSocketMessage
    target_connections: Optional[List[str]] = None
    target_users: Optional[List[str]] = None
    target_sessions: Optional[List[str]] = None
    exclude_connections: Optional[List[str]] = None
    topic: Optional[str] = None
    
    def should_send_to_connection(self, connection_info: ConnectionInfo) -> bool:
        """Check if message should be sent to a specific connection."""
        # Check exclusions first
        if self.exclude_connections and connection_info.connection_id in self.exclude_connections:
            return False
        
        # Check specific connection targets
        if self.target_connections and connection_info.connection_id not in self.target_connections:
            return False
        
        # Check user targets
        if self.target_users and connection_info.user_id not in self.target_users:
            return False
        
        # Check session targets
        if self.target_sessions and connection_info.session_id not in self.target_sessions:
            return False
        
        # Check topic subscription
        if self.topic and not connection_info.is_subscribed_to(self.topic):
            return False
        
        return True

class WebSocketError(Exception):
    """WebSocket-specific error."""
    
    def __init__(self, message: str, error_code: str = "websocket_error", details: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}

class ConnectionError(WebSocketError):
    """Connection-related error."""
    pass

class MessageError(WebSocketError):
    """Message-related error."""
    pass

class SubscriptionError(WebSocketError):
    """Subscription-related error."""
    pass
