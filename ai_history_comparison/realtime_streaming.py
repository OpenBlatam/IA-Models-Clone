"""
AI History Comparison System - Real-time Streaming

This module provides real-time streaming capabilities, WebSocket support,
and live analytics for the AI History Comparison system.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from collections import defaultdict, deque
import time

from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field
import redis
import aioredis
from aioredis import Redis

from .ai_history_analyzer import AIHistoryAnalyzer, ComparisonType, MetricType
from .advanced_ml_engine import ml_engine, AnomalyDetectionResult, AdvancedClusteringResult

logger = logging.getLogger(__name__)

class StreamEventType(Enum):
    """Types of streaming events"""
    CONTENT_ANALYZED = "content_analyzed"
    COMPARISON_COMPLETED = "comparison_completed"
    TREND_UPDATED = "trend_updated"
    ANOMALY_DETECTED = "anomaly_detected"
    CLUSTERING_UPDATED = "clustering_updated"
    QUALITY_ALERT = "quality_alert"
    SYSTEM_STATUS = "system_status"
    METRICS_UPDATE = "metrics_update"

class StreamSubscriptionType(Enum):
    """Types of stream subscriptions"""
    ALL_EVENTS = "all_events"
    CONTENT_ANALYSIS = "content_analysis"
    COMPARISONS = "comparisons"
    TRENDS = "trends"
    ANOMALIES = "anomalies"
    QUALITY_ALERTS = "quality_alerts"
    SYSTEM_METRICS = "system_metrics"

@dataclass
class StreamEvent:
    """Real-time stream event"""
    id: str
    event_type: StreamEventType
    timestamp: datetime
    data: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class StreamSubscription:
    """Stream subscription configuration"""
    id: str
    user_id: str
    subscription_types: List[StreamSubscriptionType]
    filters: Dict[str, Any]
    created_at: datetime
    last_activity: datetime

@dataclass
class RealtimeMetrics:
    """Real-time system metrics"""
    timestamp: datetime
    total_entries: int
    entries_per_minute: float
    avg_processing_time: float
    active_connections: int
    memory_usage: float
    cpu_usage: float
    error_rate: float

class WebSocketManager:
    """Manages WebSocket connections and real-time streaming"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Initialize WebSocket manager"""
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.subscription_configs: Dict[str, StreamSubscription] = {}
        self.redis_url = redis_url
        self.redis_client: Optional[Redis] = None
        self.event_queue: deque = deque(maxlen=10000)
        self.metrics_history: deque = deque(maxlen=1000)
        
        # Event handlers
        self.event_handlers: Dict[StreamEventType, List[Callable]] = defaultdict(list)
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        logger.info("WebSocket Manager initialized")

    async def connect(self, websocket: WebSocket, user_id: str, session_id: str = None):
        """Accept WebSocket connection"""
        await websocket.accept()
        
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        connection_id = f"{user_id}_{session_id}"
        self.active_connections[connection_id] = websocket
        self.user_subscriptions[user_id].add(connection_id)
        
        logger.info(f"WebSocket connected: {connection_id}")
        
        # Send welcome message
        welcome_event = StreamEvent(
            id=str(uuid.uuid4()),
            event_type=StreamEventType.SYSTEM_STATUS,
            timestamp=datetime.now(),
            data={
                "message": "Connected to AI History Comparison real-time stream",
                "connection_id": connection_id,
                "available_subscriptions": [st.value for st in StreamSubscriptionType]
            },
            user_id=user_id,
            session_id=session_id
        )
        
        await self.send_event(connection_id, welcome_event)

    async def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection"""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            
            # Remove from user subscriptions
            for user_id, connections in self.user_subscriptions.items():
                if connection_id in connections:
                    connections.remove(connection_id)
                    if not connections:
                        del self.user_subscriptions[user_id]
                    break
            
            logger.info(f"WebSocket disconnected: {connection_id}")

    async def send_event(self, connection_id: str, event: StreamEvent):
        """Send event to specific connection"""
        if connection_id in self.active_connections:
            try:
                websocket = self.active_connections[connection_id]
                await websocket.send_text(json.dumps(asdict(event), default=str))
            except Exception as e:
                logger.error(f"Failed to send event to {connection_id}: {e}")
                await self.disconnect(connection_id)

    async def broadcast_event(self, event: StreamEvent, subscription_type: StreamSubscriptionType = None):
        """Broadcast event to all relevant connections"""
        for connection_id, websocket in self.active_connections.items():
            # Check if connection should receive this event
            if self._should_send_event(connection_id, event, subscription_type):
                await self.send_event(connection_id, event)

    def _should_send_event(self, connection_id: str, event: StreamEvent, subscription_type: StreamSubscriptionType = None) -> bool:
        """Check if connection should receive the event"""
        # Extract user_id from connection_id
        user_id = connection_id.split('_')[0]
        
        # Check if user has subscription for this event type
        if user_id in self.subscription_configs:
            subscription = self.subscription_configs[user_id]
            
            # Check subscription types
            if StreamSubscriptionType.ALL_EVENTS in subscription.subscription_types:
                return True
            
            if subscription_type:
                if subscription_type in subscription.subscription_types:
                    return True
            
            # Check specific event type
            event_type_mapping = {
                StreamEventType.CONTENT_ANALYZED: StreamSubscriptionType.CONTENT_ANALYSIS,
                StreamEventType.COMPARISON_COMPLETED: StreamSubscriptionType.COMPARISONS,
                StreamEventType.TREND_UPDATED: StreamSubscriptionType.TRENDS,
                StreamEventType.ANOMALY_DETECTED: StreamSubscriptionType.ANOMALIES,
                StreamEventType.QUALITY_ALERT: StreamSubscriptionType.QUALITY_ALERTS,
                StreamEventType.SYSTEM_STATUS: StreamSubscriptionType.SYSTEM_METRICS,
                StreamEventType.METRICS_UPDATE: StreamSubscriptionType.SYSTEM_METRICS
            }
            
            mapped_type = event_type_mapping.get(event.event_type)
            if mapped_type and mapped_type in subscription.subscription_types:
                return True
        
        return False

    async def subscribe(self, user_id: str, subscription_types: List[StreamSubscriptionType], 
                       filters: Dict[str, Any] = None) -> str:
        """Create subscription for user"""
        subscription_id = str(uuid.uuid4())
        
        subscription = StreamSubscription(
            id=subscription_id,
            user_id=user_id,
            subscription_types=subscription_types,
            filters=filters or {},
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        self.subscription_configs[user_id] = subscription
        
        # Send subscription confirmation
        confirmation_event = StreamEvent(
            id=str(uuid.uuid4()),
            event_type=StreamEventType.SYSTEM_STATUS,
            timestamp=datetime.now(),
            data={
                "message": "Subscription created successfully",
                "subscription_id": subscription_id,
                "subscription_types": [st.value for st in subscription_types],
                "filters": filters
            },
            user_id=user_id
        )
        
        await self.broadcast_event(confirmation_event, StreamSubscriptionType.SYSTEM_METRICS)
        
        logger.info(f"Subscription created for user {user_id}: {subscription_types}")
        return subscription_id

    async def unsubscribe(self, user_id: str):
        """Remove subscription for user"""
        if user_id in self.subscription_configs:
            del self.subscription_configs[user_id]
            logger.info(f"Subscription removed for user {user_id}")

    async def start_background_tasks(self):
        """Start background tasks for real-time processing"""
        # Start metrics collection task
        metrics_task = asyncio.create_task(self._collect_metrics())
        self.background_tasks.add(metrics_task)
        metrics_task.add_done_callback(self.background_tasks.discard)
        
        # Start event processing task
        event_task = asyncio.create_task(self._process_events())
        self.background_tasks.add(event_task)
        event_task.add_done_callback(self.background_tasks.discard)
        
        logger.info("Background tasks started")

    async def stop_background_tasks(self):
        """Stop all background tasks"""
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        
        logger.info("Background tasks stopped")

    async def _collect_metrics(self):
        """Collect real-time system metrics"""
        while True:
            try:
                # Calculate metrics
                current_time = datetime.now()
                total_connections = len(self.active_connections)
                
                # Calculate entries per minute (simplified)
                entries_per_minute = len(self.event_queue) / 60.0 if self.event_queue else 0
                
                # Create metrics object
                metrics = RealtimeMetrics(
                    timestamp=current_time,
                    total_entries=0,  # Would be populated from analyzer
                    entries_per_minute=entries_per_minute,
                    avg_processing_time=0.1,  # Would be calculated from actual processing
                    active_connections=total_connections,
                    memory_usage=0.0,  # Would be actual memory usage
                    cpu_usage=0.0,     # Would be actual CPU usage
                    error_rate=0.0     # Would be calculated from errors
                )
                
                # Store metrics
                self.metrics_history.append(metrics)
                
                # Broadcast metrics update
                metrics_event = StreamEvent(
                    id=str(uuid.uuid4()),
                    event_type=StreamEventType.METRICS_UPDATE,
                    timestamp=current_time,
                    data=asdict(metrics)
                )
                
                await self.broadcast_event(metrics_event, StreamSubscriptionType.SYSTEM_METRICS)
                
                # Wait before next collection
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(30)

    async def _process_events(self):
        """Process queued events"""
        while True:
            try:
                if self.event_queue:
                    event = self.event_queue.popleft()
                    
                    # Process event handlers
                    handlers = self.event_handlers.get(event.event_type, [])
                    for handler in handlers:
                        try:
                            await handler(event)
                        except Exception as e:
                            logger.error(f"Error in event handler: {e}")
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Error processing events: {e}")
                await asyncio.sleep(1)

    def add_event_handler(self, event_type: StreamEventType, handler: Callable):
        """Add event handler for specific event type"""
        self.event_handlers[event_type].append(handler)

    def remove_event_handler(self, event_type: StreamEventType, handler: Callable):
        """Remove event handler"""
        if handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)

    async def emit_event(self, event_type: StreamEventType, data: Dict[str, Any], 
                        user_id: str = None, session_id: str = None):
        """Emit a new event"""
        event = StreamEvent(
            id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            user_id=user_id,
            session_id=session_id
        )
        
        # Add to queue for processing
        self.event_queue.append(event)
        
        # Broadcast immediately
        await self.broadcast_event(event)

    def get_active_connections(self) -> Dict[str, Any]:
        """Get information about active connections"""
        return {
            "total_connections": len(self.active_connections),
            "connections": list(self.active_connections.keys()),
            "user_subscriptions": dict(self.user_subscriptions),
            "subscription_configs": len(self.subscription_configs)
        }

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        if not self.metrics_history:
            return {}
        
        latest_metrics = self.metrics_history[-1]
        return asdict(latest_metrics)

class RealtimeAnalyzer:
    """Real-time content analyzer with streaming capabilities"""
    
    def __init__(self, analyzer: AIHistoryAnalyzer, websocket_manager: WebSocketManager):
        """Initialize real-time analyzer"""
        self.analyzer = analyzer
        self.websocket_manager = websocket_manager
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.anomaly_threshold = 0.8
        self.quality_thresholds = {
            "readability": {"min": 30, "max": 80},
            "sentiment": {"min": -0.5, "max": 0.5},
            "word_count": {"min": 50, "max": 1000}
        }
        
        # Register event handlers
        self.websocket_manager.add_event_handler(
            StreamEventType.CONTENT_ANALYZED, 
            self._handle_content_analyzed
        )
        
        logger.info("Realtime Analyzer initialized")

    async def analyze_content_realtime(self, content: str, model_version: str, 
                                     user_id: str = None, metadata: Dict[str, Any] = None) -> str:
        """Analyze content in real-time with streaming updates"""
        try:
            # Add to processing queue
            await self.processing_queue.put({
                "content": content,
                "model_version": model_version,
                "user_id": user_id,
                "metadata": metadata,
                "timestamp": datetime.now()
            })
            
            # Emit processing started event
            await self.websocket_manager.emit_event(
                StreamEventType.CONTENT_ANALYZED,
                {
                    "status": "processing_started",
                    "content_preview": content[:100] + "..." if len(content) > 100 else content,
                    "model_version": model_version
                },
                user_id
            )
            
            # Perform analysis
            entry_id = self.analyzer.add_history_entry(content, model_version, metadata)
            entry = self.analyzer._get_entry_by_id(entry_id)
            
            # Emit analysis completed event
            await self.websocket_manager.emit_event(
                StreamEventType.CONTENT_ANALYZED,
                {
                    "status": "completed",
                    "entry_id": entry_id,
                    "metrics": {
                        "readability_score": entry.metrics.readability_score,
                        "sentiment_score": entry.metrics.sentiment_score,
                        "word_count": entry.metrics.word_count,
                        "complexity_score": entry.metrics.complexity_score
                    },
                    "quality_assessment": self._assess_quality(entry.metrics)
                },
                user_id
            )
            
            # Check for anomalies
            await self._check_anomalies(entry, user_id)
            
            # Check quality alerts
            await self._check_quality_alerts(entry, user_id)
            
            return entry_id
            
        except Exception as e:
            logger.error(f"Error in real-time analysis: {e}")
            
            # Emit error event
            await self.websocket_manager.emit_event(
                StreamEventType.CONTENT_ANALYZED,
                {
                    "status": "error",
                    "error": str(e)
                },
                user_id
            )
            
            raise

    async def compare_content_realtime(self, entry_id_1: str, entry_id_2: str, 
                                     user_id: str = None) -> Dict[str, Any]:
        """Compare content in real-time with streaming updates"""
        try:
            # Emit comparison started event
            await self.websocket_manager.emit_event(
                StreamEventType.COMPARISON_COMPLETED,
                {
                    "status": "processing_started",
                    "entry_id_1": entry_id_1,
                    "entry_id_2": entry_id_2
                },
                user_id
            )
            
            # Perform comparison
            result = self.analyzer.compare_entries(
                entry_id_1, entry_id_2, [ComparisonType.CONTENT_SIMILARITY, ComparisonType.QUALITY_METRICS]
            )
            
            # Emit comparison completed event
            await self.websocket_manager.emit_event(
                StreamEventType.COMPARISON_COMPLETED,
                {
                    "status": "completed",
                    "entry_id_1": entry_id_1,
                    "entry_id_2": entry_id_2,
                    "similarity_score": result.similarity_score,
                    "trend_direction": result.trend_direction,
                    "significant_changes": result.significant_changes,
                    "recommendations": result.recommendations
                },
                user_id
            )
            
            return {
                "similarity_score": result.similarity_score,
                "trend_direction": result.trend_direction,
                "significant_changes": result.significant_changes,
                "recommendations": result.recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in real-time comparison: {e}")
            
            # Emit error event
            await self.websocket_manager.emit_event(
                StreamEventType.COMPARISON_COMPLETED,
                {
                    "status": "error",
                    "error": str(e)
                },
                user_id
            )
            
            raise

    async def update_trends_realtime(self, metric_type: MetricType, user_id: str = None):
        """Update trends in real-time"""
        try:
            # Perform trend analysis
            trend = self.analyzer.analyze_trends(metric_type)
            
            # Emit trend update event
            await self.websocket_manager.emit_event(
                StreamEventType.TREND_UPDATED,
                {
                    "metric_type": metric_type.value,
                    "trend_direction": trend.trend_direction,
                    "change_percentage": trend.change_percentage,
                    "significance_level": trend.significance_level,
                    "data_points": [
                        {"timestamp": point[0].isoformat(), "value": point[1]}
                        for point in trend.data_points
                    ],
                    "prediction": trend.prediction
                },
                user_id
            )
            
        except Exception as e:
            logger.error(f"Error updating trends: {e}")

    async def _handle_content_analyzed(self, event: StreamEvent):
        """Handle content analyzed event"""
        # Update trends if needed
        if event.data.get("status") == "completed":
            await self.update_trends_realtime(MetricType.READABILITY, event.user_id)
            await self.update_trends_realtime(MetricType.SENTIMENT, event.user_id)

    async def _check_anomalies(self, entry, user_id: str = None):
        """Check for anomalies in the analyzed content"""
        try:
            # Prepare entry data for anomaly detection
            entry_data = {
                "id": entry.id,
                "readability_score": entry.metrics.readability_score,
                "sentiment_score": entry.metrics.sentiment_score,
                "word_count": entry.metrics.word_count,
                "sentence_count": entry.metrics.sentence_count,
                "avg_word_length": entry.metrics.avg_word_length,
                "complexity_score": entry.metrics.complexity_score,
                "topic_diversity": entry.metrics.topic_diversity,
                "consistency_score": entry.metrics.consistency_score
            }
            
            # Detect anomalies
            anomalies = ml_engine.detect_anomalies([entry_data], method="isolation_forest")
            
            if anomalies and anomalies[0].is_anomaly:
                anomaly = anomalies[0]
                
                # Emit anomaly detected event
                await self.websocket_manager.emit_event(
                    StreamEventType.ANOMALY_DETECTED,
                    {
                        "entry_id": entry.id,
                        "anomaly_type": anomaly.anomaly_type,
                        "anomaly_score": anomaly.anomaly_score,
                        "confidence": anomaly.confidence,
                        "explanation": anomaly.explanation,
                        "recommendations": anomaly.recommendations
                    },
                    user_id
                )
                
        except Exception as e:
            logger.error(f"Error checking anomalies: {e}")

    async def _check_quality_alerts(self, entry, user_id: str = None):
        """Check for quality alerts"""
        try:
            alerts = []
            
            # Check readability
            if entry.metrics.readability_score < self.quality_thresholds["readability"]["min"]:
                alerts.append({
                    "type": "low_readability",
                    "message": f"Readability score {entry.metrics.readability_score:.1f} is below threshold",
                    "severity": "warning"
                })
            elif entry.metrics.readability_score > self.quality_thresholds["readability"]["max"]:
                alerts.append({
                    "type": "high_readability",
                    "message": f"Readability score {entry.metrics.readability_score:.1f} is above threshold",
                    "severity": "info"
                })
            
            # Check sentiment
            if abs(entry.metrics.sentiment_score) > self.quality_thresholds["sentiment"]["max"]:
                alerts.append({
                    "type": "extreme_sentiment",
                    "message": f"Sentiment score {entry.metrics.sentiment_score:.2f} is extreme",
                    "severity": "warning"
                })
            
            # Check word count
            if entry.metrics.word_count < self.quality_thresholds["word_count"]["min"]:
                alerts.append({
                    "type": "short_content",
                    "message": f"Word count {entry.metrics.word_count} is below minimum",
                    "severity": "warning"
                })
            elif entry.metrics.word_count > self.quality_thresholds["word_count"]["max"]:
                alerts.append({
                    "type": "long_content",
                    "message": f"Word count {entry.metrics.word_count} is above maximum",
                    "severity": "info"
                })
            
            # Emit quality alerts if any
            if alerts:
                await self.websocket_manager.emit_event(
                    StreamEventType.QUALITY_ALERT,
                    {
                        "entry_id": entry.id,
                        "alerts": alerts,
                        "timestamp": datetime.now().isoformat()
                    },
                    user_id
                )
                
        except Exception as e:
            logger.error(f"Error checking quality alerts: {e}")

    def _assess_quality(self, metrics) -> Dict[str, Any]:
        """Assess overall content quality"""
        quality_score = 0
        factors = []
        
        # Readability factor
        if 30 <= metrics.readability_score <= 80:
            quality_score += 25
            factors.append("good_readability")
        else:
            factors.append("poor_readability")
        
        # Sentiment factor
        if -0.5 <= metrics.sentiment_score <= 0.5:
            quality_score += 25
            factors.append("balanced_sentiment")
        else:
            factors.append("extreme_sentiment")
        
        # Length factor
        if 50 <= metrics.word_count <= 1000:
            quality_score += 25
            factors.append("appropriate_length")
        else:
            factors.append("inappropriate_length")
        
        # Complexity factor
        if 0.3 <= metrics.complexity_score <= 0.7:
            quality_score += 25
            factors.append("good_complexity")
        else:
            factors.append("poor_complexity")
        
        return {
            "quality_score": quality_score,
            "quality_grade": self._get_quality_grade(quality_score),
            "factors": factors
        }

    def _get_quality_grade(self, score: int) -> str:
        """Get quality grade based on score"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

# Global instances
websocket_manager = WebSocketManager()
realtime_analyzer = None  # Will be initialized with analyzer

# API Models
class StreamSubscriptionRequest(BaseModel):
    """Request model for stream subscription"""
    subscription_types: List[str] = Field(..., description="Types of events to subscribe to")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters")

class StreamEventResponse(BaseModel):
    """Response model for stream events"""
    id: str
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    user_id: Optional[str] = None

# WebSocket router
streaming_router = APIRouter(prefix="/stream", tags=["Real-time Streaming"])

@streaming_router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time streaming"""
    session_id = None
    
    try:
        await websocket_manager.connect(websocket, user_id, session_id)
        
        while True:
            # Keep connection alive and handle incoming messages
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "subscribe":
                    subscription_types = [
                        StreamSubscriptionType(st) for st in message.get("subscription_types", [])
                    ]
                    filters = message.get("filters", {})
                    await websocket_manager.subscribe(user_id, subscription_types, filters)
                
                elif message.get("type") == "unsubscribe":
                    await websocket_manager.unsubscribe(user_id)
                
                elif message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}))
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e)
                }))
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket_manager.disconnect(f"{user_id}_{session_id}")

@streaming_router.post("/subscribe")
async def subscribe_to_stream(request: StreamSubscriptionRequest, user_id: str = "anonymous"):
    """Subscribe to specific stream types"""
    try:
        subscription_types = [
            StreamSubscriptionType(st) for st in request.subscription_types
        ]
        
        subscription_id = await websocket_manager.subscribe(
            user_id, subscription_types, request.filters
        )
        
        return {
            "subscription_id": subscription_id,
            "subscription_types": request.subscription_types,
            "filters": request.filters,
            "status": "subscribed"
        }
        
    except Exception as e:
        logger.error(f"Error creating subscription: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@streaming_router.delete("/unsubscribe")
async def unsubscribe_from_stream(user_id: str = "anonymous"):
    """Unsubscribe from all streams"""
    try:
        await websocket_manager.unsubscribe(user_id)
        return {"status": "unsubscribed"}
        
    except Exception as e:
        logger.error(f"Error removing subscription: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@streaming_router.get("/status")
async def get_stream_status():
    """Get real-time streaming status"""
    try:
        connections = websocket_manager.get_active_connections()
        metrics = websocket_manager.get_metrics_summary()
        
        return {
            "status": "active",
            "connections": connections,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting stream status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Convenience functions
async def initialize_realtime_analyzer(analyzer: AIHistoryAnalyzer):
    """Initialize real-time analyzer with the main analyzer"""
    global realtime_analyzer
    realtime_analyzer = RealtimeAnalyzer(analyzer, websocket_manager)
    await websocket_manager.start_background_tasks()
    logger.info("Real-time analyzer initialized")

async def analyze_content_streaming(content: str, model_version: str, user_id: str = None, metadata: Dict[str, Any] = None) -> str:
    """Analyze content with real-time streaming updates"""
    if realtime_analyzer:
        return await realtime_analyzer.analyze_content_realtime(content, model_version, user_id, metadata)
    else:
        raise RuntimeError("Real-time analyzer not initialized")

async def compare_content_streaming(entry_id_1: str, entry_id_2: str, user_id: str = None) -> Dict[str, Any]:
    """Compare content with real-time streaming updates"""
    if realtime_analyzer:
        return await realtime_analyzer.compare_content_realtime(entry_id_1, entry_id_2, user_id)
    else:
        raise RuntimeError("Real-time analyzer not initialized")



























