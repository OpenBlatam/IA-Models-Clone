"""
Advanced Analytics Engine for Professional Documents System
==========================================================

This module provides comprehensive analytics capabilities including:
- Real-time performance analytics
- Predictive modeling and forecasting
- User behavior analysis
- Content quality metrics
- Business intelligence dashboards
- Machine learning insights
- Anomaly detection
- Trend analysis
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import redis
import psycopg2
from sqlalchemy import create_engine, text
import aiohttp
import websockets
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque
import hashlib
import pickle
import gzip
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalyticsEventType(Enum):
    """Types of analytics events"""
    DOCUMENT_CREATED = "document_created"
    DOCUMENT_UPDATED = "document_updated"
    DOCUMENT_DELETED = "document_deleted"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    EXPORT_REQUESTED = "export_requested"
    COLLABORATION_STARTED = "collaboration_started"
    WORKFLOW_TRIGGERED = "workflow_triggered"
    AI_GENERATION = "ai_generation"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR_OCCURRED = "error_occurred"
    SECURITY_EVENT = "security_event"

@dataclass
class AnalyticsEvent:
    """Analytics event data structure"""
    event_id: str
    event_type: AnalyticsEventType
    user_id: str
    session_id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    source: str
    version: str

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    metric_id: str
    metric_name: str
    value: float
    unit: str
    timestamp: datetime
    context: Dict[str, Any]
    tags: List[str]

@dataclass
class UserBehaviorProfile:
    """User behavior profile"""
    user_id: str
    session_duration: float
    documents_created: int
    documents_updated: int
    exports_requested: int
    collaboration_time: float
    ai_usage_frequency: float
    preferred_templates: List[str]
    peak_usage_hours: List[int]
    device_types: List[str]
    locations: List[str]
    last_updated: datetime

@dataclass
class ContentQualityMetrics:
    """Content quality metrics"""
    document_id: str
    readability_score: float
    sentiment_score: float
    topic_coherence: float
    keyword_density: float
    structure_quality: float
    ai_confidence: float
    user_satisfaction: float
    revision_count: int
    collaboration_score: float
    timestamp: datetime

class AdvancedAnalyticsEngine:
    """Advanced analytics engine with ML capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            db=config.get('redis_db', 0),
            decode_responses=True
        )
        
        # Database connections
        self.db_engine = create_engine(config['database_url'])
        
        # ML models
        self.performance_predictor = None
        self.anomaly_detector = None
        self.user_clustering_model = None
        self.content_quality_predictor = None
        
        # Data storage
        self.event_buffer = deque(maxlen=10000)
        self.metrics_buffer = deque(maxlen=5000)
        self.user_profiles = {}
        self.content_metrics = {}
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.analytics_thread = None
        self.is_running = False
        
        # WebSocket connections for real-time updates
        self.websocket_connections = set()
        
        # Initialize models
        self._initialize_ml_models()
        
    def _initialize_ml_models(self):
        """Initialize machine learning models"""
        try:
            # Performance prediction model
            self.performance_predictor = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # Anomaly detection model
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            # User clustering model
            self.user_clustering_model = KMeans(
                n_clusters=5,
                random_state=42
            )
            
            # Content quality predictor
            self.content_quality_predictor = RandomForestRegressor(
                n_estimators=50,
                random_state=42
            )
            
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
    
    async def start_analytics_engine(self):
        """Start the analytics engine"""
        if self.is_running:
            return
            
        self.is_running = True
        self.analytics_thread = threading.Thread(
            target=self._analytics_worker,
            daemon=True
        )
        self.analytics_thread.start()
        
        # Start real-time data collection
        asyncio.create_task(self._collect_real_time_metrics())
        asyncio.create_task(self._process_events())
        asyncio.create_task(self._update_ml_models())
        
        logger.info("Analytics engine started")
    
    async def stop_analytics_engine(self):
        """Stop the analytics engine"""
        self.is_running = False
        if self.analytics_thread:
            self.analytics_thread.join(timeout=5)
        logger.info("Analytics engine stopped")
    
    def _analytics_worker(self):
        """Background worker for analytics processing"""
        while self.is_running:
            try:
                # Process buffered events
                self._process_buffered_events()
                
                # Update user profiles
                self._update_user_profiles()
                
                # Calculate metrics
                self._calculate_metrics()
                
                # Detect anomalies
                self._detect_anomalies()
                
                time.sleep(1)  # Process every second
                
            except Exception as e:
                logger.error(f"Error in analytics worker: {e}")
                time.sleep(5)
    
    async def track_event(self, event: AnalyticsEvent):
        """Track an analytics event"""
        try:
            # Add to buffer
            self.event_buffer.append(event)
            
            # Store in Redis for real-time access
            event_key = f"analytics:event:{event.event_id}"
            event_data = {
                'event_type': event.event_type.value,
                'user_id': event.user_id,
                'timestamp': event.timestamp.isoformat(),
                'data': json.dumps(event.data),
                'metadata': json.dumps(event.metadata),
                'source': event.source,
                'version': event.version
            }
            
            self.redis_client.hset(event_key, mapping=event_data)
            self.redis_client.expire(event_key, 86400)  # 24 hours
            
            # Store in database
            await self._store_event_in_db(event)
            
            # Update real-time dashboards
            await self._update_real_time_dashboards(event)
            
        except Exception as e:
            logger.error(f"Error tracking event: {e}")
    
    async def track_performance_metric(self, metric: PerformanceMetrics):
        """Track a performance metric"""
        try:
            # Add to buffer
            self.metrics_buffer.append(metric)
            
            # Store in Redis
            metric_key = f"analytics:metric:{metric.metric_name}:{int(metric.timestamp.timestamp())}"
            metric_data = {
                'value': str(metric.value),
                'unit': metric.unit,
                'timestamp': metric.timestamp.isoformat(),
                'context': json.dumps(metric.context),
                'tags': json.dumps(metric.tags)
            }
            
            self.redis_client.hset(metric_key, mapping=metric_data)
            self.redis_client.expire(metric_key, 86400)
            
            # Store in database
            await self._store_metric_in_db(metric)
            
        except Exception as e:
            logger.error(f"Error tracking performance metric: {e}")
    
    def _process_buffered_events(self):
        """Process buffered events"""
        if not self.event_buffer:
            return
            
        events = list(self.event_buffer)
        self.event_buffer.clear()
        
        for event in events:
            try:
                # Update user behavior
                self._update_user_behavior(event)
                
                # Calculate event metrics
                self._calculate_event_metrics(event)
                
                # Check for patterns
                self._detect_patterns(event)
                
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    def _update_user_behavior(self, event: AnalyticsEvent):
        """Update user behavior profile"""
        user_id = event.user_id
        
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserBehaviorProfile(
                user_id=user_id,
                session_duration=0,
                documents_created=0,
                documents_updated=0,
                exports_requested=0,
                collaboration_time=0,
                ai_usage_frequency=0,
                preferred_templates=[],
                peak_usage_hours=[],
                device_types=[],
                locations=[],
                last_updated=datetime.now()
            )
        
        profile = self.user_profiles[user_id]
        
        # Update based on event type
        if event.event_type == AnalyticsEventType.DOCUMENT_CREATED:
            profile.documents_created += 1
        elif event.event_type == AnalyticsEventType.DOCUMENT_UPDATED:
            profile.documents_updated += 1
        elif event.event_type == AnalyticsEventType.EXPORT_REQUESTED:
            profile.exports_requested += 1
        elif event.event_type == AnalyticsEventType.COLLABORATION_STARTED:
            profile.collaboration_time += event.data.get('duration', 0)
        elif event.event_type == AnalyticsEventType.AI_GENERATION:
            profile.ai_usage_frequency += 1
        
        # Update metadata
        if 'template_id' in event.data:
            if event.data['template_id'] not in profile.preferred_templates:
                profile.preferred_templates.append(event.data['template_id'])
        
        if 'device_type' in event.metadata:
            if event.metadata['device_type'] not in profile.device_types:
                profile.device_types.append(event.metadata['device_type'])
        
        if 'location' in event.metadata:
            if event.metadata['location'] not in profile.locations:
                profile.locations.append(event.metadata['location'])
        
        profile.last_updated = datetime.now()
    
    def _calculate_event_metrics(self, event: AnalyticsEvent):
        """Calculate metrics from events"""
        # Calculate event frequency
        event_type = event.event_type.value
        frequency_key = f"analytics:frequency:{event_type}"
        self.redis_client.incr(frequency_key)
        self.redis_client.expire(frequency_key, 86400)
        
        # Calculate user activity
        user_activity_key = f"analytics:user_activity:{event.user_id}"
        self.redis_client.incr(user_activity_key)
        self.redis_client.expire(user_activity_key, 86400)
        
        # Calculate session metrics
        session_key = f"analytics:session:{event.session_id}"
        session_data = self.redis_client.hgetall(session_key)
        
        if not session_data:
            session_data = {
                'start_time': event.timestamp.isoformat(),
                'event_count': '0',
                'user_id': event.user_id
            }
        
        session_data['event_count'] = str(int(session_data.get('event_count', 0)) + 1)
        session_data['last_activity'] = event.timestamp.isoformat()
        
        self.redis_client.hset(session_key, mapping=session_data)
        self.redis_client.expire(session_key, 86400)
    
    def _detect_patterns(self, event: AnalyticsEvent):
        """Detect patterns in events"""
        # Detect usage patterns
        hour = event.timestamp.hour
        pattern_key = f"analytics:pattern:hourly:{hour}"
        self.redis_client.incr(pattern_key)
        self.redis_client.expire(pattern_key, 86400)
        
        # Detect feature usage patterns
        if 'feature' in event.data:
            feature = event.data['feature']
            feature_key = f"analytics:pattern:feature:{feature}"
            self.redis_client.incr(feature_key)
            self.redis_client.expire(feature_key, 86400)
    
    async def _collect_real_time_metrics(self):
        """Collect real-time system metrics"""
        while self.is_running:
            try:
                # System metrics
                cpu_usage = self._get_cpu_usage()
                memory_usage = self._get_memory_usage()
                disk_usage = self._get_disk_usage()
                network_usage = self._get_network_usage()
                
                # Application metrics
                active_users = self._get_active_users()
                active_documents = self._get_active_documents()
                queue_size = self._get_queue_size()
                response_time = self._get_avg_response_time()
                
                # Create metrics
                metrics = [
                    PerformanceMetrics(
                        metric_id=f"cpu_{int(time.time())}",
                        metric_name="cpu_usage",
                        value=cpu_usage,
                        unit="percent",
                        timestamp=datetime.now(),
                        context={"type": "system"},
                        tags=["system", "performance"]
                    ),
                    PerformanceMetrics(
                        metric_id=f"memory_{int(time.time())}",
                        metric_name="memory_usage",
                        value=memory_usage,
                        unit="percent",
                        timestamp=datetime.now(),
                        context={"type": "system"},
                        tags=["system", "performance"]
                    ),
                    PerformanceMetrics(
                        metric_id=f"active_users_{int(time.time())}",
                        metric_name="active_users",
                        value=active_users,
                        unit="count",
                        timestamp=datetime.now(),
                        context={"type": "application"},
                        tags=["application", "users"]
                    ),
                    PerformanceMetrics(
                        metric_id=f"response_time_{int(time.time())}",
                        metric_name="response_time",
                        value=response_time,
                        unit="milliseconds",
                        timestamp=datetime.now(),
                        context={"type": "application"},
                        tags=["application", "performance"]
                    )
                ]
                
                # Track metrics
                for metric in metrics:
                    await self.track_performance_metric(metric)
                
                await asyncio.sleep(5)  # Collect every 5 seconds
                
            except Exception as e:
                logger.error(f"Error collecting real-time metrics: {e}")
                await asyncio.sleep(10)
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get memory usage percentage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent
        except ImportError:
            return 0.0
    
    def _get_disk_usage(self) -> float:
        """Get disk usage percentage"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            return (disk.used / disk.total) * 100
        except ImportError:
            return 0.0
    
    def _get_network_usage(self) -> Dict[str, float]:
        """Get network usage statistics"""
        try:
            import psutil
            network = psutil.net_io_counters()
            return {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
        except ImportError:
            return {}
    
    def _get_active_users(self) -> int:
        """Get number of active users"""
        try:
            # Count active sessions in Redis
            pattern = "analytics:session:*"
            keys = self.redis_client.keys(pattern)
            return len(keys)
        except Exception:
            return 0
    
    def _get_active_documents(self) -> int:
        """Get number of active documents"""
        try:
            # Count active documents in Redis
            pattern = "analytics:document:*"
            keys = self.redis_client.keys(pattern)
            return len(keys)
        except Exception:
            return 0
    
    def _get_queue_size(self) -> int:
        """Get queue size"""
        try:
            return self.redis_client.llen('analytics:queue')
        except Exception:
            return 0
    
    def _get_avg_response_time(self) -> float:
        """Get average response time"""
        try:
            # Get response times from Redis
            response_times = self.redis_client.lrange('analytics:response_times', 0, -1)
            if response_times:
                return sum(float(rt) for rt in response_times) / len(response_times)
            return 0.0
        except Exception:
            return 0.0
    
    async def _process_events(self):
        """Process events from queue"""
        while self.is_running:
            try:
                # Get events from queue
                event_data = self.redis_client.lpop('analytics:queue')
                if event_data:
                    event = json.loads(event_data)
                    await self._process_single_event(event)
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing events: {e}")
                await asyncio.sleep(1)
    
    async def _process_single_event(self, event_data: Dict[str, Any]):
        """Process a single event"""
        try:
            # Create event object
            event = AnalyticsEvent(
                event_id=event_data['event_id'],
                event_type=AnalyticsEventType(event_data['event_type']),
                user_id=event_data['user_id'],
                session_id=event_data['session_id'],
                timestamp=datetime.fromisoformat(event_data['timestamp']),
                data=event_data['data'],
                metadata=event_data['metadata'],
                source=event_data['source'],
                version=event_data['version']
            )
            
            # Track event
            await self.track_event(event)
            
        except Exception as e:
            logger.error(f"Error processing single event: {e}")
    
    async def _update_ml_models(self):
        """Update ML models with new data"""
        while self.is_running:
            try:
                # Get training data
                training_data = await self._get_training_data()
                
                if training_data:
                    # Update models
                    await self._update_performance_predictor(training_data)
                    await self._update_anomaly_detector(training_data)
                    await self._update_user_clustering_model(training_data)
                    await self._update_content_quality_predictor(training_data)
                
                await asyncio.sleep(3600)  # Update every hour
                
            except Exception as e:
                logger.error(f"Error updating ML models: {e}")
                await asyncio.sleep(3600)
    
    async def _get_training_data(self) -> Optional[pd.DataFrame]:
        """Get training data for ML models"""
        try:
            # Get recent events and metrics
            events = await self._get_recent_events(days=7)
            metrics = await self._get_recent_metrics(days=7)
            
            if not events or not metrics:
                return None
            
            # Combine data
            df = pd.DataFrame(events)
            metrics_df = pd.DataFrame(metrics)
            
            # Merge data
            df = df.merge(metrics_df, on='timestamp', how='left')
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return None
    
    async def _get_recent_events(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent events from database"""
        try:
            query = """
                SELECT * FROM analytics_events 
                WHERE timestamp >= NOW() - INTERVAL '%s days'
                ORDER BY timestamp DESC
                LIMIT 10000
            """
            
            with self.db_engine.connect() as conn:
                result = conn.execute(text(query), (days,))
                return [dict(row) for row in result]
                
        except Exception as e:
            logger.error(f"Error getting recent events: {e}")
            return []
    
    async def _get_recent_metrics(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent metrics from database"""
        try:
            query = """
                SELECT * FROM performance_metrics 
                WHERE timestamp >= NOW() - INTERVAL '%s days'
                ORDER BY timestamp DESC
                LIMIT 10000
            """
            
            with self.db_engine.connect() as conn:
                result = conn.execute(text(query), (days,))
                return [dict(row) for row in result]
                
        except Exception as e:
            logger.error(f"Error getting recent metrics: {e}")
            return []
    
    async def _update_performance_predictor(self, training_data: pd.DataFrame):
        """Update performance prediction model"""
        try:
            if len(training_data) < 100:
                return
            
            # Prepare features
            features = ['cpu_usage', 'memory_usage', 'active_users', 'queue_size']
            X = training_data[features].fillna(0)
            y = training_data['response_time'].fillna(0)
            
            # Train model
            self.performance_predictor.fit(X, y)
            
            logger.info("Performance predictor updated")
            
        except Exception as e:
            logger.error(f"Error updating performance predictor: {e}")
    
    async def _update_anomaly_detector(self, training_data: pd.DataFrame):
        """Update anomaly detection model"""
        try:
            if len(training_data) < 100:
                return
            
            # Prepare features
            features = ['cpu_usage', 'memory_usage', 'response_time', 'error_rate']
            X = training_data[features].fillna(0)
            
            # Train model
            self.anomaly_detector.fit(X)
            
            logger.info("Anomaly detector updated")
            
        except Exception as e:
            logger.error(f"Error updating anomaly detector: {e}")
    
    async def _update_user_clustering_model(self, training_data: pd.DataFrame):
        """Update user clustering model"""
        try:
            if len(training_data) < 100:
                return
            
            # Prepare features
            features = ['documents_created', 'documents_updated', 'exports_requested', 'collaboration_time']
            X = training_data[features].fillna(0)
            
            # Train model
            self.user_clustering_model.fit(X)
            
            logger.info("User clustering model updated")
            
        except Exception as e:
            logger.error(f"Error updating user clustering model: {e}")
    
    async def _update_content_quality_predictor(self, training_data: pd.DataFrame):
        """Update content quality prediction model"""
        try:
            if len(training_data) < 100:
                return
            
            # Prepare features
            features = ['readability_score', 'sentiment_score', 'topic_coherence', 'keyword_density']
            X = training_data[features].fillna(0)
            y = training_data['user_satisfaction'].fillna(0)
            
            # Train model
            self.content_quality_predictor.fit(X, y)
            
            logger.info("Content quality predictor updated")
            
        except Exception as e:
            logger.error(f"Error updating content quality predictor: {e}")
    
    def _detect_anomalies(self):
        """Detect anomalies in current data"""
        try:
            if not self.anomaly_detector:
                return
            
            # Get current metrics
            current_metrics = self._get_current_metrics()
            if not current_metrics:
                return
            
            # Predict anomalies
            anomalies = self.anomaly_detector.predict([current_metrics])
            
            if anomalies[0] == -1:  # Anomaly detected
                self._handle_anomaly(current_metrics)
                
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
    
    def _get_current_metrics(self) -> List[float]:
        """Get current system metrics"""
        try:
            return [
                self._get_cpu_usage(),
                self._get_memory_usage(),
                self._get_avg_response_time(),
                0.0  # error_rate placeholder
            ]
        except Exception:
            return []
    
    def _handle_anomaly(self, metrics: List[float]):
        """Handle detected anomaly"""
        try:
            anomaly_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'severity': 'high' if max(metrics) > 80 else 'medium',
                'type': 'performance_anomaly'
            }
            
            # Store anomaly
            anomaly_key = f"analytics:anomaly:{int(time.time())}"
            self.redis_client.hset(anomaly_key, mapping=anomaly_data)
            self.redis_client.expire(anomaly_key, 86400)
            
            # Send alert
            self._send_anomaly_alert(anomaly_data)
            
            logger.warning(f"Anomaly detected: {anomaly_data}")
            
        except Exception as e:
            logger.error(f"Error handling anomaly: {e}")
    
    def _send_anomaly_alert(self, anomaly_data: Dict[str, Any]):
        """Send anomaly alert"""
        try:
            # Store alert in Redis
            alert_key = f"analytics:alert:{int(time.time())}"
            self.redis_client.hset(alert_key, mapping=anomaly_data)
            self.redis_client.expire(alert_key, 86400)
            
            # Notify WebSocket connections
            asyncio.create_task(self._notify_websocket_clients(anomaly_data))
            
        except Exception as e:
            logger.error(f"Error sending anomaly alert: {e}")
    
    async def _notify_websocket_clients(self, data: Dict[str, Any]):
        """Notify WebSocket clients of updates"""
        try:
            message = json.dumps({
                'type': 'analytics_update',
                'data': data,
                'timestamp': datetime.now().isoformat()
            })
            
            # Send to all connected clients
            disconnected = set()
            for websocket in self.websocket_connections:
                try:
                    await websocket.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(websocket)
            
            # Remove disconnected clients
            self.websocket_connections -= disconnected
            
        except Exception as e:
            logger.error(f"Error notifying WebSocket clients: {e}")
    
    async def _update_real_time_dashboards(self, event: AnalyticsEvent):
        """Update real-time dashboards"""
        try:
            # Update dashboard data in Redis
            dashboard_key = f"analytics:dashboard:{event.event_type.value}"
            self.redis_client.incr(dashboard_key)
            self.redis_client.expire(dashboard_key, 86400)
            
            # Notify WebSocket clients
            await self._notify_websocket_clients({
                'event_type': event.event_type.value,
                'user_id': event.user_id,
                'timestamp': event.timestamp.isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error updating real-time dashboards: {e}")
    
    async def _store_event_in_db(self, event: AnalyticsEvent):
        """Store event in database"""
        try:
            query = """
                INSERT INTO analytics_events 
                (event_id, event_type, user_id, session_id, timestamp, data, metadata, source, version)
                VALUES (:event_id, :event_type, :user_id, :session_id, :timestamp, :data, :metadata, :source, :version)
            """
            
            with self.db_engine.connect() as conn:
                conn.execute(text(query), {
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'user_id': event.user_id,
                    'session_id': event.session_id,
                    'timestamp': event.timestamp,
                    'data': json.dumps(event.data),
                    'metadata': json.dumps(event.metadata),
                    'source': event.source,
                    'version': event.version
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing event in database: {e}")
    
    async def _store_metric_in_db(self, metric: PerformanceMetrics):
        """Store metric in database"""
        try:
            query = """
                INSERT INTO performance_metrics 
                (metric_id, metric_name, value, unit, timestamp, context, tags)
                VALUES (:metric_id, :metric_name, :value, :unit, :timestamp, :context, :tags)
            """
            
            with self.db_engine.connect() as conn:
                conn.execute(text(query), {
                    'metric_id': metric.metric_id,
                    'metric_name': metric.metric_name,
                    'value': metric.value,
                    'unit': metric.unit,
                    'timestamp': metric.timestamp,
                    'context': json.dumps(metric.context),
                    'tags': json.dumps(metric.tags)
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing metric in database: {e}")
    
    async def get_analytics_dashboard(self, time_range: str = "24h") -> Dict[str, Any]:
        """Get analytics dashboard data"""
        try:
            # Get time range
            if time_range == "1h":
                hours = 1
            elif time_range == "24h":
                hours = 24
            elif time_range == "7d":
                hours = 168
            elif time_range == "30d":
                hours = 720
            else:
                hours = 24
            
            # Get metrics
            metrics = await self._get_dashboard_metrics(hours)
            
            # Get user activity
            user_activity = await self._get_user_activity(hours)
            
            # Get content metrics
            content_metrics = await self._get_content_metrics(hours)
            
            # Get performance metrics
            performance_metrics = await self._get_performance_metrics(hours)
            
            # Get trends
            trends = await self._get_trends(hours)
            
            return {
                'time_range': time_range,
                'metrics': metrics,
                'user_activity': user_activity,
                'content_metrics': content_metrics,
                'performance_metrics': performance_metrics,
                'trends': trends,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics dashboard: {e}")
            return {}
    
    async def _get_dashboard_metrics(self, hours: int) -> Dict[str, Any]:
        """Get dashboard metrics"""
        try:
            # Get event counts
            event_counts = {}
            for event_type in AnalyticsEventType:
                key = f"analytics:frequency:{event_type.value}"
                count = self.redis_client.get(key) or 0
                event_counts[event_type.value] = int(count)
            
            # Get active users
            active_users = self._get_active_users()
            
            # Get active documents
            active_documents = self._get_active_documents()
            
            # Get system health
            system_health = {
                'cpu_usage': self._get_cpu_usage(),
                'memory_usage': self._get_memory_usage(),
                'disk_usage': self._get_disk_usage(),
                'response_time': self._get_avg_response_time()
            }
            
            return {
                'event_counts': event_counts,
                'active_users': active_users,
                'active_documents': active_documents,
                'system_health': system_health
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard metrics: {e}")
            return {}
    
    async def _get_user_activity(self, hours: int) -> Dict[str, Any]:
        """Get user activity data"""
        try:
            # Get user sessions
            session_keys = self.redis_client.keys("analytics:session:*")
            sessions = []
            
            for key in session_keys:
                session_data = self.redis_client.hgetall(key)
                if session_data:
                    sessions.append(session_data)
            
            # Get user profiles
            user_profiles = list(self.user_profiles.values())
            
            # Calculate activity metrics
            total_sessions = len(sessions)
            active_users = len(set(s['user_id'] for s in sessions if 'user_id' in s))
            avg_session_duration = sum(float(s.get('event_count', 0)) for s in sessions) / max(total_sessions, 1)
            
            return {
                'total_sessions': total_sessions,
                'active_users': active_users,
                'avg_session_duration': avg_session_duration,
                'user_profiles': [asdict(profile) for profile in user_profiles]
            }
            
        except Exception as e:
            logger.error(f"Error getting user activity: {e}")
            return {}
    
    async def _get_content_metrics(self, hours: int) -> Dict[str, Any]:
        """Get content metrics"""
        try:
            # Get content quality metrics
            content_metrics = list(self.content_metrics.values())
            
            if not content_metrics:
                return {}
            
            # Calculate averages
            avg_readability = sum(m.readability_score for m in content_metrics) / len(content_metrics)
            avg_sentiment = sum(m.sentiment_score for m in content_metrics) / len(content_metrics)
            avg_quality = sum(m.structure_quality for m in content_metrics) / len(content_metrics)
            avg_satisfaction = sum(m.user_satisfaction for m in content_metrics) / len(content_metrics)
            
            return {
                'avg_readability': avg_readability,
                'avg_sentiment': avg_sentiment,
                'avg_quality': avg_quality,
                'avg_satisfaction': avg_satisfaction,
                'total_documents': len(content_metrics)
            }
            
        except Exception as e:
            logger.error(f"Error getting content metrics: {e}")
            return {}
    
    async def _get_performance_metrics(self, hours: int) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            # Get recent metrics from buffer
            recent_metrics = list(self.metrics_buffer)[-100:]  # Last 100 metrics
            
            if not recent_metrics:
                return {}
            
            # Group by metric name
            metrics_by_name = defaultdict(list)
            for metric in recent_metrics:
                metrics_by_name[metric.metric_name].append(metric.value)
            
            # Calculate statistics
            performance_stats = {}
            for name, values in metrics_by_name.items():
                if values:
                    performance_stats[name] = {
                        'current': values[-1],
                        'average': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'trend': self._calculate_trend(values)
                    }
            
            return performance_stats
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend from values"""
        if len(values) < 2:
            return "stable"
        
        # Simple trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg * 1.1:
            return "increasing"
        elif second_avg < first_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    async def _get_trends(self, hours: int) -> Dict[str, Any]:
        """Get trend data"""
        try:
            # Get hourly patterns
            hourly_patterns = {}
            for hour in range(24):
                key = f"analytics:pattern:hourly:{hour}"
                count = self.redis_client.get(key) or 0
                hourly_patterns[hour] = int(count)
            
            # Get feature usage patterns
            feature_patterns = {}
            feature_keys = self.redis_client.keys("analytics:pattern:feature:*")
            for key in feature_keys:
                feature = key.split(":")[-1]
                count = self.redis_client.get(key) or 0
                feature_patterns[feature] = int(count)
            
            return {
                'hourly_patterns': hourly_patterns,
                'feature_patterns': feature_patterns
            }
            
        except Exception as e:
            logger.error(f"Error getting trends: {e}")
            return {}
    
    async def predict_performance(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """Predict performance based on input data"""
        try:
            if not self.performance_predictor:
                return {'error': 'Performance predictor not available'}
            
            # Prepare input
            features = ['cpu_usage', 'memory_usage', 'active_users', 'queue_size']
            X = np.array([[input_data.get(f, 0) for f in features]])
            
            # Make prediction
            prediction = self.performance_predictor.predict(X)[0]
            
            # Get confidence interval
            predictions = self.performance_predictor.predict(X)
            confidence = np.std(predictions)
            
            return {
                'predicted_response_time': prediction,
                'confidence': confidence,
                'input_data': input_data,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting performance: {e}")
            return {'error': str(e)}
    
    async def detect_anomalies(self, data: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Detect anomalies in data"""
        try:
            if not self.anomaly_detector:
                return []
            
            # Prepare data
            features = ['cpu_usage', 'memory_usage', 'response_time', 'error_rate']
            X = np.array([[d.get(f, 0) for f in features] for d in data])
            
            # Detect anomalies
            anomalies = self.anomaly_detector.predict(X)
            scores = self.anomaly_detector.decision_function(X)
            
            # Format results
            results = []
            for i, (anomaly, score) in enumerate(zip(anomalies, scores)):
                if anomaly == -1:  # Anomaly detected
                    results.append({
                        'index': i,
                        'data': data[i],
                        'anomaly_score': score,
                        'severity': 'high' if score < -0.5 else 'medium',
                        'timestamp': datetime.now().isoformat()
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    async def cluster_users(self, user_data: List[Dict[str, float]]) -> Dict[str, Any]:
        """Cluster users based on behavior"""
        try:
            if not self.user_clustering_model:
                return {'error': 'User clustering model not available'}
            
            # Prepare data
            features = ['documents_created', 'documents_updated', 'exports_requested', 'collaboration_time']
            X = np.array([[d.get(f, 0) for f in features] for d in user_data])
            
            # Cluster users
            clusters = self.user_clustering_model.predict(X)
            
            # Calculate cluster centers
            cluster_centers = self.user_clustering_model.cluster_centers_
            
            # Calculate silhouette score
            silhouette = silhouette_score(X, clusters)
            
            return {
                'clusters': clusters.tolist(),
                'cluster_centers': cluster_centers.tolist(),
                'silhouette_score': silhouette,
                'n_clusters': len(cluster_centers),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error clustering users: {e}")
            return {'error': str(e)}
    
    async def predict_content_quality(self, content_data: Dict[str, float]) -> Dict[str, Any]:
        """Predict content quality"""
        try:
            if not self.content_quality_predictor:
                return {'error': 'Content quality predictor not available'}
            
            # Prepare input
            features = ['readability_score', 'sentiment_score', 'topic_coherence', 'keyword_density']
            X = np.array([[content_data.get(f, 0) for f in features]])
            
            # Make prediction
            prediction = self.content_quality_predictor.predict(X)[0]
            
            # Get feature importance
            importance = self.content_quality_predictor.feature_importances_
            feature_importance = dict(zip(features, importance))
            
            return {
                'predicted_quality': prediction,
                'feature_importance': feature_importance,
                'input_data': content_data,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error predicting content quality: {e}")
            return {'error': str(e)}
    
    async def generate_insights(self, time_range: str = "24h") -> Dict[str, Any]:
        """Generate AI-powered insights"""
        try:
            # Get data
            dashboard_data = await self.get_analytics_dashboard(time_range)
            
            # Analyze patterns
            patterns = self._analyze_patterns(dashboard_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(dashboard_data)
            
            # Calculate KPIs
            kpis = self._calculate_kpis(dashboard_data)
            
            # Generate summary
            summary = self._generate_summary(dashboard_data, patterns, recommendations)
            
            return {
                'time_range': time_range,
                'patterns': patterns,
                'recommendations': recommendations,
                'kpis': kpis,
                'summary': summary,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {'error': str(e)}
    
    def _analyze_patterns(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze patterns in data"""
        patterns = []
        
        try:
            # Analyze user activity patterns
            if 'user_activity' in data:
                user_activity = data['user_activity']
                if user_activity.get('active_users', 0) > 100:
                    patterns.append({
                        'type': 'high_user_activity',
                        'description': 'High user activity detected',
                        'severity': 'info',
                        'impact': 'positive'
                    })
            
            # Analyze performance patterns
            if 'performance_metrics' in data:
                perf_metrics = data['performance_metrics']
                if 'response_time' in perf_metrics:
                    response_time = perf_metrics['response_time']
                    if response_time.get('current', 0) > 1000:
                        patterns.append({
                            'type': 'high_response_time',
                            'description': 'High response time detected',
                            'severity': 'warning',
                            'impact': 'negative'
                        })
            
            # Analyze content patterns
            if 'content_metrics' in data:
                content_metrics = data['content_metrics']
                if content_metrics.get('avg_satisfaction', 0) < 3.0:
                    patterns.append({
                        'type': 'low_satisfaction',
                        'description': 'Low user satisfaction detected',
                        'severity': 'warning',
                        'impact': 'negative'
                    })
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
        
        return patterns
    
    def _generate_recommendations(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on data"""
        recommendations = []
        
        try:
            # Performance recommendations
            if 'performance_metrics' in data:
                perf_metrics = data['performance_metrics']
                if 'cpu_usage' in perf_metrics:
                    cpu_usage = perf_metrics['cpu_usage']
                    if cpu_usage.get('current', 0) > 80:
                        recommendations.append({
                            'type': 'performance',
                            'title': 'High CPU Usage',
                            'description': 'Consider scaling up or optimizing resource usage',
                            'priority': 'high',
                            'category': 'infrastructure'
                        })
            
            # User experience recommendations
            if 'user_activity' in data:
                user_activity = data['user_activity']
                if user_activity.get('avg_session_duration', 0) < 300:
                    recommendations.append({
                        'type': 'user_experience',
                        'title': 'Short Session Duration',
                        'description': 'Users are leaving quickly. Consider improving onboarding',
                        'priority': 'medium',
                        'category': 'user_experience'
                    })
            
            # Content recommendations
            if 'content_metrics' in data:
                content_metrics = data['content_metrics']
                if content_metrics.get('avg_quality', 0) < 0.7:
                    recommendations.append({
                        'type': 'content',
                        'title': 'Content Quality',
                        'description': 'Consider improving content templates and AI models',
                        'priority': 'medium',
                        'category': 'content'
                    })
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def _calculate_kpis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key performance indicators"""
        kpis = {}
        
        try:
            # User engagement KPI
            if 'user_activity' in data:
                user_activity = data['user_activity']
                kpis['user_engagement'] = {
                    'value': user_activity.get('active_users', 0),
                    'target': 100,
                    'status': 'good' if user_activity.get('active_users', 0) > 50 else 'needs_improvement'
                }
            
            # Performance KPI
            if 'performance_metrics' in data:
                perf_metrics = data['performance_metrics']
                if 'response_time' in perf_metrics:
                    response_time = perf_metrics['response_time']
                    kpis['performance'] = {
                        'value': response_time.get('current', 0),
                        'target': 500,
                        'status': 'good' if response_time.get('current', 0) < 500 else 'needs_improvement'
                    }
            
            # Content quality KPI
            if 'content_metrics' in data:
                content_metrics = data['content_metrics']
                kpis['content_quality'] = {
                    'value': content_metrics.get('avg_quality', 0),
                    'target': 0.8,
                    'status': 'good' if content_metrics.get('avg_quality', 0) > 0.8 else 'needs_improvement'
                }
            
        except Exception as e:
            logger.error(f"Error calculating KPIs: {e}")
        
        return kpis
    
    def _generate_summary(self, data: Dict[str, Any], patterns: List[Dict[str, Any]], recommendations: List[Dict[str, Any]]) -> str:
        """Generate AI summary"""
        try:
            summary_parts = []
            
            # System status
            if 'metrics' in data:
                metrics = data['metrics']
                summary_parts.append(f"System is running with {metrics.get('active_users', 0)} active users and {metrics.get('active_documents', 0)} active documents.")
            
            # Performance status
            if 'performance_metrics' in data:
                perf_metrics = data['performance_metrics']
                if 'response_time' in perf_metrics:
                    response_time = perf_metrics['response_time']
                    summary_parts.append(f"Average response time is {response_time.get('current', 0):.1f}ms.")
            
            # Patterns
            if patterns:
                pattern_count = len(patterns)
                summary_parts.append(f"Detected {pattern_count} patterns in system behavior.")
            
            # Recommendations
            if recommendations:
                rec_count = len(recommendations)
                summary_parts.append(f"Generated {rec_count} recommendations for improvement.")
            
            return " ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Unable to generate summary due to data processing error."
    
    async def export_analytics_data(self, format: str = "json", time_range: str = "24h") -> str:
        """Export analytics data"""
        try:
            # Get data
            dashboard_data = await self.get_analytics_dashboard(time_range)
            insights = await self.generate_insights(time_range)
            
            # Combine data
            export_data = {
                'dashboard': dashboard_data,
                'insights': insights,
                'export_info': {
                    'format': format,
                    'time_range': time_range,
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
            
            if format == "json":
                return json.dumps(export_data, indent=2)
            elif format == "csv":
                return self._convert_to_csv(export_data)
            else:
                return json.dumps(export_data, indent=2)
                
        except Exception as e:
            logger.error(f"Error exporting analytics data: {e}")
            return json.dumps({'error': str(e)})
    
    def _convert_to_csv(self, data: Dict[str, Any]) -> str:
        """Convert data to CSV format"""
        try:
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['Metric', 'Value', 'Timestamp'])
            
            # Write data
            if 'dashboard' in data:
                dashboard = data['dashboard']
                if 'metrics' in dashboard:
                    metrics = dashboard['metrics']
                    for key, value in metrics.items():
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                writer.writerow([f"{key}.{sub_key}", sub_value, datetime.now().isoformat()])
                        else:
                            writer.writerow([key, value, datetime.now().isoformat()])
            
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error converting to CSV: {e}")
            return "Error converting data to CSV format"
    
    async def create_analytics_report(self, report_type: str = "comprehensive", time_range: str = "24h") -> Dict[str, Any]:
        """Create comprehensive analytics report"""
        try:
            # Get all data
            dashboard_data = await self.get_analytics_dashboard(time_range)
            insights = await self.generate_insights(time_range)
            
            # Generate visualizations
            visualizations = await self._generate_visualizations(dashboard_data)
            
            # Create report
            report = {
                'report_id': f"report_{int(time.time())}",
                'report_type': report_type,
                'time_range': time_range,
                'generated_at': datetime.now().isoformat(),
                'executive_summary': insights.get('summary', ''),
                'dashboard_data': dashboard_data,
                'insights': insights,
                'visualizations': visualizations,
                'recommendations': insights.get('recommendations', []),
                'kpis': insights.get('kpis', {}),
                'metadata': {
                    'version': '1.0',
                    'generator': 'Advanced Analytics Engine',
                    'data_sources': ['events', 'metrics', 'user_behavior', 'content_quality']
                }
            }
            
            # Store report
            report_key = f"analytics:report:{report['report_id']}"
            self.redis_client.setex(
                report_key,
                86400,  # 24 hours
                json.dumps(report)
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Error creating analytics report: {e}")
            return {'error': str(e)}
    
    async def _generate_visualizations(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate visualization data"""
        try:
            visualizations = {}
            
            # User activity chart
            if 'user_activity' in data:
                user_activity = data['user_activity']
                if 'user_profiles' in user_activity:
                    profiles = user_activity['user_profiles']
                    if profiles:
                        # Create user activity chart
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=[p['user_id'] for p in profiles[:10]],  # Top 10 users
                            y=[p['documents_created'] for p in profiles[:10]],
                            name='Documents Created'
                        ))
                        fig.update_layout(
                            title='Top Users by Document Creation',
                            xaxis_title='User ID',
                            yaxis_title='Documents Created'
                        )
                        visualizations['user_activity_chart'] = fig.to_json()
            
            # Performance metrics chart
            if 'performance_metrics' in data:
                perf_metrics = data['performance_metrics']
                if perf_metrics:
                    # Create performance chart
                    fig = go.Figure()
                    for metric_name, metric_data in perf_metrics.items():
                        if isinstance(metric_data, dict) and 'current' in metric_data:
                            fig.add_trace(go.Scatter(
                                x=[metric_name],
                                y=[metric_data['current']],
                                mode='markers',
                                name=metric_name
                            ))
                    fig.update_layout(
                        title='Current Performance Metrics',
                        xaxis_title='Metric',
                        yaxis_title='Value'
                    )
                    visualizations['performance_chart'] = fig.to_json()
            
            # System health chart
            if 'metrics' in data and 'system_health' in data['metrics']:
                system_health = data['metrics']['system_health']
                if system_health:
                    # Create system health chart
                    fig = go.Figure()
                    fig.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=system_health.get('cpu_usage', 0),
                        title={'text': "CPU Usage (%)"},
                        domain={'x': [0, 1], 'y': [0, 1]}
                    ))
                    visualizations['system_health_chart'] = fig.to_json()
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            return {}
    
    async def add_websocket_connection(self, websocket):
        """Add WebSocket connection for real-time updates"""
        self.websocket_connections.add(websocket)
        logger.info(f"WebSocket connection added. Total connections: {len(self.websocket_connections)}")
    
    async def remove_websocket_connection(self, websocket):
        """Remove WebSocket connection"""
        self.websocket_connections.discard(websocket)
        logger.info(f"WebSocket connection removed. Total connections: {len(self.websocket_connections)}")
    
    async def get_real_time_updates(self, websocket):
        """Handle real-time updates for WebSocket connection"""
        try:
            await self.add_websocket_connection(websocket)
            
            # Send initial data
            dashboard_data = await self.get_analytics_dashboard("1h")
            await websocket.send(json.dumps({
                'type': 'initial_data',
                'data': dashboard_data
            }))
            
            # Keep connection alive and send updates
            while True:
                try:
                    # Wait for client message or timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=30)
                    data = json.loads(message)
                    
                    # Handle client requests
                    if data.get('type') == 'get_dashboard':
                        dashboard_data = await self.get_analytics_dashboard(data.get('time_range', '1h'))
                        await websocket.send(json.dumps({
                            'type': 'dashboard_update',
                            'data': dashboard_data
                        }))
                    
                except asyncio.TimeoutError:
                    # Send heartbeat
                    await websocket.send(json.dumps({
                        'type': 'heartbeat',
                        'timestamp': datetime.now().isoformat()
                    }))
                except websockets.exceptions.ConnectionClosed:
                    break
                    
        except Exception as e:
            logger.error(f"Error in real-time updates: {e}")
        finally:
            await self.remove_websocket_connection(websocket)

# Example usage and testing
async def main():
    """Example usage of the Advanced Analytics Engine"""
    
    # Configuration
    config = {
        'redis_host': 'localhost',
        'redis_port': 6379,
        'redis_db': 0,
        'database_url': 'postgresql://user:password@localhost/analytics_db'
    }
    
    # Initialize engine
    analytics_engine = AdvancedAnalyticsEngine(config)
    
    # Start engine
    await analytics_engine.start_analytics_engine()
    
    # Create sample events
    sample_events = [
        AnalyticsEvent(
            event_id=f"event_{i}",
            event_type=AnalyticsEventType.DOCUMENT_CREATED,
            user_id=f"user_{i % 10}",
            session_id=f"session_{i}",
            timestamp=datetime.now(),
            data={'template_id': f'template_{i % 5}', 'feature': 'ai_generation'},
            metadata={'device_type': 'desktop', 'location': 'US'},
            source='web_app',
            version='1.0'
        )
        for i in range(100)
    ]
    
    # Track events
    for event in sample_events:
        await analytics_engine.track_event(event)
    
    # Create sample metrics
    sample_metrics = [
        PerformanceMetrics(
            metric_id=f"metric_{i}",
            metric_name="response_time",
            value=100 + (i % 50),
            unit="milliseconds",
            timestamp=datetime.now(),
            context={"endpoint": f"/api/endpoint_{i % 10}"},
            tags=["api", "performance"]
        )
        for i in range(50)
    ]
    
    # Track metrics
    for metric in sample_metrics:
        await analytics_engine.track_performance_metric(metric)
    
    # Wait for processing
    await asyncio.sleep(5)
    
    # Get dashboard
    dashboard = await analytics_engine.get_analytics_dashboard("1h")
    print("Dashboard data:", json.dumps(dashboard, indent=2))
    
    # Generate insights
    insights = await analytics_engine.generate_insights("1h")
    print("Insights:", json.dumps(insights, indent=2))
    
    # Create report
    report = await analytics_engine.create_analytics_report("comprehensive", "1h")
    print("Report created:", report['report_id'])
    
    # Test predictions
    prediction = await analytics_engine.predict_performance({
        'cpu_usage': 75,
        'memory_usage': 60,
        'active_users': 50,
        'queue_size': 10
    })
    print("Performance prediction:", prediction)
    
    # Test anomaly detection
    test_data = [
        {'cpu_usage': 20, 'memory_usage': 30, 'response_time': 100, 'error_rate': 0.01},
        {'cpu_usage': 90, 'memory_usage': 95, 'response_time': 2000, 'error_rate': 0.5}
    ]
    anomalies = await analytics_engine.detect_anomalies(test_data)
    print("Anomalies detected:", anomalies)
    
    # Export data
    exported_data = await analytics_engine.export_analytics_data("json", "1h")
    print("Exported data length:", len(exported_data))
    
    # Stop engine
    await analytics_engine.stop_analytics_engine()
    
    print("Analytics engine test completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())

























