"""
Real-Time Analytics System
Ultra-modular Facebook Posts System v6.0

Advanced real-time analytics and monitoring:
- Live data streaming
- Real-time dashboards
- Predictive analytics
- Anomaly detection
- Performance monitoring
- Business intelligence
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)

class AnalyticsEventType(Enum):
    """Analytics event types"""
    CONTENT_CREATED = "content_created"
    CONTENT_PUBLISHED = "content_published"
    ENGAGEMENT_RECEIVED = "engagement_received"
    USER_INTERACTION = "user_interaction"
    SYSTEM_METRIC = "system_metric"
    PERFORMANCE_UPDATE = "performance_update"
    ANOMALY_DETECTED = "anomaly_detected"
    PREDICTION_UPDATE = "prediction_update"

class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class AnalyticsEvent:
    """Analytics event data structure"""
    event_type: AnalyticsEventType
    timestamp: datetime
    data: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class Metric:
    """Metric data structure"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    labels: Optional[Dict[str, str]] = None

@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    id: str
    title: str
    widget_type: str
    data_source: str
    refresh_interval: int
    config: Dict[str, Any]

class RealTimeAnalytics:
    """Advanced real-time analytics system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_running = False
        self.events = deque(maxlen=10000)  # Keep last 10k events
        self.metrics = defaultdict(list)
        self.dashboards = {}
        self.websocket_clients = set()
        self.anomaly_detectors = {}
        self.predictors = {}
        self.alert_rules = {}
        self.is_initialized = False
        
        # Performance tracking
        self.performance_metrics = {
            "events_processed": 0,
            "metrics_collected": 0,
            "alerts_triggered": 0,
            "predictions_made": 0,
            "anomalies_detected": 0
        }
        
    async def initialize(self) -> bool:
        """Initialize real-time analytics system"""
        try:
            logger.info("Initializing Real-Time Analytics System...")
            
            # Initialize data stores
            await self._initialize_data_stores()
            
            # Initialize anomaly detectors
            await self._initialize_anomaly_detectors()
            
            # Initialize predictors
            await self._initialize_predictors()
            
            # Initialize alert rules
            await self._initialize_alert_rules()
            
            # Initialize dashboards
            await self._initialize_dashboards()
            
            self.is_initialized = True
            logger.info("✓ Real-Time Analytics System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Real-Time Analytics System: {e}")
            return False
    
    async def start(self) -> bool:
        """Start real-time analytics system"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            logger.info("Starting Real-Time Analytics System...")
            
            # Start event processing
            self.event_processor_task = asyncio.create_task(self._process_events())
            
            # Start metrics collection
            self.metrics_collector_task = asyncio.create_task(self._collect_metrics())
            
            # Start anomaly detection
            self.anomaly_detection_task = asyncio.create_task(self._detect_anomalies())
            
            # Start prediction engine
            self.prediction_task = asyncio.create_task(self._run_predictions())
            
            # Start alert processing
            self.alert_processor_task = asyncio.create_task(self._process_alerts())
            
            self.is_running = True
            logger.info("✓ Real-Time Analytics System started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Real-Time Analytics System: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop real-time analytics system"""
        try:
            logger.info("Stopping Real-Time Analytics System...")
            
            self.is_running = False
            
            # Cancel all tasks
            tasks = [
                self.event_processor_task,
                self.metrics_collector_task,
                self.anomaly_detection_task,
                self.prediction_task,
                self.alert_processor_task
            ]
            
            for task in tasks:
                if task and not task.done():
                    task.cancel()
            
            # Close WebSocket connections
            for client in self.websocket_clients:
                await client.close()
            
            logger.info("✓ Real-Time Analytics System stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop Real-Time Analytics System: {e}")
            return False
    
    async def _initialize_data_stores(self) -> None:
        """Initialize data stores"""
        logger.info("Initializing data stores...")
        
        # Initialize event store
        self.event_store = {
            "events": deque(maxlen=100000),
            "indexes": defaultdict(list)
        }
        
        # Initialize metrics store
        self.metrics_store = {
            "counters": defaultdict(int),
            "gauges": defaultdict(float),
            "histograms": defaultdict(list),
            "summaries": defaultdict(list)
        }
        
        logger.info("✓ Data stores initialized")
    
    async def _initialize_anomaly_detectors(self) -> None:
        """Initialize anomaly detectors"""
        logger.info("Initializing anomaly detectors...")
        
        # Statistical anomaly detector
        self.anomaly_detectors["statistical"] = {
            "type": "statistical",
            "threshold": 2.0,  # 2 standard deviations
            "window_size": 100,
            "enabled": True
        }
        
        # Machine learning anomaly detector
        self.anomaly_detectors["ml"] = {
            "type": "machine_learning",
            "model": "isolation_forest",
            "contamination": 0.1,
            "enabled": True
        }
        
        # Time series anomaly detector
        self.anomaly_detectors["time_series"] = {
            "type": "time_series",
            "method": "seasonal_decomposition",
            "enabled": True
        }
        
        logger.info("✓ Anomaly detectors initialized")
    
    async def _initialize_predictors(self) -> None:
        """Initialize prediction models"""
        logger.info("Initializing predictors...")
        
        # Engagement prediction
        self.predictors["engagement"] = {
            "type": "regression",
            "model": "linear_regression",
            "features": ["content_length", "hashtags", "time_of_day"],
            "accuracy": 0.85
        }
        
        # Viral potential prediction
        self.predictors["viral_potential"] = {
            "type": "classification",
            "model": "random_forest",
            "features": ["content_type", "engagement_rate", "user_followers"],
            "accuracy": 0.78
        }
        
        # Performance prediction
        self.predictors["performance"] = {
            "type": "time_series",
            "model": "arima",
            "features": ["historical_performance", "trends"],
            "accuracy": 0.82
        }
        
        logger.info("✓ Predictors initialized")
    
    async def _initialize_alert_rules(self) -> None:
        """Initialize alert rules"""
        logger.info("Initializing alert rules...")
        
        # High engagement alert
        self.alert_rules["high_engagement"] = {
            "condition": "engagement_rate > 0.1",
            "severity": "info",
            "enabled": True
        }
        
        # Low performance alert
        self.alert_rules["low_performance"] = {
            "condition": "engagement_rate < 0.01",
            "severity": "warning",
            "enabled": True
        }
        
        # System error alert
        self.alert_rules["system_error"] = {
            "condition": "error_rate > 0.05",
            "severity": "critical",
            "enabled": True
        }
        
        logger.info("✓ Alert rules initialized")
    
    async def _initialize_dashboards(self) -> None:
        """Initialize dashboards"""
        logger.info("Initializing dashboards...")
        
        # Main analytics dashboard
        self.dashboards["main"] = {
            "id": "main",
            "title": "Main Analytics Dashboard",
            "widgets": [
                {
                    "id": "engagement_chart",
                    "title": "Engagement Over Time",
                    "type": "line_chart",
                    "data_source": "engagement_metrics",
                    "refresh_interval": 30
                },
                {
                    "id": "content_performance",
                    "title": "Content Performance",
                    "type": "bar_chart",
                    "data_source": "content_metrics",
                    "refresh_interval": 60
                },
                {
                    "id": "user_activity",
                    "title": "User Activity",
                    "type": "heatmap",
                    "data_source": "user_metrics",
                    "refresh_interval": 120
                }
            ]
        }
        
        # Performance dashboard
        self.dashboards["performance"] = {
            "id": "performance",
            "title": "Performance Dashboard",
            "widgets": [
                {
                    "id": "response_time",
                    "title": "Response Time",
                    "type": "gauge",
                    "data_source": "performance_metrics",
                    "refresh_interval": 10
                },
                {
                    "id": "throughput",
                    "title": "Throughput",
                    "type": "counter",
                    "data_source": "performance_metrics",
                    "refresh_interval": 10
                }
            ]
        }
        
        logger.info("✓ Dashboards initialized")
    
    async def _process_events(self) -> None:
        """Process analytics events"""
        while self.is_running:
            try:
                # Process events from queue
                if self.events:
                    event = self.events.popleft()
                    await self._handle_event(event)
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Event processing error: {e}")
                await asyncio.sleep(1)
    
    async def _handle_event(self, event: AnalyticsEvent) -> None:
        """Handle a single analytics event"""
        try:
            # Store event
            self.event_store["events"].append(event)
            
            # Update indexes
            self.event_store["indexes"][event.event_type.value].append(len(self.event_store["events"]) - 1)
            
            # Update metrics
            await self._update_metrics_from_event(event)
            
            # Check for anomalies
            await self._check_event_for_anomalies(event)
            
            # Broadcast to WebSocket clients
            await self._broadcast_event(event)
            
            self.performance_metrics["events_processed"] += 1
            
        except Exception as e:
            logger.error(f"Error handling event: {e}")
    
    async def _update_metrics_from_event(self, event: AnalyticsEvent) -> None:
        """Update metrics based on event"""
        try:
            # Update event counters
            self.metrics_store["counters"][f"events_{event.event_type.value}"] += 1
            
            # Update engagement metrics
            if event.event_type == AnalyticsEventType.ENGAGEMENT_RECEIVED:
                engagement_rate = event.data.get("engagement_rate", 0)
                self.metrics_store["gauges"]["current_engagement_rate"] = engagement_rate
                self.metrics_store["histograms"]["engagement_rates"].append(engagement_rate)
            
            # Update content metrics
            if event.event_type == AnalyticsEventType.CONTENT_CREATED:
                content_length = event.data.get("content_length", 0)
                self.metrics_store["histograms"]["content_lengths"].append(content_length)
            
            self.performance_metrics["metrics_collected"] += 1
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def _check_event_for_anomalies(self, event: AnalyticsEvent) -> None:
        """Check event for anomalies"""
        try:
            # Statistical anomaly detection
            if self.anomaly_detectors["statistical"]["enabled"]:
                await self._check_statistical_anomaly(event)
            
            # ML anomaly detection
            if self.anomaly_detectors["ml"]["enabled"]:
                await self._check_ml_anomaly(event)
            
            # Time series anomaly detection
            if self.anomaly_detectors["time_series"]["enabled"]:
                await self._check_time_series_anomaly(event)
            
        except Exception as e:
            logger.error(f"Error checking anomalies: {e}")
    
    async def _check_statistical_anomaly(self, event: AnalyticsEvent) -> None:
        """Check for statistical anomalies"""
        if event.event_type == AnalyticsEventType.ENGAGEMENT_RECEIVED:
            engagement_rate = event.data.get("engagement_rate", 0)
            recent_rates = list(self.metrics_store["histograms"]["engagement_rates"])[-100:]
            
            if len(recent_rates) > 10:
                mean_rate = np.mean(recent_rates)
                std_rate = np.std(recent_rates)
                threshold = self.anomaly_detectors["statistical"]["threshold"]
                
                if abs(engagement_rate - mean_rate) > threshold * std_rate:
                    await self._trigger_anomaly_alert("statistical", event, {
                        "engagement_rate": engagement_rate,
                        "mean_rate": mean_rate,
                        "std_rate": std_rate
                    })
    
    async def _check_ml_anomaly(self, event: AnalyticsEvent) -> None:
        """Check for ML-based anomalies"""
        # Simulate ML anomaly detection
        features = self._extract_features(event)
        anomaly_score = np.random.random()  # Simulate ML prediction
        
        if anomaly_score > 0.8:  # High anomaly score
            await self._trigger_anomaly_alert("ml", event, {
                "anomaly_score": anomaly_score,
                "features": features
            })
    
    async def _check_time_series_anomaly(self, event: AnalyticsEvent) -> None:
        """Check for time series anomalies"""
        # Simulate time series anomaly detection
        if event.event_type == AnalyticsEventType.SYSTEM_METRIC:
            metric_value = event.data.get("value", 0)
            timestamp = event.timestamp
            
            # Simple trend analysis
            recent_values = [e.data.get("value", 0) for e in 
                           list(self.event_store["events"])[-10:] 
                           if e.event_type == AnalyticsEventType.SYSTEM_METRIC]
            
            if len(recent_values) > 5:
                trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                if abs(trend) > 0.5:  # Significant trend change
                    await self._trigger_anomaly_alert("time_series", event, {
                        "trend": trend,
                        "metric_value": metric_value
                    })
    
    async def _trigger_anomaly_alert(self, detector_type: str, event: AnalyticsEvent, details: Dict[str, Any]) -> None:
        """Trigger anomaly alert"""
        alert = {
            "type": "anomaly",
            "detector": detector_type,
            "event": asdict(event),
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store alert
        self.metrics_store["summaries"]["anomaly_alerts"].append(alert)
        
        # Broadcast alert
        await self._broadcast_alert(alert)
        
        self.performance_metrics["anomalies_detected"] += 1
        logger.warning(f"Anomaly detected by {detector_type}: {details}")
    
    async def _extract_features(self, event: AnalyticsEvent) -> List[float]:
        """Extract features from event for ML analysis"""
        features = []
        
        # Basic features
        features.append(len(str(event.data)))
        features.append(event.timestamp.hour)
        features.append(event.timestamp.weekday())
        
        # Event-specific features
        if event.event_type == AnalyticsEventType.ENGAGEMENT_RECEIVED:
            features.append(event.data.get("engagement_rate", 0))
            features.append(event.data.get("likes", 0))
            features.append(event.data.get("shares", 0))
        
        return features
    
    async def _collect_metrics(self) -> None:
        """Collect system metrics"""
        while self.is_running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Collect application metrics
                await self._collect_application_metrics()
                
                # Collect business metrics
                await self._collect_business_metrics()
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system-level metrics"""
        import psutil
        
        # CPU usage
        cpu_usage = psutil.cpu_percent()
        self.metrics_store["gauges"]["system_cpu_usage"] = cpu_usage
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics_store["gauges"]["system_memory_usage"] = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        self.metrics_store["gauges"]["system_disk_usage"] = disk.percent
        
        # Network I/O
        network = psutil.net_io_counters()
        self.metrics_store["counters"]["network_bytes_sent"] = network.bytes_sent
        self.metrics_store["counters"]["network_bytes_recv"] = network.bytes_recv
    
    async def _collect_application_metrics(self) -> None:
        """Collect application-level metrics"""
        # Event processing rate
        events_per_second = self.performance_metrics["events_processed"] / 60
        self.metrics_store["gauges"]["events_per_second"] = events_per_second
        
        # WebSocket connections
        self.metrics_store["gauges"]["websocket_connections"] = len(self.websocket_clients)
        
        # Queue sizes
        self.metrics_store["gauges"]["event_queue_size"] = len(self.events)
    
    async def _collect_business_metrics(self) -> None:
        """Collect business-level metrics"""
        # Engagement rate
        if self.metrics_store["histograms"]["engagement_rates"]:
            avg_engagement = np.mean(self.metrics_store["histograms"]["engagement_rates"])
            self.metrics_store["gauges"]["average_engagement_rate"] = avg_engagement
        
        # Content creation rate
        content_events = self.metrics_store["counters"]["events_content_created"]
        self.metrics_store["gauges"]["content_creation_rate"] = content_events / 60
    
    async def _detect_anomalies(self) -> None:
        """Run anomaly detection"""
        while self.is_running:
            try:
                # Run anomaly detection on recent events
                recent_events = list(self.event_store["events"])[-100:]
                
                for event in recent_events:
                    await self._check_event_for_anomalies(event)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Anomaly detection error: {e}")
                await asyncio.sleep(60)
    
    async def _run_predictions(self) -> None:
        """Run prediction models"""
        while self.is_running:
            try:
                # Run engagement predictions
                await self._predict_engagement()
                
                # Run viral potential predictions
                await self._predict_viral_potential()
                
                # Run performance predictions
                await self._predict_performance()
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                await asyncio.sleep(300)
    
    async def _predict_engagement(self) -> None:
        """Predict engagement for recent content"""
        try:
            # Simulate engagement prediction
            prediction = {
                "type": "engagement_prediction",
                "predicted_engagement": np.random.uniform(0.01, 0.15),
                "confidence": np.random.uniform(0.7, 0.95),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store prediction
            self.metrics_store["summaries"]["predictions"].append(prediction)
            
            # Broadcast prediction
            await self._broadcast_prediction(prediction)
            
            self.performance_metrics["predictions_made"] += 1
            
        except Exception as e:
            logger.error(f"Engagement prediction error: {e}")
    
    async def _predict_viral_potential(self) -> None:
        """Predict viral potential for content"""
        try:
            # Simulate viral potential prediction
            prediction = {
                "type": "viral_potential_prediction",
                "viral_score": np.random.uniform(0.1, 0.9),
                "confidence": np.random.uniform(0.6, 0.9),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store prediction
            self.metrics_store["summaries"]["predictions"].append(prediction)
            
            # Broadcast prediction
            await self._broadcast_prediction(prediction)
            
        except Exception as e:
            logger.error(f"Viral potential prediction error: {e}")
    
    async def _predict_performance(self) -> None:
        """Predict system performance"""
        try:
            # Simulate performance prediction
            prediction = {
                "type": "performance_prediction",
                "predicted_load": np.random.uniform(0.3, 0.9),
                "recommended_scaling": "scale_up" if np.random.random() > 0.7 else "maintain",
                "timestamp": datetime.now().isoformat()
            }
            
            # Store prediction
            self.metrics_store["summaries"]["predictions"].append(prediction)
            
            # Broadcast prediction
            await self._broadcast_prediction(prediction)
            
        except Exception as e:
            logger.error(f"Performance prediction error: {e}")
    
    async def _process_alerts(self) -> None:
        """Process alert rules"""
        while self.is_running:
            try:
                # Check alert rules
                for rule_name, rule in self.alert_rules.items():
                    if rule["enabled"]:
                        await self._check_alert_rule(rule_name, rule)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(30)
    
    async def _check_alert_rule(self, rule_name: str, rule: Dict[str, Any]) -> None:
        """Check a specific alert rule"""
        try:
            condition = rule["condition"]
            
            # Simple condition evaluation (in real implementation, use proper expression evaluator)
            if "engagement_rate" in condition:
                current_engagement = self.metrics_store["gauges"].get("current_engagement_rate", 0)
                
                if "> 0.1" in condition and current_engagement > 0.1:
                    await self._trigger_alert(rule_name, "high_engagement", {
                        "engagement_rate": current_engagement
                    })
                elif "< 0.01" in condition and current_engagement < 0.01:
                    await self._trigger_alert(rule_name, "low_engagement", {
                        "engagement_rate": current_engagement
                    })
            
        except Exception as e:
            logger.error(f"Alert rule check error for {rule_name}: {e}")
    
    async def _trigger_alert(self, rule_name: str, alert_type: str, details: Dict[str, Any]) -> None:
        """Trigger an alert"""
        alert = {
            "rule": rule_name,
            "type": alert_type,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store alert
        self.metrics_store["summaries"]["alerts"].append(alert)
        
        # Broadcast alert
        await self._broadcast_alert(alert)
        
        self.performance_metrics["alerts_triggered"] += 1
        logger.warning(f"Alert triggered: {rule_name} - {alert_type}")
    
    async def _broadcast_event(self, event: AnalyticsEvent) -> None:
        """Broadcast event to WebSocket clients"""
        if self.websocket_clients:
            message = {
                "type": "event",
                "data": asdict(event)
            }
            
            # Send to all connected clients
            disconnected_clients = set()
            for client in self.websocket_clients:
                try:
                    await client.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.websocket_clients -= disconnected_clients
    
    async def _broadcast_alert(self, alert: Dict[str, Any]) -> None:
        """Broadcast alert to WebSocket clients"""
        if self.websocket_clients:
            message = {
                "type": "alert",
                "data": alert
            }
            
            # Send to all connected clients
            disconnected_clients = set()
            for client in self.websocket_clients:
                try:
                    await client.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.websocket_clients -= disconnected_clients
    
    async def _broadcast_prediction(self, prediction: Dict[str, Any]) -> None:
        """Broadcast prediction to WebSocket clients"""
        if self.websocket_clients:
            message = {
                "type": "prediction",
                "data": prediction
            }
            
            # Send to all connected clients
            disconnected_clients = set()
            for client in self.websocket_clients:
                try:
                    await client.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.websocket_clients -= disconnected_clients
    
    # Public API methods
    
    async def track_event(self, event_type: AnalyticsEventType, data: Dict[str, Any], 
                         user_id: Optional[str] = None, session_id: Optional[str] = None) -> None:
        """Track an analytics event"""
        event = AnalyticsEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            user_id=user_id,
            session_id=session_id
        )
        
        self.events.append(event)
    
    async def get_metrics(self, metric_type: Optional[str] = None) -> Dict[str, Any]:
        """Get analytics metrics"""
        if metric_type:
            return dict(self.metrics_store[metric_type])
        else:
            return dict(self.metrics_store)
    
    async def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Get dashboard data"""
        if dashboard_id not in self.dashboards:
            return {"error": "Dashboard not found"}
        
        dashboard = self.dashboards[dashboard_id]
        widgets_data = {}
        
        for widget in dashboard["widgets"]:
            widget_id = widget["id"]
            data_source = widget["data_source"]
            
            if data_source == "engagement_metrics":
                widgets_data[widget_id] = {
                    "data": list(self.metrics_store["histograms"]["engagement_rates"])[-100:],
                    "labels": [f"Point {i}" for i in range(100)]
                }
            elif data_source == "content_metrics":
                widgets_data[widget_id] = {
                    "data": list(self.metrics_store["histograms"]["content_lengths"])[-50:],
                    "labels": [f"Content {i}" for i in range(50)]
                }
            elif data_source == "performance_metrics":
                widgets_data[widget_id] = {
                    "data": {
                        "response_time": self.metrics_store["gauges"].get("system_cpu_usage", 0),
                        "throughput": self.metrics_store["gauges"].get("events_per_second", 0)
                    }
                }
        
        return {
            "dashboard": dashboard,
            "widgets_data": widgets_data,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get analytics system health status"""
        return {
            "status": "healthy" if self.is_running else "unhealthy",
            "running": self.is_running,
            "events_processed": self.performance_metrics["events_processed"],
            "metrics_collected": self.performance_metrics["metrics_collected"],
            "websocket_clients": len(self.websocket_clients),
            "event_queue_size": len(self.events)
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.performance_metrics,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "real_time_analytics": {
                "status": "running" if self.is_running else "stopped",
                "performance": self.performance_metrics,
                "dashboards": list(self.dashboards.keys()),
                "anomaly_detectors": list(self.anomaly_detectors.keys()),
                "predictors": list(self.predictors.keys())
            },
            "timestamp": datetime.now().isoformat()
        }
