from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import os
import prometheus_client as prometheus
from prometheus_client import Counter, Histogram, Gauge, Summary, Info
import structlog
from structlog import get_logger
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
import opentelemetry
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
import elasticsearch
from elasticsearch import AsyncElasticsearch
import redis.asyncio as redis
import aiohttp
import asyncio_mqtt as mqtt
import aiokafka
from kafka import KafkaProducer, KafkaConsumer
import influxdb
from influxdb import InfluxDBClient
import grafana_api
from grafana_api.grafana_face import GrafanaFace
import alertmanager
from alertmanager import AlertManager
import pagerduty
from pagerduty import PagerDuty
import slack
from slack import WebClient
import discord
from discord import Client
import telegram
from telegram import Bot
import email
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
Advanced Monitoring System
Comprehensive monitoring and observability with real-time metrics, alerting, and distributed tracing.
"""


# Monitoring and Observability Libraries

logger = get_logger()


class MonitoringLevel(Enum):
    """Monitoring levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MonitoringConfig:
    """Configuration for monitoring system."""
    # Prometheus
    prometheus_enabled: bool = True
    prometheus_port: int = 9090
    prometheus_host: str = "0.0.0.0"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: str = "logs/monitoring.log"
    
    # Sentry
    sentry_enabled: bool = True
    sentry_dsn: str = ""
    sentry_environment: str = "production"
    
    # OpenTelemetry
    opentelemetry_enabled: bool = True
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    prometheus_endpoint: str = "http://localhost:9090"
    
    # Elasticsearch
    elasticsearch_enabled: bool = True
    elasticsearch_hosts: List[str] = field(default_factory=lambda: ["http://localhost:9200"])
    elasticsearch_index: str = "math_platform_logs"
    
    # Redis
    redis_enabled: bool = True
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Kafka
    kafka_enabled: bool = False
    kafka_bootstrap_servers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    kafka_topic: str = "math_platform_metrics"
    
    # InfluxDB
    influxdb_enabled: bool = False
    influxdb_host: str = "localhost"
    influxdb_port: int = 8086
    influxdb_database: str = "math_platform"
    
    # Grafana
    grafana_enabled: bool = False
    grafana_host: str = "localhost"
    grafana_port: int = 3000
    grafana_token: str = ""
    
    # Alerting
    alerting_enabled: bool = True
    alertmanager_url: str = "http://localhost:9093"
    pagerduty_token: str = ""
    slack_token: str = ""
    discord_token: str = ""
    telegram_token: str = ""
    email_smtp_server: str = ""
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    
    # Health checks
    health_check_interval: int = 30
    health_check_timeout: int = 10
    
    # Metrics collection
    metrics_interval: int = 60
    metrics_retention_days: int = 30
    
    # Performance monitoring
    performance_monitoring: bool = True
    memory_monitoring: bool = True
    cpu_monitoring: bool = True
    network_monitoring: bool = True
    disk_monitoring: bool = True


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used: int
    memory_total: int
    disk_percent: float
    disk_used: int
    disk_total: int
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    thread_count: int
    load_average: List[float] = field(default_factory=list)


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    timestamp: datetime
    request_count: int
    request_latency: float
    error_count: int
    success_rate: float
    active_connections: int
    cache_hit_rate: float
    optimization_success_rate: float
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert definition."""
    alert_id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: Callable[[Dict[str, Any]], bool]
    threshold: float
    cooldown: int = 300  # seconds
    last_triggered: Optional[datetime] = None
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)


class PrometheusMetrics:
    """Prometheus metrics collection."""
    
    def __init__(self) -> Any:
        # System metrics
        self.cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage')
        self.memory_used = Gauge('system_memory_used_bytes', 'Memory used in bytes')
        self.disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percentage')
        self.disk_used = Gauge('system_disk_used_bytes', 'Disk used in bytes')
        self.network_sent = Counter('system_network_bytes_sent_total', 'Total bytes sent')
        self.network_recv = Counter('system_network_bytes_recv_total', 'Total bytes received')
        
        # Application metrics
        self.request_total = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
        self.request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')
        self.request_latency = Histogram('http_request_latency_seconds', 'HTTP request latency')
        self.active_connections = Gauge('active_connections', 'Number of active connections')
        self.error_total = Counter('errors_total', 'Total errors', ['type', 'severity'])
        self.cache_hits = Counter('cache_hits_total', 'Total cache hits')
        self.cache_misses = Counter('cache_misses_total', 'Total cache misses')
        self.optimization_success = Counter('optimization_success_total', 'Total successful optimizations')
        self.optimization_failure = Counter('optimization_failure_total', 'Total failed optimizations')
        
        # Business metrics
        self.operations_total = Counter('math_operations_total', 'Total mathematical operations', ['operation_type'])
        self.operations_duration = Histogram('math_operations_duration_seconds', 'Mathematical operation duration')
        self.workflow_executions = Counter('workflow_executions_total', 'Total workflow executions', ['workflow_type'])
        self.workflow_duration = Histogram('workflow_duration_seconds', 'Workflow execution duration')
        
        # Custom metrics
        self.custom_metrics = {}
    
    def add_custom_metric(self, name: str, metric_type: str, description: str = ""):
        """Add a custom metric."""
        if metric_type == "counter":
            self.custom_metrics[name] = Counter(name, description)
        elif metric_type == "gauge":
            self.custom_metrics[name] = Gauge(name, description)
        elif metric_type == "histogram":
            self.custom_metrics[name] = Histogram(name, description)
        elif metric_type == "summary":
            self.custom_metrics[name] = Summary(name, description)


class StructuredLogger:
    """Structured logging with multiple backends."""
    
    def __init__(self, config: MonitoringConfig):
        
    """__init__ function."""
self.config = config
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer() if config.log_format == "json" else structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = get_logger()
        
        # Setup Sentry if enabled
        if config.sentry_enabled and config.sentry_dsn:
            sentry_sdk.init(
                dsn=config.sentry_dsn,
                environment=config.sentry_environment,
                traces_sample_rate=1.0,
                profiles_sample_rate=1.0,
            )
    
    def log_event(self, event: str, level: MonitoringLevel = MonitoringLevel.INFO, 
                  **kwargs):
        """Log an event with structured data."""
        log_method = getattr(self.logger, level.value)
        log_method(event, **kwargs)
        
        # Send to Sentry for errors
        if level in [MonitoringLevel.ERROR, MonitoringLevel.CRITICAL]:
            sentry_sdk.capture_message(event, level=level.value, extra=kwargs)
    
    def log_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Log a metric."""
        self.logger.info("metric", name=metric_name, value=value, labels=labels or {})
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log an error with context."""
        self.logger.error("error", 
                         error_type=type(error).__name__,
                         error_message=str(error),
                         context=context or {})
        
        # Send to Sentry
        sentry_sdk.capture_exception(error, extra=context)


class DistributedTracer:
    """Distributed tracing with OpenTelemetry."""
    
    def __init__(self, config: MonitoringConfig):
        
    """__init__ function."""
self.config = config
        
        if not config.opentelemetry_enabled:
            return
        
        # Setup tracer provider
        trace.set_tracer_provider(TracerProvider())
        self.tracer = trace.get_tracer(__name__)
        
        # Setup Jaeger exporter
        if config.jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
            trace.get_tracer_provider().add_span_processor(
                BatchSpanProcessor(jaeger_exporter)
            )
        
        # Setup Prometheus exporter
        if config.prometheus_endpoint:
            prometheus_exporter = PrometheusExporter()
            trace.get_tracer_provider().add_span_processor(
                BatchSpanProcessor(prometheus_exporter)
            )
    
    def start_span(self, name: str, attributes: Dict[str, Any] = None):
        """Start a new span."""
        if not self.config.opentelemetry_enabled:
            return None
        
        return self.tracer.start_span(name, attributes=attributes or {})
    
    def add_event(self, span, name: str, attributes: Dict[str, Any] = None):
        """Add an event to a span."""
        if span and self.config.opentelemetry_enabled:
            span.add_event(name, attributes=attributes or {})
    
    def set_attribute(self, span, key: str, value: Any):
        """Set an attribute on a span."""
        if span and self.config.opentelemetry_enabled:
            span.set_attribute(key, value)


class MetricsCollector:
    """Collect and store metrics from various sources."""
    
    def __init__(self, config: MonitoringConfig):
        
    """__init__ function."""
self.config = config
        self.prometheus_metrics = PrometheusMetrics()
        self.logger = StructuredLogger(config)
        self.tracer = DistributedTracer(config)
        
        # Storage backends
        self.elasticsearch_client = None
        self.redis_client = None
        self.kafka_producer = None
        self.influxdb_client = None
        
        self._setup_storage_backends()
    
    def _setup_storage_backends(self) -> Any:
        """Setup storage backends for metrics."""
        # Elasticsearch
        if self.config.elasticsearch_enabled:
            try:
                self.elasticsearch_client = AsyncElasticsearch(self.config.elasticsearch_hosts)
                logger.info("Elasticsearch client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Elasticsearch: {e}")
        
        # Redis
        if self.config.redis_enabled:
            try:
                self.redis_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    db=self.config.redis_db,
                    decode_responses=True
                )
                logger.info("Redis client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Redis: {e}")
        
        # Kafka
        if self.config.kafka_enabled:
            try:
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=self.config.kafka_bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
                )
                logger.info("Kafka producer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Kafka: {e}")
        
        # InfluxDB
        if self.config.influxdb_enabled:
            try:
                self.influxdb_client = InfluxDBClient(
                    host=self.config.influxdb_host,
                    port=self.config.influxdb_port,
                    database=self.config.influxdb_database
                )
                logger.info("InfluxDB client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize InfluxDB: {e}")
    
    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used=memory.used,
            memory_total=memory.total,
            disk_percent=disk.percent,
            disk_used=disk.used,
            disk_total=disk.total,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            process_count=len(psutil.pids()),
            thread_count=psutil.cpu_count(),
            load_average=psutil.getloadavg() if hasattr(psutil, 'getloadavg') else []
        )
        
        # Update Prometheus metrics
        self.prometheus_metrics.cpu_usage.set(cpu_percent)
        self.prometheus_metrics.memory_usage.set(memory.percent)
        self.prometheus_metrics.memory_used.set(memory.used)
        self.prometheus_metrics.disk_usage.set(disk.percent)
        self.prometheus_metrics.disk_used.set(disk.used)
        self.prometheus_metrics.network_sent.inc(network.bytes_sent)
        self.prometheus_metrics.network_recv.inc(network.bytes_recv)
        
        return metrics
    
    async def collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics."""
        # This would be populated from the actual application
        metrics = ApplicationMetrics(
            timestamp=datetime.now(),
            request_count=0,
            request_latency=0.0,
            error_count=0,
            success_rate=1.0,
            active_connections=0,
            cache_hit_rate=0.0,
            optimization_success_rate=0.0
        )
        
        return metrics
    
    async def store_metrics(self, system_metrics: SystemMetrics, 
                          app_metrics: ApplicationMetrics):
        """Store metrics in various backends."""
        metrics_data = {
            "timestamp": system_metrics.timestamp.isoformat(),
            "system": {
                "cpu_percent": system_metrics.cpu_percent,
                "memory_percent": system_metrics.memory_percent,
                "memory_used": system_metrics.memory_used,
                "memory_total": system_metrics.memory_total,
                "disk_percent": system_metrics.disk_percent,
                "disk_used": system_metrics.disk_used,
                "disk_total": system_metrics.disk_total,
                "network_bytes_sent": system_metrics.network_bytes_sent,
                "network_bytes_recv": system_metrics.network_bytes_recv,
                "process_count": system_metrics.process_count,
                "thread_count": system_metrics.thread_count,
                "load_average": system_metrics.load_average
            },
            "application": {
                "request_count": app_metrics.request_count,
                "request_latency": app_metrics.request_latency,
                "error_count": app_metrics.error_count,
                "success_rate": app_metrics.success_rate,
                "active_connections": app_metrics.active_connections,
                "cache_hit_rate": app_metrics.cache_hit_rate,
                "optimization_success_rate": app_metrics.optimization_success_rate,
                "custom_metrics": app_metrics.custom_metrics
            }
        }
        
        # Store in Elasticsearch
        if self.elasticsearch_client:
            try:
                await self.elasticsearch_client.index(
                    index=self.config.elasticsearch_index,
                    body=metrics_data
                )
            except Exception as e:
                logger.error(f"Failed to store metrics in Elasticsearch: {e}")
        
        # Store in Redis
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"metrics:{system_metrics.timestamp.timestamp()}",
                    86400,  # 24 hours
                    json.dumps(metrics_data)
                )
            except Exception as e:
                logger.error(f"Failed to store metrics in Redis: {e}")
        
        # Send to Kafka
        if self.kafka_producer:
            try:
                self.kafka_producer.send(self.config.kafka_topic, metrics_data)
            except Exception as e:
                logger.error(f"Failed to send metrics to Kafka: {e}")
        
        # Store in InfluxDB
        if self.influxdb_client:
            try:
                points = [
                    {
                        "measurement": "system_metrics",
                        "time": system_metrics.timestamp,
                        "fields": {
                            "cpu_percent": system_metrics.cpu_percent,
                            "memory_percent": system_metrics.memory_percent,
                            "disk_percent": system_metrics.disk_percent
                        }
                    }
                ]
                self.influxdb_client.write_points(points)
            except Exception as e:
                logger.error(f"Failed to store metrics in InfluxDB: {e}")


class AlertManager:
    """Alert management system."""
    
    def __init__(self, config: MonitoringConfig):
        
    """__init__ function."""
self.config = config
        self.alerts: List[Alert] = []
        self.notification_clients = {}
        
        self._setup_notification_clients()
        self._setup_default_alerts()
    
    def _setup_notification_clients(self) -> Any:
        """Setup notification clients."""
        # PagerDuty
        if self.config.pagerduty_token:
            try:
                self.notification_clients['pagerduty'] = PagerDuty(self.config.pagerduty_token)
            except Exception as e:
                logger.error(f"Failed to setup PagerDuty: {e}")
        
        # Slack
        if self.config.slack_token:
            try:
                self.notification_clients['slack'] = WebClient(token=self.config.slack_token)
            except Exception as e:
                logger.error(f"Failed to setup Slack: {e}")
        
        # Discord
        if self.config.discord_token:
            try:
                self.notification_clients['discord'] = Client()
            except Exception as e:
                logger.error(f"Failed to setup Discord: {e}")
        
        # Telegram
        if self.config.telegram_token:
            try:
                self.notification_clients['telegram'] = Bot(token=self.config.telegram_token)
            except Exception as e:
                logger.error(f"Failed to setup Telegram: {e}")
    
    def _setup_default_alerts(self) -> Any:
        """Setup default alert rules."""
        # High CPU usage
        self.add_alert(Alert(
            alert_id="high_cpu_usage",
            name="High CPU Usage",
            description="CPU usage is above 80%",
            severity=AlertSeverity.WARNING,
            condition=lambda metrics: metrics.get('cpu_percent', 0) > 80,
            threshold=80.0,
            notification_channels=['slack', 'email']
        ))
        
        # High memory usage
        self.add_alert(Alert(
            alert_id="high_memory_usage",
            name="High Memory Usage",
            description="Memory usage is above 85%",
            severity=AlertSeverity.WARNING,
            condition=lambda metrics: metrics.get('memory_percent', 0) > 85,
            threshold=85.0,
            notification_channels=['slack', 'email']
        ))
        
        # High error rate
        self.add_alert(Alert(
            alert_id="high_error_rate",
            name="High Error Rate",
            description="Error rate is above 5%",
            severity=AlertSeverity.CRITICAL,
            condition=lambda metrics: metrics.get('error_rate', 0) > 5,
            threshold=5.0,
            notification_channels=['pagerduty', 'slack', 'email']
        ))
    
    def add_alert(self, alert: Alert):
        """Add a new alert rule."""
        self.alerts.append(alert)
        logger.info(f"Added alert: {alert.name}")
    
    def remove_alert(self, alert_id: str):
        """Remove an alert rule."""
        self.alerts = [alert for alert in self.alerts if alert.alert_id != alert_id]
        logger.info(f"Removed alert: {alert_id}")
    
    async def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert conditions."""
        for alert in self.alerts:
            if not alert.enabled:
                continue
            
            # Check cooldown
            if (alert.last_triggered and 
                datetime.now() - alert.last_triggered < timedelta(seconds=alert.cooldown)):
                continue
            
            # Check condition
            if alert.condition(metrics):
                await self._trigger_alert(alert, metrics)
                alert.last_triggered = datetime.now()
    
    async def _trigger_alert(self, alert: Alert, metrics: Dict[str, Any]):
        """Trigger an alert notification."""
        message = f"ALERT: {alert.name}\n{alert.description}\nSeverity: {alert.severity.value}\nMetrics: {metrics}"
        
        logger.warning(f"Alert triggered: {alert.name}", 
                      alert_id=alert.alert_id,
                      severity=alert.severity.value,
                      metrics=metrics)
        
        # Send notifications
        for channel in alert.notification_channels:
            if channel in self.notification_clients:
                try:
                    await self._send_notification(channel, message, alert)
                except Exception as e:
                    logger.error(f"Failed to send notification via {channel}: {e}")
    
    async def _send_notification(self, channel: str, message: str, alert: Alert):
        """Send notification via specified channel."""
        if channel == 'pagerduty':
            # PagerDuty notification
            pass
        elif channel == 'slack':
            # Slack notification
            pass
        elif channel == 'discord':
            # Discord notification
            pass
        elif channel == 'telegram':
            # Telegram notification
            pass
        elif channel == 'email':
            # Email notification
            await self._send_email_notification(message, alert)


class AdvancedMonitoringSystem:
    """Advanced monitoring system with comprehensive features."""
    
    def __init__(self, config: MonitoringConfig):
        
    """__init__ function."""
self.config = config
        self.metrics_collector = MetricsCollector(config)
        self.alert_manager = AlertManager(config)
        self.logger = StructuredLogger(config)
        self.tracer = DistributedTracer(config)
        
        self.running = False
        self.collection_task = None
        self.alert_task = None
    
    async def start(self) -> Any:
        """Start the monitoring system."""
        self.running = True
        
        # Start metrics collection
        self.collection_task = asyncio.create_task(self._metrics_collection_loop())
        
        # Start alert checking
        self.alert_task = asyncio.create_task(self._alert_checking_loop())
        
        logger.info("Advanced monitoring system started")
    
    async def stop(self) -> Any:
        """Stop the monitoring system."""
        self.running = False
        
        if self.collection_task:
            self.collection_task.cancel()
        if self.alert_task:
            self.alert_task.cancel()
        
        logger.info("Advanced monitoring system stopped")
    
    async def _metrics_collection_loop(self) -> Any:
        """Main metrics collection loop."""
        while self.running:
            try:
                # Collect metrics
                system_metrics = await self.metrics_collector.collect_system_metrics()
                app_metrics = await self.metrics_collector.collect_application_metrics()
                
                # Store metrics
                await self.metrics_collector.store_metrics(system_metrics, app_metrics)
                
                # Log metrics
                self.logger.log_metric("system_cpu_percent", system_metrics.cpu_percent)
                self.logger.log_metric("system_memory_percent", system_metrics.memory_percent)
                self.logger.log_metric("system_disk_percent", system_metrics.disk_percent)
                
                await asyncio.sleep(self.config.metrics_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _alert_checking_loop(self) -> Any:
        """Main alert checking loop."""
        while self.running:
            try:
                # Collect current metrics
                system_metrics = await self.metrics_collector.collect_system_metrics()
                app_metrics = await self.metrics_collector.collect_application_metrics()
                
                # Combine metrics
                combined_metrics = {
                    "cpu_percent": system_metrics.cpu_percent,
                    "memory_percent": system_metrics.memory_percent,
                    "disk_percent": system_metrics.disk_percent,
                    "error_rate": (app_metrics.error_count / max(app_metrics.request_count, 1)) * 100,
                    "success_rate": app_metrics.success_rate,
                    "active_connections": app_metrics.active_connections
                }
                
                # Check alerts
                await self.alert_manager.check_alerts(combined_metrics)
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in alert checking: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        return {
            "status": "healthy" if self.running else "stopped",
            "timestamp": datetime.now().isoformat(),
            "metrics_collection": "running" if self.collection_task and not self.collection_task.done() else "stopped",
            "alert_checking": "running" if self.alert_task and not self.alert_task.done() else "stopped",
            "storage_backends": {
                "elasticsearch": self.metrics_collector.elasticsearch_client is not None,
                "redis": self.metrics_collector.redis_client is not None,
                "kafka": self.metrics_collector.kafka_producer is not None,
                "influxdb": self.metrics_collector.influxdb_client is not None
            }
        }


async def main():
    """Main function for testing the monitoring system."""
    # Create configuration
    config = MonitoringConfig(
        prometheus_enabled=True,
        elasticsearch_enabled=False,
        redis_enabled=False,
        kafka_enabled=False,
        influxdb_enabled=False,
        alerting_enabled=True
    )
    
    # Create monitoring system
    monitoring_system = AdvancedMonitoringSystem(config)
    
    # Start monitoring
    await monitoring_system.start()
    
    try:
        # Run for some time
        await asyncio.sleep(300)  # 5 minutes
    finally:
        # Stop monitoring
        await monitoring_system.stop()


match __name__:
    case "__main__":
    asyncio.run(main()) 