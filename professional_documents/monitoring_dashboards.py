"""
Dashboards de Monitoreo Avanzados para el Sistema de Documentos Profesionales

Este módulo implementa dashboards completos de monitoreo con métricas en tiempo real,
alertas inteligentes, y visualizaciones avanzadas para el sistema.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import redis
from fastapi import WebSocket, WebSocketDisconnect
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Tipos de métricas"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    """Severidad de alertas"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricData:
    """Datos de métrica"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str]
    type: MetricType

@dataclass
class Alert:
    """Alerta del sistema"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class DashboardWidget:
    """Widget del dashboard"""
    id: str
    title: str
    type: str
    config: Dict[str, Any]
    data: Optional[Dict[str, Any]] = None
    last_updated: Optional[datetime] = None

class MetricsCollector:
    """Recolector de métricas"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.registry = CollectorRegistry()
        self.metrics = {}
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Configurar métricas de Prometheus"""
        # Métricas del sistema
        self.system_cpu = Gauge('system_cpu_usage_percent', 'CPU usage percentage', registry=self.registry)
        self.system_memory = Gauge('system_memory_usage_percent', 'Memory usage percentage', registry=self.registry)
        self.system_disk = Gauge('system_disk_usage_percent', 'Disk usage percentage', registry=self.registry)
        self.system_load = Gauge('system_load_average', 'System load average', registry=self.registry)
        
        # Métricas de la aplicación
        self.app_requests = Counter('app_requests_total', 'Total requests', ['method', 'endpoint', 'status'], registry=self.registry)
        self.app_response_time = Histogram('app_response_time_seconds', 'Response time', ['endpoint'], registry=self.registry)
        self.app_errors = Counter('app_errors_total', 'Total errors', ['type', 'endpoint'], registry=self.registry)
        self.app_active_users = Gauge('app_active_users', 'Active users', registry=self.registry)
        
        # Métricas de base de datos
        self.db_connections = Gauge('db_connections_active', 'Active database connections', registry=self.registry)
        self.db_queries = Counter('db_queries_total', 'Total database queries', ['type'], registry=self.registry)
        self.db_query_time = Histogram('db_query_time_seconds', 'Database query time', ['type'], registry=self.registry)
        
        # Métricas de caché
        self.cache_hits = Counter('cache_hits_total', 'Total cache hits', ['cache_type'], registry=self.registry)
        self.cache_misses = Counter('cache_misses_total', 'Total cache misses', ['cache_type'], registry=self.registry)
        self.cache_size = Gauge('cache_size_bytes', 'Cache size in bytes', ['cache_type'], registry=self.registry)
        
        # Métricas de IA
        self.ai_predictions = Counter('ai_predictions_total', 'Total AI predictions', ['model', 'type'], registry=self.registry)
        self.ai_prediction_time = Histogram('ai_prediction_time_seconds', 'AI prediction time', ['model'], registry=self.registry)
        self.ai_model_accuracy = Gauge('ai_model_accuracy', 'AI model accuracy', ['model'], registry=self.registry)
        
        # Métricas de servicios avanzados
        self.quantum_tasks = Gauge('quantum_tasks_pending', 'Pending quantum tasks', registry=self.registry)
        self.quantum_task_time = Histogram('quantum_task_time_seconds', 'Quantum task execution time', registry=self.registry)
        self.blockchain_transactions = Counter('blockchain_transactions_total', 'Total blockchain transactions', ['network'], registry=self.registry)
        self.blockchain_confirmation_time = Histogram('blockchain_confirmation_time_seconds', 'Blockchain confirmation time', ['network'], registry=self.registry)
        self.metaverse_sessions = Gauge('metaverse_active_sessions', 'Active metaverse sessions', registry=self.registry)
        self.metaverse_users = Gauge('metaverse_concurrent_users', 'Concurrent metaverse users', registry=self.registry)
        self.workflow_executions = Counter('workflow_executions_total', 'Total workflow executions', ['workflow_type'], registry=self.registry)
        self.workflow_execution_time = Histogram('workflow_execution_time_seconds', 'Workflow execution time', ['workflow_type'], registry=self.registry)
        
        # Métricas de seguridad
        self.security_events = Counter('security_events_total', 'Total security events', ['type', 'severity'], registry=self.registry)
        self.failed_logins = Counter('failed_logins_total', 'Total failed login attempts', ['ip'], registry=self.registry)
        self.blocked_ips = Gauge('blocked_ips_count', 'Number of blocked IPs', registry=self.registry)
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Recopilar métricas del sistema"""
        try:
            import psutil
            
            # Métricas del sistema
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
            
            # Actualizar métricas de Prometheus
            self.system_cpu.set(cpu_percent)
            self.system_memory.set(memory.percent)
            self.system_disk.set(disk.percent)
            self.system_load.set(load_avg)
            
            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "disk_usage": disk.percent,
                "load_average": load_avg,
                "memory_total": memory.total,
                "memory_available": memory.available,
                "disk_total": disk.total,
                "disk_free": disk.free
            }
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    async def collect_application_metrics(self) -> Dict[str, Any]:
        """Recopilar métricas de la aplicación"""
        try:
            # Obtener métricas de Redis
            total_requests = await self.redis.get("total_requests") or 0
            total_errors = await self.redis.get("total_errors") or 0
            active_users = await self.redis.scard("active_users")
            avg_response_time = await self.redis.get("avg_response_time") or 0
            
            # Obtener métricas por endpoint
            endpoint_metrics = {}
            endpoints = await self.redis.keys("endpoint_metrics:*")
            for endpoint_key in endpoints:
                endpoint_name = endpoint_key.decode().replace("endpoint_metrics:", "")
                metrics_data = await self.redis.hgetall(endpoint_key)
                endpoint_metrics[endpoint_name] = {
                    "requests": int(metrics_data.get(b"requests", 0)),
                    "errors": int(metrics_data.get(b"errors", 0)),
                    "avg_response_time": float(metrics_data.get(b"avg_response_time", 0))
                }
            
            return {
                "total_requests": int(total_requests),
                "total_errors": int(total_errors),
                "active_users": active_users,
                "avg_response_time": float(avg_response_time),
                "error_rate": (int(total_errors) / max(int(total_requests), 1)) * 100,
                "endpoints": endpoint_metrics
            }
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
            return {}
    
    async def collect_database_metrics(self) -> Dict[str, Any]:
        """Recopilar métricas de base de datos"""
        try:
            # Obtener métricas de base de datos
            db_metrics = await self.redis.hgetall("db_metrics")
            
            return {
                "active_connections": int(db_metrics.get(b"active_connections", 0)),
                "total_queries": int(db_metrics.get(b"total_queries", 0)),
                "slow_queries": int(db_metrics.get(b"slow_queries", 0)),
                "avg_query_time": float(db_metrics.get(b"avg_query_time", 0)),
                "cache_hit_rate": float(db_metrics.get(b"cache_hit_rate", 0))
            }
            
        except Exception as e:
            logger.error(f"Error collecting database metrics: {e}")
            return {}
    
    async def collect_cache_metrics(self) -> Dict[str, Any]:
        """Recopilar métricas de caché"""
        try:
            # Obtener métricas de caché
            cache_hits = await self.redis.get("cache_hits") or 0
            cache_misses = await self.redis.get("cache_misses") or 0
            cache_size = await self.redis.memory_usage("cache_data")
            
            total_requests = int(cache_hits) + int(cache_misses)
            hit_rate = (int(cache_hits) / max(total_requests, 1)) * 100
            
            return {
                "hits": int(cache_hits),
                "misses": int(cache_misses),
                "hit_rate": hit_rate,
                "size_bytes": cache_size or 0,
                "size_mb": (cache_size or 0) / (1024 * 1024)
            }
            
        except Exception as e:
            logger.error(f"Error collecting cache metrics: {e}")
            return {}
    
    async def collect_ai_metrics(self) -> Dict[str, Any]:
        """Recopilar métricas de IA"""
        try:
            ai_metrics = {}
            
            # Obtener métricas por modelo
            models = ["gpt-4", "claude-3", "dall-e-3", "whisper", "bert"]
            for model in models:
                model_metrics = await self.redis.hgetall(f"ai_model_metrics:{model}")
                ai_metrics[model] = {
                    "predictions": int(model_metrics.get(b"predictions", 0)),
                    "avg_response_time": float(model_metrics.get(b"avg_response_time", 0)),
                    "accuracy": float(model_metrics.get(b"accuracy", 0)),
                    "errors": int(model_metrics.get(b"errors", 0))
                }
            
            return ai_metrics
            
        except Exception as e:
            logger.error(f"Error collecting AI metrics: {e}")
            return {}
    
    async def collect_advanced_service_metrics(self) -> Dict[str, Any]:
        """Recopilar métricas de servicios avanzados"""
        try:
            metrics = {}
            
            # Métricas cuánticas
            quantum_tasks = await self.redis.llen("quantum_task_queue")
            quantum_metrics = await self.redis.hgetall("quantum_metrics")
            metrics["quantum"] = {
                "pending_tasks": quantum_tasks,
                "completed_tasks": int(quantum_metrics.get(b"completed_tasks", 0)),
                "avg_execution_time": float(quantum_metrics.get(b"avg_execution_time", 0)),
                "success_rate": float(quantum_metrics.get(b"success_rate", 0))
            }
            
            # Métricas blockchain
            blockchain_metrics = await self.redis.hgetall("blockchain_metrics")
            metrics["blockchain"] = {
                "total_transactions": int(blockchain_metrics.get(b"total_transactions", 0)),
                "pending_transactions": int(blockchain_metrics.get(b"pending_transactions", 0)),
                "avg_confirmation_time": float(blockchain_metrics.get(b"avg_confirmation_time", 0)),
                "gas_usage": float(blockchain_metrics.get(b"gas_usage", 0))
            }
            
            # Métricas metaverso
            metaverse_sessions = await self.redis.scard("metaverse_sessions")
            metaverse_users = await self.redis.scard("metaverse_users")
            metaverse_metrics = await self.redis.hgetall("metaverse_metrics")
            metrics["metaverse"] = {
                "active_sessions": metaverse_sessions,
                "concurrent_users": metaverse_users,
                "vr_sessions": int(metaverse_metrics.get(b"vr_sessions", 0)),
                "ar_sessions": int(metaverse_metrics.get(b"ar_sessions", 0)),
                "avg_session_duration": float(metaverse_metrics.get(b"avg_session_duration", 0))
            }
            
            # Métricas de flujos de trabajo
            workflow_metrics = await self.redis.hgetall("workflow_metrics")
            metrics["workflow"] = {
                "total_executions": int(workflow_metrics.get(b"total_executions", 0)),
                "active_workflows": int(workflow_metrics.get(b"active_workflows", 0)),
                "failed_workflows": int(workflow_metrics.get(b"failed_workflows", 0)),
                "avg_execution_time": float(workflow_metrics.get(b"avg_execution_time", 0)),
                "success_rate": float(workflow_metrics.get(b"success_rate", 0))
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting advanced service metrics: {e}")
            return {}
    
    async def collect_all_metrics(self) -> Dict[str, Any]:
        """Recopilar todas las métricas"""
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "system": await self.collect_system_metrics(),
                "application": await self.collect_application_metrics(),
                "database": await self.collect_database_metrics(),
                "cache": await self.collect_cache_metrics(),
                "ai": await self.collect_ai_metrics(),
                "advanced_services": await self.collect_advanced_service_metrics()
            }
            
            # Guardar métricas en Redis para historial
            await self.redis.lpush("metrics_history", json.dumps(metrics))
            await self.redis.ltrim("metrics_history", 0, 999)  # Mantener solo las últimas 1000 métricas
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting all metrics: {e}")
            return {}

class AlertManager:
    """Gestor de alertas"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.alerts = []
        self.alert_rules = self._load_alert_rules()
    
    def _load_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """Cargar reglas de alertas"""
        return {
            "high_cpu_usage": {
                "metric": "system.cpu_usage",
                "threshold": 80,
                "severity": AlertSeverity.WARNING,
                "title": "High CPU Usage",
                "description": "CPU usage is above 80%"
            },
            "critical_cpu_usage": {
                "metric": "system.cpu_usage",
                "threshold": 95,
                "severity": AlertSeverity.CRITICAL,
                "title": "Critical CPU Usage",
                "description": "CPU usage is above 95%"
            },
            "high_memory_usage": {
                "metric": "system.memory_usage",
                "threshold": 85,
                "severity": AlertSeverity.WARNING,
                "title": "High Memory Usage",
                "description": "Memory usage is above 85%"
            },
            "critical_memory_usage": {
                "metric": "system.memory_usage",
                "threshold": 95,
                "severity": AlertSeverity.CRITICAL,
                "title": "Critical Memory Usage",
                "description": "Memory usage is above 95%"
            },
            "high_disk_usage": {
                "metric": "system.disk_usage",
                "threshold": 90,
                "severity": AlertSeverity.WARNING,
                "title": "High Disk Usage",
                "description": "Disk usage is above 90%"
            },
            "high_error_rate": {
                "metric": "application.error_rate",
                "threshold": 5,
                "severity": AlertSeverity.WARNING,
                "title": "High Error Rate",
                "description": "Application error rate is above 5%"
            },
            "critical_error_rate": {
                "metric": "application.error_rate",
                "threshold": 10,
                "severity": AlertSeverity.CRITICAL,
                "title": "Critical Error Rate",
                "description": "Application error rate is above 10%"
            },
            "slow_response_time": {
                "metric": "application.avg_response_time",
                "threshold": 2.0,
                "severity": AlertSeverity.WARNING,
                "title": "Slow Response Time",
                "description": "Average response time is above 2 seconds"
            },
            "low_cache_hit_rate": {
                "metric": "cache.hit_rate",
                "threshold": 70,
                "severity": AlertSeverity.WARNING,
                "title": "Low Cache Hit Rate",
                "description": "Cache hit rate is below 70%",
                "comparison": "less_than"
            }
        }
    
    async def check_alerts(self, metrics: Dict[str, Any]) -> List[Alert]:
        """Verificar alertas basadas en métricas"""
        try:
            new_alerts = []
            
            for rule_id, rule in self.alert_rules.items():
                metric_path = rule["metric"].split(".")
                current_value = self._get_metric_value(metrics, metric_path)
                
                if current_value is None:
                    continue
                
                threshold = rule["threshold"]
                comparison = rule.get("comparison", "greater_than")
                
                should_alert = False
                if comparison == "greater_than" and current_value > threshold:
                    should_alert = True
                elif comparison == "less_than" and current_value < threshold:
                    should_alert = True
                elif comparison == "equals" and current_value == threshold:
                    should_alert = True
                
                if should_alert:
                    # Verificar si ya existe una alerta activa para esta regla
                    existing_alert = await self._get_active_alert(rule_id)
                    
                    if not existing_alert:
                        alert = Alert(
                            id=f"{rule_id}_{int(time.time())}",
                            title=rule["title"],
                            description=rule["description"],
                            severity=rule["severity"],
                            metric_name=rule["metric"],
                            threshold=threshold,
                            current_value=current_value,
                            timestamp=datetime.now()
                        )
                        
                        new_alerts.append(alert)
                        await self._save_alert(alert)
            
            # Verificar si hay alertas que se han resuelto
            await self._check_resolved_alerts(metrics)
            
            return new_alerts
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
            return []
    
    def _get_metric_value(self, metrics: Dict[str, Any], metric_path: List[str]) -> Optional[float]:
        """Obtener valor de métrica desde path anidado"""
        try:
            value = metrics
            for key in metric_path:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            
            if isinstance(value, (int, float)):
                return float(value)
            return None
            
        except Exception:
            return None
    
    async def _get_active_alert(self, rule_id: str) -> Optional[Alert]:
        """Obtener alerta activa para una regla"""
        try:
            alert_data = await self.redis.get(f"active_alert:{rule_id}")
            if alert_data:
                alert_dict = json.loads(alert_data)
                return Alert(**alert_dict)
            return None
        except Exception:
            return None
    
    async def _save_alert(self, alert: Alert):
        """Guardar alerta"""
        try:
            # Guardar como alerta activa
            await self.redis.setex(
                f"active_alert:{alert.metric_name}",
                3600,  # 1 hora
                json.dumps(asdict(alert))
            )
            
            # Agregar a historial de alertas
            await self.redis.lpush("alerts_history", json.dumps(asdict(alert)))
            await self.redis.ltrim("alerts_history", 0, 999)  # Mantener solo las últimas 1000 alertas
            
            # Enviar notificación si es crítica
            if alert.severity == AlertSeverity.CRITICAL:
                await self._send_critical_alert(alert)
            
        except Exception as e:
            logger.error(f"Error saving alert: {e}")
    
    async def _check_resolved_alerts(self, metrics: Dict[str, Any]):
        """Verificar alertas resueltas"""
        try:
            active_alerts = await self.redis.keys("active_alert:*")
            
            for alert_key in active_alerts:
                alert_data = await self.redis.get(alert_key)
                if alert_data:
                    alert_dict = json.loads(alert_data)
                    alert = Alert(**alert_dict)
                    
                    # Verificar si la condición ya no se cumple
                    metric_path = alert.metric_name.split(".")
                    current_value = self._get_metric_value(metrics, metric_path)
                    
                    if current_value is not None:
                        threshold = alert.threshold
                        comparison = "greater_than"  # Por defecto
                        
                        # Determinar si la alerta se ha resuelto
                        resolved = False
                        if comparison == "greater_than" and current_value <= threshold:
                            resolved = True
                        elif comparison == "less_than" and current_value >= threshold:
                            resolved = True
                        
                        if resolved:
                            alert.resolved = True
                            alert.resolved_at = datetime.now()
                            
                            # Actualizar en historial
                            await self.redis.lpush("alerts_history", json.dumps(asdict(alert)))
                            
                            # Eliminar alerta activa
                            await self.redis.delete(alert_key)
            
        except Exception as e:
            logger.error(f"Error checking resolved alerts: {e}")
    
    async def _send_critical_alert(self, alert: Alert):
        """Enviar alerta crítica"""
        try:
            # Implementar envío de alertas críticas (email, Slack, etc.)
            logger.critical(f"CRITICAL ALERT: {alert.title} - {alert.description}")
            
            # Notificar a administradores
            await self.redis.publish("critical_alerts", json.dumps(asdict(alert)))
            
        except Exception as e:
            logger.error(f"Error sending critical alert: {e}")
    
    async def get_active_alerts(self) -> List[Alert]:
        """Obtener alertas activas"""
        try:
            active_alerts = []
            alert_keys = await self.redis.keys("active_alert:*")
            
            for alert_key in alert_keys:
                alert_data = await self.redis.get(alert_key)
                if alert_data:
                    alert_dict = json.loads(alert_data)
                    alert = Alert(**alert_dict)
                    active_alerts.append(alert)
            
            return active_alerts
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    async def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Obtener historial de alertas"""
        try:
            alerts_data = await self.redis.lrange("alerts_history", 0, limit - 1)
            alerts = []
            
            for alert_data in alerts_data:
                alert_dict = json.loads(alert_data)
                alert = Alert(**alert_dict)
                alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting alert history: {e}")
            return []

class DashboardGenerator:
    """Generador de dashboards"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.metrics_collector = MetricsCollector(redis_client)
        self.alert_manager = AlertManager(redis_client)
    
    async def generate_system_overview_dashboard(self) -> Dict[str, Any]:
        """Generar dashboard de resumen del sistema"""
        try:
            metrics = await self.metrics_collector.collect_all_metrics()
            alerts = await self.alert_manager.get_active_alerts()
            
            # Calcular estado general del sistema
            system_health = self._calculate_system_health(metrics)
            
            # Generar widgets
            widgets = [
                await self._create_system_health_widget(metrics, system_health),
                await self._create_performance_widget(metrics),
                await self._create_alerts_widget(alerts),
                await self._create_usage_widget(metrics),
                await self._create_ai_performance_widget(metrics),
                await self._create_advanced_services_widget(metrics)
            ]
            
            return {
                "title": "System Overview Dashboard",
                "timestamp": datetime.now().isoformat(),
                "system_health": system_health,
                "widgets": widgets,
                "alerts": [asdict(alert) for alert in alerts]
            }
            
        except Exception as e:
            logger.error(f"Error generating system overview dashboard: {e}")
            return {"error": str(e)}
    
    def _calculate_system_health(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calcular salud del sistema"""
        try:
            system_metrics = metrics.get("system", {})
            app_metrics = metrics.get("application", {})
            
            health_score = 100
            issues = []
            
            # CPU
            cpu_usage = system_metrics.get("cpu_usage", 0)
            if cpu_usage > 95:
                health_score -= 30
                issues.append("Critical CPU usage")
            elif cpu_usage > 80:
                health_score -= 15
                issues.append("High CPU usage")
            
            # Memory
            memory_usage = system_metrics.get("memory_usage", 0)
            if memory_usage > 95:
                health_score -= 30
                issues.append("Critical memory usage")
            elif memory_usage > 80:
                health_score -= 15
                issues.append("High memory usage")
            
            # Disk
            disk_usage = system_metrics.get("disk_usage", 0)
            if disk_usage > 95:
                health_score -= 20
                issues.append("Critical disk usage")
            elif disk_usage > 85:
                health_score -= 10
                issues.append("High disk usage")
            
            # Error rate
            error_rate = app_metrics.get("error_rate", 0)
            if error_rate > 10:
                health_score -= 25
                issues.append("High error rate")
            elif error_rate > 5:
                health_score -= 10
                issues.append("Elevated error rate")
            
            # Response time
            response_time = app_metrics.get("avg_response_time", 0)
            if response_time > 5:
                health_score -= 20
                issues.append("Slow response time")
            elif response_time > 2:
                health_score -= 10
                issues.append("Elevated response time")
            
            # Determinar estado
            if health_score >= 90:
                status = "healthy"
            elif health_score >= 70:
                status = "warning"
            elif health_score >= 50:
                status = "critical"
            else:
                status = "down"
            
            return {
                "status": status,
                "score": health_score,
                "issues": issues,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating system health: {e}")
            return {"status": "unknown", "score": 0, "issues": ["Health calculation error"]}
    
    async def _create_system_health_widget(self, metrics: Dict[str, Any], system_health: Dict[str, Any]) -> DashboardWidget:
        """Crear widget de salud del sistema"""
        try:
            system_metrics = metrics.get("system", {})
            
            # Crear gráfico de uso de recursos
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('CPU Usage', 'Memory Usage', 'Disk Usage', 'Load Average'),
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}]]
            )
            
            # CPU
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=system_metrics.get("cpu_usage", 0),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "CPU %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "red"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ), row=1, col=1)
            
            # Memory
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=system_metrics.get("memory_usage", 0),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Memory %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "red"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ), row=1, col=2)
            
            # Disk
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=system_metrics.get("disk_usage", 0),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Disk %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "red"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 90}}
            ), row=2, col=1)
            
            # Load Average
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=system_metrics.get("load_average", 0),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Load Avg"},
                gauge={'axis': {'range': [None, 10]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 2], 'color': "lightgray"},
                                {'range': [2, 5], 'color': "yellow"},
                                {'range': [5, 10], 'color': "red"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 8}}
            ), row=2, col=2)
            
            fig.update_layout(height=400, showlegend=False)
            
            return DashboardWidget(
                id="system_health",
                title="System Health",
                type="plotly",
                config={"figure": fig.to_dict()},
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error creating system health widget: {e}")
            return DashboardWidget(
                id="system_health",
                title="System Health",
                type="error",
                config={"error": str(e)}
            )
    
    async def _create_performance_widget(self, metrics: Dict[str, Any]) -> DashboardWidget:
        """Crear widget de rendimiento"""
        try:
            app_metrics = metrics.get("application", {})
            
            # Crear gráfico de rendimiento
            fig = go.Figure()
            
            # Agregar métricas de rendimiento
            fig.add_trace(go.Bar(
                name="Requests/sec",
                x=["Current"],
                y=[app_metrics.get("total_requests", 0) / 3600],  # Requests por segundo
                marker_color="blue"
            ))
            
            fig.add_trace(go.Bar(
                name="Error Rate %",
                x=["Current"],
                y=[app_metrics.get("error_rate", 0)],
                marker_color="red"
            ))
            
            fig.update_layout(
                title="Application Performance",
                xaxis_title="Time",
                yaxis_title="Value",
                barmode="group"
            )
            
            return DashboardWidget(
                id="performance",
                title="Application Performance",
                type="plotly",
                config={"figure": fig.to_dict()},
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error creating performance widget: {e}")
            return DashboardWidget(
                id="performance",
                title="Application Performance",
                type="error",
                config={"error": str(e)}
            )
    
    async def _create_alerts_widget(self, alerts: List[Alert]) -> DashboardWidget:
        """Crear widget de alertas"""
        try:
            # Agrupar alertas por severidad
            alert_counts = {
                "critical": 0,
                "error": 0,
                "warning": 0,
                "info": 0
            }
            
            for alert in alerts:
                alert_counts[alert.severity.value] += 1
            
            # Crear gráfico de alertas
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(alert_counts.keys()),
                    values=list(alert_counts.values()),
                    hole=0.3,
                    marker_colors=["red", "orange", "yellow", "green"]
                )
            ])
            
            fig.update_layout(title="Active Alerts by Severity")
            
            return DashboardWidget(
                id="alerts",
                title="Active Alerts",
                type="plotly",
                config={"figure": fig.to_dict()},
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error creating alerts widget: {e}")
            return DashboardWidget(
                id="alerts",
                title="Active Alerts",
                type="error",
                config={"error": str(e)}
            )
    
    async def _create_usage_widget(self, metrics: Dict[str, Any]) -> DashboardWidget:
        """Crear widget de uso"""
        try:
            app_metrics = metrics.get("application", {})
            cache_metrics = metrics.get("cache", {})
            
            # Crear gráfico de uso
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Active Users', 'Cache Hit Rate'),
                specs=[[{"type": "indicator"}, {"type": "indicator"}]]
            )
            
            # Active Users
            fig.add_trace(go.Indicator(
                mode="number",
                value=app_metrics.get("active_users", 0),
                title={'text': "Active Users"},
                number={'font': {'size': 50}}
            ), row=1, col=1)
            
            # Cache Hit Rate
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=cache_metrics.get("hit_rate", 0),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Cache Hit Rate %"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 70], 'color': "red"},
                                {'range': [70, 90], 'color': "yellow"},
                                {'range': [90, 100], 'color': "green"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 70}}
            ), row=1, col=2)
            
            fig.update_layout(height=300, showlegend=False)
            
            return DashboardWidget(
                id="usage",
                title="Usage Metrics",
                type="plotly",
                config={"figure": fig.to_dict()},
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error creating usage widget: {e}")
            return DashboardWidget(
                id="usage",
                title="Usage Metrics",
                type="error",
                config={"error": str(e)}
            )
    
    async def _create_ai_performance_widget(self, metrics: Dict[str, Any]) -> DashboardWidget:
        """Crear widget de rendimiento de IA"""
        try:
            ai_metrics = metrics.get("ai", {})
            
            if not ai_metrics:
                return DashboardWidget(
                    id="ai_performance",
                    title="AI Performance",
                    type="empty",
                    config={"message": "No AI metrics available"}
                )
            
            # Crear gráfico de rendimiento de modelos de IA
            models = list(ai_metrics.keys())
            predictions = [ai_metrics[model].get("predictions", 0) for model in models]
            response_times = [ai_metrics[model].get("avg_response_time", 0) for model in models]
            accuracies = [ai_metrics[model].get("accuracy", 0) for model in models]
            
            fig = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Predictions', 'Response Time (s)', 'Accuracy %'),
                specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
            )
            
            # Predictions
            fig.add_trace(go.Bar(x=models, y=predictions, name="Predictions"), row=1, col=1)
            
            # Response Time
            fig.add_trace(go.Bar(x=models, y=response_times, name="Response Time"), row=1, col=2)
            
            # Accuracy
            fig.add_trace(go.Bar(x=models, y=accuracies, name="Accuracy"), row=1, col=3)
            
            fig.update_layout(height=400, showlegend=False)
            
            return DashboardWidget(
                id="ai_performance",
                title="AI Model Performance",
                type="plotly",
                config={"figure": fig.to_dict()},
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error creating AI performance widget: {e}")
            return DashboardWidget(
                id="ai_performance",
                title="AI Model Performance",
                type="error",
                config={"error": str(e)}
            )
    
    async def _create_advanced_services_widget(self, metrics: Dict[str, Any]) -> DashboardWidget:
        """Crear widget de servicios avanzados"""
        try:
            advanced_metrics = metrics.get("advanced_services", {})
            
            if not advanced_metrics:
                return DashboardWidget(
                    id="advanced_services",
                    title="Advanced Services",
                    type="empty",
                    config={"message": "No advanced services metrics available"}
                )
            
            # Crear gráfico de servicios avanzados
            services = []
            values = []
            
            # Quantum
            if "quantum" in advanced_metrics:
                services.append("Quantum Tasks")
                values.append(advanced_metrics["quantum"].get("pending_tasks", 0))
            
            # Blockchain
            if "blockchain" in advanced_metrics:
                services.append("Blockchain TX")
                values.append(advanced_metrics["blockchain"].get("total_transactions", 0))
            
            # Metaverse
            if "metaverse" in advanced_metrics:
                services.append("Metaverse Sessions")
                values.append(advanced_metrics["metaverse"].get("active_sessions", 0))
            
            # Workflow
            if "workflow" in advanced_metrics:
                services.append("Workflow Executions")
                values.append(advanced_metrics["workflow"].get("total_executions", 0))
            
            fig = go.Figure(data=[
                go.Bar(x=services, y=values, marker_color="purple")
            ])
            
            fig.update_layout(
                title="Advanced Services Activity",
                xaxis_title="Service",
                yaxis_title="Count"
            )
            
            return DashboardWidget(
                id="advanced_services",
                title="Advanced Services",
                type="plotly",
                config={"figure": fig.to_dict()},
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error creating advanced services widget: {e}")
            return DashboardWidget(
                id="advanced_services",
                title="Advanced Services",
                type="error",
                config={"error": str(e)}
            )

class RealTimeDashboard:
    """Dashboard en tiempo real"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.dashboard_generator = DashboardGenerator(redis_client)
        self.connected_clients = set()
    
    async def connect_websocket(self, websocket: WebSocket):
        """Conectar WebSocket para dashboard en tiempo real"""
        try:
            await websocket.accept()
            self.connected_clients.add(websocket)
            
            # Enviar datos iniciales
            dashboard_data = await self.dashboard_generator.generate_system_overview_dashboard()
            await websocket.send_text(json.dumps(dashboard_data))
            
            # Mantener conexión y enviar actualizaciones
            while True:
                try:
                    # Esperar mensaje del cliente
                    message = await websocket.receive_text()
                    
                    if message == "ping":
                        await websocket.send_text("pong")
                    elif message == "get_dashboard":
                        dashboard_data = await self.dashboard_generator.generate_system_overview_dashboard()
                        await websocket.send_text(json.dumps(dashboard_data))
                    
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    break
            
        except Exception as e:
            logger.error(f"Error connecting WebSocket: {e}")
        finally:
            self.connected_clients.discard(websocket)
    
    async def broadcast_dashboard_update(self):
        """Transmitir actualización del dashboard a todos los clientes conectados"""
        try:
            dashboard_data = await self.dashboard_generator.generate_system_overview_dashboard()
            
            # Enviar a todos los clientes conectados
            disconnected_clients = set()
            for websocket in self.connected_clients:
                try:
                    await websocket.send_text(json.dumps(dashboard_data))
                except Exception:
                    disconnected_clients.add(websocket)
            
            # Limpiar clientes desconectados
            self.connected_clients -= disconnected_clients
            
        except Exception as e:
            logger.error(f"Error broadcasting dashboard update: {e}")
    
    async def start_real_time_updates(self):
        """Iniciar actualizaciones en tiempo real"""
        try:
            while True:
                await self.broadcast_dashboard_update()
                await asyncio.sleep(30)  # Actualizar cada 30 segundos
        except Exception as e:
            logger.error(f"Error in real-time updates: {e}")

# Funciones de utilidad
async def get_metrics_summary(redis_client: redis.Redis) -> Dict[str, Any]:
    """Obtener resumen de métricas"""
    try:
        collector = MetricsCollector(redis_client)
        metrics = await collector.collect_all_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "system_health": "healthy" if metrics.get("system", {}).get("cpu_usage", 0) < 80 else "warning",
                "active_users": metrics.get("application", {}).get("active_users", 0),
                "total_requests": metrics.get("application", {}).get("total_requests", 0),
                "error_rate": metrics.get("application", {}).get("error_rate", 0),
                "cache_hit_rate": metrics.get("cache", {}).get("hit_rate", 0)
            },
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        return {"error": str(e)}

async def export_metrics_to_prometheus(redis_client: redis.Redis) -> str:
    """Exportar métricas a formato Prometheus"""
    try:
        collector = MetricsCollector(redis_client)
        await collector.collect_all_metrics()
        
        from prometheus_client import generate_latest
        return generate_latest(collector.registry).decode('utf-8')
    except Exception as e:
        logger.error(f"Error exporting metrics to Prometheus: {e}")
        return ""

# Configuración de dashboards por defecto
DEFAULT_DASHBOARD_CONFIG = {
    "refresh_interval": 30,
    "retention_days": 30,
    "widgets": [
        {"id": "system_health", "title": "System Health", "type": "gauge", "enabled": True},
        {"id": "performance", "title": "Performance", "type": "chart", "enabled": True},
        {"id": "alerts", "title": "Alerts", "type": "list", "enabled": True},
        {"id": "usage", "title": "Usage", "type": "metric", "enabled": True},
        {"id": "ai_performance", "title": "AI Performance", "type": "chart", "enabled": True},
        {"id": "advanced_services", "title": "Advanced Services", "type": "chart", "enabled": True}
    ],
    "alerts": {
        "enabled": True,
        "channels": ["email", "slack", "webhook"],
        "severity_levels": ["critical", "error", "warning"]
    }
}



























