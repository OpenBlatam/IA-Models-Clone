"""
Real-time Analyzer for AI History Comparison System
Sistema de análisis en tiempo real para monitoreo continuo del rendimiento de IA
"""

import asyncio
import json
import logging
import redis
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Niveles de alerta"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class MetricType(Enum):
    """Tipos de métricas"""
    QUALITY_SCORE = "quality_score"
    READABILITY = "readability"
    ORIGINALITY = "originality"
    PROCESSING_TIME = "processing_time"
    USER_SATISFACTION = "user_satisfaction"
    ERROR_RATE = "error_rate"

@dataclass
class RealtimeMetric:
    """Métrica en tiempo real"""
    metric_type: MetricType
    value: float
    timestamp: datetime
    document_id: Optional[str] = None
    query: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Alerta del sistema"""
    id: str
    level: AlertLevel
    title: str
    message: str
    metric_type: MetricType
    threshold: float
    current_value: float
    timestamp: datetime
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrendAnalysis:
    """Análisis de tendencias"""
    metric_type: MetricType
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0.0 - 1.0
    change_rate: float
    confidence: float
    time_window: str
    data_points: int

class RealtimeAnalyzer:
    """
    Analizador en tiempo real para monitoreo continuo del sistema de IA
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        analysis_interval: int = 10,  # segundos
        alert_thresholds: Optional[Dict[str, float]] = None,
        websocket_port: int = 8765,
        max_history_size: int = 1000
    ):
        self.redis_url = redis_url
        self.analysis_interval = analysis_interval
        self.websocket_port = websocket_port
        self.max_history_size = max_history_size
        
        # Configurar umbrales de alerta
        self.alert_thresholds = alert_thresholds or {
            "quality_score_low": 0.3,
            "quality_score_critical": 0.2,
            "readability_low": 0.4,
            "originality_low": 0.3,
            "processing_time_high": 30.0,
            "error_rate_high": 0.1,
            "user_satisfaction_low": 2.0
        }
        
        # Inicializar Redis
        self.redis_client = redis.from_url(redis_url)
        
        # Almacenamiento de métricas
        self.metrics_history: Dict[MetricType, deque] = {
            metric_type: deque(maxlen=max_history_size)
            for metric_type in MetricType
        }
        
        # Alertas activas
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Análisis de tendencias
        self.trend_analysis: Dict[MetricType, TrendAnalysis] = {}
        
        # Callbacks para eventos
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.metric_callbacks: List[Callable[[RealtimeMetric], None]] = []
        
        # Estado del sistema
        self.is_running = False
        self.websocket_clients: set = set()
        
        # Estadísticas
        self.stats = {
            "total_metrics_processed": 0,
            "total_alerts_generated": 0,
            "active_alerts_count": 0,
            "last_analysis_time": None,
            "system_uptime": datetime.now()
        }
    
    async def start(self):
        """Iniciar el analizador en tiempo real"""
        logger.info("Iniciando analizador en tiempo real")
        
        self.is_running = True
        
        # Iniciar tareas en paralelo
        tasks = [
            asyncio.create_task(self._metrics_processing_loop()),
            asyncio.create_task(self._trend_analysis_loop()),
            asyncio.create_task(self._alert_monitoring_loop()),
            asyncio.create_task(self._websocket_server()),
            asyncio.create_task(self._cleanup_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error en analizador en tiempo real: {e}")
        finally:
            self.is_running = False
    
    async def stop(self):
        """Detener el analizador"""
        logger.info("Deteniendo analizador en tiempo real")
        self.is_running = False
    
    async def add_metric(self, metric: RealtimeMetric):
        """
        Agregar una nueva métrica para análisis
        
        Args:
            metric: Métrica a agregar
        """
        try:
            # Agregar a historial
            self.metrics_history[metric.metric_type].append(metric)
            
            # Actualizar estadísticas
            self.stats["total_metrics_processed"] += 1
            
            # Ejecutar callbacks
            for callback in self.metric_callbacks:
                try:
                    await callback(metric)
                except Exception as e:
                    logger.error(f"Error en callback de métrica: {e}")
            
            # Almacenar en Redis para persistencia
            await self._store_metric_in_redis(metric)
            
            # Verificar alertas inmediatamente
            await self._check_immediate_alerts(metric)
            
        except Exception as e:
            logger.error(f"Error agregando métrica: {e}")
    
    async def _store_metric_in_redis(self, metric: RealtimeMetric):
        """Almacenar métrica en Redis"""
        try:
            metric_data = {
                "metric_type": metric.metric_type.value,
                "value": metric.value,
                "timestamp": metric.timestamp.isoformat(),
                "document_id": metric.document_id,
                "query": metric.query,
                "metadata": json.dumps(metric.metadata)
            }
            
            # Almacenar en lista ordenada por timestamp
            key = f"metrics:{metric.metric_type.value}"
            self.redis_client.zadd(key, {json.dumps(metric_data): metric.timestamp.timestamp()})
            
            # Mantener solo los últimos 1000 elementos
            self.redis_client.zremrangebyrank(key, 0, -1001)
            
        except Exception as e:
            logger.error(f"Error almacenando métrica en Redis: {e}")
    
    async def _metrics_processing_loop(self):
        """Loop de procesamiento de métricas"""
        while self.is_running:
            try:
                await asyncio.sleep(self.analysis_interval)
                
                # Procesar métricas acumuladas
                await self._process_accumulated_metrics()
                
                # Actualizar estadísticas
                self.stats["last_analysis_time"] = datetime.now()
                
            except Exception as e:
                logger.error(f"Error en loop de procesamiento: {e}")
                await asyncio.sleep(5)
    
    async def _process_accumulated_metrics(self):
        """Procesar métricas acumuladas"""
        for metric_type, metrics in self.metrics_history.items():
            if not metrics:
                continue
            
            # Calcular estadísticas básicas
            values = [m.value for m in metrics]
            recent_values = values[-10:] if len(values) >= 10 else values
            
            if recent_values:
                avg_value = np.mean(recent_values)
                std_value = np.std(recent_values)
                min_value = np.min(recent_values)
                max_value = np.max(recent_values)
                
                # Verificar anomalías
                await self._detect_anomalies(metric_type, recent_values, avg_value, std_value)
                
                # Actualizar análisis de tendencias
                await self._update_trend_analysis(metric_type, recent_values)
    
    async def _detect_anomalies(
        self, 
        metric_type: MetricType, 
        values: List[float], 
        mean: float, 
        std: float
    ):
        """Detectar anomalías en las métricas"""
        if std == 0:
            return
        
        # Detectar valores atípicos usando regla 3-sigma
        threshold = 3 * std
        
        for i, value in enumerate(values):
            if abs(value - mean) > threshold:
                # Crear alerta de anomalía
                alert = Alert(
                    id=f"anomaly_{metric_type.value}_{datetime.now().timestamp()}",
                    level=AlertLevel.WARNING,
                    title=f"Anomalía detectada en {metric_type.value}",
                    message=f"Valor atípico detectado: {value:.3f} (promedio: {mean:.3f}, std: {std:.3f})",
                    metric_type=metric_type,
                    threshold=threshold,
                    current_value=value,
                    timestamp=datetime.now(),
                    metadata={
                        "anomaly_type": "statistical_outlier",
                        "mean": mean,
                        "std": std,
                        "position": i
                    }
                )
                
                await self._create_alert(alert)
    
    async def _update_trend_analysis(self, metric_type: MetricType, values: List[float]):
        """Actualizar análisis de tendencias"""
        if len(values) < 5:
            return
        
        # Calcular tendencia usando regresión lineal simple
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calcular pendiente
        slope = np.polyfit(x, y, 1)[0]
        
        # Determinar dirección de tendencia
        if slope > 0.01:
            trend_direction = "increasing"
        elif slope < -0.01:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        # Calcular fuerza de tendencia (R²)
        correlation_matrix = np.corrcoef(x, y)
        correlation = correlation_matrix[0, 1]
        trend_strength = abs(correlation)
        
        # Calcular tasa de cambio
        change_rate = (values[-1] - values[0]) / len(values) if len(values) > 1 else 0
        
        # Calcular confianza
        confidence = min(trend_strength * len(values) / 10, 1.0)
        
        self.trend_analysis[metric_type] = TrendAnalysis(
            metric_type=metric_type,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            change_rate=change_rate,
            confidence=confidence,
            time_window=f"{len(values)} puntos",
            data_points=len(values)
        )
    
    async def _trend_analysis_loop(self):
        """Loop de análisis de tendencias"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Análisis cada minuto
                
                # Analizar tendencias críticas
                await self._analyze_critical_trends()
                
            except Exception as e:
                logger.error(f"Error en análisis de tendencias: {e}")
                await asyncio.sleep(10)
    
    async def _analyze_critical_trends(self):
        """Analizar tendencias críticas"""
        for metric_type, trend in self.trend_analysis.items():
            # Verificar tendencias preocupantes
            if trend.confidence > 0.7:  # Tendencia confiable
                if trend.trend_direction == "decreasing" and trend.change_rate < -0.05:
                    # Tendencia descendente significativa
                    alert = Alert(
                        id=f"trend_{metric_type.value}_{datetime.now().timestamp()}",
                        level=AlertLevel.WARNING,
                        title=f"Tendencia descendente en {metric_type.value}",
                        message=f"Tendencia descendente detectada: {trend.change_rate:.3f} por punto de datos",
                        metric_type=metric_type,
                        threshold=0.05,
                        current_value=trend.change_rate,
                        timestamp=datetime.now(),
                        metadata={
                            "trend_type": "declining",
                            "confidence": trend.confidence,
                            "data_points": trend.data_points
                        }
                    )
                    
                    await self._create_alert(alert)
                
                elif trend.trend_direction == "increasing" and trend.change_rate > 0.1:
                    # Tendencia ascendente significativa (puede ser buena o mala dependiendo de la métrica)
                    if metric_type in [MetricType.ERROR_RATE, MetricType.PROCESSING_TIME]:
                        alert = Alert(
                            id=f"trend_{metric_type.value}_{datetime.now().timestamp()}",
                            level=AlertLevel.WARNING,
                            title=f"Tendencia ascendente preocupante en {metric_type.value}",
                            message=f"Tendencia ascendente detectada: {trend.change_rate:.3f} por punto de datos",
                            metric_type=metric_type,
                            threshold=0.1,
                            current_value=trend.change_rate,
                            timestamp=datetime.now(),
                            metadata={
                                "trend_type": "increasing_negative",
                                "confidence": trend.confidence,
                                "data_points": trend.data_points
                            }
                        )
                        
                        await self._create_alert(alert)
    
    async def _alert_monitoring_loop(self):
        """Loop de monitoreo de alertas"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Verificar alertas cada 30 segundos
                
                # Verificar alertas activas
                await self._check_active_alerts()
                
                # Limpiar alertas resueltas
                await self._cleanup_resolved_alerts()
                
            except Exception as e:
                logger.error(f"Error en monitoreo de alertas: {e}")
                await asyncio.sleep(10)
    
    async def _check_immediate_alerts(self, metric: RealtimeMetric):
        """Verificar alertas inmediatas basadas en umbrales"""
        metric_type = metric.metric_type
        value = metric.value
        
        # Verificar umbrales específicos
        if metric_type == MetricType.QUALITY_SCORE:
            if value < self.alert_thresholds["quality_score_critical"]:
                await self._create_alert(Alert(
                    id=f"quality_critical_{datetime.now().timestamp()}",
                    level=AlertLevel.CRITICAL,
                    title="Calidad crítica de documento",
                    message=f"Calidad del documento extremadamente baja: {value:.3f}",
                    metric_type=metric_type,
                    threshold=self.alert_thresholds["quality_score_critical"],
                    current_value=value,
                    timestamp=datetime.now(),
                    metadata={"document_id": metric.document_id}
                ))
            elif value < self.alert_thresholds["quality_score_low"]:
                await self._create_alert(Alert(
                    id=f"quality_low_{datetime.now().timestamp()}",
                    level=AlertLevel.WARNING,
                    title="Calidad baja de documento",
                    message=f"Calidad del documento por debajo del umbral: {value:.3f}",
                    metric_type=metric_type,
                    threshold=self.alert_thresholds["quality_score_low"],
                    current_value=value,
                    timestamp=datetime.now(),
                    metadata={"document_id": metric.document_id}
                ))
        
        elif metric_type == MetricType.PROCESSING_TIME:
            if value > self.alert_thresholds["processing_time_high"]:
                await self._create_alert(Alert(
                    id=f"processing_slow_{datetime.now().timestamp()}",
                    level=AlertLevel.WARNING,
                    title="Tiempo de procesamiento alto",
                    message=f"Tiempo de procesamiento excesivo: {value:.2f}s",
                    metric_type=metric_type,
                    threshold=self.alert_thresholds["processing_time_high"],
                    current_value=value,
                    timestamp=datetime.now(),
                    metadata={"document_id": metric.document_id}
                ))
    
    async def _check_active_alerts(self):
        """Verificar alertas activas"""
        current_time = datetime.now()
        
        for alert_id, alert in list(self.active_alerts.items()):
            # Verificar si la alerta ha estado activa por mucho tiempo
            time_since_alert = (current_time - alert.timestamp).total_seconds()
            
            if time_since_alert > 3600:  # 1 hora
                # Escalar alerta
                if alert.level == AlertLevel.WARNING:
                    alert.level = AlertLevel.CRITICAL
                    alert.title = f"[ESCALADA] {alert.title}"
                    alert.message = f"Alerta escalada después de {time_since_alert/60:.0f} minutos: {alert.message}"
                    
                    # Notificar escalación
                    await self._notify_alert_escalation(alert)
    
    async def _cleanup_resolved_alerts(self):
        """Limpiar alertas resueltas"""
        resolved_alerts = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.resolved
        ]
        
        for alert_id in resolved_alerts:
            alert = self.active_alerts.pop(alert_id)
            self.alert_history.append(alert)
    
    async def _create_alert(self, alert: Alert):
        """Crear nueva alerta"""
        # Verificar si ya existe una alerta similar
        similar_alert = await self._find_similar_alert(alert)
        
        if similar_alert:
            # Actualizar alerta existente
            similar_alert.timestamp = alert.timestamp
            similar_alert.current_value = alert.current_value
            similar_alert.metadata.update(alert.metadata)
        else:
            # Crear nueva alerta
            self.active_alerts[alert.id] = alert
            self.stats["total_alerts_generated"] += 1
            self.stats["active_alerts_count"] = len(self.active_alerts)
            
            # Notificar alerta
            await self._notify_new_alert(alert)
    
    async def _find_similar_alert(self, new_alert: Alert) -> Optional[Alert]:
        """Encontrar alerta similar existente"""
        for alert in self.active_alerts.values():
            if (alert.metric_type == new_alert.metric_type and 
                alert.level == new_alert.level and
                not alert.resolved):
                return alert
        return None
    
    async def _notify_new_alert(self, alert: Alert):
        """Notificar nueva alerta"""
        logger.warning(f"Nueva alerta: {alert.title} - {alert.message}")
        
        # Ejecutar callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error en callback de alerta: {e}")
        
        # Notificar clientes WebSocket
        await self._broadcast_to_websockets({
            "type": "new_alert",
            "alert": {
                "id": alert.id,
                "level": alert.level.value,
                "title": alert.title,
                "message": alert.message,
                "metric_type": alert.metric_type.value,
                "timestamp": alert.timestamp.isoformat()
            }
        })
    
    async def _notify_alert_escalation(self, alert: Alert):
        """Notificar escalación de alerta"""
        logger.critical(f"Alerta escalada: {alert.title} - {alert.message}")
        
        # Notificar clientes WebSocket
        await self._broadcast_to_websockets({
            "type": "alert_escalation",
            "alert": {
                "id": alert.id,
                "level": alert.level.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat()
            }
        })
    
    async def _websocket_server(self):
        """Servidor WebSocket para notificaciones en tiempo real"""
        async def handle_client(websocket, path):
            self.websocket_clients.add(websocket)
            logger.info(f"Cliente WebSocket conectado: {websocket.remote_address}")
            
            try:
                # Enviar estado inicial
                await websocket.send(json.dumps({
                    "type": "initial_state",
                    "data": {
                        "active_alerts": len(self.active_alerts),
                        "trends": {t.metric_type.value: {
                            "direction": t.trend_direction,
                            "strength": t.trend_strength,
                            "confidence": t.confidence
                        } for t in self.trend_analysis.values()},
                        "stats": self.stats
                    }
                }))
                
                # Mantener conexión
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self._handle_websocket_message(websocket, data)
                    except json.JSONDecodeError:
                        await websocket.send(json.dumps({"error": "Invalid JSON"}))
                    except Exception as e:
                        logger.error(f"Error procesando mensaje WebSocket: {e}")
                        
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.websocket_clients.discard(websocket)
                logger.info(f"Cliente WebSocket desconectado: {websocket.remote_address}")
        
        try:
            server = await websockets.serve(handle_client, "0.0.0.0", self.websocket_port)
            logger.info(f"Servidor WebSocket iniciado en puerto {self.websocket_port}")
            await server.wait_closed()
        except Exception as e:
            logger.error(f"Error en servidor WebSocket: {e}")
    
    async def _handle_websocket_message(self, websocket, data: Dict[str, Any]):
        """Manejar mensajes de WebSocket"""
        message_type = data.get("type")
        
        if message_type == "get_metrics":
            # Enviar métricas recientes
            metrics_data = {}
            for metric_type, metrics in self.metrics_history.items():
                if metrics:
                    recent_metrics = list(metrics)[-10:]  # Últimos 10
                    metrics_data[metric_type.value] = [
                        {
                            "value": m.value,
                            "timestamp": m.timestamp.isoformat()
                        }
                        for m in recent_metrics
                    ]
            
            await websocket.send(json.dumps({
                "type": "metrics_data",
                "data": metrics_data
            }))
        
        elif message_type == "get_alerts":
            # Enviar alertas activas
            alerts_data = [
                {
                    "id": alert.id,
                    "level": alert.level.value,
                    "title": alert.title,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in self.active_alerts.values()
            ]
            
            await websocket.send(json.dumps({
                "type": "alerts_data",
                "data": alerts_data
            }))
    
    async def _broadcast_to_websockets(self, message: Dict[str, Any]):
        """Transmitir mensaje a todos los clientes WebSocket"""
        if not self.websocket_clients:
            return
        
        message_json = json.dumps(message)
        disconnected_clients = set()
        
        for websocket in self.websocket_clients:
            try:
                await websocket.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(websocket)
            except Exception as e:
                logger.error(f"Error enviando mensaje WebSocket: {e}")
                disconnected_clients.add(websocket)
        
        # Limpiar clientes desconectados
        self.websocket_clients -= disconnected_clients
    
    async def _cleanup_loop(self):
        """Loop de limpieza y mantenimiento"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Limpieza cada 5 minutos
                
                # Limpiar historial de métricas antiguas
                cutoff_time = datetime.now() - timedelta(hours=24)
                
                for metric_type, metrics in self.metrics_history.items():
                    # Remover métricas más antiguas que 24 horas
                    while metrics and metrics[0].timestamp < cutoff_time:
                        metrics.popleft()
                
                # Limpiar alertas antiguas del historial
                self.alert_history = [
                    alert for alert in self.alert_history
                    if alert.timestamp > cutoff_time
                ]
                
            except Exception as e:
                logger.error(f"Error en loop de limpieza: {e}")
                await asyncio.sleep(60)
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Agregar callback para alertas"""
        self.alert_callbacks.append(callback)
    
    def add_metric_callback(self, callback: Callable[[RealtimeMetric], None]):
        """Agregar callback para métricas"""
        self.metric_callbacks.append(callback)
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolver alerta"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            self.stats["active_alerts_count"] = len([
                a for a in self.active_alerts.values() if not a.resolved
            ])
            return True
        return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtener estado del sistema"""
        return {
            "is_running": self.is_running,
            "active_alerts": len(self.active_alerts),
            "total_alerts_generated": self.stats["total_alerts_generated"],
            "metrics_processed": self.stats["total_metrics_processed"],
            "websocket_clients": len(self.websocket_clients),
            "uptime": (datetime.now() - self.stats["system_uptime"]).total_seconds(),
            "last_analysis": self.stats["last_analysis_time"].isoformat() if self.stats["last_analysis_time"] else None,
            "trends": {
                metric_type.value: {
                    "direction": trend.trend_direction,
                    "strength": trend.trend_strength,
                    "confidence": trend.confidence,
                    "change_rate": trend.change_rate
                }
                for metric_type, trend in self.trend_analysis.items()
            }
        }
    
    async def export_realtime_report(self) -> Dict[str, Any]:
        """Exportar reporte en tiempo real"""
        return {
            "timestamp": datetime.now().isoformat(),
            "system_status": self.get_system_status(),
            "active_alerts": [
                {
                    "id": alert.id,
                    "level": alert.level.value,
                    "title": alert.title,
                    "message": alert.message,
                    "metric_type": alert.metric_type.value,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in self.active_alerts.values()
            ],
            "recent_metrics": {
                metric_type.value: [
                    {
                        "value": metric.value,
                        "timestamp": metric.timestamp.isoformat()
                    }
                    for metric in list(metrics)[-10:]  # Últimos 10
                ]
                for metric_type, metrics in self.metrics_history.items()
                if metrics
            },
            "trend_analysis": {
                metric_type.value: {
                    "direction": trend.trend_direction,
                    "strength": trend.trend_strength,
                    "confidence": trend.confidence,
                    "change_rate": trend.change_rate,
                    "data_points": trend.data_points
                }
                for metric_type, trend in self.trend_analysis.items()
            }
        }



























