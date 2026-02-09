"""
Endpoints para métricas y estadísticas avanzadas

Incluye:
- Métricas de generación en tiempo real
- Estadísticas del sistema
- Métricas de rendimiento
- Dashboard de monitoreo
- Alertas y notificaciones
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Query, Depends, HTTPException, status
from datetime import datetime, timedelta

try:
    from ...utils.monitoring import GenerationMetrics, SystemMonitor, PerformanceTracker
    from ...utils.alerting import get_alert_manager
except ImportError:
    from utils.monitoring import GenerationMetrics, SystemMonitor, PerformanceTracker
    from utils.alerting import get_alert_manager

from ..dependencies import MetricsServiceDep

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/metrics",
    tags=["metrics"],
    responses={
        200: {"description": "Métricas obtenidas exitosamente"},
        500: {"description": "Error al obtener métricas"}
    }
)

# Instancias globales de monitoreo
_generation_metrics = GenerationMetrics()
_system_monitor = SystemMonitor()
_performance_tracker = PerformanceTracker(window_size=1000)
_alert_manager = get_alert_manager()


@router.get("/stats")
async def get_stats(
    days: int = Query(7, ge=1, le=365, description="Número de días"),
    metrics_service: Optional[MetricsServiceDep] = Depends()
) -> Dict[str, Any]:
    """
    Obtiene estadísticas generales del sistema.
    
    Combina métricas del servicio de métricas (si está disponible)
    con métricas en tiempo real del sistema de monitoreo.
    """
    try:
        stats = {}
        
        # Métricas del servicio (si está disponible)
        if metrics_service:
            try:
                service_stats = metrics_service.get_stats(days=days)
                stats["service_stats"] = service_stats
            except Exception as e:
                logger.warning(f"Could not get service stats: {e}")
        
        # Métricas en tiempo real
        real_time_metrics = {
            "generation_metrics": _generation_metrics.get_stats(),
            "system_info": _system_monitor.get_system_info(),
            "performance_trend": _performance_tracker.get_performance_trend(),
            "timestamp": datetime.now().isoformat()
        }
        stats["real_time"] = real_time_metrics
        
        # Verificar alertas
        _alert_manager.check_metrics(real_time_metrics)
        
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stats: {str(e)}"
        )


@router.get("/user/{user_id}")
async def get_user_stats(
    user_id: str,
    days: int = Query(30, ge=1, le=365, description="Número de días"),
    metrics_service: Optional[MetricsServiceDep] = Depends()
) -> Dict[str, Any]:
    """
    Obtiene estadísticas de un usuario específico.
    """
    try:
        if metrics_service:
            return metrics_service.get_user_stats(user_id, days=days)
        else:
            return {
                "user_id": user_id,
                "message": "Metrics service not available",
                "days": days
            }
    except Exception as e:
        logger.error(f"Error getting user stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving user stats: {str(e)}"
        )


@router.get("/realtime")
async def get_realtime_metrics() -> Dict[str, Any]:
    """
    Obtiene métricas en tiempo real del sistema.
    
    Incluye:
    - Métricas de generación (tiempos, cache hits, errores)
    - Información del sistema (CPU, memoria, GPU)
    - Tendencias de rendimiento
    """
    try:
        return {
            "generation_metrics": _generation_metrics.get_stats(),
            "system_info": _system_monitor.get_system_info(),
            "performance_trend": _performance_tracker.get_performance_trend(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting realtime metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving realtime metrics: {str(e)}"
        )


@router.get("/dashboard")
async def get_dashboard_data(
    hours: int = Query(24, ge=1, le=168, description="Horas de datos históricos")
) -> Dict[str, Any]:
    """
    Obtiene datos completos para un dashboard de monitoreo.
    
    Incluye:
    - Métricas de generación
    - Estado del sistema
    - Tendencias de rendimiento
    - Alertas activas
    - Resumen de actividad
    """
    try:
        generation_stats = _generation_metrics.get_stats()
        system_info = _system_monitor.get_system_info()
        performance_trend = _performance_tracker.get_performance_trend()
        
        # Calcular resumen
        total_generations = generation_stats.get("total_generations", 0)
        avg_time = generation_stats.get("avg_generation_time_seconds", 0)
        cache_hit_rate = generation_stats.get("cache_hit_rate_percent", 0)
        
        # Determinar estado del sistema
        system_status = "healthy"
        if system_info.get("memory_percent", 0) > 90:
            system_status = "warning"
        if system_info.get("memory_percent", 0) > 95:
            system_status = "critical"
        
        return {
            "summary": {
                "total_generations": total_generations,
                "avg_generation_time_seconds": round(avg_time, 3),
                "cache_hit_rate_percent": round(cache_hit_rate, 2),
                "system_status": system_status,
                "timestamp": datetime.now().isoformat()
            },
            "generation_metrics": generation_stats,
            "system_info": system_info,
            "performance_trend": performance_trend,
            "alerts": [alert.to_dict() for alert in _alert_manager.get_active_alerts()],
            "time_range_hours": hours
        }
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving dashboard data: {str(e)}"
        )


@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """
    Obtiene métricas detalladas de rendimiento.
    
    Incluye:
    - Tendencias de rendimiento
    - Análisis de cache
    - Distribución de tiempos de generación
    """
    try:
        generation_stats = _generation_metrics.get_stats()
        performance_trend = _performance_tracker.get_performance_trend()
        
        return {
            "performance_trend": performance_trend,
            "generation_times": {
                "avg": generation_stats.get("avg_generation_time_seconds", 0),
                "min": generation_stats.get("min_generation_time_seconds", 0),
                "max": generation_stats.get("max_generation_time_seconds", 0)
            },
            "cache_performance": {
                "hits": generation_stats.get("cache_hits", 0),
                "misses": generation_stats.get("cache_misses", 0),
                "hit_rate_percent": generation_stats.get("cache_hit_rate_percent", 0)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving performance metrics: {str(e)}"
        )


@router.get("/health/detailed")
async def get_detailed_health() -> Dict[str, Any]:
    """
    Health check detallado con métricas del sistema.
    """
    try:
        system_info = _system_monitor.get_system_info()
        generation_stats = _generation_metrics.get_stats()
        
        # Determinar estado de salud
        health_status = "healthy"
        issues = []
        
        if system_info.get("memory_percent", 0) > 90:
            health_status = "degraded"
            issues.append("High memory usage")
        
        if system_info.get("cpu_percent", 0) > 90:
            health_status = "degraded"
            issues.append("High CPU usage")
        
        error_count = sum(generation_stats.get("error_counts", {}).values())
        if error_count > 100:
            health_status = "degraded"
            issues.append(f"High error count: {error_count}")
        
        return {
            "status": health_status,
            "system_info": system_info,
            "generation_stats": {
                "total": generation_stats.get("total_generations", 0),
                "errors": error_count
            },
            "issues": issues,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting detailed health: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def _get_active_alerts(system_info: Dict[str, Any], generation_stats: Dict[str, Any]) -> list:
    """
    Genera alertas basadas en métricas del sistema.
    """
    alerts = []
    
    # Alerta de memoria
    if system_info.get("memory_percent", 0) > 90:
        alerts.append({
            "level": "warning" if system_info.get("memory_percent", 0) < 95 else "critical",
            "type": "memory",
            "message": f"High memory usage: {system_info.get('memory_percent', 0):.1f}%",
            "timestamp": datetime.now().isoformat()
        })
    
    # Alerta de CPU
    if system_info.get("cpu_percent", 0) > 90:
        alerts.append({
            "level": "warning",
            "type": "cpu",
            "message": f"High CPU usage: {system_info.get('cpu_percent', 0):.1f}%",
            "timestamp": datetime.now().isoformat()
        })
    
    # Alerta de errores
    error_count = sum(generation_stats.get("error_counts", {}).values())
    if error_count > 50:
        alerts.append({
            "level": "warning" if error_count < 100 else "critical",
            "type": "errors",
            "message": f"High error count: {error_count}",
            "timestamp": datetime.now().isoformat()
        })
    
    # Alerta de rendimiento
    avg_time = generation_stats.get("avg_generation_time_seconds", 0)
    if avg_time > 60:
        alerts.append({
            "level": "warning",
            "type": "performance",
            "message": f"Slow generation times: {avg_time:.2f}s average",
            "timestamp": datetime.now().isoformat()
        })
    
    return alerts


# Funciones helper para registrar métricas (usadas por otros módulos)
def record_generation_metric(duration: float, text_length: int, cache_hit: bool = False):
    """Registra una métrica de generación"""
    _generation_metrics.record_generation(duration, text_length, cache_hit)
    _performance_tracker.track(duration, cache_hit)


def record_error_metric(error_type: str):
    """Registra un error"""
    _generation_metrics.record_error(error_type)


@router.get("/alerts")
async def get_alerts(
    hours: int = Query(24, ge=1, le=168, description="Horas de historial"),
    level: Optional[str] = Query(None, description="Filtrar por nivel (info, warning, critical, error)")
) -> Dict[str, Any]:
    """
    Obtiene alertas del sistema.
    
    Incluye alertas activas e historial de alertas.
    """
    try:
        from ...utils.alerting import AlertLevel
        
        alert_level = None
        if level:
            try:
                alert_level = AlertLevel(level)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid alert level: {level}"
                )
        
        history = _alert_manager.get_alert_history(hours=hours, level=alert_level)
        
        return {
            "active_alerts": [alert.to_dict() for alert in _alert_manager.get_active_alerts()],
            "history": [alert.to_dict() for alert in history],
            "stats": _alert_manager.get_stats(),
            "time_range_hours": hours
        }
    except Exception as e:
        logger.error(f"Error getting alerts: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving alerts: {str(e)}"
        )


@router.post("/alerts/{alert_key}/resolve")
async def resolve_alert(alert_key: str) -> Dict[str, Any]:
    """
    Resuelve una alerta activa.
    """
    try:
        _alert_manager.resolve_alert(alert_key)
        return {
            "message": f"Alert {alert_key} resolved",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error resolving alert: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error resolving alert: {str(e)}"
        )

