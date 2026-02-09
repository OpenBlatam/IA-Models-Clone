"""
Integración de todos los componentes del sistema de pipelines
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PipelineSystem:
    """Sistema integrado de pipelines con todas las funcionalidades"""
    
    def __init__(
        self,
        enable_cache: bool = True,
        enable_metrics: bool = True,
        enable_monitoring: bool = True,
        cache_ttl: float = 300.0,
        monitor_interval: float = 60.0,
        alert_threshold: float = 0.8
    ):
        """
        Inicializa el sistema completo de pipelines.
        
        Args:
            enable_cache: Habilitar sistema de cache
            enable_metrics: Habilitar sistema de métricas
            enable_monitoring: Habilitar monitoreo automático
            cache_ttl: TTL del cache en segundos
            monitor_interval: Intervalo de monitoreo en segundos
            alert_threshold: Umbral de alertas
        """
        self.enable_cache = enable_cache
        self.enable_metrics = enable_metrics
        self.enable_monitoring = enable_monitoring
        
        # Inicializar componentes
        self._cache = None
        self._metrics = None
        self._monitor = None
        
        if enable_cache:
            try:
                from .pipelines_cache import get_cache
                self._cache = get_cache(default_ttl=cache_ttl)
                logger.info("Cache enabled")
            except ImportError:
                logger.warning("Cache module not available")
        
        if enable_metrics:
            try:
                from .pipelines_metrics import get_metrics_collector
                self._metrics = get_metrics_collector()
                logger.info("Metrics enabled")
            except ImportError:
                logger.warning("Metrics module not available")
        
        if enable_monitoring:
            try:
                from .pipelines_monitor import get_monitor
                self._monitor = get_monitor(
                    check_interval=monitor_interval,
                    alert_threshold=alert_threshold
                )
                logger.info("Monitoring enabled")
            except ImportError:
                logger.warning("Monitoring module not available")
    
    def check_health(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Realiza un check de salud completo.
        
        Args:
            use_cache: Usar cache si está disponible
            
        Returns:
            Reporte de salud completo
        """
        from .pipelines import check_compatibility, get_import_statistics
        
        # Intentar obtener del cache
        if use_cache and self._cache:
            cached = self._cache.get("health_check")
            if cached is not None:
                if self._metrics:
                    self._metrics.increment_counter("health_check_cache_hits")
                return cached
        
        # Medir tiempo
        if self._metrics:
            with self._metrics.time_operation("health_check"):
                compatibility = check_compatibility()
                statistics = get_import_statistics()
        else:
            compatibility = check_compatibility()
            statistics = get_import_statistics()
        
        result = {
            "compatibility": compatibility,
            "statistics": statistics,
            "cache_enabled": self._cache is not None,
            "metrics_enabled": self._metrics is not None,
            "monitoring_enabled": self._monitor is not None
        }
        
        # Guardar en cache
        if use_cache and self._cache:
            self._cache.set("health_check", result, ttl=60.0)
        
        if self._metrics:
            self._metrics.increment_counter("health_check_total")
        
        return result
    
    def start_monitoring(self) -> None:
        """Inicia el monitoreo automático"""
        if self._monitor:
            self._monitor.start_monitoring()
            logger.info("Monitoring started")
        else:
            logger.warning("Monitoring not available")
    
    def stop_monitoring(self) -> None:
        """Detiene el monitoreo automático"""
        if self._monitor:
            self._monitor.stop_monitoring()
            logger.info("Monitoring stopped")
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """
        Obtiene un reporte completo del sistema.
        
        Returns:
            Reporte con todas las métricas y estadísticas
        """
        report = {
            "health": self.check_health(),
            "components": {
                "cache": {},
                "metrics": {},
                "monitoring": {}
            }
        }
        
        # Estadísticas de cache
        if self._cache:
            report["components"]["cache"] = self._cache.get_stats()
        
        # Estadísticas de métricas
        if self._metrics:
            report["components"]["metrics"] = self._metrics.get_stats()
        
        # Estado de monitoreo
        if self._monitor:
            summary = self._monitor.get_summary()
            report["components"]["monitoring"] = summary
            
            # Alertas activas
            active_alerts = self._monitor.get_active_alerts()
            report["components"]["monitoring"]["active_alerts"] = [
                {
                    "level": alert.level.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in active_alerts
            ]
        
        return report
    
    def export_report(
        self,
        output_path: Path,
        format: str = "json"
    ) -> None:
        """
        Exporta un reporte completo.
        
        Args:
            output_path: Ruta del archivo de salida
            format: Formato (json, txt, md)
        """
        from .pipelines_utils import export_compatibility_report
        
        report = self.get_comprehensive_report()
        
        if format == "json":
            import json
            output_path.write_text(json.dumps(report, indent=2, default=str))
        else:
            # Usar utilidad existente para formato texto/markdown
            export_compatibility_report(output_path, format=format)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado actual del sistema.
        
        Returns:
            Estado del sistema
        """
        status = {
            "cache": {
                "enabled": self._cache is not None,
                "stats": self._cache.get_stats() if self._cache else None
            },
            "metrics": {
                "enabled": self._metrics is not None,
                "stats": self._metrics.get_stats() if self._metrics else None
            },
            "monitoring": {
                "enabled": self._monitor is not None,
                "active": self._monitor._running if self._monitor else False,
                "summary": self._monitor.get_summary() if self._monitor else None
            }
        }
        
        return status


# Instancia global del sistema
_global_system: Optional[PipelineSystem] = None


def get_pipeline_system(
    enable_cache: bool = True,
    enable_metrics: bool = True,
    enable_monitoring: bool = False,
    **kwargs
) -> PipelineSystem:
    """
    Obtiene la instancia global del sistema de pipelines.
    
    Args:
        enable_cache: Habilitar cache
        enable_metrics: Habilitar métricas
        enable_monitoring: Habilitar monitoreo
        **kwargs: Argumentos adicionales para PipelineSystem
        
    Returns:
        PipelineSystem instance
    """
    global _global_system
    
    if _global_system is None:
        _global_system = PipelineSystem(
            enable_cache=enable_cache,
            enable_metrics=enable_metrics,
            enable_monitoring=enable_monitoring,
            **kwargs
        )
    
    return _global_system


def quick_system_check() -> Dict[str, Any]:
    """
    Realiza un check rápido del sistema completo.
    
    Returns:
        Resumen del estado del sistema
    """
    system = get_pipeline_system()
    health = system.check_health()
    
    return {
        "status": health["compatibility"]["status"],
        "health_score": health["compatibility"]["health_score"],
        "coverage": health["statistics"]["coverage_percentage"],
        "cache_hit_rate": (
            system._cache.get_stats()["hit_rate"]
            if system._cache else None
        ),
        "monitoring_active": (
            system._monitor._running
            if system._monitor else False
        )
    }

