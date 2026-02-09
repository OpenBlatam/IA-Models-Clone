"""
Quality Assurance Utilities
============================

Utilidades para asegurar calidad del código y sistema.
"""

import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityMetric:
    """Métrica de calidad."""
    name: str
    value: float
    threshold: float
    unit: str = ""
    description: str = ""


class QualityChecker:
    """
    Verificador de calidad del sistema.
    
    Verifica que el sistema cumple con estándares de calidad.
    """
    
    def __init__(self):
        """Inicializar verificador de calidad."""
        self.metrics: List[QualityMetric] = []
        self.checks: List[Callable[[], bool]] = []
    
    def add_metric(
        self,
        name: str,
        value: float,
        threshold: float,
        unit: str = "",
        description: str = ""
    ) -> None:
        """Agregar métrica de calidad."""
        metric = QualityMetric(
            name=name,
            value=value,
            threshold=threshold,
            unit=unit,
            description=description
        )
        self.metrics.append(metric)
    
    def add_check(self, check_func: Callable[[], bool], name: str = "") -> None:
        """Agregar check de calidad."""
        self.checks.append((name or check_func.__name__, check_func))
    
    def check_all(self) -> Dict[str, Any]:
        """
        Ejecutar todos los checks.
        
        Returns:
            Reporte de calidad
        """
        results = {
            "metrics": [],
            "checks": [],
            "overall_quality": 0.0
        }
        
        # Verificar métricas
        passed_metrics = 0
        for metric in self.metrics:
            passed = metric.value >= metric.threshold
            results["metrics"].append({
                "name": metric.name,
                "value": metric.value,
                "threshold": metric.threshold,
                "unit": metric.unit,
                "passed": passed,
                "description": metric.description
            })
            if passed:
                passed_metrics += 1
        
        # Ejecutar checks
        passed_checks = 0
        for name, check_func in self.checks:
            try:
                passed = check_func()
                results["checks"].append({
                    "name": name,
                    "passed": passed
                })
                if passed:
                    passed_checks += 1
            except Exception as e:
                logger.error(f"Error in quality check {name}: {e}")
                results["checks"].append({
                    "name": name,
                    "passed": False,
                    "error": str(e)
                })
        
        # Calcular calidad general
        total_items = len(self.metrics) + len(self.checks)
        if total_items > 0:
            results["overall_quality"] = (passed_metrics + passed_checks) / total_items
        
        results["summary"] = {
            "total_metrics": len(self.metrics),
            "passed_metrics": passed_metrics,
            "total_checks": len(self.checks),
            "passed_checks": passed_checks,
            "overall_quality": results["overall_quality"]
        }
        
        return results


def check_performance_quality() -> Dict[str, Any]:
    """
    Verificar calidad de performance.
    
    Returns:
        Reporte de calidad de performance
    """
    from .metrics import get_metrics_collector
    
    collector = get_metrics_collector()
    all_metrics = collector.get_all_metrics()
    
    quality_checker = QualityChecker()
    
    # Verificar tiempos de optimización
    opt_time_metric = all_metrics.get("trajectory_optimization.total_time")
    if opt_time_metric:
        avg_time = opt_time_metric.get("average", 0.0)
        quality_checker.add_metric(
            "optimization_time",
            avg_time,
            threshold=1.0,  # Menos de 1 segundo
            unit="seconds",
            description="Average trajectory optimization time"
        )
    
    # Verificar cache hit rate
    cache_hits = all_metrics.get("trajectory_optimization.cache_hits", {}).get("latest", 0)
    cache_misses = all_metrics.get("trajectory_optimization.cache_misses", {}).get("latest", 0)
    total = cache_hits + cache_misses
    if total > 0:
        hit_rate = cache_hits / total
        quality_checker.add_metric(
            "cache_hit_rate",
            hit_rate,
            threshold=0.5,  # Al menos 50% hit rate
            unit="ratio",
            description="Cache hit rate"
        )
    
    return quality_checker.check_all()


def check_system_health_quality() -> Dict[str, Any]:
    """
    Verificar calidad de salud del sistema.
    
    Returns:
        Reporte de calidad de salud
    """
    from .health_check import get_health_check_system
    
    health_system = get_health_check_system()
    report = health_system.get_health_report()
    
    quality_checker = QualityChecker()
    
    # Verificar estado general
    status = report.get("status", "unknown")
    quality_checker.add_check(
        lambda: status == "healthy",
        "system_health"
    )
    
    # Verificar checks individuales
    checks = report.get("checks", {})
    for check_name, check_data in checks.items():
        check_status = check_data.get("status", "unknown")
        quality_checker.add_check(
            lambda s=check_status: s == "healthy",
            f"check_{check_name}"
        )
    
    return quality_checker.check_all()






