"""
Metrics Collector - Recolección de métricas y monitoring
=========================================================
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Recolecta métricas de uso y performance.
    """
    
    def __init__(self, metrics_dir: str = "data/metrics"):
        """
        Inicializar recolector de métricas.
        
        Args:
            metrics_dir: Directorio para almacenar métricas
        """
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            "requests": [],
            "improvements": [],
            "papers_used": defaultdict(int),
            "errors": [],
            "performance": []
        }
    
    def record_request(
        self,
        endpoint: str,
        method: str = "POST",
        duration_ms: Optional[float] = None,
        success: bool = True
    ):
        """
        Registra una petición.
        
        Args:
            endpoint: Endpoint llamado
            method: Método HTTP
            duration_ms: Duración en milisegundos
            success: Si fue exitosa
        """
        request_metric = {
            "timestamp": datetime.now().isoformat(),
            "endpoint": endpoint,
            "method": method,
            "duration_ms": duration_ms,
            "success": success
        }
        
        self.metrics["requests"].append(request_metric)
        
        # Mantener solo últimas 1000 peticiones
        if len(self.metrics["requests"]) > 1000:
            self.metrics["requests"] = self.metrics["requests"][-1000:]
    
    def record_improvement(
        self,
        file_path: str,
        improvements_applied: int,
        papers_used: List[str],
        duration_ms: Optional[float] = None
    ):
        """
        Registra una mejora aplicada.
        
        Args:
            file_path: Archivo mejorado
            improvements_applied: Número de mejoras
            papers_used: Papers usados
            duration_ms: Duración en milisegundos
        """
        improvement_metric = {
            "timestamp": datetime.now().isoformat(),
            "file_path": file_path,
            "improvements_applied": improvements_applied,
            "papers_used": papers_used,
            "duration_ms": duration_ms
        }
        
        self.metrics["improvements"].append(improvement_metric)
        
        # Contar papers usados
        for paper in papers_used:
            self.metrics["papers_used"][paper] += 1
        
        # Mantener solo últimas 500 mejoras
        if len(self.metrics["improvements"]) > 500:
            self.metrics["improvements"] = self.metrics["improvements"][-500:]
    
    def record_error(self, error_type: str, error_message: str, endpoint: Optional[str] = None):
        """
        Registra un error.
        
        Args:
            error_type: Tipo de error
            error_message: Mensaje de error
            endpoint: Endpoint donde ocurrió (opcional)
        """
        error_metric = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "endpoint": endpoint
        }
        
        self.metrics["errors"].append(error_metric)
        
        # Mantener solo últimos 100 errores
        if len(self.metrics["errors"]) > 100:
            self.metrics["errors"] = self.metrics["errors"][-100:]
    
    def record_performance(
        self,
        operation: str,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Registra métricas de performance.
        
        Args:
            operation: Operación realizada
            duration_ms: Duración en milisegundos
            metadata: Metadata adicional
        """
        perf_metric = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "duration_ms": duration_ms,
            "metadata": metadata or {}
        }
        
        self.metrics["performance"].append(perf_metric)
        
        # Mantener solo últimas 200 métricas
        if len(self.metrics["performance"]) > 200:
            self.metrics["performance"] = self.metrics["performance"][-200:]
    
    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Obtiene estadísticas de las últimas N horas.
        
        Args:
            hours: Número de horas a analizar
            
        Returns:
            Estadísticas
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Filtrar métricas recientes
        recent_requests = [
            r for r in self.metrics["requests"]
            if datetime.fromisoformat(r["timestamp"]) > cutoff
        ]
        
        recent_improvements = [
            i for i in self.metrics["improvements"]
            if datetime.fromisoformat(i["timestamp"]) > cutoff
        ]
        
        recent_errors = [
            e for e in self.metrics["errors"]
            if datetime.fromisoformat(e["timestamp"]) > cutoff
        ]
        
        # Calcular estadísticas
        total_requests = len(recent_requests)
        successful_requests = sum(1 for r in recent_requests if r.get("success", False))
        
        avg_duration = None
        durations = [r.get("duration_ms") for r in recent_requests if r.get("duration_ms")]
        if durations:
            avg_duration = sum(durations) / len(durations)
        
        total_improvements = len(recent_improvements)
        total_improvements_applied = sum(
            i.get("improvements_applied", 0) for i in recent_improvements
        )
        
        # Papers más usados
        top_papers = sorted(
            self.metrics["papers_used"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "period_hours": hours,
            "requests": {
                "total": total_requests,
                "successful": successful_requests,
                "failed": total_requests - successful_requests,
                "success_rate": round(successful_requests / total_requests * 100, 2) if total_requests > 0 else 0,
                "avg_duration_ms": round(avg_duration, 2) if avg_duration else None
            },
            "improvements": {
                "total": total_improvements,
                "total_applied": total_improvements_applied,
                "avg_per_file": round(total_improvements_applied / total_improvements, 2) if total_improvements > 0 else 0
            },
            "errors": {
                "total": len(recent_errors),
                "by_type": self._count_errors_by_type(recent_errors)
            },
            "top_papers": [
                {"paper": paper, "usage_count": count}
                for paper, count in top_papers
            ]
        }
    
    def _count_errors_by_type(self, errors: List[Dict[str, Any]]) -> Dict[str, int]:
        """Cuenta errores por tipo"""
        error_counts = defaultdict(int)
        for error in errors:
            error_type = error.get("error_type", "unknown")
            error_counts[error_type] += 1
        return dict(error_counts)
    
    def save_metrics(self, filename: Optional[str] = None):
        """Guarda métricas en disco"""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"metrics_{timestamp}.json"
            
            filepath = self.metrics_dir / filename
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.metrics, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Métricas guardadas: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error guardando métricas: {e}")
            return None
    
    def load_metrics(self, filename: str):
        """Carga métricas desde disco"""
        try:
            filepath = self.metrics_dir / filename
            
            if not filepath.exists():
                logger.warning(f"Archivo no existe: {filepath}")
                return False
            
            with open(filepath, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)
            
            logger.info(f"Métricas cargadas: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando métricas: {e}")
            return False




