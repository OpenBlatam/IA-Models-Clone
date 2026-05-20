"""
Analytics Engine - Motor de Análisis Avanzado
==============================================

Motor de análisis y reportes avanzados para proyectos generados.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """Motor de análisis avanzado"""

    def __init__(self, data_dir: Path = None):
        """
        Inicializa el motor de análisis.

        Args:
            data_dir: Directorio para almacenar datos de analytics
        """
        if data_dir is None:
            data_dir = Path("analytics")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            "projects_by_type": defaultdict(int),
            "projects_by_framework": defaultdict(int),
            "projects_by_author": defaultdict(int),
            "generation_times": [],
            "success_rates": [],
            "daily_counts": defaultdict(int),
        }

    def record_project(
        self,
        project_info: Dict[str, Any],
        generation_time: float,
        success: bool,
    ):
        """
        Registra un proyecto generado.

        Args:
            project_info: Información del proyecto
            generation_time: Tiempo de generación en segundos
            success: Si fue exitoso
        """
        ai_type = project_info.get("ai_type", "unknown")
        backend = project_info.get("backend_framework", "unknown")
        frontend = project_info.get("frontend_framework", "unknown")
        author = project_info.get("author", "unknown")
        date = datetime.now().strftime("%Y-%m-%d")

        self.metrics["projects_by_type"][ai_type] += 1
        self.metrics["projects_by_framework"][f"{backend}+{frontend}"] += 1
        self.metrics["projects_by_author"][author] += 1
        self.metrics["generation_times"].append(generation_time)
        self.metrics["daily_counts"][date] += 1
        
        if success:
            self.metrics["success_rates"].append(1)
        else:
            self.metrics["success_rates"].append(0)

        # Limitar tamaño de listas
        if len(self.metrics["generation_times"]) > 10000:
            self.metrics["generation_times"] = self.metrics["generation_times"][-10000:]
        if len(self.metrics["success_rates"]) > 10000:
            self.metrics["success_rates"] = self.metrics["success_rates"][-10000:]

    def get_trends(
        self,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Obtiene tendencias de los últimos días.

        Args:
            days: Número de días

        Returns:
            Tendencias
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        daily_data = {}
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            daily_data[date_str] = self.metrics["daily_counts"].get(date_str, 0)
            current_date += timedelta(days=1)

        return {
            "period": f"{days} days",
            "daily_counts": daily_data,
            "total_projects": sum(daily_data.values()),
            "average_per_day": sum(daily_data.values()) / max(days, 1),
        }

    def get_top_ai_types(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtiene tipos de IA más populares"""
        sorted_types = sorted(
            self.metrics["projects_by_type"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

        return [
            {"ai_type": ai_type, "count": count}
            for ai_type, count in sorted_types
        ]

    def get_performance_report(self) -> Dict[str, Any]:
        """Obtiene reporte de performance"""
        if not self.metrics["generation_times"]:
            return {"error": "No hay datos suficientes"}

        times = self.metrics["generation_times"]
        success_rates = self.metrics["success_rates"]

        return {
            "generation_time": {
                "average": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "median": sorted(times)[len(times) // 2],
            },
            "success_rate": {
                "overall": sum(success_rates) / len(success_rates) * 100 if success_rates else 0,
                "total": len(success_rates),
                "successful": sum(success_rates),
                "failed": len(success_rates) - sum(success_rates),
            },
        }

    def get_framework_usage(self) -> Dict[str, Any]:
        """Obtiene uso de frameworks"""
        return {
            "frameworks": dict(self.metrics["projects_by_framework"]),
            "total_combinations": len(self.metrics["projects_by_framework"]),
        }

    def get_author_stats(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Obtiene estadísticas por autor"""
        sorted_authors = sorted(
            self.metrics["projects_by_author"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

        return [
            {"author": author, "project_count": count}
            for author, count in sorted_authors
        ]

    def generate_report(
        self,
        report_type: str = "comprehensive",
    ) -> Dict[str, Any]:
        """
        Genera un reporte completo.

        Args:
            report_type: Tipo de reporte (comprehensive, summary, detailed)

        Returns:
            Reporte completo
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "report_type": report_type,
        }

        if report_type in ["comprehensive", "summary"]:
            report["summary"] = {
                "total_projects": sum(self.metrics["projects_by_type"].values()),
                "unique_ai_types": len(self.metrics["projects_by_type"]),
                "unique_frameworks": len(self.metrics["projects_by_framework"]),
                "unique_authors": len(self.metrics["projects_by_author"]),
            }

        if report_type in ["comprehensive", "detailed"]:
            report["trends"] = self.get_trends(30)
            report["top_ai_types"] = self.get_top_ai_types(10)
            report["performance"] = self.get_performance_report()
            report["frameworks"] = self.get_framework_usage()
            report["authors"] = self.get_author_stats(10)

        return report


