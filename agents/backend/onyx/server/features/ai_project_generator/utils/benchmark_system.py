"""
Benchmark System - Sistema de Benchmarking
===========================================

Sistema para comparar y hacer benchmarking de proyectos.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class BenchmarkSystem:
    """Sistema de benchmarking"""

    def __init__(self):
        """Inicializa el sistema"""
        self.benchmark_results: Dict[str, Dict[str, Any]] = {}

    def benchmark_project_generation(
        self,
        project_id: str,
        generation_time: float,
        project_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Hace benchmarking de la generación de un proyecto.

        Args:
            project_id: ID del proyecto
            generation_time: Tiempo de generación
            project_info: Información del proyecto

        Returns:
            Resultados del benchmark
        """
        benchmark = {
            "project_id": project_id,
            "generation_time": generation_time,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "description_length": len(project_info.get("description", "")),
                "features_count": len(project_info.get("features", [])),
                "backend_framework": project_info.get("backend_framework"),
                "frontend_framework": project_info.get("frontend_framework"),
            },
            "performance_score": self._calculate_performance_score(generation_time, project_info),
        }

        self.benchmark_results[project_id] = benchmark
        logger.info(f"Benchmark registrado para proyecto {project_id}")
        return benchmark

    def _calculate_performance_score(
        self,
        generation_time: float,
        project_info: Dict[str, Any],
    ) -> float:
        """Calcula score de performance"""
        # Score base (más rápido = mejor)
        base_score = max(0, 100 - (generation_time / 10))
        
        # Ajustar por complejidad
        complexity = len(project_info.get("description", "").split()) / 10
        adjusted_score = base_score * (1 + complexity * 0.1)
        
        return min(100, max(0, adjusted_score))

    def compare_projects(
        self,
        project_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Compara múltiples proyectos.

        Args:
            project_ids: IDs de proyectos a comparar

        Returns:
            Comparación
        """
        projects_data = []
        for project_id in project_ids:
            if project_id in self.benchmark_results:
                projects_data.append(self.benchmark_results[project_id])

        if not projects_data:
            return {"error": "No hay datos de benchmark para los proyectos especificados"}

        comparison = {
            "projects": projects_data,
            "average_generation_time": sum(p["generation_time"] for p in projects_data) / len(projects_data),
            "fastest_project": min(projects_data, key=lambda x: x["generation_time"])["project_id"],
            "slowest_project": max(projects_data, key=lambda x: x["generation_time"])["project_id"],
            "average_performance_score": sum(p["performance_score"] for p in projects_data) / len(projects_data),
        }

        return comparison

    def get_benchmark_leaderboard(
        self,
        limit: int = 10,
        sort_by: str = "performance_score",
    ) -> List[Dict[str, Any]]:
        """
        Obtiene leaderboard de benchmarks.

        Args:
            limit: Límite de resultados
            sort_by: Campo para ordenar

        Returns:
            Leaderboard
        """
        results = list(self.benchmark_results.values())
        
        if sort_by == "performance_score":
            results.sort(key=lambda x: x["performance_score"], reverse=True)
        elif sort_by == "generation_time":
            results.sort(key=lambda x: x["generation_time"])

        return results[:limit]


