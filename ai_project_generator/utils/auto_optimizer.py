"""
Auto Optimizer - Optimizador Automático
========================================

Optimiza automáticamente la generación de proyectos.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class AutoOptimizer:
    """Optimizador automático"""

    def __init__(self):
        """Inicializa el optimizador"""
        self.optimization_rules: List[Dict[str, Any]] = []
        self.optimization_history: List[Dict[str, Any]] = []

    def analyze_project(
        self,
        project_info: Dict[str, Any],
        generation_time: float,
    ) -> Dict[str, Any]:
        """
        Analiza un proyecto y sugiere optimizaciones.

        Args:
            project_info: Información del proyecto
            generation_time: Tiempo de generación

        Returns:
            Análisis y sugerencias
        """
        suggestions = []
        issues = []

        # Analizar tiempo de generación
        if generation_time > 120:
            suggestions.append({
                "type": "performance",
                "priority": "high",
                "message": "Tiempo de generación muy alto. Considerar optimización.",
                "suggestion": "Revisar dependencias y procesos pesados",
            })

        # Analizar features
        features = project_info.get("features", [])
        if len(features) > 10:
            suggestions.append({
                "type": "complexity",
                "priority": "medium",
                "message": "Muchas features. Considerar modularización.",
                "suggestion": "Dividir en módulos más pequeños",
            })

        # Analizar frameworks
        backend = project_info.get("backend_framework", "")
        frontend = project_info.get("frontend_framework", "")

        if backend == "django" and len(features) < 3:
            suggestions.append({
                "type": "framework",
                "priority": "low",
                "message": "Django puede ser excesivo para proyectos simples",
                "suggestion": "Considerar FastAPI o Flask para proyectos pequeños",
            })

        return {
            "analysis_date": datetime.now().isoformat(),
            "generation_time": generation_time,
            "suggestions": suggestions,
            "issues": issues,
            "optimization_score": self._calculate_score(generation_time, len(features)),
        }

    def _calculate_score(
        self,
        generation_time: float,
        features_count: int,
    ) -> float:
        """Calcula score de optimización"""
        # Score basado en tiempo y complejidad
        time_score = max(0, 100 - (generation_time / 2))
        complexity_score = max(0, 100 - (features_count * 5))
        
        return (time_score + complexity_score) / 2

    def optimize_project_config(
        self,
        project_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Optimiza la configuración de un proyecto.

        Args:
            project_info: Información del proyecto

        Returns:
            Configuración optimizada
        """
        optimized = project_info.copy()

        # Optimizar framework según features
        features = project_info.get("features", [])
        if len(features) < 3:
            if optimized.get("backend_framework") == "django":
                optimized["backend_framework"] = "fastapi"
                optimized["optimization_note"] = "Cambiado a FastAPI para proyecto simple"

        # Optimizar según tipo de IA
        ai_type = project_info.get("ai_type", "")
        if ai_type == "chat" and "websocket" not in features:
            optimized["features"] = features + ["websocket"]
            optimized["optimization_note"] = "Agregado WebSocket para chat en tiempo real"

        return optimized

    def get_optimization_recommendations(
        self,
        project_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Obtiene recomendaciones de optimización.

        Args:
            project_info: Información del proyecto

        Returns:
            Lista de recomendaciones
        """
        recommendations = []

        # Recomendaciones basadas en tipo de IA
        ai_type = project_info.get("ai_type", "")
        if ai_type == "vision":
            recommendations.append({
                "category": "dependencies",
                "message": "Considerar agregar bibliotecas de procesamiento de imágenes",
                "priority": "high",
            })

        # Recomendaciones basadas en features
        features = project_info.get("features", [])
        if "database" in features and "cache" not in features:
            recommendations.append({
                "category": "performance",
                "message": "Agregar cache para mejorar performance de base de datos",
                "priority": "medium",
            })

        return recommendations


