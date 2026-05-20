"""
Recommendation Engine - Motor de Recomendaciones
=================================================

Sistema inteligente de recomendaciones basado en proyectos existentes.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Motor de recomendaciones inteligentes"""

    def __init__(self):
        """Inicializa el motor de recomendaciones"""
        self.project_patterns: Dict[str, Any] = defaultdict(lambda: {
            "count": 0,
            "features": defaultdict(int),
            "frameworks": defaultdict(int),
            "success_rate": 0.0,
        })
        self.user_preferences: Dict[str, Dict[str, Any]] = {}

    def learn_from_project(
        self,
        project_info: Dict[str, Any],
        success: bool,
    ):
        """
        Aprende de un proyecto generado.

        Args:
            project_info: Información del proyecto
            success: Si fue exitoso
        """
        ai_type = project_info.get("ai_type", "unknown")
        features = project_info.get("features", [])
        backend = project_info.get("backend_framework", "unknown")
        frontend = project_info.get("frontend_framework", "unknown")

        pattern = self.project_patterns[ai_type]
        pattern["count"] += 1

        for feature in features:
            pattern["features"][feature] += 1

        pattern["frameworks"][f"{backend}+{frontend}"] += 1

        # Actualizar tasa de éxito
        if success:
            pattern["success_rate"] = (
                (pattern["success_rate"] * (pattern["count"] - 1) + 1) / pattern["count"]
            )
        else:
            pattern["success_rate"] = (
                (pattern["success_rate"] * (pattern["count"] - 1)) / pattern["count"]
            )

    def recommend_features(
        self,
        ai_type: str,
        limit: int = 5,
    ) -> List[str]:
        """
        Recomienda features basado en el tipo de IA.

        Args:
            ai_type: Tipo de IA
            limit: Límite de recomendaciones

        Returns:
            Lista de features recomendadas
        """
        if ai_type not in self.project_patterns:
            return []

        pattern = self.project_patterns[ai_type]
        sorted_features = sorted(
            pattern["features"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

        return [feature for feature, count in sorted_features]

    def recommend_framework(
        self,
        ai_type: str,
    ) -> Optional[str]:
        """
        Recomienda framework basado en el tipo de IA.

        Args:
            ai_type: Tipo de IA

        Returns:
            Framework recomendado
        """
        if ai_type not in self.project_patterns:
            return None

        pattern = self.project_patterns[ai_type]
        if not pattern["frameworks"]:
            return None

        # Retornar framework más usado y exitoso
        sorted_frameworks = sorted(
            pattern["frameworks"].items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_frameworks[0][0] if sorted_frameworks else None

    def get_similar_projects(
        self,
        project_info: Dict[str, Any],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Encuentra proyectos similares.

        Args:
            project_info: Información del proyecto
            limit: Límite de resultados

        Returns:
            Lista de proyectos similares
        """
        ai_type = project_info.get("ai_type", "unknown")
        features = set(project_info.get("features", []))

        # Calcular similitud basado en tipo y features
        similarities = []
        for other_type, pattern in self.project_patterns.items():
            if other_type == ai_type:
                continue

            other_features = set(pattern["features"].keys())
            similarity = len(features & other_features) / max(len(features | other_features), 1)

            similarities.append({
                "ai_type": other_type,
                "similarity": similarity,
                "count": pattern["count"],
                "success_rate": pattern["success_rate"],
            })

        # Ordenar por similitud
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        return similarities[:limit]

    def get_recommendations(
        self,
        description: str,
        ai_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Obtiene recomendaciones completas.

        Args:
            description: Descripción del proyecto
            ai_type: Tipo de IA (opcional)

        Returns:
            Recomendaciones
        """
        if not ai_type:
            # Intentar inferir del tipo más común
            if self.project_patterns:
                ai_type = max(
                    self.project_patterns.items(),
                    key=lambda x: x[1]["count"]
                )[0]
            else:
                ai_type = "chat"

        recommendations = {
            "ai_type": ai_type,
            "recommended_features": self.recommend_features(ai_type, 5),
            "recommended_framework": self.recommend_framework(ai_type),
            "similar_projects": [],
        }

        # Agregar proyectos similares si hay datos
        if ai_type in self.project_patterns:
            project_info = {"ai_type": ai_type, "features": recommendations["recommended_features"]}
            recommendations["similar_projects"] = self.get_similar_projects(project_info, 3)

        return recommendations


