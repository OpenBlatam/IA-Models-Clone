"""
Intelligent Suggestions - Sugerencias Inteligentes
=================================================

Sistema de sugerencias inteligentes basado en IA.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class IntelligentSuggestions:
    """Sistema de sugerencias inteligentes"""

    def __init__(self):
        """Inicializa el sistema"""
        self.suggestion_history: List[Dict[str, Any]] = []
        self.user_preferences: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.pattern_analysis: Dict[str, int] = defaultdict(int)

    def generate_suggestions(
        self,
        project_info: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Genera sugerencias inteligentes para un proyecto.

        Args:
            project_info: Información del proyecto
            user_id: ID del usuario (opcional)

        Returns:
            Lista de sugerencias
        """
        suggestions = []

        # Analizar descripción
        description = project_info.get("description", "").lower()
        
        # Sugerir framework basado en complejidad
        if len(description.split()) < 20:
            suggestions.append({
                "type": "framework",
                "priority": "high",
                "message": "Proyecto simple detectado. Considera usar Flask en lugar de FastAPI para mayor simplicidad.",
                "recommendation": "flask",
            })
        elif "api" in description or "rest" in description:
            suggestions.append({
                "type": "framework",
                "priority": "medium",
                "message": "Proyecto API detectado. FastAPI es ideal para APIs REST modernas.",
                "recommendation": "fastapi",
            })

        # Sugerir features basado en palabras clave
        if "auth" in description or "login" in description:
            suggestions.append({
                "type": "feature",
                "priority": "high",
                "message": "Sistema de autenticación detectado. Considera agregar JWT tokens.",
                "recommendation": "add_auth",
            })

        if "database" in description or "data" in description:
            suggestions.append({
                "type": "feature",
                "priority": "medium",
                "message": "Manejo de datos detectado. Considera agregar una base de datos.",
                "recommendation": "add_database",
            })

        # Sugerir testing
        if project_info.get("generate_tests") is False:
            suggestions.append({
                "type": "testing",
                "priority": "medium",
                "message": "Se recomienda generar tests para garantizar calidad del código.",
                "recommendation": "enable_tests",
            })

        # Aplicar preferencias del usuario si existe
        if user_id and user_id in self.user_preferences:
            user_prefs = self.user_preferences[user_id]
            for suggestion in suggestions:
                if suggestion["type"] in user_prefs.get("preferred_types", []):
                    suggestion["priority"] = "high"

        # Guardar sugerencias
        self.suggestion_history.append({
            "project_info": project_info,
            "suggestions": suggestions,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
        })

        return suggestions

    def record_suggestion_feedback(
        self,
        suggestion_id: str,
        accepted: bool,
        user_id: Optional[str] = None,
    ):
        """
        Registra feedback sobre una sugerencia.

        Args:
            suggestion_id: ID de la sugerencia
            accepted: Si fue aceptada
            user_id: ID del usuario
        """
        feedback = {
            "suggestion_id": suggestion_id,
            "accepted": accepted,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
        }

        if accepted:
            self.pattern_analysis[suggestion_id] = self.pattern_analysis.get(suggestion_id, 0) + 1

        logger.info(f"Feedback registrado para sugerencia {suggestion_id}: {'aceptada' if accepted else 'rechazada'}")

    def get_suggestion_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de sugerencias"""
        total_suggestions = len(self.suggestion_history)
        accepted_count = sum(self.pattern_analysis.values())
        
        return {
            "total_suggestions": total_suggestions,
            "accepted_suggestions": accepted_count,
            "acceptance_rate": accepted_count / total_suggestions if total_suggestions > 0 else 0,
            "top_suggestions": dict(sorted(self.pattern_analysis.items(), key=lambda x: x[1], reverse=True)[:10]),
        }


