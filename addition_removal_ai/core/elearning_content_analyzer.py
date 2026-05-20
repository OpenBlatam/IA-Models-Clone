"""
E-Learning Content Analyzer - Sistema de análisis de contenido de e-learning
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class ELearningContentAnalyzer:
    """Analizador de contenido de e-learning"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos de e-learning
        self.elearning_elements = {
            "learning_objectives": [
                r'(?:objetivo|objective|meta|goal|aprenderás|you will learn)',
                r'(?:al finalizar|at the end|al completar|upon completion)'
            ],
            "modules": [
                r'(?:módulo|module|unidad|unit|lección|lesson)',
                r'(?:módulo \d+|module \d+|unidad \d+|unit \d+)'
            ],
            "quizzes": [
                r'(?:cuestionario|quiz|pregunta|question|ejercicio|exercise)',
                r'(?:evalúa|evaluate|prueba|test)'
            ],
            "videos": [
                r'(?:video|vídeo|grabación|recording|clase|class)',
                r'<video[^>]*>',  # HTML video
            ],
            "resources": [
                r'(?:recurso|resource|material|material|descarga|download)',
                r'(?:lectura|reading|artículo|article|documento|document)'
            ],
            "interactive": [
                r'(?:interactivo|interactive|simulación|simulation|práctica|practice)',
                r'(?:laboratorio|lab|workshop|taller)'
            ],
            "assessments": [
                r'(?:evaluación|assessment|examen|exam|prueba|test)',
                r'(?:calificación|grade|puntuación|score)'
            ]
        }

    def analyze_elearning_content(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar contenido de e-learning.

        Args:
            content: Contenido

        Returns:
            Análisis de contenido de e-learning
        """
        element_counts = {}
        
        # Contar elementos de e-learning
        for element_type, patterns in self.elearning_elements.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                count += len(matches)
            element_counts[element_type] = count
        
        # Calcular score de e-learning
        total_elements = sum(element_counts.values())
        elearning_score = min(1.0, total_elements / 25)  # Normalizar
        
        # Verificar si es contenido de e-learning
        is_elearning = (
            element_counts.get("learning_objectives", 0) > 0 or
            element_counts.get("modules", 0) > 0 or
            element_counts.get("quizzes", 0) > 0 or
            element_counts.get("videos", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "elearning_score": elearning_score,
            "total_elements": total_elements,
            "is_elearning": is_elearning
        }

    def analyze_elearning_quality(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar calidad del contenido de e-learning.

        Args:
            content: Contenido

        Returns:
            Análisis de calidad de e-learning
        """
        elearning_analysis = self.analyze_elearning_content(content)
        element_counts = elearning_analysis["element_counts"]
        
        # Verificar elementos de calidad
        has_objectives = element_counts.get("learning_objectives", 0) > 0
        has_modules = element_counts.get("modules", 0) > 0
        has_quizzes = element_counts.get("quizzes", 0) > 0
        has_videos = element_counts.get("videos", 0) > 0
        has_resources = element_counts.get("resources", 0) > 0
        has_interactive = element_counts.get("interactive", 0) > 0
        has_assessments = element_counts.get("assessments", 0) > 0
        
        # Calcular score de calidad
        quality_score = (
            (1.0 if has_objectives else 0.0) * 0.2 +
            (1.0 if has_modules else 0.0) * 0.15 +
            (1.0 if has_quizzes else 0.0) * 0.15 +
            (1.0 if has_videos else 0.0) * 0.15 +
            (1.0 if has_resources else 0.0) * 0.15 +
            (1.0 if has_interactive else 0.0) * 0.1 +
            (1.0 if has_assessments else 0.0) * 0.1
        )
        
        return {
            "quality_score": quality_score,
            "has_objectives": has_objectives,
            "has_modules": has_modules,
            "has_quizzes": has_quizzes,
            "has_videos": has_videos,
            "has_resources": has_resources,
            "has_interactive": has_interactive,
            "has_assessments": has_assessments,
            "quality_level": (
                "high" if quality_score > 0.7 else
                "medium" if quality_score > 0.4 else
                "low"
            )
        }


