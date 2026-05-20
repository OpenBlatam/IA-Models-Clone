"""
Educational Content Analyzer - Sistema de análisis de contenido educativo
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class EducationalContentAnalyzer:
    """Analizador de contenido educativo"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos educativos
        self.educational_elements = {
            "definitions": [
                r'(?:es|significa|definición|definition|means)',
                r':\s*[A-Z]'  # Definiciones después de dos puntos
            ],
            "examples": [
                r'(?:por ejemplo|ejemplo|example|for instance|such as)',
                r'ej\.|e\.g\.'
            ],
            "explanations": [
                r'(?:explicar|explicación|explain|explanation|because|porque)',
                r'(?:cómo|how|por qué|why)'
            ],
            "exercises": [
                r'(?:ejercicio|exercise|práctica|practice|actividad|activity)',
                r'(?:resuelve|solve|completa|complete)'
            ],
            "questions": [
                r'\?',
                r'(?:pregunta|question|cuál|what|quién|who)'
            ],
            "summaries": [
                r'(?:resumen|summary|en resumen|in summary|conclusión|conclusion)'
            ]
        }

    def analyze_educational_structure(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar estructura educativa del contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de estructura educativa
        """
        content_lower = content.lower()
        element_counts = {}
        
        # Contar cada elemento educativo
        for element_type, patterns in self.educational_elements.items():
            count = 0
            for pattern in patterns:
                matches = re.findall(pattern, content_lower, re.IGNORECASE)
                count += len(matches)
            element_counts[element_type] = count
        
        # Detectar preguntas
        questions = len(re.findall(r'\?', content))
        element_counts["questions"] = questions
        
        # Calcular score educativo
        total_elements = sum(element_counts.values())
        educational_score = min(1.0, total_elements / 15)  # Normalizar
        
        # Verificar si tiene estructura educativa completa
        has_structure = (
            element_counts.get("definitions", 0) > 0 and
            element_counts.get("explanations", 0) > 0
        )
        
        return {
            "element_counts": element_counts,
            "educational_score": educational_score,
            "total_elements": total_elements,
            "has_educational_structure": has_structure
        }

    def analyze_learning_objectives(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar objetivos de aprendizaje en el contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de objetivos de aprendizaje
        """
        # Patrones de objetivos de aprendizaje
        objective_patterns = [
            r'(?:aprender|learn|entender|understand|conocer|know)',
            r'(?:objetivo|objective|meta|goal)',
            r'(?:al final|at the end|después|after)'
        ]
        
        content_lower = content.lower()
        objective_indicators = sum(
            len(re.findall(pattern, content_lower, re.IGNORECASE))
            for pattern in objective_patterns
        )
        
        # Detectar verbos de acción educativos
        action_verbs = [
            "identificar", "describir", "explicar", "aplicar", "analizar", "evaluar",
            "identify", "describe", "explain", "apply", "analyze", "evaluate"
        ]
        
        verb_count = sum(1 for verb in action_verbs if verb in content_lower)
        
        # Calcular score de objetivos
        objectives_score = min(1.0, (objective_indicators + verb_count) / 10)
        
        return {
            "objective_indicators": objective_indicators,
            "action_verbs": verb_count,
            "objectives_score": objectives_score,
            "has_clear_objectives": objectives_score > 0.5
        }

    def suggest_educational_improvements(
        self,
        content: str
    ) -> List[Dict[str, Any]]:
        """
        Sugerir mejoras educativas.

        Args:
            content: Contenido

        Returns:
            Sugerencias de mejora educativa
        """
        structure_analysis = self.analyze_educational_structure(content)
        objectives_analysis = self.analyze_learning_objectives(content)
        
        suggestions = []
        
        if structure_analysis["educational_score"] < 0.4:
            suggestions.append({
                "type": "structure",
                "priority": "high",
                "issue": "Falta de estructura educativa",
                "suggestion": "Agrega definiciones, ejemplos y explicaciones para mejorar el valor educativo"
            })
        
        if structure_analysis["element_counts"].get("examples", 0) == 0:
            suggestions.append({
                "type": "examples",
                "priority": "medium",
                "issue": "No hay ejemplos",
                "suggestion": "Incluye ejemplos prácticos para facilitar la comprensión"
            })
        
        if structure_analysis["element_counts"].get("questions", 0) < 2:
            suggestions.append({
                "type": "questions",
                "priority": "medium",
                "issue": "Pocas preguntas",
                "suggestion": "Agrega preguntas para fomentar la reflexión y el aprendizaje activo"
            })
        
        if not objectives_analysis["has_clear_objectives"]:
            suggestions.append({
                "type": "objectives",
                "priority": "high",
                "issue": "Objetivos de aprendizaje no claros",
                "suggestion": "Define claramente los objetivos de aprendizaje al inicio del contenido"
            })
        
        return suggestions






