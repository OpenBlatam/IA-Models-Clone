"""
Interactive Content Analyzer - Sistema de análisis de contenido interactivo
"""

import logging
import re
from typing import Dict, Any, Optional, List
from collections import Counter

logger = logging.getLogger(__name__)


class InteractiveContentAnalyzer:
    """Analizador de contenido interactivo"""

    def __init__(self):
        """Inicializar analizador"""
        # Elementos interactivos comunes
        self.interactive_elements = {
            "forms": [
                r'<form[^>]*>[\s\S]*?</form>',
                r'<input[^>]*>',
                r'<textarea[^>]*>[\s\S]*?</textarea>',
                r'<select[^>]*>[\s\S]*?</select>',
                r'<button[^>]*>[\s\S]*?</button>'
            ],
            "links": [
                r'<a[^>]+href=["\']([^"\']+)["\']',
                r'\[([^\]]+)\]\((https?://[^\)]+)\)'
            ],
            "buttons": [
                r'<button[^>]*>[\s\S]*?</button>',
                r'<input[^>]+type=["\']button["\']',
                r'<input[^>]+type=["\']submit["\']'
            ],
            "quizzes": [
                r'quiz|pregunta|question|test',
                r'<input[^>]+type=["\']radio["\']',
                r'<input[^>]+type=["\']checkbox["\']'
            ],
            "polls": [
                r'poll|encuesta|votación',
                r'<input[^>]+type=["\']radio["\']'
            ],
            "comments": [
                r'<div[^>]*class=["\'][^"\']*comment[^"\']*["\']',
                r'comment|comentario'
            ],
            "social_sharing": [
                r'share|compartir',
                r'<div[^>]*class=["\'][^"\']*share[^"\']*["\']',
                r'social|redes'
            ]
        }

    def analyze_interactive_elements(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar elementos interactivos en el contenido.

        Args:
            content: Contenido

        Returns:
            Análisis de elementos interactivos
        """
        analysis = {
            "forms": [],
            "links": [],
            "buttons": [],
            "quizzes": [],
            "polls": [],
            "comments": [],
            "social_sharing": []
        }
        
        # Detectar cada tipo de elemento interactivo
        for element_type, patterns in self.interactive_elements.items():
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    if isinstance(matches[0], tuple):
                        analysis[element_type].extend([m[0] if m[0] else m[1] for m in matches])
                    else:
                        analysis[element_type].extend(matches)
        
        # Eliminar duplicados
        for key in analysis:
            analysis[key] = list(set(analysis[key]))
        
        # Calcular estadísticas
        total_interactive = sum(len(elements) for elements in analysis.values())
        
        # Calcular score de interactividad (0-1)
        interactivity_score = min(1.0, total_interactive / 15)  # Normalizar a 15 elementos
        
        return {
            "interactive_elements": analysis,
            "statistics": {
                "total_forms": len(analysis["forms"]),
                "total_links": len(analysis["links"]),
                "total_buttons": len(analysis["buttons"]),
                "total_quizzes": len(analysis["quizzes"]),
                "total_polls": len(analysis["polls"]),
                "total_comments": len(analysis["comments"]),
                "total_social_sharing": len(analysis["social_sharing"]),
                "total_interactive": total_interactive
            },
            "interactivity_score": interactivity_score,
            "has_interactive": total_interactive > 0
        }

    def analyze_user_engagement_potential(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar potencial de engagement del usuario.

        Args:
            content: Contenido

        Returns:
            Análisis de potencial de engagement
        """
        interactive_analysis = self.analyze_interactive_elements(content)
        
        # Palabras que invitan a la acción
        action_words = {
            'click', 'haz clic', 'comparte', 'comenta', 'vota', 'responde',
            'click', 'share', 'comment', 'vote', 'respond', 'subscribe'
        }
        
        content_lower = content.lower()
        action_word_count = sum(1 for word in action_words if word in content_lower)
        
        # Preguntas (invitan a responder)
        questions = len(re.findall(r'\?', content))
        
        # Calcular score de engagement potencial
        engagement_score = (
            interactive_analysis["interactivity_score"] * 0.5 +
            min(1.0, action_word_count / 5) * 0.3 +
            min(1.0, questions / 10) * 0.2
        )
        
        return {
            "engagement_score": engagement_score,
            "interactive_elements": interactive_analysis["statistics"]["total_interactive"],
            "action_words": action_word_count,
            "questions": questions,
            "recommendations": self._generate_engagement_recommendations(
                engagement_score,
                interactive_analysis,
                action_word_count,
                questions
            )
        }

    def _generate_engagement_recommendations(
        self,
        engagement_score: float,
        interactive_analysis: Dict[str, Any],
        action_words: int,
        questions: int
    ) -> List[str]:
        """Generar recomendaciones de engagement"""
        recommendations = []
        
        if engagement_score < 0.3:
            recommendations.append("El contenido tiene bajo potencial de engagement")
        
        if interactive_analysis["statistics"]["total_interactive"] == 0:
            recommendations.append("Agrega elementos interactivos (formularios, botones, enlaces)")
        
        if action_words < 2:
            recommendations.append("Incluye más palabras de acción (click, comparte, comenta)")
        
        if questions == 0:
            recommendations.append("Agrega preguntas para invitar a los usuarios a responder")
        
        return recommendations






