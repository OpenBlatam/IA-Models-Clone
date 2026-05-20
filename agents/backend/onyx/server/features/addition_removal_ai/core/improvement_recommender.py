"""
Improvement Recommender - Sistema de recomendaciones de mejora
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RecommendationPriority(Enum):
    """Prioridad de recomendación"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Recommendation:
    """Recomendación de mejora"""
    category: str
    priority: RecommendationPriority
    title: str
    description: str
    suggestion: str
    impact: str


class ImprovementRecommender:
    """Recomendador de mejoras"""

    def __init__(self):
        """Inicializar recomendador"""
        pass

    def generate_recommendations(
        self,
        content: str,
        analysis_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generar recomendaciones de mejora.

        Args:
            content: Contenido
            analysis_results: Resultados de análisis previos (opcional)

        Returns:
            Recomendaciones
        """
        recommendations = []
        
        # Análisis básico
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        
        # Recomendaciones de longitud
        length_recs = self._analyze_length(content, word_count, sentence_count, paragraph_count)
        recommendations.extend(length_recs)
        
        # Recomendaciones de estructura
        structure_recs = self._analyze_structure(content, paragraph_count)
        recommendations.extend(structure_recs)
        
        # Recomendaciones de formato
        format_recs = self._analyze_format(content)
        recommendations.extend(format_recs)
        
        # Recomendaciones de contenido
        content_recs = self._analyze_content_quality(content, word_count)
        recommendations.extend(content_recs)
        
        # Ordenar por prioridad
        recommendations.sort(key=lambda r: (
            0 if r.priority == RecommendationPriority.HIGH else
            1 if r.priority == RecommendationPriority.MEDIUM else 2
        ))
        
        return {
            "total_recommendations": len(recommendations),
            "high_priority": len([r for r in recommendations if r.priority == RecommendationPriority.HIGH]),
            "medium_priority": len([r for r in recommendations if r.priority == RecommendationPriority.MEDIUM]),
            "low_priority": len([r for r in recommendations if r.priority == RecommendationPriority.LOW]),
            "recommendations": [
                {
                    "category": r.category,
                    "priority": r.priority.value,
                    "title": r.title,
                    "description": r.description,
                    "suggestion": r.suggestion,
                    "impact": r.impact
                }
                for r in recommendations
            ]
        }

    def _analyze_length(
        self,
        content: str,
        word_count: int,
        sentence_count: int,
        paragraph_count: int
    ) -> List[Recommendation]:
        """Analizar longitud"""
        recommendations = []
        
        if word_count < 100:
            recommendations.append(Recommendation(
                category="length",
                priority=RecommendationPriority.HIGH,
                title="Contenido muy corto",
                description=f"El contenido tiene solo {word_count} palabras",
                suggestion="Expande el contenido con más detalles, ejemplos o explicaciones",
                impact="Mejorará la completitud y valor del contenido"
            ))
        elif word_count > 3000:
            recommendations.append(Recommendation(
                category="length",
                priority=RecommendationPriority.MEDIUM,
                title="Contenido muy largo",
                description=f"El contenido tiene {word_count} palabras",
                suggestion="Considera dividir en múltiples secciones o artículos",
                impact="Mejorará la legibilidad y retención del lector"
            ))
        
        if sentence_count < 3:
            recommendations.append(Recommendation(
                category="length",
                priority=RecommendationPriority.MEDIUM,
                title="Pocas oraciones",
                description=f"El contenido tiene solo {sentence_count} oraciones",
                suggestion="Agrega más oraciones para desarrollar mejor las ideas",
                impact="Mejorará la claridad y desarrollo del contenido"
            ))
        
        if paragraph_count < 2 and word_count > 200:
            recommendations.append(Recommendation(
                category="structure",
                priority=RecommendationPriority.MEDIUM,
                title="Falta estructura",
                description="El contenido largo no está dividido en párrafos",
                suggestion="Divide el contenido en párrafos para mejorar legibilidad",
                impact="Mejorará la legibilidad y organización"
            ))
        
        return recommendations

    def _analyze_structure(
        self,
        content: str,
        paragraph_count: int
    ) -> List[Recommendation]:
        """Analizar estructura"""
        recommendations = []
        
        # Verificar título
        lines = content.split('\n')
        first_line = lines[0].strip() if lines else ""
        if not first_line.startswith('#') and len(content) > 200:
            recommendations.append(Recommendation(
                category="structure",
                priority=RecommendationPriority.HIGH,
                title="Falta título",
                description="El documento no tiene un título claro",
                suggestion="Agrega un título usando # al inicio",
                impact="Mejorará la navegación y comprensión"
            ))
        
        # Verificar introducción
        content_lower = content.lower()
        intro_keywords = ['introducción', 'introduction', 'resumen', 'summary']
        has_intro = any(kw in content_lower[:500] for kw in intro_keywords)
        
        if not has_intro and len(content) > 500:
            recommendations.append(Recommendation(
                category="structure",
                priority=RecommendationPriority.MEDIUM,
                title="Falta introducción",
                description="El contenido no tiene una introducción clara",
                suggestion="Agrega una introducción al inicio del documento",
                impact="Mejorará la comprensión del contexto"
            ))
        
        # Verificar conclusión
        concl_keywords = ['conclusión', 'conclusion', 'resumen', 'summary', 'finalmente']
        has_concl = any(kw in content_lower[-500:] for kw in concl_keywords)
        
        if not has_concl and len(content) > 500:
            recommendations.append(Recommendation(
                category="structure",
                priority=RecommendationPriority.MEDIUM,
                title="Falta conclusión",
                description="El contenido no tiene una conclusión clara",
                suggestion="Agrega una conclusión al final del documento",
                impact="Mejorará el cierre y resumen del contenido"
            ))
        
        return recommendations

    def _analyze_format(self, content: str) -> List[Recommendation]:
        """Analizar formato"""
        recommendations = []
        
        # Verificar espacios dobles
        if '  ' in content:
            recommendations.append(Recommendation(
                category="format",
                priority=RecommendationPriority.LOW,
                title="Espacios dobles",
                description="El contenido contiene espacios dobles",
                suggestion="Elimina espacios duplicados",
                impact="Mejorará la presentación profesional"
            ))
        
        # Verificar puntuación al final
        if content and content[-1] not in '.!?':
            recommendations.append(Recommendation(
                category="format",
                priority=RecommendationPriority.MEDIUM,
                title="Falta puntuación final",
                description="El contenido no termina con puntuación",
                suggestion="Agrega un punto, signo de exclamación o interrogación al final",
                impact="Mejorará la completitud del contenido"
            ))
        
        return recommendations

    def _analyze_content_quality(
        self,
        content: str,
        word_count: int
    ) -> List[Recommendation]:
        """Analizar calidad de contenido"""
        recommendations = []
        
        # Verificar repetición
        words = content.lower().split()
        if words:
            from collections import Counter
            word_freq = Counter(words)
            most_common = word_freq.most_common(1)[0]
            ratio = most_common[1] / len(words)
            
            if ratio > 0.1:  # Más del 10%
                recommendations.append(Recommendation(
                    category="content",
                    priority=RecommendationPriority.MEDIUM,
                    title="Repetición excesiva",
                    description=f"La palabra '{most_common[0]}' aparece {most_common[1]} veces",
                    suggestion="Usa sinónimos y variaciones para evitar repetición",
                    impact="Mejorará la variedad y calidad del contenido"
                ))
        
        # Verificar conectores
        connectors = ['además', 'también', 'sin embargo', 'por lo tanto', 'also', 'however', 'therefore']
        connector_count = sum(1 for conn in connectors if conn in content.lower())
        
        if connector_count == 0 and word_count > 100:
            recommendations.append(Recommendation(
                category="content",
                priority=RecommendationPriority.MEDIUM,
                title="Faltan conectores",
                description="El contenido no usa palabras de transición",
                suggestion="Agrega conectores como 'además', 'sin embargo', 'por lo tanto'",
                impact="Mejorará la fluidez y coherencia"
            ))
        
        return recommendations






