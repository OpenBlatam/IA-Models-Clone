"""
Document Intelligent Recommendations - Recomendaciones Inteligentes
====================================================================

Sistema de recomendaciones inteligentes basado en ML y análisis de patrones.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """Recomendación."""
    recommendation_type: str
    title: str
    description: str
    priority: str  # 'high', 'medium', 'low'
    confidence: float
    action_items: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecommendationSet:
    """Conjunto de recomendaciones."""
    document_id: str
    recommendations: List[Recommendation]
    overall_score: float
    generated_at: datetime = field(default_factory=datetime.now)


class IntelligentRecommendationEngine:
    """Motor de recomendaciones inteligentes."""
    
    def __init__(self, analyzer):
        """Inicializar motor."""
        self.analyzer = analyzer
        self.user_preferences: Dict[str, Any] = {}
        self.recommendation_history: List[RecommendationSet] = []
    
    async def generate_recommendations(
        self,
        document_id: str,
        content: str,
        analysis_result: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RecommendationSet:
        """
        Generar recomendaciones inteligentes.
        
        Args:
            document_id: ID del documento
            content: Contenido del documento
            analysis_result: Resultado de análisis (opcional)
            context: Contexto adicional (opcional)
        
        Returns:
            RecommendationSet con recomendaciones
        """
        recommendations = []
        
        # Analizar contenido si no hay resultado
        if analysis_result is None:
            try:
                analysis_result = await self.analyzer.analyze_document(document_content=content)
            except Exception as e:
                logger.warning(f"No se pudo analizar documento: {e}")
        
        # Recomendaciones de calidad
        quality_recs = await self._analyze_quality_recommendations(content, analysis_result)
        recommendations.extend(quality_recs)
        
        # Recomendaciones de estructura
        structure_recs = await self._analyze_structure_recommendations(content)
        recommendations.extend(structure_recs)
        
        # Recomendaciones de contenido
        content_recs = await self._analyze_content_recommendations(content, analysis_result)
        recommendations.extend(content_recs)
        
        # Recomendaciones personalizadas basadas en contexto
        if context:
            context_recs = await self._analyze_context_recommendations(content, context)
            recommendations.extend(context_recs)
        
        # Calcular score general
        if recommendations:
            overall_score = sum(r.confidence for r in recommendations) / len(recommendations)
        else:
            overall_score = 1.0
        
        recommendation_set = RecommendationSet(
            document_id=document_id,
            recommendations=recommendations,
            overall_score=overall_score
        )
        
        self.recommendation_history.append(recommendation_set)
        
        return recommendation_set
    
    async def _analyze_quality_recommendations(
        self,
        content: str,
        analysis_result: Optional[Any]
    ) -> List[Recommendation]:
        """Analizar recomendaciones de calidad."""
        recommendations = []
        
        # Longitud del documento
        if len(content) < 200:
            recommendations.append(Recommendation(
                recommendation_type="quality",
                title="Documento muy corto",
                description="El documento es demasiado corto. Considera agregar más contenido.",
                priority="high",
                confidence=0.9,
                action_items=["Agregar más detalles", "Expandir secciones principales"]
            ))
        
        # Estructura
        paragraphs = content.split('\n\n')
        if len(paragraphs) < 3:
            recommendations.append(Recommendation(
                recommendation_type="structure",
                title="Falta de estructura",
                description="El documento necesita más párrafos para mejor organización.",
                priority="medium",
                confidence=0.8,
                action_items=["Dividir contenido en más párrafos", "Agregar encabezados"]
            ))
        
        # Análisis de calidad si está disponible
        if analysis_result and hasattr(analysis_result, 'quality_score'):
            if analysis_result.quality_score < 70:
                recommendations.append(Recommendation(
                    recommendation_type="quality",
                    title="Calidad mejorable",
                    description=f"El documento tiene una calidad de {analysis_result.quality_score:.1f}%. Hay espacio para mejoras.",
                    priority="high",
                    confidence=0.85,
                    action_items=["Revisar gramática", "Mejorar claridad", "Agregar más detalles"]
                ))
        
        return recommendations
    
    async def _analyze_structure_recommendations(self, content: str) -> List[Recommendation]:
        """Analizar recomendaciones de estructura."""
        recommendations = []
        
        # Detectar encabezados
        lines = content.split('\n')
        has_headers = any(line.strip().startswith('#') or line.strip().isupper() for line in lines[:10])
        
        if not has_headers and len(lines) > 10:
            recommendations.append(Recommendation(
                recommendation_type="structure",
                title="Agregar encabezados",
                description="Los encabezados mejoran la navegación y organización.",
                priority="medium",
                confidence=0.75,
                action_items=["Agregar encabezados principales", "Usar jerarquía de títulos"]
            ))
        
        return recommendations
    
    async def _analyze_content_recommendations(
        self,
        content: str,
        analysis_result: Optional[Any]
    ) -> List[Recommendation]:
        """Analizar recomendaciones de contenido."""
        recommendations = []
        
        # Detectar palabras de relleno
        filler_words = ['muy', 'bastante', 'realmente', 'definitivamente']
        filler_count = sum(content.lower().count(word) for word in filler_words)
        
        if filler_count > len(content.split()) * 0.05:  # Más del 5%
            recommendations.append(Recommendation(
                recommendation_type="content",
                title="Reducir palabras de relleno",
                description="El documento tiene muchas palabras de relleno que reducen la claridad.",
                priority="low",
                confidence=0.7,
                action_items=["Revisar y eliminar palabras innecesarias", "Usar lenguaje más directo"]
            ))
        
        return recommendations
    
    async def _analyze_context_recommendations(
        self,
        content: str,
        context: Dict[str, Any]
    ) -> List[Recommendation]:
        """Analizar recomendaciones basadas en contexto."""
        recommendations = []
        
        # Recomendaciones basadas en tipo de documento
        doc_type = context.get("document_type")
        if doc_type == "technical" and len(content.split()) < 500:
            recommendations.append(Recommendation(
                recommendation_type="context",
                title="Documento técnico incompleto",
                description="Los documentos técnicos generalmente requieren más detalle.",
                priority="high",
                confidence=0.8,
                action_items=["Agregar ejemplos técnicos", "Incluir diagramas o referencias"]
            ))
        
        return recommendations
    
    def set_user_preferences(self, preferences: Dict[str, Any]):
        """Establecer preferencias del usuario."""
        self.user_preferences.update(preferences)
    
    def get_recommendation_history(self, document_id: Optional[str] = None) -> List[RecommendationSet]:
        """Obtener historial de recomendaciones."""
        if document_id:
            return [r for r in self.recommendation_history if r.document_id == document_id]
        return self.recommendation_history


__all__ = [
    "IntelligentRecommendationEngine",
    "Recommendation",
    "RecommendationSet"
]


