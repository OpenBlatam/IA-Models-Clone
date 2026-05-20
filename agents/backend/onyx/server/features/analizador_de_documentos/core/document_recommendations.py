"""
Document Recommendations - Sistema de Recomendaciones Inteligente
================================================================

Sistema de recomendaciones basado en análisis de documentos.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecommendation:
    """Recomendación de documento."""
    recommendation_type: str  # 'improvement', 'similar', 'related', 'action'
    title: str
    description: str
    priority: str = "medium"  # 'low', 'medium', 'high', 'critical'
    confidence: float = 0.0
    action_items: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RecommendationEngine:
    """Motor de recomendaciones."""
    
    def __init__(self, analyzer):
        """Inicializar motor."""
        self.analyzer = analyzer
        self.recommendation_history: List[DocumentRecommendation] = []
    
    async def generate_recommendations(
        self,
        document_analysis: Any,
        quality_analysis: Optional[Any] = None,
        grammar_analysis: Optional[Any] = None,
        version_history: Optional[Any] = None
    ) -> List[DocumentRecommendation]:
        """
        Generar recomendaciones basadas en análisis.
        
        Args:
            document_analysis: Resultado de análisis de documento
            quality_analysis: Análisis de calidad (opcional)
            grammar_analysis: Análisis de gramática (opcional)
            version_history: Historial de versiones (opcional)
        
        Returns:
            Lista de recomendaciones
        """
        recommendations = []
        
        # Recomendaciones basadas en análisis básico
        recommendations.extend(await self._recommendations_from_analysis(document_analysis))
        
        # Recomendaciones basadas en calidad
        if quality_analysis:
            recommendations.extend(await self._recommendations_from_quality(quality_analysis))
        
        # Recomendaciones basadas en gramática
        if grammar_analysis:
            recommendations.extend(await self._recommendations_from_grammar(grammar_analysis))
        
        # Recomendaciones basadas en versiones
        if version_history:
            recommendations.extend(await self._recommendations_from_versions(version_history))
        
        # Ordenar por prioridad
        recommendations.sort(key=lambda r: self._priority_score(r.priority), reverse=True)
        
        self.recommendation_history.extend(recommendations)
        
        return recommendations
    
    async def _recommendations_from_analysis(
        self,
        analysis: Any
    ) -> List[DocumentRecommendation]:
        """Recomendaciones basadas en análisis básico."""
        recommendations = []
        
        # Recomendación de keywords
        if hasattr(analysis, 'keywords') and analysis.keywords:
            if len(analysis.keywords) < 5:
                recommendations.append(DocumentRecommendation(
                    recommendation_type="improvement",
                    title="Agregar más keywords",
                    description=f"Documento tiene solo {len(analysis.keywords)} keywords. Agregar más para mejor SEO y descubribilidad.",
                    priority="medium",
                    confidence=0.8,
                    action_items=["Identificar términos clave adicionales", "Agregar keywords relevantes"]
                ))
        
        # Recomendación de resumen
        if hasattr(analysis, 'summary'):
            if not analysis.summary or len(analysis.summary) < 50:
                recommendations.append(DocumentRecommendation(
                    recommendation_type="improvement",
                    title="Mejorar resumen",
                    description="El resumen es muy corto o no existe. Un buen resumen ayuda a los lectores a entender el contenido rápidamente.",
                    priority="high",
                    confidence=0.9,
                    action_items=["Escribir un resumen de 2-3 oraciones", "Incluir puntos principales"]
                ))
        
        # Recomendación de clasificación
        if hasattr(analysis, 'classification') and analysis.classification:
            max_score = max(analysis.classification.values())
            if max_score < 0.7:
                recommendations.append(DocumentRecommendation(
                    recommendation_type="improvement",
                    title="Clarificar tipo de documento",
                    description=f"La clasificación más alta tiene solo {max_score:.1%} de confianza. El documento podría necesitar más contenido específico.",
                    priority="medium",
                    confidence=0.7,
                    action_items=["Agregar contenido más específico", "Clarificar propósito del documento"]
                ))
        
        return recommendations
    
    async def _recommendations_from_quality(
        self,
        quality: Any
    ) -> List[DocumentRecommendation]:
        """Recomendaciones basadas en calidad."""
        recommendations = []
        
        if quality.overall_score < 60:
            recommendations.append(DocumentRecommendation(
                recommendation_type="improvement",
                title="Mejorar calidad general del documento",
                description=f"Score de calidad: {quality.overall_score:.1f}/100. El documento necesita mejoras significativas.",
                priority="high",
                confidence=0.95,
                action_items=quality.recommendations[:3]
            ))
        
        if quality.readability_score < 50:
            recommendations.append(DocumentRecommendation(
                recommendation_type="improvement",
                title="Mejorar legibilidad",
                description=f"Legibilidad: {quality.readability_score:.1f}/100. El documento es difícil de leer.",
                priority="high",
                confidence=0.9,
                action_items=[
                    "Usar oraciones más cortas",
                    "Simplificar vocabulario",
                    "Agregar más párrafos"
                ]
            ))
        
        if quality.structure_score < 50:
            recommendations.append(DocumentRecommendation(
                recommendation_type="improvement",
                title="Mejorar estructura",
                description=f"Estructura: {quality.structure_score:.1f}/100. El documento necesita mejor organización.",
                priority="medium",
                confidence=0.85,
                action_items=[
                    "Agregar títulos y secciones",
                    "Organizar contenido lógicamente",
                    "Usar listas y tablas donde sea apropiado"
                ]
            ))
        
        return recommendations
    
    async def _recommendations_from_grammar(
        self,
        grammar: Any
    ) -> List[DocumentRecommendation]:
        """Recomendaciones basadas en gramática."""
        recommendations = []
        
        if grammar.overall_score < 70:
            recommendations.append(DocumentRecommendation(
                recommendation_type="improvement",
                title="Corregir errores gramaticales",
                description=f"Score gramatical: {grammar.overall_score:.1f}/100. Se detectaron {len(grammar.issues)} problemas.",
                priority="high",
                confidence=0.9,
                action_items=grammar.recommendations[:3]
            ))
        
        critical_issues = [i for i in grammar.issues if i.severity == 'critical']
        if critical_issues:
            recommendations.append(DocumentRecommendation(
                recommendation_type="action",
                title="Corregir errores críticos",
                description=f"Se encontraron {len(critical_issues)} errores críticos que deben corregirse.",
                priority="critical",
                confidence=1.0,
                action_items=[i.message for i in critical_issues[:5]]
            ))
        
        return recommendations
    
    async def _recommendations_from_versions(
        self,
        version_history: Dict[str, Any]
    ) -> List[DocumentRecommendation]:
        """Recomendaciones basadas en historial de versiones."""
        recommendations = []
        
        if version_history.get('similarity_trend') == 'decreasing':
            recommendations.append(DocumentRecommendation(
                recommendation_type="improvement",
                title="Documento está cambiando mucho",
                description="Las versiones recientes muestran cambios significativos. Considerar consolidar cambios antes de más ediciones.",
                priority="medium",
                confidence=0.7
            ))
        
        comparisons = version_history.get('comparisons', [])
        if comparisons:
            most_changed = max(comparisons, key=lambda c: len(c.get('changes_count', [])))
            if most_changed.get('changes_count', 0) > 50:
                recommendations.append(DocumentRecommendation(
                    recommendation_type="action",
                    title="Revisar versión con muchos cambios",
                    description=f"Versión {most_changed.get('to_version')} tiene {most_changed.get('changes_count')} cambios. Revisar cuidadosamente.",
                    priority="high",
                    confidence=0.8
                ))
        
        return recommendations
    
    def _priority_score(self, priority: str) -> int:
        """Calcular score de prioridad."""
        scores = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'critical': 4
        }
        return scores.get(priority, 2)
    
    async def find_similar_documents_recommendations(
        self,
        current_document: str,
        document_corpus: List[Tuple[str, str]],
        top_k: int = 5
    ) -> List[DocumentRecommendation]:
        """
        Recomendar documentos similares.
        
        Args:
            current_document: Contenido del documento actual
            document_corpus: Lista de (doc_id, content)
            top_k: Número de recomendaciones
        
        Returns:
            Lista de recomendaciones de documentos similares
        """
        if not hasattr(self.analyzer, 'comparator') or not self.analyzer.comparator:
            return []
        
        similar_docs = await self.analyzer.find_similar_documents(
            current_document,
            document_corpus,
            threshold=0.6,
            top_k=top_k
        )
        
        recommendations = []
        for sim_doc in similar_docs:
            recommendations.append(DocumentRecommendation(
                recommendation_type="similar",
                title=f"Documento similar: {sim_doc.document2_id}",
                description=f"Similitud: {sim_doc.similarity_score:.1%}. Puede contener información relacionada.",
                priority="low",
                confidence=sim_doc.similarity_score,
                metadata={"document_id": sim_doc.document2_id, "similarity": sim_doc.similarity_score}
            ))
        
        return recommendations


__all__ = [
    "RecommendationEngine",
    "DocumentRecommendation"
]
















