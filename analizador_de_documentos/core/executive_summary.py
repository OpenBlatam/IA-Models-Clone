"""
Generador de Resúmenes Ejecutivos
==================================

Sistema para generar resúmenes ejecutivos inteligentes y estructurados.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .document_analyzer import DocumentAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ExecutiveSummary:
    """Resumen ejecutivo"""
    title: str
    overview: str
    key_findings: List[str]
    recommendations: List[str]
    metrics: Dict[str, Any]
    insights: List[str]
    confidence: float
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ExecutiveSummaryGenerator:
    """
    Generador de resúmenes ejecutivos
    
    Genera resúmenes estructurados y accionables basados en análisis
    completos de documentos.
    """
    
    def __init__(self, analyzer: DocumentAnalyzer):
        """
        Inicializar generador
        
        Args:
            analyzer: Instancia de DocumentAnalyzer
        """
        self.analyzer = analyzer
        logger.info("ExecutiveSummaryGenerator inicializado")
    
    async def generate_summary(
        self,
        analysis_result: Dict[str, Any],
        document_content: Optional[str] = None,
        max_findings: int = 5,
        max_recommendations: int = 5
    ) -> ExecutiveSummary:
        """
        Generar resumen ejecutivo
        
        Args:
            analysis_result: Resultado de análisis completo
            document_content: Contenido del documento (opcional)
            max_findings: Número máximo de hallazgos clave
            max_recommendations: Número máximo de recomendaciones
        
        Returns:
            ExecutiveSummary
        """
        # Extraer información del análisis
        summary = analysis_result.get("summary", "")
        classification = analysis_result.get("classification", {})
        keywords = analysis_result.get("keywords", [])
        sentiment = analysis_result.get("sentiment", {})
        topics = analysis_result.get("topics", [])
        entities = analysis_result.get("entities", [])
        confidence = analysis_result.get("confidence", 0.0)
        
        # Generar título
        title = self._generate_title(classification, topics)
        
        # Generar overview
        overview = self._generate_overview(summary, classification, sentiment)
        
        # Extraer hallazgos clave
        key_findings = self._extract_key_findings(
            analysis_result,
            max_findings
        )
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(
            analysis_result,
            max_recommendations
        )
        
        # Compilar métricas
        metrics = {
            "confidence": confidence,
            "sentiment_score": self._calculate_sentiment_score(sentiment),
            "topic_count": len(topics),
            "keyword_count": len(keywords),
            "entity_count": len(entities)
        }
        
        # Generar insights
        insights = self._generate_insights(analysis_result)
        
        return ExecutiveSummary(
            title=title,
            overview=overview,
            key_findings=key_findings,
            recommendations=recommendations,
            metrics=metrics,
            insights=insights,
            confidence=confidence
        )
    
    def _generate_title(
        self,
        classification: Dict[str, float],
        topics: List[Dict[str, Any]]
    ) -> str:
        """Generar título del resumen"""
        if classification:
            main_category = max(classification.items(), key=lambda x: x[1])[0]
            return f"Resumen Ejecutivo: {main_category}"
        elif topics:
            main_topic = topics[0].get("keywords", [])
            if main_topic:
                return f"Resumen Ejecutivo: {', '.join(main_topic[:3])}"
        return "Resumen Ejecutivo del Documento"
    
    def _generate_overview(
        self,
        summary: str,
        classification: Dict[str, float],
        sentiment: Dict[str, float]
    ) -> str:
        """Generar overview"""
        overview_parts = []
        
        if summary:
            # Usar resumen si está disponible
            overview_parts.append(summary[:200] + "..." if len(summary) > 200 else summary)
        else:
            overview_parts.append("Análisis completo del documento realizado.")
        
        if classification:
            main_category = max(classification.items(), key=lambda x: x[1])
            overview_parts.append(f"Clasificado como: {main_category[0]} (confianza: {main_category[1]:.1%})")
        
        if sentiment:
            dominant_sentiment = max(sentiment.items(), key=lambda x: x[1])
            overview_parts.append(f"Sentimiento predominante: {dominant_sentiment[0]} ({dominant_sentiment[1]:.1%})")
        
        return " ".join(overview_parts)
    
    def _extract_key_findings(
        self,
        analysis_result: Dict[str, Any],
        max_findings: int
    ) -> List[str]:
        """Extraer hallazgos clave"""
        findings = []
        
        # De keywords
        keywords = analysis_result.get("keywords", [])
        if keywords:
            findings.append(f"Palabras clave principales: {', '.join(keywords[:5])}")
        
        # De entidades
        entities = analysis_result.get("entities", [])
        if entities:
            entity_types = {}
            for entity in entities[:10]:
                entity_type = entity.get("label", "unknown")
                if entity_type not in entity_types:
                    entity_types[entity_type] = []
                entity_types[entity_type].append(entity.get("text", ""))
            
            for entity_type, texts in list(entity_types.items())[:3]:
                findings.append(f"{entity_type}: {', '.join(texts[:3])}")
        
        # De temas
        topics = analysis_result.get("topics", [])
        if topics:
            for topic in topics[:3]:
                topic_keywords = topic.get("keywords", [])
                if topic_keywords:
                    findings.append(f"Tema identificado: {', '.join(topic_keywords[:3])}")
        
        # De sentimiento
        sentiment = analysis_result.get("sentiment", {})
        if sentiment:
            dominant = max(sentiment.items(), key=lambda x: x[1])
            findings.append(f"Sentimiento: {dominant[0]} ({dominant[1]:.1%})")
        
        return findings[:max_findings]
    
    def _generate_recommendations(
        self,
        analysis_result: Dict[str, Any],
        max_recommendations: int
    ) -> List[str]:
        """Generar recomendaciones"""
        recommendations = []
        
        # Basado en sentimiento
        sentiment = analysis_result.get("sentiment", {})
        if sentiment:
            negative = sentiment.get("negative", 0)
            if negative > 0.5:
                recommendations.append("Considerar revisar el tono negativo del documento")
        
        # Basado en confianza
        confidence = analysis_result.get("confidence", 0.0)
        if confidence < 0.5:
            recommendations.append("El análisis tiene baja confianza - considerar revisión manual")
        
        # Basado en keywords
        keywords = analysis_result.get("keywords", [])
        if len(keywords) < 3:
            recommendations.append("El documento tiene pocas palabras clave - considerar expandir contenido")
        
        # Basado en entidades
        entities = analysis_result.get("entities", [])
        if not entities:
            recommendations.append("No se detectaron entidades nombradas - verificar contenido")
        
        return recommendations[:max_recommendations]
    
    def _calculate_sentiment_score(self, sentiment: Dict[str, float]) -> float:
        """Calcular score de sentimiento (-1 a 1)"""
        if not sentiment:
            return 0.0
        
        positive = sentiment.get("positive", 0)
        negative = sentiment.get("negative", 0)
        neutral = sentiment.get("neutral", 0)
        
        # Score normalizado: -1 (muy negativo) a 1 (muy positivo)
        if positive + negative > 0:
            return (positive - negative) / (positive + negative)
        return 0.0
    
    def _generate_insights(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Generar insights"""
        insights = []
        
        # Insight de confianza
        confidence = analysis_result.get("confidence", 0.0)
        if confidence > 0.8:
            insights.append("Alta confianza en el análisis realizado")
        elif confidence < 0.5:
            insights.append("Baja confianza - puede requerir revisión adicional")
        
        # Insight de completitud
        has_summary = bool(analysis_result.get("summary"))
        has_classification = bool(analysis_result.get("classification"))
        has_keywords = bool(analysis_result.get("keywords"))
        
        completeness = sum([has_summary, has_classification, has_keywords]) / 3
        if completeness > 0.8:
            insights.append("Análisis completo con múltiples métricas")
        
        # Insight de temas
        topics = analysis_result.get("topics", [])
        if len(topics) > 1:
            insights.append(f"Documento abarca múltiples temas ({len(topics)})")
        
        return insights
















