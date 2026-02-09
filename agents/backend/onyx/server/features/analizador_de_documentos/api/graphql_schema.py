"""
GraphQL Schema para Analizador de Documentos
==============================================

Schema GraphQL para acceso flexible a datos.
"""

import logging
from typing import Optional, List, Dict, Any
import strawberry
from strawberry.fastapi import GraphQLRouter

from ..core.document_analyzer import DocumentAnalyzer, AnalysisTask
from .routes import get_analyzer

logger = logging.getLogger(__name__)


@strawberry.type
class ClassificationResult:
    """Resultado de clasificación"""
    label: str
    score: float


@strawberry.type
class SentimentResult:
    """Resultado de sentimiento"""
    positive: float
    negative: float
    neutral: float


@strawberry.type
class DocumentAnalysis:
    """Análisis de documento"""
    document_id: str
    summary: Optional[str]
    classification: Optional[List[ClassificationResult]]
    sentiment: Optional[SentimentResult]
    keywords: Optional[List[str]]
    confidence: float
    processing_time: float


@strawberry.type
class Query:
    """Queries GraphQL"""
    
    @strawberry.field
    async def analyze_document(
        self,
        content: str,
        tasks: Optional[List[str]] = None
    ) -> DocumentAnalysis:
        """Analizar documento"""
        analyzer = get_analyzer()
        
        analysis_tasks = None
        if tasks:
            analysis_tasks = [AnalysisTask(task) for task in tasks]
        
        result = await analyzer.analyze_document(
            document_content=content,
            tasks=analysis_tasks
        )
        
        # Convertir a tipos GraphQL
        classification = None
        if result.classification:
            classification = [
                ClassificationResult(label=k, score=v)
                for k, v in result.classification.items()
            ]
        
        sentiment = None
        if result.sentiment:
            sentiment = SentimentResult(
                positive=result.sentiment.get("positive", 0.0),
                negative=result.sentiment.get("negative", 0.0),
                neutral=result.sentiment.get("neutral", 0.0)
            )
        
        return DocumentAnalysis(
            document_id=result.document_id,
            summary=result.summary,
            classification=classification,
            sentiment=sentiment,
            keywords=result.keywords,
            confidence=result.confidence,
            processing_time=result.processing_time
        )
    
    @strawberry.field
    def health(self) -> str:
        """Health check"""
        return "healthy"


# Crear schema
schema = strawberry.Schema(Query)

# Crear router GraphQL
graphql_router = GraphQLRouter(schema, path="/graphql")
















