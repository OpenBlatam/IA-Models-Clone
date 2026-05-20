"""
Document Analyzer Enhanced - Mejoras Avanzadas
==============================================

Mejoras adicionales para análisis de documentos:
- Procesamiento paralelo de múltiples documentos
- Análisis comparativo
- Detección de similitud
- Extracción avanzada de información estructurada
- Análisis de lenguaje avanzado
- Procesamiento en batch optimizado
"""

import asyncio
import logging
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Importar tipos del analizador principal
try:
    from .document_analyzer import AnalysisTask
except ImportError:
    from enum import Enum
    class AnalysisTask(Enum):
        CLASSIFICATION = "classification"
        SUMMARIZATION = "summarization"
        KEYWORD_EXTRACTION = "keyword_extraction"
        ENTITY_RECOGNITION = "entity_recognition"


@dataclass
class DocumentSimilarity:
    """Resultado de similitud entre documentos."""
    document1_id: str
    document2_id: str
    similarity_score: float
    similarity_type: str  # "semantic", "structural", "content"
    common_keywords: List[str] = field(default_factory=list)
    common_entities: List[str] = field(default_factory=list)
    differences: List[str] = field(default_factory=list)


@dataclass
class BatchAnalysisResult:
    """Resultado de análisis en batch."""
    total_documents: int
    processed: int
    failed: int
    results: List[Any] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    processing_time: float = 0.0
    average_confidence: float = 0.0
    statistics: Dict[str, Any] = field(default_factory=dict)


class DocumentComparator:
    """Comparador de documentos avanzado."""
    
    def __init__(self, analyzer):
        """Inicializar comparador."""
        self.analyzer = analyzer
        self.cache: Dict[str, Any] = {}
    
    async def compare_documents(
        self,
        doc1_content: str,
        doc2_content: str,
        doc1_id: Optional[str] = None,
        doc2_id: Optional[str] = None
    ) -> DocumentSimilarity:
        """
        Comparar dos documentos.
        
        Args:
            doc1_content: Contenido del primer documento
            doc2_content: Contenido del segundo documento
            doc1_id: ID del primer documento
            doc2_id: ID del segundo documento
        
        Returns:
            DocumentSimilarity con resultados de comparación
        """
        doc1_id = doc1_id or self._generate_id(doc1_content)
        doc2_id = doc2_id or self._generate_id(doc2_content)
        
        # Análisis paralelo de ambos documentos
        results = await asyncio.gather(
            self.analyzer.analyze_document(
                document_content=doc1_content,
                tasks=[
                    AnalysisTask.KEYWORD_EXTRACTION,
                    AnalysisTask.ENTITY_RECOGNITION,
                    AnalysisTask.CLASSIFICATION
                ]
            ),
            self.analyzer.analyze_document(
                document_content=doc2_content,
                tasks=[
                    AnalysisTask.KEYWORD_EXTRACTION,
                    AnalysisTask.ENTITY_RECOGNITION,
                    AnalysisTask.CLASSIFICATION
                ]
            ),
            return_exceptions=True
        )
        
        result1, result2 = results
        
        if isinstance(result1, Exception) or isinstance(result2, Exception):
            logger.error(f"Error comparing documents: {result1 if isinstance(result1, Exception) else result2}")
            return DocumentSimilarity(
                document1_id=doc1_id,
                document2_id=doc2_id,
                similarity_score=0.0,
                similarity_type="error"
            )
        
        # Calcular similitud semántica usando embeddings
        similarity_score = await self._calculate_semantic_similarity(
            doc1_content, doc2_content
        )
        
        # Keywords comunes
        keywords1 = set(result1.keywords or [])
        keywords2 = set(result2.keywords or [])
        common_keywords = list(keywords1.intersection(keywords2))
        
        # Entidades comunes
        entities1 = {e.get("text", "") for e in (result1.entities or [])}
        entities2 = {e.get("text", "") for e in (result2.entities or [])}
        common_entities = list(entities1.intersection(entities2))
        
        # Diferencias
        differences = []
        if result1.classification != result2.classification:
            differences.append("classification")
        if result1.sentiment != result2.sentiment:
            differences.append("sentiment")
        
        return DocumentSimilarity(
            document1_id=doc1_id,
            document2_id=doc2_id,
            similarity_score=similarity_score,
            similarity_type="semantic",
            common_keywords=common_keywords,
            common_entities=common_entities,
            differences=differences
        )
    
    async def _calculate_semantic_similarity(
        self,
        content1: str,
        content2: str
    ) -> float:
        """Calcular similitud semántica usando embeddings."""
        try:
            # Generar embeddings
            embeddings1 = await self.analyzer.embedding_generator.generate_embeddings([content1])
            embeddings2 = await self.analyzer.embedding_generator.generate_embeddings([content2])
            
            if len(embeddings1) == 0 or len(embeddings2) == 0:
                return 0.0
            
            # Calcular cosine similarity
            emb1 = np.array(embeddings1[0])
            emb2 = np.array(embeddings2[0])
            
            cosine_sim = np.dot(emb1, emb2) / (
                np.linalg.norm(emb1) * np.linalg.norm(emb2)
            )
            
            return float(cosine_sim)
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _generate_id(self, content: str) -> str:
        """Generar ID único para contenido."""
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def find_similar_documents(
        self,
        target_doc: str,
        document_corpus: List[Tuple[str, str]],  # List of (doc_id, content)
        threshold: float = 0.7,
        top_k: int = 10
    ) -> List[DocumentSimilarity]:
        """
        Encontrar documentos similares en un corpus.
        
        Args:
            target_doc: Documento objetivo
            document_corpus: Lista de (doc_id, content) del corpus
            threshold: Umbral mínimo de similitud
            top_k: Número máximo de resultados
        
        Returns:
            Lista de DocumentSimilarity ordenada por similitud
        """
        target_id = self._generate_id(target_doc)
        
        # Comparar con todos los documentos en paralelo
        tasks = [
            self.compare_documents(target_doc, content, target_id, doc_id)
            for doc_id, content in document_corpus
        ]
        
        similarities = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrar errores y aplicar threshold
        valid_similarities = [
            sim for sim in similarities
            if isinstance(sim, DocumentSimilarity) and sim.similarity_score >= threshold
        ]
        
        # Ordenar por similitud
        valid_similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return valid_similarities[:top_k]


class BatchDocumentProcessor:
    """Procesador de documentos en batch optimizado."""
    
    def __init__(self, analyzer, max_workers: int = 10):
        """Inicializar procesador batch."""
        self.analyzer = analyzer
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
    
    async def process_batch(
        self,
        documents: List[Dict[str, Any]],  # List of {id, content/path, type, tasks}
        tasks: Optional[List] = None,
        on_progress: Optional[callable] = None
    ) -> BatchAnalysisResult:
        """
        Procesar múltiples documentos en paralelo.
        
        Args:
            documents: Lista de documentos a procesar
            tasks: Tareas de análisis a realizar
            on_progress: Callback de progreso (processed, total)
        
        Returns:
            BatchAnalysisResult con todos los resultados
        """
        start_time = time.time()
        total = len(documents)
        results = []
        errors = []
        
        # Procesar documentos con semáforo para control de concurrencia
        async def process_single(doc_data, index):
            async with self.semaphore:
                try:
                    doc_id = doc_data.get("id", f"doc_{index}")
                    content = doc_data.get("content")
                    path = doc_data.get("path")
                    doc_type = doc_data.get("type")
                    
                    result = await self.analyzer.analyze_document(
                        document_path=path,
                        document_content=content,
                        document_type=doc_type,
                        tasks=tasks or None,
                        document_id=doc_id
                    )
                    
                    results.append(result)
                    
                    if on_progress:
                        on_progress(len(results), total)
                    
                    return result
                except Exception as e:
                    error_info = {
                        "doc_id": doc_data.get("id", f"doc_{index}"),
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    errors.append(error_info)
                    logger.error(f"Error processing document {doc_data.get('id')}: {e}")
                    
                    if on_progress:
                        on_progress(len(results) + len(errors), total)
                    
                    return None
        
        # Procesar todos los documentos
        await asyncio.gather(*[
            process_single(doc, i) for i, doc in enumerate(documents)
        ])
        
        # Calcular estadísticas
        processing_time = time.time() - start_time
        confidences = [r.confidence for r in results if r.confidence > 0]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Estadísticas adicionales
        statistics = self._calculate_statistics(results)
        
        return BatchAnalysisResult(
            total_documents=total,
            processed=len(results),
            failed=len(errors),
            results=results,
            errors=errors,
            processing_time=processing_time,
            average_confidence=avg_confidence,
            statistics=statistics
        )
    
    def _calculate_statistics(self, results: List[Any]) -> Dict[str, Any]:
        """Calcular estadísticas del batch."""
        if not results:
            return {}
        
        # Estadísticas por tipo de análisis
        stats = {
            "total_documents": len(results),
            "processing_times": {
                "avg": np.mean([r.processing_time for r in results]),
                "min": np.min([r.processing_time for r in results]),
                "max": np.max([r.processing_time for r in results]),
                "p95": np.percentile([r.processing_time for r in results], 95)
            },
            "confidences": {
                "avg": np.mean([r.confidence for r in results]),
                "min": np.min([r.confidence for r in results]),
                "max": np.max([r.confidence for r in results])
            },
            "document_types": Counter([r.document_type for r in results]),
            "classifications": defaultdict(int),
            "sentiments": defaultdict(int)
        }
        
        # Contar clasificaciones
        for result in results:
            if result.classification:
                top_class = max(result.classification.items(), key=lambda x: x[1])[0]
                stats["classifications"][top_class] += 1
        
        # Contar sentimientos
        for result in results:
            if result.sentiment:
                top_sentiment = max(result.sentiment.items(), key=lambda x: x[1])[0]
                stats["sentiments"][top_sentiment] += 1
        
        # Convertir defaultdicts a dicts
        stats["classifications"] = dict(stats["classifications"])
        stats["sentiments"] = dict(stats["sentiments"])
        
        return stats


class AdvancedInformationExtractor:
    """Extractor avanzado de información estructurada."""
    
    def __init__(self, analyzer):
        """Inicializar extractor."""
        self.analyzer = analyzer
    
    async def extract_structured_data(
        self,
        content: str,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extraer información estructurada según un schema.
        
        Args:
            content: Contenido del documento
            schema: Schema con campos a extraer y sus tipos
        
        Returns:
            Diccionario con datos extraídos
        """
        extracted = {}
        
        # Analizar documento completo
        analysis = await self.analyzer.analyze_document(
            document_content=content,
            tasks=[
                AnalysisTask.ENTITY_RECOGNITION,
                AnalysisTask.KEYWORD_EXTRACTION,
                AnalysisTask.CLASSIFICATION
            ]
        )
        
        # Extraer según schema
        for field_name, field_config in schema.items():
            field_type = field_config.get("type", "string")
            extraction_method = field_config.get("method", "auto")
            
            if extraction_method == "entity":
                # Extraer de entidades
                entities = analysis.entities or []
                matching_entities = [
                    e for e in entities
                    if e.get("label", "").lower() == field_type.lower()
                ]
                extracted[field_name] = [e.get("text") for e in matching_entities]
            
            elif extraction_method == "keyword":
                # Extraer de keywords
                keywords = analysis.keywords or []
                extracted[field_name] = keywords[:field_config.get("limit", 5)]
            
            elif extraction_method == "classification":
                # Extraer de clasificación
                if analysis.classification:
                    extracted[field_name] = max(
                        analysis.classification.items(),
                        key=lambda x: x[1]
                    )[0]
            
            elif extraction_method == "qa":
                # Usar Q&A para extraer
                question = field_config.get("question", f"What is {field_name}?")
                answer = await self.analyzer.answer_question(content, question)
                extracted[field_name] = answer.get("answer", "")
            
            else:
                # Auto - intentar todos los métodos
                extracted[field_name] = self._auto_extract(
                    field_name, field_type, analysis
                )
        
        return extracted
    
    def _auto_extract(
        self,
        field_name: str,
        field_type: str,
        analysis: Any
    ) -> Any:
        """Extracción automática basada en nombre y tipo."""
        # Buscar en entidades
        if analysis.entities:
            for entity in analysis.entities:
                if field_name.lower() in entity.get("text", "").lower():
                    return entity.get("text")
        
        # Buscar en keywords
        if analysis.keywords:
            for keyword in analysis.keywords:
                if field_name.lower() in keyword.lower():
                    return keyword
        
        return None


class DocumentLanguageAnalyzer:
    """Analizador avanzado de lenguaje."""
    
    def __init__(self, analyzer):
        """Inicializar analizador de lenguaje."""
        self.analyzer = analyzer
    
    async def analyze_writing_style(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Analizar estilo de escritura.
        
        Args:
            content: Contenido del documento
        
        Returns:
            Diccionario con análisis de estilo
        """
        sentences = content.split(". ")
        words = content.split()
        
        # Métricas básicas
        metrics = {
            "total_words": len(words),
            "total_sentences": len(sentences),
            "avg_words_per_sentence": len(words) / len(sentences) if sentences else 0,
            "avg_chars_per_word": np.mean([len(w) for w in words]) if words else 0,
            "readability_score": self._calculate_readability(content),
            "complexity": self._assess_complexity(content)
        }
        
        # Análisis de sentimiento detallado
        sentiment = await self.analyzer.analyze_sentiment(content)
        metrics["sentiment"] = sentiment
        
        # Análisis de tono
        metrics["tone"] = self._analyze_tone(content)
        
        return metrics
    
    def _calculate_readability(self, content: str) -> float:
        """Calcular score de legibilidad (simplificado)."""
        sentences = content.split(". ")
        words = content.split()
        
        if not sentences or not words:
            return 0.0
        
        # Fórmula simplificada (en producción usar Flesch-Kincaid, etc.)
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = np.mean([len(w) for w in words])
        
        # Score más alto = más legible
        readability = 100 - (avg_sentence_length * 2) - (avg_word_length * 3)
        return max(0, min(100, readability))
    
    def _assess_complexity(self, content: str) -> str:
        """Evaluar complejidad del texto."""
        words = content.split()
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        if avg_word_length < 4:
            return "simple"
        elif avg_word_length < 6:
            return "moderate"
        else:
            return "complex"
    
    def _analyze_tone(self, content: str) -> str:
        """Analizar tono del texto."""
        # Palabras clave para diferentes tonos
        formal_words = ["therefore", "furthermore", "consequently", "however"]
        casual_words = ["hey", "cool", "awesome", "yeah"]
        technical_words = ["implementation", "algorithm", "optimization", "architecture"]
        
        content_lower = content.lower()
        
        formal_count = sum(1 for w in formal_words if w in content_lower)
        casual_count = sum(1 for w in casual_words if w in content_lower)
        technical_count = sum(1 for w in technical_words if w in content_lower)
        
        if technical_count > max(formal_count, casual_count):
            return "technical"
        elif formal_count > casual_count:
            return "formal"
        elif casual_count > 0:
            return "casual"
        else:
            return "neutral"


# Funciones de utilidad

async def analyze_document_batch_optimized(
    analyzer,
    documents: List[Dict[str, Any]],
    max_workers: int = 10,
    tasks: Optional[List] = None
) -> BatchAnalysisResult:
    """Función de utilidad para análisis batch optimizado."""
    processor = BatchDocumentProcessor(analyzer, max_workers=max_workers)
    return await processor.process_batch(documents, tasks=tasks)


def create_extraction_schema(
    fields: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Crear schema de extracción desde lista de campos."""
    schema = {}
    for field in fields:
        schema[field["name"]] = {
            "type": field.get("type", "string"),
            "method": field.get("method", "auto"),
            "question": field.get("question"),
            "limit": field.get("limit", 5)
        }
    return schema


# Exportar componentes principales
__all__ = [
    "DocumentComparator",
    "BatchDocumentProcessor",
    "AdvancedInformationExtractor",
    "DocumentLanguageAnalyzer",
    "DocumentSimilarity",
    "BatchAnalysisResult",
    "analyze_document_batch_optimized",
    "create_extraction_schema"
]

