"""
Comparison Microservice - Microservicio de Comparación
====================================================

Microservicio especializado en comparación y análisis de entradas de IA.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize

from ..shared.models import HistoryEntry, ComparisonResult, ModelType, AnalysisStatus
from ..shared.database import DatabaseManager
from ..shared.config import Settings
from ..shared.messaging import MessageBroker
from ..shared.monitoring import MetricsCollector

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración
settings = Settings()

# Servicios compartidos
db_manager = DatabaseManager()
message_broker = MessageBroker()
metrics = MetricsCollector()

# Analizadores de texto
sia = SentimentIntensityAnalyzer()
vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    ngram_range=(1, 2)
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestión del ciclo de vida del microservicio."""
    # Startup
    logger.info("Starting Comparison Microservice...")
    await db_manager.initialize()
    await message_broker.initialize()
    await metrics.initialize()
    logger.info("Comparison Microservice initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Comparison Microservice...")
    await db_manager.close()
    await message_broker.close()
    await metrics.close()
    logger.info("Comparison Microservice shutdown complete")


# Crear aplicación FastAPI
app = FastAPI(
    title="Comparison Microservice",
    description="Microservicio especializado en comparación y análisis de entradas de IA",
    version="7.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints del microservicio
@app.get("/health")
async def health_check():
    """Verificación de salud del microservicio."""
    try:
        db_health = await db_manager.health_check()
        broker_health = await message_broker.health_check()
        
        return {
            "service": "comparison",
            "status": "healthy" if db_health and broker_health else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "connected" if db_health else "disconnected",
            "message_broker": "connected" if broker_health else "disconnected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.post("/compare", response_model=ComparisonResult, status_code=201)
async def compare_entries(
    entry_1_id: str,
    entry_2_id: str,
    comparison_type: str = "comprehensive"
):
    """Comparar dos entradas de historial."""
    try:
        # Obtener entradas
        entry1 = await db_manager.get_history_entry(entry_1_id)
        entry2 = await db_manager.get_history_entry(entry_2_id)
        
        if not entry1 or not entry2:
            raise HTTPException(status_code=404, detail="One or both history entries not found")
        
        # Realizar comparación
        comparison = await perform_comparison(entry1, entry2, comparison_type)
        
        # Guardar comparación
        saved_comparison = await db_manager.create_comparison(comparison)
        
        # Enviar evento de comparación completada
        await message_broker.publish_event("comparison.completed", {
            "comparison_id": saved_comparison.id,
            "entry_1_id": entry_1_id,
            "entry_2_id": entry_2_id,
            "similarity_score": saved_comparison.overall_similarity,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Métricas
        await metrics.increment_counter("comparisons_created")
        await metrics.record_histogram("comparison_similarity", saved_comparison.overall_similarity)
        
        logger.info(f"Created comparison: {saved_comparison.id}")
        return saved_comparison
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating comparison: {e}")
        await metrics.increment_counter("comparison_creation_errors")
        raise HTTPException(status_code=500, detail="Failed to create comparison")


@app.post("/compare/batch", response_model=List[ComparisonResult], status_code=201)
async def compare_entries_batch(
    entry_pairs: List[Dict[str, str]],
    comparison_type: str = "comprehensive"
):
    """Comparar múltiples pares de entradas en lote."""
    try:
        if len(entry_pairs) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 pairs per batch")
        
        comparisons = []
        
        for pair in entry_pairs:
            entry_1_id = pair.get("entry_1_id")
            entry_2_id = pair.get("entry_2_id")
            
            if not entry_1_id or not entry_2_id:
                continue
            
            try:
                # Obtener entradas
                entry1 = await db_manager.get_history_entry(entry_1_id)
                entry2 = await db_manager.get_history_entry(entry_2_id)
                
                if not entry1 or not entry2:
                    continue
                
                # Realizar comparación
                comparison = await perform_comparison(entry1, entry2, comparison_type)
                
                # Guardar comparación
                saved_comparison = await db_manager.create_comparison(comparison)
                comparisons.append(saved_comparison)
                
            except Exception as e:
                logger.error(f"Error comparing pair {entry_1_id}-{entry_2_id}: {e}")
                continue
        
        # Enviar evento de comparación en lote completada
        await message_broker.publish_event("comparison.batch.completed", {
            "total_comparisons": len(comparisons),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Métricas
        await metrics.increment_counter("batch_comparisons_created")
        await metrics.record_histogram("batch_comparisons_size", len(comparisons))
        
        logger.info(f"Created {len(comparisons)} batch comparisons")
        return comparisons
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating batch comparisons: {e}")
        await metrics.increment_counter("batch_comparison_creation_errors")
        raise HTTPException(status_code=500, detail="Failed to create batch comparisons")


@app.get("/comparisons", response_model=List[ComparisonResult])
async def get_comparisons(
    skip: int = 0,
    limit: int = 100,
    entry_1_id: Optional[str] = None,
    entry_2_id: Optional[str] = None,
    min_similarity: Optional[float] = None,
    max_similarity: Optional[float] = None
):
    """Obtener comparaciones con filtros avanzados."""
    try:
        if limit > 1000:
            limit = 1000
        
        comparisons = await db_manager.get_comparisons(
            skip=skip,
            limit=limit,
            entry_1_id=entry_1_id,
            entry_2_id=entry_2_id,
            min_similarity=min_similarity,
            max_similarity=max_similarity
        )
        
        # Métricas
        await metrics.increment_counter("comparisons_retrieved")
        await metrics.record_histogram("comparisons_retrieved_count", len(comparisons))
        
        return comparisons
        
    except Exception as e:
        logger.error(f"Error getting comparisons: {e}")
        await metrics.increment_counter("comparisons_retrieval_errors")
        raise HTTPException(status_code=500, detail="Failed to retrieve comparisons")


@app.get("/comparisons/{comparison_id}", response_model=ComparisonResult)
async def get_comparison(comparison_id: str):
    """Obtener una comparación específica."""
    try:
        comparison = await db_manager.get_comparison(comparison_id)
        if not comparison:
            raise HTTPException(status_code=404, detail="Comparison not found")
        
        # Métricas
        await metrics.increment_counter("comparison_retrieved")
        
        return comparison
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting comparison {comparison_id}: {e}")
        await metrics.increment_counter("comparison_retrieval_errors")
        raise HTTPException(status_code=500, detail="Failed to retrieve comparison")


@app.get("/comparisons/{comparison_id}/analysis")
async def get_comparison_analysis(comparison_id: str):
    """Obtener análisis detallado de una comparación."""
    try:
        comparison = await db_manager.get_comparison(comparison_id)
        if not comparison:
            raise HTTPException(status_code=404, detail="Comparison not found")
        
        # Generar análisis detallado
        analysis = await generate_comparison_analysis(comparison)
        
        # Métricas
        await metrics.increment_counter("comparison_analyses_generated")
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating comparison analysis {comparison_id}: {e}")
        await metrics.increment_counter("comparison_analysis_errors")
        raise HTTPException(status_code=500, detail="Failed to generate comparison analysis")


@app.get("/similarities/search")
async def find_similar_entries(
    entry_id: str,
    similarity_threshold: float = 0.7,
    limit: int = 10
):
    """Encontrar entradas similares a una entrada específica."""
    try:
        # Obtener entrada de referencia
        reference_entry = await db_manager.get_history_entry(entry_id)
        if not reference_entry:
            raise HTTPException(status_code=404, detail="Reference entry not found")
        
        # Buscar entradas similares
        similar_entries = await find_similar_entries_async(
            reference_entry, similarity_threshold, limit
        )
        
        # Métricas
        await metrics.increment_counter("similarity_searches_performed")
        await metrics.record_histogram("similar_entries_found", len(similar_entries))
        
        return {
            "reference_entry_id": entry_id,
            "similarity_threshold": similarity_threshold,
            "similar_entries": similar_entries,
            "total_found": len(similar_entries),
            "search_timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar entries for {entry_id}: {e}")
        await metrics.increment_counter("similarity_search_errors")
        raise HTTPException(status_code=500, detail="Failed to find similar entries")


@app.get("/statistics/similarity")
async def get_similarity_statistics(
    model_type: Optional[ModelType] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """Obtener estadísticas de similitud."""
    try:
        stats = await db_manager.get_similarity_statistics(
            model_type=model_type,
            start_date=start_date,
            end_date=end_date
        )
        
        # Métricas
        await metrics.increment_counter("similarity_statistics_retrieved")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting similarity statistics: {e}")
        await metrics.increment_counter("similarity_statistics_errors")
        raise HTTPException(status_code=500, detail="Failed to retrieve similarity statistics")


# Funciones auxiliares
async def perform_comparison(
    entry1: HistoryEntry, 
    entry2: HistoryEntry, 
    comparison_type: str
) -> ComparisonResult:
    """Realizar comparación entre dos entradas."""
    
    # Análisis de contenido
    analysis1 = analyze_content(entry1.response)
    analysis2 = analyze_content(entry2.response)
    
    # Similitud semántica usando TF-IDF
    semantic_similarity = calculate_semantic_similarity(
        entry1.response, entry2.response
    )
    
    # Similitud léxica
    lexical_similarity = calculate_lexical_similarity(
        entry1.response, entry2.response
    )
    
    # Similitud estructural
    structural_similarity = calculate_structural_similarity(analysis1, analysis2)
    
    # Similitud general (promedio ponderado)
    overall_similarity = (
        semantic_similarity * 0.5 +
        lexical_similarity * 0.3 +
        structural_similarity * 0.2
    )
    
    # Detectar diferencias
    differences = detect_differences(analysis1, analysis2)
    improvements = detect_improvements(analysis1, analysis2)
    
    return ComparisonResult(
        entry_1_id=entry1.id,
        entry_2_id=entry2.id,
        semantic_similarity=semantic_similarity,
        lexical_similarity=lexical_similarity,
        structural_similarity=structural_similarity,
        overall_similarity=overall_similarity,
        differences=differences,
        improvements=improvements,
        analysis_details={
            "entry1_analysis": analysis1,
            "entry2_analysis": analysis2,
            "comparison_type": comparison_type,
            "comparison_timestamp": datetime.utcnow().isoformat()
        }
    )


def analyze_content(content: str) -> Dict[str, Any]:
    """Analizar contenido de texto."""
    try:
        # Métricas básicas
        word_count = len(word_tokenize(content))
        sentence_count = len(sent_tokenize(content))
        char_count = len(content)
        
        # Análisis de sentimiento
        sentiment_scores = sia.polarity_scores(content)
        
        # Análisis de complejidad
        avg_word_length = sum(len(word) for word in word_tokenize(content)) / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Análisis de vocabulario
        unique_words = len(set(word.lower() for word in word_tokenize(content)))
        vocabulary_richness = unique_words / word_count if word_count > 0 else 0
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "char_count": char_count,
            "sentiment_positive": sentiment_scores["pos"],
            "sentiment_neutral": sentiment_scores["neu"],
            "sentiment_negative": sentiment_scores["neg"],
            "sentiment_compound": sentiment_scores["compound"],
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "vocabulary_richness": vocabulary_richness,
            "unique_words": unique_words
        }
    except Exception as e:
        logger.error(f"Error analyzing content: {e}")
        return get_default_analysis()


def get_default_analysis() -> Dict[str, Any]:
    """Obtener análisis por defecto."""
    return {
        "word_count": 0,
        "sentence_count": 0,
        "char_count": 0,
        "sentiment_positive": 0.0,
        "sentiment_neutral": 1.0,
        "sentiment_negative": 0.0,
        "sentiment_compound": 0.0,
        "avg_word_length": 0.0,
        "avg_sentence_length": 0.0,
        "vocabulary_richness": 0.0,
        "unique_words": 0
    }


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calcular similitud semántica usando TF-IDF."""
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(similarity)
    except Exception:
        return 0.0


def calculate_lexical_similarity(text1: str, text2: str) -> float:
    """Calcular similitud léxica."""
    try:
        words1 = set(word.lower() for word in word_tokenize(text1))
        words2 = set(word.lower() for word in word_tokenize(text2))
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    except Exception:
        return 0.0


def calculate_structural_similarity(analysis1: Dict[str, Any], analysis2: Dict[str, Any]) -> float:
    """Calcular similitud estructural."""
    try:
        # Comparar métricas estructurales
        metrics = [
            'word_count', 'sentence_count', 'avg_word_length', 
            'avg_sentence_length', 'vocabulary_richness'
        ]
        
        similarities = []
        for metric in metrics:
            val1 = analysis1.get(metric, 0)
            val2 = analysis2.get(metric, 0)
            
            if val1 == 0 and val2 == 0:
                similarity = 1.0
            elif val1 == 0 or val2 == 0:
                similarity = 0.0
            else:
                similarity = 1.0 - abs(val1 - val2) / max(val1, val2)
            
            similarities.append(similarity)
        
        return np.mean(similarities)
    except Exception:
        return 0.0


def detect_differences(analysis1: Dict[str, Any], analysis2: Dict[str, Any]) -> List[str]:
    """Detectar diferencias significativas."""
    differences = []
    
    # Diferencia en longitud
    word_diff = abs(analysis1.get('word_count', 0) - analysis2.get('word_count', 0))
    if word_diff > 50:
        differences.append(f"Significant word count difference: {word_diff} words")
    
    # Diferencia en sentimiento
    sentiment_diff = abs(analysis1.get('sentiment_compound', 0) - analysis2.get('sentiment_compound', 0))
    if sentiment_diff > 0.3:
        differences.append(f"Significant sentiment difference: {sentiment_diff:.2f}")
    
    # Diferencia en riqueza de vocabulario
    vocab_diff = abs(analysis1.get('vocabulary_richness', 0) - analysis2.get('vocabulary_richness', 0))
    if vocab_diff > 0.2:
        differences.append(f"Significant vocabulary richness difference: {vocab_diff:.2f}")
    
    return differences


def detect_improvements(analysis1: Dict[str, Any], analysis2: Dict[str, Any]) -> List[str]:
    """Detectar mejoras."""
    improvements = []
    
    # Mejora en riqueza de vocabulario
    if analysis2.get('vocabulary_richness', 0) > analysis1.get('vocabulary_richness', 0) + 0.1:
        improvements.append("Enhanced vocabulary richness")
    
    # Mejora en sentimiento positivo
    if analysis2.get('sentiment_compound', 0) > analysis1.get('sentiment_compound', 0) + 0.2:
        improvements.append("More positive sentiment")
    
    # Mejora en longitud apropiada
    if 50 <= analysis2.get('word_count', 0) <= 500 and not (50 <= analysis1.get('word_count', 0) <= 500):
        improvements.append("Improved content length")
    
    return improvements


async def generate_comparison_analysis(comparison: ComparisonResult) -> Dict[str, Any]:
    """Generar análisis detallado de una comparación."""
    analysis = {
        "comparison_id": comparison.id,
        "similarity_breakdown": {
            "semantic": comparison.semantic_similarity,
            "lexical": comparison.lexical_similarity,
            "structural": comparison.structural_similarity,
            "overall": comparison.overall_similarity
        },
        "differences": comparison.differences,
        "improvements": comparison.improvements,
        "analysis_details": comparison.analysis_details,
        "generated_at": datetime.utcnow().isoformat()
    }
    
    return analysis


async def find_similar_entries_async(
    reference_entry: HistoryEntry, 
    threshold: float, 
    limit: int
) -> List[Dict[str, Any]]:
    """Encontrar entradas similares de forma asíncrona."""
    try:
        # Obtener todas las entradas (esto se puede optimizar con índices)
        all_entries = await db_manager.get_history_entries(limit=10000)
        
        similar_entries = []
        
        for entry in all_entries:
            if entry.id == reference_entry.id:
                continue
            
            # Calcular similitud
            similarity = calculate_semantic_similarity(
                reference_entry.response, entry.response
            )
            
            if similarity >= threshold:
                similar_entries.append({
                    "entry_id": entry.id,
                    "similarity_score": similarity,
                    "model_type": entry.model_type.value,
                    "timestamp": entry.timestamp.isoformat()
                })
        
        # Ordenar por similitud y limitar resultados
        similar_entries.sort(key=lambda x: x["similarity_score"], reverse=True)
        return similar_entries[:limit]
        
    except Exception as e:
        logger.error(f"Error finding similar entries: {e}")
        return []


# Manejo de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Manejador global de excepciones."""
    logger.error(f"Unhandled exception: {exc}")
    await metrics.increment_counter("unhandled_exceptions")
    return {"detail": "Internal server error"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )




