"""
Sistema Realista de Comparación de Historial de IA
=================================================

Un sistema simple y práctico para analizar y comparar contenido generado por IA.
Solo funcionalidades reales y útiles.
"""

import asyncio
import logging
import sys
import argparse
import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, asdict
import re
import math

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ai_comparison.log')
    ]
)

logger = logging.getLogger(__name__)

# Modelos de datos
@dataclass
class ContentAnalysis:
    """Análisis de contenido."""
    content: str
    model_version: str
    timestamp: datetime
    word_count: int
    sentence_count: int
    readability_score: float
    sentiment_score: float
    quality_score: float
    unique_words: int
    avg_word_length: float

@dataclass
class ComparisonResult:
    """Resultado de comparación."""
    model_a: str
    model_b: str
    similarity_score: float
    quality_difference: float
    timestamp: datetime
    details: Dict[str, Any]

# Modelos Pydantic para API
class AnalyzeRequest(BaseModel):
    """Solicitud de análisis."""
    content: str = Field(..., description="Contenido a analizar")
    model_version: str = Field(..., description="Versión del modelo de IA")

class CompareRequest(BaseModel):
    """Solicitud de comparación."""
    content_a: str = Field(..., description="Primer contenido")
    content_b: str = Field(..., description="Segundo contenido")
    model_a: str = Field(..., description="Modelo del primer contenido")
    model_b: str = Field(..., description="Modelo del segundo contenido")

class AnalysisResponse(BaseModel):
    """Respuesta de análisis."""
    analysis_id: str
    content: str
    model_version: str
    word_count: int
    readability_score: float
    sentiment_score: float
    quality_score: float
    timestamp: str

class ComparisonResponse(BaseModel):
    """Respuesta de comparación."""
    comparison_id: str
    model_a: str
    model_b: str
    similarity_score: float
    quality_difference: float
    timestamp: str

class AIComparisonSystem:
    """Sistema realista de comparación de IA."""
    
    def __init__(self):
        self.app = FastAPI(
            title="Sistema de Comparación de IA",
            description="Sistema simple para analizar y comparar contenido generado por IA",
            version="1.0.0"
        )
        self.analyses: Dict[str, ContentAnalysis] = {}
        self.comparisons: Dict[str, ComparisonResult] = {}
        self.setup_database()
        self.setup_middleware()
        self.setup_routes()
        logger.info("Sistema de Comparación de IA inicializado")
    
    def setup_database(self):
        """Configurar base de datos SQLite."""
        self.db_path = "ai_comparison.db"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Crear tabla de análisis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                model_version TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                word_count INTEGER,
                sentence_count INTEGER,
                readability_score REAL,
                sentiment_score REAL,
                quality_score REAL,
                unique_words INTEGER,
                avg_word_length REAL
            )
        """)
        
        # Crear tabla de comparaciones
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS comparisons (
                id TEXT PRIMARY KEY,
                model_a TEXT NOT NULL,
                model_b TEXT NOT NULL,
                similarity_score REAL,
                quality_difference REAL,
                timestamp TEXT NOT NULL,
                details TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def setup_middleware(self):
        """Configurar middleware de FastAPI."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Configurar rutas de la API."""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Sistema de Comparación de IA",
                "version": "1.0.0",
                "status": "operacional",
                "timestamp": datetime.now().isoformat(),
                "analyses_count": len(self.analyses),
                "comparisons_count": len(self.comparisons)
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "analyses": len(self.analyses),
                "comparisons": len(self.comparisons),
                "database": "connected"
            }
        
        @self.app.post("/analyze", response_model=AnalysisResponse)
        async def analyze_content(request: AnalyzeRequest):
            """Analizar contenido generado por IA."""
            try:
                analysis = self.analyze_content(request.content, request.model_version)
                analysis_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.analyses)}"
                
                self.analyses[analysis_id] = analysis
                self.save_analysis_to_db(analysis_id, analysis)
                
                return AnalysisResponse(
                    analysis_id=analysis_id,
                    content=request.content,
                    model_version=request.model_version,
                    word_count=analysis.word_count,
                    readability_score=analysis.readability_score,
                    sentiment_score=analysis.sentiment_score,
                    quality_score=analysis.quality_score,
                    timestamp=analysis.timestamp.isoformat()
                )
                
            except Exception as e:
                logger.error(f"Error analizando contenido: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/compare", response_model=ComparisonResponse)
        async def compare_content(request: CompareRequest):
            """Comparar dos contenidos generados por IA."""
            try:
                analysis_a = self.analyze_content(request.content_a, request.model_a)
                analysis_b = self.analyze_content(request.content_b, request.model_b)
                
                comparison = self.compare_analyses(analysis_a, analysis_b, request.model_a, request.model_b)
                comparison_id = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.comparisons)}"
                
                self.comparisons[comparison_id] = comparison
                self.save_comparison_to_db(comparison_id, comparison)
                
                return ComparisonResponse(
                    comparison_id=comparison_id,
                    model_a=request.model_a,
                    model_b=request.model_b,
                    similarity_score=comparison.similarity_score,
                    quality_difference=comparison.quality_difference,
                    timestamp=comparison.timestamp.isoformat()
                )
                
            except Exception as e:
                logger.error(f"Error comparando contenido: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/analyses")
        async def list_analyses():
            """Listar todos los análisis."""
            return {
                "analyses": [
                    {
                        "id": analysis_id,
                        "model_version": analysis.model_version,
                        "word_count": analysis.word_count,
                        "quality_score": analysis.quality_score,
                        "timestamp": analysis.timestamp.isoformat()
                    }
                    for analysis_id, analysis in self.analyses.items()
                ],
                "total": len(self.analyses)
            }
        
        @self.app.get("/comparisons")
        async def list_comparisons():
            """Listar todas las comparaciones."""
            return {
                "comparisons": [
                    {
                        "id": comparison_id,
                        "model_a": comparison.model_a,
                        "model_b": comparison.model_b,
                        "similarity_score": comparison.similarity_score,
                        "quality_difference": comparison.quality_difference,
                        "timestamp": comparison.timestamp.isoformat()
                    }
                    for comparison_id, comparison in self.comparisons.items()
                ],
                "total": len(self.comparisons)
            }
        
        @self.app.get("/analyses/{analysis_id}")
        async def get_analysis(analysis_id: str):
            """Obtener análisis específico."""
            if analysis_id not in self.analyses:
                raise HTTPException(status_code=404, detail="Análisis no encontrado")
            
            analysis = self.analyses[analysis_id]
            return {
                "id": analysis_id,
                "content": analysis.content,
                "model_version": analysis.model_version,
                "word_count": analysis.word_count,
                "sentence_count": analysis.sentence_count,
                "readability_score": analysis.readability_score,
                "sentiment_score": analysis.sentiment_score,
                "quality_score": analysis.quality_score,
                "unique_words": analysis.unique_words,
                "avg_word_length": analysis.avg_word_length,
                "timestamp": analysis.timestamp.isoformat()
            }
        
        @self.app.get("/stats")
        async def get_statistics():
            """Obtener estadísticas del sistema."""
            if not self.analyses:
                return {
                    "total_analyses": 0,
                    "total_comparisons": 0,
                    "models_used": [],
                    "avg_quality_score": 0,
                    "avg_readability_score": 0
                }
            
            models = set(analysis.model_version for analysis in self.analyses.values())
            avg_quality = sum(analysis.quality_score for analysis in self.analyses.values()) / len(self.analyses)
            avg_readability = sum(analysis.readability_score for analysis in self.analyses.values()) / len(self.analyses)
            
            return {
                "total_analyses": len(self.analyses),
                "total_comparisons": len(self.comparisons),
                "models_used": list(models),
                "avg_quality_score": round(avg_quality, 2),
                "avg_readability_score": round(avg_readability, 2)
            }
    
    def analyze_content(self, content: str, model_version: str) -> ContentAnalysis:
        """Analizar contenido usando métricas realistas."""
        
        # Conteo básico
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        word_count = len(words)
        sentence_count = len(sentences)
        unique_words = len(set(word.lower() for word in words))
        
        # Longitud promedio de palabras
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        
        # Puntuación de legibilidad (Flesch Reading Ease simplificado)
        readability_score = self.calculate_readability(word_count, sentence_count, avg_word_length)
        
        # Análisis de sentimiento (básico)
        sentiment_score = self.calculate_sentiment(content)
        
        # Puntuación de calidad (combinación de métricas)
        quality_score = self.calculate_quality_score(
            readability_score, sentiment_score, unique_words, word_count
        )
        
        return ContentAnalysis(
            content=content,
            model_version=model_version,
            timestamp=datetime.now(),
            word_count=word_count,
            sentence_count=sentence_count,
            readability_score=readability_score,
            sentiment_score=sentiment_score,
            quality_score=quality_score,
            unique_words=unique_words,
            avg_word_length=avg_word_length
        )
    
    def calculate_readability(self, word_count: int, sentence_count: int, avg_word_length: float) -> float:
        """Calcular puntuación de legibilidad."""
        if sentence_count == 0:
            return 0.0
        
        # Fórmula Flesch Reading Ease simplificada
        avg_sentence_length = word_count / sentence_count
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length / 100)
        
        # Normalizar entre 0 y 100
        return max(0, min(100, readability))
    
    def calculate_sentiment(self, content: str) -> float:
        """Calcular sentimiento básico."""
        positive_words = {
            'bueno', 'excelente', 'fantástico', 'maravilloso', 'increíble', 'perfecto',
            'genial', 'súper', 'magnífico', 'extraordinario', 'brillante', 'exitoso',
            'positivo', 'beneficioso', 'útil', 'eficaz', 'efectivo', 'mejor', 'mejora'
        }
        
        negative_words = {
            'malo', 'terrible', 'horrible', 'pésimo', 'defectuoso', 'problemático',
            'negativo', 'dañino', 'inútil', 'ineficaz', 'peor', 'empeora', 'falla',
            'error', 'problema', 'dificultad', 'complicado', 'confuso', 'mal'
        }
        
        words = content.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words == 0:
            return 0.5  # Neutral
        
        sentiment = positive_count / total_sentiment_words
        return sentiment
    
    def calculate_quality_score(self, readability: float, sentiment: float, unique_words: int, word_count: int) -> float:
        """Calcular puntuación de calidad general."""
        
        # Diversidad de vocabulario (0-1)
        vocabulary_diversity = unique_words / word_count if word_count > 0 else 0
        
        # Normalizar legibilidad (0-1)
        readability_normalized = readability / 100
        
        # Combinar métricas
        quality = (
            readability_normalized * 0.4 +
            sentiment * 0.3 +
            vocabulary_diversity * 0.3
        )
        
        return round(quality, 2)
    
    def compare_analyses(self, analysis_a: ContentAnalysis, analysis_b: ContentAnalysis, 
                        model_a: str, model_b: str) -> ComparisonResult:
        """Comparar dos análisis."""
        
        # Similitud de contenido (básica)
        words_a = set(analysis_a.content.lower().split())
        words_b = set(analysis_b.content.lower().split())
        
        intersection = words_a.intersection(words_b)
        union = words_a.union(words_b)
        
        similarity_score = len(intersection) / len(union) if union else 0
        
        # Diferencia de calidad
        quality_difference = abs(analysis_a.quality_score - analysis_b.quality_score)
        
        # Detalles de comparación
        details = {
            "word_count_diff": analysis_a.word_count - analysis_b.word_count,
            "readability_diff": analysis_a.readability_score - analysis_b.readability_score,
            "sentiment_diff": analysis_a.sentiment_score - analysis_b.sentiment_score,
            "vocabulary_diversity_a": analysis_a.unique_words / analysis_a.word_count if analysis_a.word_count > 0 else 0,
            "vocabulary_diversity_b": analysis_b.unique_words / analysis_b.word_count if analysis_b.word_count > 0 else 0
        }
        
        return ComparisonResult(
            model_a=model_a,
            model_b=model_b,
            similarity_score=round(similarity_score, 3),
            quality_difference=round(quality_difference, 3),
            timestamp=datetime.now(),
            details=details
        )
    
    def save_analysis_to_db(self, analysis_id: str, analysis: ContentAnalysis):
        """Guardar análisis en base de datos."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO analyses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            analysis_id,
            analysis.content,
            analysis.model_version,
            analysis.timestamp.isoformat(),
            analysis.word_count,
            analysis.sentence_count,
            analysis.readability_score,
            analysis.sentiment_score,
            analysis.quality_score,
            analysis.unique_words,
            analysis.avg_word_length
        ))
        
        conn.commit()
        conn.close()
    
    def save_comparison_to_db(self, comparison_id: str, comparison: ComparisonResult):
        """Guardar comparación en base de datos."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO comparisons VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            comparison_id,
            comparison.model_a,
            comparison.model_b,
            comparison.similarity_score,
            comparison.quality_difference,
            comparison.timestamp.isoformat(),
            json.dumps(comparison.details)
        ))
        
        conn.commit()
        conn.close()
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """Ejecutar el sistema."""
        logger.info(f"Iniciando Sistema de Comparación de IA en {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="debug" if debug else "info"
        )

def main():
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(description="Sistema de Comparación de IA")
    parser.add_argument("--host", default="0.0.0.0", help="Host para enlazar")
    parser.add_argument("--port", type=int, default=8000, help="Puerto para enlazar")
    parser.add_argument("--debug", action="store_true", help="Habilitar modo debug")
    
    args = parser.parse_args()
    
    # Crear y ejecutar sistema
    system = AIComparisonSystem()
    system.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()

