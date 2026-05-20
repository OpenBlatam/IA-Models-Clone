"""
Document ML - Machine Learning para Mejoras Automáticas
========================================================

Sistema de ML para mejorar automáticamente el análisis de documentos.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class MLModel:
    """Modelo de ML."""
    model_name: str
    model_type: str  # 'classification', 'regression', 'clustering'
    accuracy: float = 0.0
    last_trained: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Resultado de predicción."""
    prediction: Any
    confidence: float
    model_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentMLPredictor:
    """Predictor ML para documentos."""
    
    def __init__(self, analyzer):
        """Inicializar predictor."""
        self.analyzer = analyzer
        self.models: Dict[str, MLModel] = {}
        self.training_data: List[Dict[str, Any]] = []
        self.feature_cache: Dict[str, Any] = {}
    
    async def predict_document_quality(
        self,
        features: Dict[str, Any]
    ) -> PredictionResult:
        """
        Predecir calidad de documento.
        
        Args:
            features: Features del documento
        
        Returns:
            PredictionResult con predicción
        """
        # Modelo simplificado - en producción usar modelo real entrenado
        quality_score = self._calculate_predicted_quality(features)
        
        return PredictionResult(
            prediction=quality_score,
            confidence=0.85,
            model_name="quality_predictor",
            metadata={"method": "rule_based"}
        )
    
    async def predict_processing_time(
        self,
        document_length: int,
        tasks_count: int
    ) -> PredictionResult:
        """
        Predecir tiempo de procesamiento.
        
        Args:
            document_length: Longitud del documento
            tasks_count: Número de tareas
        
        Returns:
            PredictionResult con predicción
        """
        # Modelo simplificado
        base_time = 0.5  # segundos base
        length_factor = document_length / 1000 * 0.1
        tasks_factor = tasks_count * 0.2
        
        predicted_time = base_time + length_factor + tasks_factor
        
        return PredictionResult(
            prediction=predicted_time,
            confidence=0.75,
            model_name="processing_time_predictor",
            metadata={
                "document_length": document_length,
                "tasks_count": tasks_count
            }
        )
    
    async def predict_optimal_batch_size(
        self,
        total_documents: int,
        avg_document_size: int,
        available_memory: int
    ) -> PredictionResult:
        """
        Predecir tamaño óptimo de batch.
        
        Args:
            total_documents: Total de documentos
            avg_document_size: Tamaño promedio
            available_memory: Memoria disponible
        
        Returns:
            PredictionResult con predicción
        """
        # Calcular tamaño óptimo basado en memoria
        memory_per_doc = avg_document_size * 3  # Factor de seguridad
        max_batch = int(available_memory * 0.7 / memory_per_doc) if memory_per_doc > 0 else 10
        
        optimal_size = min(max_batch, total_documents, 32)  # Limitar a 32
        
        return PredictionResult(
            prediction=optimal_size,
            confidence=0.8,
            model_name="batch_size_predictor",
            metadata={
                "total_documents": total_documents,
                "avg_document_size": avg_document_size,
                "available_memory": available_memory
            }
        )
    
    def _calculate_predicted_quality(self, features: Dict[str, Any]) -> float:
        """Calcular calidad predicha (modelo simplificado)."""
        score = 50.0  # Base
        
        # Word count
        word_count = features.get("word_count", 0)
        if 100 < word_count < 5000:
            score += 10
        elif word_count > 5000:
            score += 5
        
        # Sentence count
        sentence_count = features.get("sentence_count", 0)
        if sentence_count > 5:
            score += 10
        
        # Paragraph count
        paragraph_count = features.get("paragraph_count", 0)
        if paragraph_count > 1:
            score += 10
        
        # Has structure
        if features.get("has_structure", False):
            score += 15
        
        # Has keywords
        if features.get("has_keywords", False):
            score += 5
        
        return min(100, score)
    
    def record_training_data(
        self,
        features: Dict[str, Any],
        actual_quality: float,
        actual_processing_time: float
    ):
        """Registrar datos de entrenamiento."""
        self.training_data.append({
            "features": features,
            "actual_quality": actual_quality,
            "actual_processing_time": actual_processing_time,
            "timestamp": datetime.now()
        })
        
        # Mantener solo últimos 1000
        if len(self.training_data) > 1000:
            self.training_data = self.training_data[-1000:]
    
    async def train_models(self):
        """Entrenar modelos (placeholder)."""
        # En producción, entrenar modelos reales con sklearn, tensorflow, etc.
        logger.info("Entrenando modelos...")
        
        if len(self.training_data) < 10:
            logger.warning("Datos insuficientes para entrenamiento")
            return
        
        # Placeholder para entrenamiento real
        logger.info(f"Modelos entrenados con {len(self.training_data)} ejemplos")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información de modelos."""
        return {
            "total_models": len(self.models),
            "models": {
                name: {
                    "type": model.model_type,
                    "accuracy": model.accuracy,
                    "last_trained": model.last_trained.isoformat() if model.last_trained else None
                }
                for name, model in self.models.items()
            },
            "training_data_count": len(self.training_data)
        }


class DocumentTrendAnalyzer:
    """Analizador de tendencias."""
    
    def __init__(self, analyzer):
        """Inicializar analizador."""
        self.analyzer = analyzer
    
    async def analyze_trends(
        self,
        metrics_data: List[Dict[str, Any]],
        period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Analizar tendencias.
        
        Args:
            metrics_data: Datos de métricas
            period_days: Período en días
        
        Returns:
            Diccionario con análisis de tendencias
        """
        if not metrics_data:
            return {"error": "No hay datos"}
        
        # Agrupar por fecha
        daily_data = defaultdict(list)
        for metric in metrics_data:
            date = metric.get("date")
            if date:
                daily_data[date].append(metric)
        
        # Calcular tendencias
        quality_trend = self._calculate_trend(
            [m.get("quality_score", 0) for metrics in daily_data.values() for m in metrics]
        )
        
        grammar_trend = self._calculate_trend(
            [m.get("grammar_score", 0) for metrics in daily_data.values() for m in metrics]
        )
        
        processing_trend = self._calculate_trend(
            [m.get("processing_time", 0) for metrics in daily_data.values() for m in metrics]
        )
        
        return {
            "period_days": period_days,
            "total_data_points": len(metrics_data),
            "trends": {
                "quality": quality_trend,
                "grammar": grammar_trend,
                "processing_time": processing_trend
            },
            "predictions": {
                "next_quality": quality_trend.get("predicted_next", 0),
                "next_grammar": grammar_trend.get("predicted_next", 0)
            }
        }
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calcular tendencia de valores."""
        if len(values) < 2:
            return {"direction": "stable", "change": 0}
        
        # Calcular dirección
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        avg_first = sum(first_half) / len(first_half) if first_half else 0
        avg_second = sum(second_half) / len(second_half) if second_half else 0
        
        change = avg_second - avg_first
        direction = "up" if change > 0 else ("down" if change < 0 else "stable")
        
        # Predicción simple (promedio móvil)
        if len(values) >= 3:
            recent = values[-3:]
            predicted_next = sum(recent) / len(recent)
        else:
            predicted_next = values[-1] if values else 0
        
        return {
            "direction": direction,
            "change": change,
            "change_percentage": (change / avg_first * 100) if avg_first > 0 else 0,
            "predicted_next": predicted_next,
            "current": values[-1] if values else 0,
            "average": sum(values) / len(values) if values else 0
        }


__all__ = [
    "DocumentMLPredictor",
    "DocumentTrendAnalyzer",
    "MLModel",
    "PredictionResult"
]
















