"""
Document Predictive - Análisis Predictivo Avanzado
==================================================

Sistema de análisis predictivo para predecir resultados y tendencias.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Resultado de predicción."""
    prediction_type: str
    predicted_value: float
    confidence: float
    factors: Dict[str, float] = field(default_factory=dict)
    timeframe: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendPrediction:
    """Predicción de tendencia."""
    metric_name: str
    current_value: float
    predicted_value: float
    trend_direction: str  # 'up', 'down', 'stable'
    confidence: float
    timeframe_days: int
    factors: List[str] = field(default_factory=list)


class PredictiveAnalyzer:
    """Analizador predictivo."""
    
    def __init__(self, analyzer):
        """Inicializar analizador."""
        self.analyzer = analyzer
        self.historical_data: List[Dict[str, Any]] = []
    
    async def predict_document_quality(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PredictionResult:
        """
        Predecir calidad de documento antes de analizar.
        
        Args:
            content: Contenido del documento
            metadata: Metadatos adicionales
        
        Returns:
            PredictionResult con predicción
        """
        # Factores que influyen en la calidad
        factors = {}
        
        # Longitud del documento
        length_factor = min(len(content) / 1000, 1.0) * 0.3
        factors["length"] = length_factor
        
        # Estructura (detección básica)
        has_paragraphs = '\n\n' in content or '\n' in content
        structure_factor = 0.2 if has_paragraphs else 0.1
        factors["structure"] = structure_factor
        
        # Complejidad (palabras únicas)
        words = content.split()
        unique_words = len(set(words))
        complexity_factor = min(unique_words / len(words) if words else 0, 1.0) * 0.2
        factors["complexity"] = complexity_factor
        
        # Formato (mayúsculas, puntuación)
        has_capitals = any(c.isupper() for c in content)
        has_punctuation = any(c in '.,!?;:' for c in content)
        format_factor = 0.15 if (has_capitals and has_punctuation) else 0.05
        factors["format"] = format_factor
        
        # Metadatos (si están disponibles)
        metadata_factor = 0.15 if metadata and len(metadata) > 0 else 0.0
        factors["metadata"] = metadata_factor
        
        # Calcular predicción
        predicted_quality = sum(factors.values())
        confidence = min(len(content) / 500, 1.0)  # Más confianza con más contenido
        
        return PredictionResult(
            prediction_type="quality",
            predicted_value=predicted_quality * 100,
            confidence=confidence,
            factors=factors,
            metadata={
                "content_length": len(content),
                "word_count": len(words)
            }
        )
    
    async def predict_processing_time(
        self,
        content: str,
        tasks: Optional[List[str]] = None
    ) -> PredictionResult:
        """
        Predecir tiempo de procesamiento.
        
        Args:
            content: Contenido del documento
            tasks: Lista de tareas a realizar
        
        Returns:
            PredictionResult con predicción
        """
        tasks = tasks or ["classification", "summarization"]
        
        base_time_per_char = 0.001  # segundos por carácter
        task_multiplier = len(tasks)
        
        estimated_time = len(content) * base_time_per_char * task_multiplier
        
        # Ajustar según complejidad
        complexity = len(set(content.split())) / len(content.split()) if content.split() else 0
        complexity_multiplier = 1 + (complexity * 0.5)
        
        estimated_time *= complexity_multiplier
        
        confidence = 0.75  # Confianza moderada
        
        return PredictionResult(
            prediction_type="processing_time",
            predicted_value=estimated_time,
            confidence=confidence,
            factors={
                "content_length": len(content),
                "task_count": len(tasks),
                "complexity": complexity
            },
            metadata={
                "tasks": tasks
            }
        )
    
    async def predict_trend(
        self,
        metric_name: str,
        historical_values: List[float],
        timeframe_days: int = 30
    ) -> TrendPrediction:
        """
        Predecir tendencia de una métrica.
        
        Args:
            metric_name: Nombre de la métrica
            historical_values: Valores históricos
            timeframe_days: Días hacia el futuro
        
        Returns:
            TrendPrediction con predicción
        """
        if not historical_values:
            raise ValueError("Se requieren valores históricos")
        
        current_value = historical_values[-1]
        
        # Calcular tendencia simple (promedio móvil)
        if len(historical_values) >= 3:
            recent_avg = statistics.mean(historical_values[-3:])
            older_avg = statistics.mean(historical_values[:-3]) if len(historical_values) > 3 else historical_values[0]
            
            trend = recent_avg - older_avg
            trend_direction = "up" if trend > 0 else "down" if trend < 0 else "stable"
            
            # Proyectar valor futuro
            predicted_value = current_value + (trend * (timeframe_days / 30))
        else:
            trend_direction = "stable"
            predicted_value = current_value
        
        # Calcular confianza basada en variabilidad
        if len(historical_values) > 1:
            variance = statistics.variance(historical_values)
            confidence = max(0.5, 1.0 - (variance / 100))
        else:
            confidence = 0.5
        
        factors = []
        if len(historical_values) >= 5:
            factors.append("Tendencia basada en datos históricos")
        if variance < 10:
            factors.append("Baja variabilidad")
        
        return TrendPrediction(
            metric_name=metric_name,
            current_value=current_value,
            predicted_value=predicted_value,
            trend_direction=trend_direction,
            confidence=confidence,
            timeframe_days=timeframe_days,
            factors=factors
        )
    
    def add_historical_data_point(self, data: Dict[str, Any]):
        """Agregar punto de datos histórico."""
        self.historical_data.append({
            **data,
            "timestamp": datetime.now()
        })
    
    def get_historical_data(self, metric_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtener datos históricos."""
        if metric_name:
            return [d for d in self.historical_data if d.get("metric") == metric_name]
        return self.historical_data


__all__ = [
    "PredictiveAnalyzer",
    "PredictionResult",
    "TrendPrediction"
]


