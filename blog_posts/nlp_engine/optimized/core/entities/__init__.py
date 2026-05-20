from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from .models import (
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🎯 ENTITIES - Domain Models
===========================

Entidades del dominio NLP.
"""

    TextInput,
    AnalysisResult,
    BatchResult,
    AnalysisType,
    OptimizationTier,
    PerformanceMetrics
)

__all__ = [
    'TextInput',
    'AnalysisResult',
    'BatchResult', 
    'AnalysisType',
    'OptimizationTier',
    'PerformanceMetrics'
]



class AnalysisType(Enum):
    """Tipos de análisis disponibles."""
    SENTIMENT = "sentiment"
    QUALITY = "quality"
    EMOTION = "emotion"
    LANGUAGE = "language"


class OptimizationTier(Enum):
    """Niveles de optimización."""
    STANDARD = "standard"
    ADVANCED = "advanced" 
    ULTRA = "ultra"
    EXTREME = "extreme"


@dataclass
class TextInput:
    """Entrada de texto para análisis."""
    content: str
    id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> Any:
        if self.id is None:
            self.id = f"text_{hash(self.content) % 1000000}"


@dataclass
class AnalysisResult:
    """Resultado de análisis individual."""
    text_id: str
    analysis_type: AnalysisType
    score: float
    confidence: float
    processing_time_ms: float
    metadata: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text_id': self.text_id,
            'analysis_type': self.analysis_type.value,
            'score': self.score,
            'confidence': self.confidence,
            'processing_time_ms': self.processing_time_ms,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class BatchResult:
    """Resultado de análisis en lote."""
    results: List[AnalysisResult]
    total_processing_time_ms: float
    optimization_tier: OptimizationTier
    success_rate: float
    metadata: Dict[str, Any]
    
    @property
    def average_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)
    
    @property
    def total_texts(self) -> int:
        return len(self.results)
    
    @property
    def throughput_ops_per_second(self) -> float:
        if self.total_processing_time_ms <= 0:
            return 0.0
        return self.total_texts / (self.total_processing_time_ms / 1000)


@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento."""
    latency_ms: float
    throughput_ops_per_second: float
    memory_usage_mb: float
    cpu_utilization_percent: float
    cache_hit_ratio: float
    optimization_factor: float
    timestamp: datetime
    
    def is_healthy(self) -> bool:
        """Verificar si las métricas están dentro de rangos saludables."""
        return (
            self.latency_ms < 100 and  # < 100ms
            self.throughput_ops_per_second > 100 and  # > 100 ops/s
            self.cpu_utilization_percent < 80 and  # < 80% CPU
            self.cache_hit_ratio > 0.7  # > 70% cache hits
        )


@dataclass
class SystemStatus:
    """Estado del sistema."""
    is_initialized: bool
    optimization_tier: OptimizationTier
    available_optimizers: Dict[str, bool]
    performance_metrics: Optional[PerformanceMetrics]
    error_count: int
    total_requests: int
    uptime_seconds: float
    
    def get_health_status(self) -> str:
        """Obtener estado de salud del sistema."""
        if not self.is_initialized:
            return "unhealthy"
        
        if self.performance_metrics and self.performance_metrics.is_healthy():
            return "healthy"
        
        return "degraded" 