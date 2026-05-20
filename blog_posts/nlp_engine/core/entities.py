from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from .enums import AnalysisType, ProcessingTier, AnalysisStatus, ErrorType
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🎯 DOMAIN ENTITIES - Entidades del Dominio NLP
==============================================

Entidades y Value Objects que representan los conceptos
centrales del dominio NLP.
"""



@dataclass(frozen=True)
class TextFingerprint:
    """Value Object para identificación única del texto."""
    hash_value: str
    length: int
    language_hint: Optional[str] = None
    encoding: str = "utf-8"
    
    @classmethod
    def create(cls, text: str, language_hint: Optional[str] = None) -> 'TextFingerprint':
        """Factory method para crear fingerprint del texto."""
        if not text:
            raise ValueError("Text cannot be empty")
        
        # Usar BLAKE2b para hash rápido y seguro
        hash_value = hashlib.blake2b(
            text.encode('utf-8'), 
            digest_size=16
        ).hexdigest()
        
        return cls(
            hash_value=hash_value,
            length=len(text),
            language_hint=language_hint
        )
    
    @property
    def short_hash(self) -> str:
        """Hash corto para logging."""
        return self.hash_value[:8]


@dataclass(frozen=True)
class AnalysisScore:
    """Value Object para puntuaciones con validación de dominio."""
    value: float
    confidence: float = 1.0
    method: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> Any:
        """Validar reglas de dominio."""
        if not 0 <= self.value <= 100:
            raise ValueError(f"Score debe estar entre 0-100, recibido: {self.value}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence debe estar entre 0-1, recibido: {self.confidence}")
    
    @property
    def weighted_value(self) -> float:
        """Valor ponderado por confianza."""
        return self.value * self.confidence
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Verificar si tiene alta confianza."""
        return self.confidence >= threshold


@dataclass(frozen=True)
class ProcessingMetrics:
    """Value Object para métricas de procesamiento."""
    start_time_ns: int
    end_time_ns: int
    cache_hit: bool
    cache_level: str
    model_used: str
    tier: ProcessingTier
    optimization_applied: List[str] = field(default_factory=list)
    memory_used_mb: Optional[float] = None
    
    @property
    def duration_ms(self) -> float:
        """Duración en milisegundos."""
        return (self.end_time_ns - self.start_time_ns) / 1_000_000
    
    @property
    def duration_ns(self) -> int:
        """Duración en nanosegundos."""
        return self.end_time_ns - self.start_time_ns
    
    @property
    def is_ultra_fast(self) -> bool:
        """Verificar si cumple target ultra-fast."""
        return self.duration_ms < 0.1
    
    def meets_tier_target(self) -> bool:
        """Verificar si cumple target del tier."""
        targets = {
            ProcessingTier.ULTRA_FAST: 0.1,
            ProcessingTier.BALANCED: 1.0,
            ProcessingTier.HIGH_QUALITY: 10.0,
            ProcessingTier.RESEARCH_GRADE: 100.0
        }
        target = targets.get(self.tier, 1.0)
        return self.duration_ms <= target


@dataclass
class AnalysisError:
    """Entity para errores de análisis."""
    error_type: ErrorType
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    recoverable: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización."""
        return {
            'type': self.error_type.value,
            'message': self.message,
            'context': self.context,
            'timestamp': self.timestamp,
            'recoverable': self.recoverable
        }


@dataclass
class AnalysisResult:
    """Aggregate Root para resultados de análisis NLP."""
    fingerprint: TextFingerprint
    status: AnalysisStatus = AnalysisStatus.PENDING
    scores: Dict[AnalysisType, AnalysisScore] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Optional[ProcessingMetrics] = None
    errors: List[AnalysisError] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    _version: int = field(default=1, init=False)
    
    def add_score(
        self, 
        analysis_type: AnalysisType, 
        value: float, 
        confidence: float = 1.0, 
        method: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Añadir score aplicando reglas de dominio."""
        if self.status == AnalysisStatus.FAILED:
            raise ValueError("Cannot add score to failed analysis")
        
        score = AnalysisScore(
            value=value, 
            confidence=confidence, 
            method=method,
            metadata=metadata or {}
        )
        
        self.scores[analysis_type] = score
        self.updated_at = time.time()
        
        # Transición de estado automática
        if self.status == AnalysisStatus.PENDING:
            self.status = AnalysisStatus.PROCESSING
    
    def add_error(
        self, 
        error_type: ErrorType, 
        message: str, 
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ) -> None:
        """Añadir error aplicando reglas de dominio."""
        error = AnalysisError(
            error_type=error_type,
            message=message,
            context=context or {},
            recoverable=recoverable
        )
        
        self.errors.append(error)
        self.updated_at = time.time()
        
        # Cambiar estado a failed si es error no recuperable
        if not recoverable:
            self.status = AnalysisStatus.FAILED
    
    def complete(self, metrics: Optional[ProcessingMetrics] = None) -> None:
        """Marcar análisis como completado."""
        if not self.scores:
            raise ValueError("Cannot complete analysis without scores")
        
        self.status = AnalysisStatus.COMPLETED
        self.metrics = metrics
        self.updated_at = time.time()
    
    def get_score(self, analysis_type: AnalysisType) -> Optional[AnalysisScore]:
        """Obtener score de forma segura."""
        return self.scores.get(analysis_type)
    
    def get_overall_quality(self) -> float:
        """Calcular calidad general usando lógica de dominio."""
        if not self.scores:
            return 0.0
        
        # Promedio ponderado por confianza
        total_weighted = sum(score.weighted_value for score in self.scores.values())
        total_confidence = sum(score.confidence for score in self.scores.values())
        
        return total_weighted / total_confidence if total_confidence > 0 else 0.0
    
    def is_valid(self) -> bool:
        """Verificar validez según reglas de dominio."""
        if self.status == AnalysisStatus.FAILED:
            return False
        
        # Verificar que no hay errores críticos
        critical_errors = [e for e in self.errors if not e.recoverable]
        if critical_errors:
            return False
        
        # Debe tener al menos un score para ser válido
        return len(self.scores) > 0
    
    def has_high_confidence_scores(self, threshold: float = 0.8) -> bool:
        """Verificar si todos los scores tienen alta confianza."""
        if not self.scores:
            return False
        
        return all(score.is_high_confidence(threshold) for score in self.scores.values())
    
    def get_performance_grade(self) -> str:
        """Obtener grado de performance basado en métricas."""
        if not self.metrics:
            return "unknown"
        
        if self.metrics.duration_ms < 0.1:
            return "A+"  # Ultra-fast
        elif self.metrics.duration_ms < 1.0:
            return "A"   # Fast
        elif self.metrics.duration_ms < 10.0:
            return "B"   # Good
        elif self.metrics.duration_ms < 100.0:
            return "C"   # Acceptable
        else:
            return "D"   # Slow
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización."""
        return {
            'fingerprint': {
                'hash': self.fingerprint.hash_value,
                'length': self.fingerprint.length,
                'language_hint': self.fingerprint.language_hint
            },
            'status': self.status.value,
            'scores': {
                analysis_type.name.lower(): {
                    'value': score.value,
                    'confidence': score.confidence,
                    'method': score.method,
                    'metadata': score.metadata
                }
                for analysis_type, score in self.scores.items()
            },
            'metadata': self.metadata,
            'metrics': {
                'duration_ms': self.metrics.duration_ms,
                'cache_hit': self.metrics.cache_hit,
                'tier': self.metrics.tier.value,
                'performance_grade': self.get_performance_grade()
            } if self.metrics else None,
            'errors': [error.to_dict() for error in self.errors],
            'overall_quality': self.get_overall_quality(),
            'is_valid': self.is_valid(),
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'version': self._version
        } 