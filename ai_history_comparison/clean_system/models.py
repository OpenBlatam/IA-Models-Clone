"""
Modelos de datos simples y funcionales
=====================================

Solo los modelos esenciales para el sistema.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid
import hashlib


@dataclass
class HistoryEntry:
    """Entrada de historial de IA - simple y directa."""
    id: str
    content: str
    model_version: str
    timestamp: datetime
    quality_score: float
    word_count: int
    readability_score: float
    sentiment_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, content: str, model_version: str, **metrics) -> 'HistoryEntry':
        """Crear nueva entrada de historial."""
        return cls(
            id=str(uuid.uuid4()),
            content=content,
            model_version=model_version,
            timestamp=datetime.utcnow(),
            quality_score=metrics.get('quality_score', 0.0),
            word_count=metrics.get('word_count', len(content.split())),
            readability_score=metrics.get('readability_score', 0.0),
            sentiment_score=metrics.get('sentiment_score', 0.0),
            metadata=metrics.get('metadata', {})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'id': self.id,
            'content': self.content,
            'model_version': self.model_version,
            'timestamp': self.timestamp.isoformat(),
            'quality_score': self.quality_score,
            'word_count': self.word_count,
            'readability_score': self.readability_score,
            'sentiment_score': self.sentiment_score,
            'metadata': self.metadata
        }


@dataclass
class ComparisonResult:
    """Resultado de comparación entre modelos."""
    id: str
    model_a: str
    model_b: str
    similarity_score: float
    quality_difference: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, model_a: str, model_b: str, similarity: float, quality_diff: float) -> 'ComparisonResult':
        """Crear resultado de comparación."""
        return cls(
            id=str(uuid.uuid4()),
            model_a=model_a,
            model_b=model_b,
            similarity_score=similarity,
            quality_difference=quality_diff,
            timestamp=datetime.utcnow()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'id': self.id,
            'model_a': self.model_a,
            'model_b': self.model_b,
            'similarity_score': self.similarity_score,
            'quality_difference': self.quality_difference,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details
        }


@dataclass
class AnalysisJob:
    """Trabajo de análisis."""
    id: str
    status: str  # pending, running, completed, failed
    content: str
    model_version: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[HistoryEntry] = None
    error: Optional[str] = None
    
    @classmethod
    def create(cls, content: str, model_version: str) -> 'AnalysisJob':
        """Crear trabajo de análisis."""
        return cls(
            id=str(uuid.uuid4()),
            status='pending',
            content=content,
            model_version=model_version,
            created_at=datetime.utcnow()
        )
    
    def complete(self, result: HistoryEntry) -> None:
        """Marcar como completado."""
        self.status = 'completed'
        self.result = result
        self.completed_at = datetime.utcnow()
    
    def fail(self, error: str) -> None:
        """Marcar como fallido."""
        self.status = 'failed'
        self.error = error
        self.completed_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            'id': self.id,
            'status': self.status,
            'content': self.content,
            'model_version': self.model_version,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'result': self.result.to_dict() if self.result else None,
            'error': self.error
        }




