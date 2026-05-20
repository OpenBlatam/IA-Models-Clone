"""
Comparison Result Entity
========================

Entidad que representa el resultado de una comparación entre modelos.
Responsabilidad única: Gestionar datos de comparación entre modelos.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional
import uuid


@dataclass
class ComparisonResult:
    """
    Entidad que representa el resultado de una comparación entre modelos.
    
    Responsabilidad única: Gestionar los datos y comportamiento
    de un resultado de comparación entre modelos de IA.
    """
    id: str
    model_a: str
    model_b: str
    similarity: float
    quality_diff: float
    timestamp: datetime
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validar datos después de la inicialización."""
        if not self.model_a or not self.model_a.strip():
            raise ValueError("Model A cannot be empty")
        
        if not self.model_b or not self.model_b.strip():
            raise ValueError("Model B cannot be empty")
        
        if not (0.0 <= self.similarity <= 1.0):
            raise ValueError("Similarity must be between 0.0 and 1.0")
        
        if self.quality_diff < 0:
            raise ValueError("Quality difference cannot be negative")
        
        if self.details is None:
            self.details = {}
    
    @classmethod
    def create(
        cls,
        model_a: str,
        model_b: str,
        similarity: float,
        quality_diff: float,
        details: Optional[Dict[str, Any]] = None
    ) -> 'ComparisonResult':
        """
        Factory method para crear un nuevo resultado de comparación.
        
        Args:
            model_a: Primer modelo
            model_b: Segundo modelo
            similarity: Puntuación de similitud
            quality_diff: Diferencia de calidad
            details: Detalles opcionales
            
        Returns:
            Nueva instancia de ComparisonResult
        """
        return cls(
            id=str(uuid.uuid4()),
            model_a=model_a,
            model_b=model_b,
            similarity=similarity,
            quality_diff=quality_diff,
            timestamp=datetime.utcnow(),
            details=details or {}
        )
    
    def is_high_similarity(self, threshold: float = 0.8) -> bool:
        """
        Verificar si la similitud es alta.
        
        Args:
            threshold: Umbral de similitud
            
        Returns:
            True si la similitud es alta
        """
        return self.similarity >= threshold
    
    def is_significant_quality_diff(self, threshold: float = 0.2) -> bool:
        """
        Verificar si la diferencia de calidad es significativa.
        
        Args:
            threshold: Umbral de diferencia
            
        Returns:
            True si la diferencia es significativa
        """
        return self.quality_diff >= threshold
    
    def get_winner(self) -> Optional[str]:
        """
        Obtener el modelo ganador basado en la calidad.
        
        Returns:
            Nombre del modelo ganador o None si es empate
        """
        if self.quality_diff < 0.01:  # Empate
            return None
        
        # El modelo con mayor calidad se determina por los detalles
        if "model_a_quality" in self.details and "model_b_quality" in self.details:
            if self.details["model_a_quality"] > self.details["model_b_quality"]:
                return self.model_a
            else:
                return self.model_b
        
        return None
    
    def is_recent(self, days: int = 7) -> bool:
        """
        Verificar si la comparación es reciente.
        
        Args:
            days: Número de días
            
        Returns:
            True si la comparación es reciente
        """
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        return self.timestamp >= cutoff
    
    def has_detail(self, key: str) -> bool:
        """
        Verificar si existe una clave en los detalles.
        
        Args:
            key: Clave a verificar
            
        Returns:
            True si la clave existe
        """
        return key in self.details
    
    def get_detail(self, key: str, default: Any = None) -> Any:
        """
        Obtener valor de detalles por clave.
        
        Args:
            key: Clave de detalles
            default: Valor por defecto
            
        Returns:
            Valor de detalles o valor por defecto
        """
        return self.details.get(key, default)
    
    def update_detail(self, key: str, value: Any) -> None:
        """
        Actualizar valor de detalles.
        
        Args:
            key: Clave de detalles
            value: Nuevo valor
        """
        self.details[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertir a diccionario.
        
        Returns:
            Representación en diccionario
        """
        return {
            "id": self.id,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "similarity": self.similarity,
            "quality_diff": self.quality_diff,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComparisonResult':
        """
        Crear desde diccionario.
        
        Args:
            data: Datos del diccionario
            
        Returns:
            Instancia de ComparisonResult
        """
        return cls(
            id=data["id"],
            model_a=data["model_a"],
            model_b=data["model_b"],
            similarity=data["similarity"],
            quality_diff=data["quality_diff"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            details=data.get("details", {})
        )
    
    def __str__(self) -> str:
        """Representación de cadena."""
        return f"ComparisonResult(id={self.id}, models={self.model_a} vs {self.model_b}, similarity={self.similarity:.2f})"
    
    def __repr__(self) -> str:
        """Representación detallada."""
        return f"ComparisonResult(id='{self.id}', model_a='{self.model_a}', model_b='{self.model_b}', similarity={self.similarity:.2f})"




