"""
Anomaly DTO

Data transfer object for anomaly information.
"""

from dataclasses import dataclass
from typing import Optional, Dict
from datetime import datetime


@dataclass
class AnomalyDTO:
    """
    DTO for anomaly information.
    
    Attributes:
        id: Anomaly identifier
        type: Type of anomaly detection method
        severity: Severity level
        location: Location dictionary with x, y, width, height
        score: Anomaly score (0.0 to 1.0)
        penalty_score: Penalty score for quality calculation
        detected_at: Timestamp when anomaly was detected
    """
    
    id: str
    type: str
    severity: str
    location: Dict[str, int]  # {"x": int, "y": int, "width": int, "height": int}
    score: float
    penalty_score: float = 0.0
    detected_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """
        Convert anomaly DTO to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "type": self.type,
            "severity": self.severity,
            "location": self.location,
            "score": self.score,
            "penalty_score": self.penalty_score,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None,
        }
    
    @classmethod
    def from_domain_entity(cls, anomaly) -> 'AnomalyDTO':
        """
        Create DTO from domain entity.
        
        Args:
            anomaly: Anomaly domain entity
        
        Returns:
            AnomalyDTO instance
        """
        return cls(
            id=anomaly.id,
            type=anomaly.type.value,
            severity=anomaly.severity.value,
            location={
                "x": anomaly.location.x,
                "y": anomaly.location.y,
                "width": anomaly.location.width,
                "height": anomaly.location.height,
            },
            score=anomaly.score,
            penalty_score=anomaly.penalty_score,
            detected_at=anomaly.detected_at,
        )

