"""
Anomaly Entity

Represents an anomaly detected in an inspection.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""
    STATISTICAL = "statistical"
    AUTOENCODER = "autoencoder"
    EDGE_BASED = "edge_based"
    COLOR_BASED = "color_based"
    DIFFUSION = "diffusion"
    OTHER = "other"


class AnomalySeverity(str, Enum):
    """Severity levels for anomalies."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class AnomalyLocation:
    """Location of an anomaly in an image."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def area(self) -> int:
        """Calculate the area of the anomaly bounding box."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get the center point of the anomaly."""
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class Anomaly:
    """
    Anomaly entity representing a detected anomaly.
    
    Attributes:
        id: Unique identifier for the anomaly
        type: Type of anomaly detection method
        severity: Severity level
        location: Location in the image (bounding box)
        score: Anomaly score (0.0 to 1.0, higher = more anomalous)
        detected_at: Timestamp when anomaly was detected
    """
    
    id: str
    type: AnomalyType
    severity: AnomalySeverity
    location: AnomalyLocation
    score: float
    detected_at: datetime = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.detected_at is None:
            self.detected_at = datetime.utcnow()
        
        # Validate score
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Anomaly score must be between 0.0 and 1.0, got {self.score}")
    
    @property
    def penalty_score(self) -> float:
        """
        Calculate penalty score based on severity.
        
        Returns:
            Penalty score (higher = worse)
        """
        penalties = {
            AnomalySeverity.HIGH: 10.0,
            AnomalySeverity.MEDIUM: 5.0,
            AnomalySeverity.LOW: 2.0,
        }
        return penalties.get(self.severity, 0.0)
    
    def to_dict(self) -> dict:
        """
        Convert anomaly to dictionary.
        
        Returns:
            Dictionary representation of the anomaly
        """
        return {
            "id": self.id,
            "type": self.type.value,
            "severity": self.severity.value,
            "location": {
                "x": self.location.x,
                "y": self.location.y,
                "width": self.location.width,
                "height": self.location.height,
            },
            "score": self.score,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None,
            "penalty_score": self.penalty_score,
        }



