"""
Defect Entity

Represents a defect detected in an inspection.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime


class DefectType(str, Enum):
    """Types of defects that can be detected."""
    SCRATCH = "scratch"
    CRACK = "crack"
    DENT = "dent"
    DISCOLORATION = "discoloration"
    DEFORMATION = "deformation"
    MISSING_PART = "missing_part"
    SURFACE_IMPERFECTION = "surface_imperfection"
    CONTAMINATION = "contamination"
    SIZE_VARIATION = "size_variation"
    OTHER = "other"


class DefectSeverity(str, Enum):
    """Severity levels for defects."""
    CRITICAL = "critical"
    SEVERE = "severe"
    MODERATE = "moderate"
    MINOR = "minor"


@dataclass(frozen=True)
class DefectLocation:
    """Location of a defect in an image."""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def area(self) -> int:
        """Calculate the area of the defect bounding box."""
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get the center point of the defect."""
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class Defect:
    """
    Defect entity representing a detected defect.
    
    Attributes:
        id: Unique identifier for the defect
        type: Type of defect
        severity: Severity level
        location: Location in the image (bounding box)
        confidence: Confidence score (0.0 to 1.0)
        description: Optional description
        detected_at: Timestamp when defect was detected
    """
    
    id: str
    type: DefectType
    severity: DefectSeverity
    location: DefectLocation
    confidence: float
    description: Optional[str] = None
    detected_at: datetime = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.detected_at is None:
            self.detected_at = datetime.utcnow()
        
        # Validate confidence
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
    
    @property
    def penalty_score(self) -> float:
        """
        Calculate penalty score based on severity.
        
        Returns:
            Penalty score (higher = worse)
        """
        penalties = {
            DefectSeverity.CRITICAL: 20.0,
            DefectSeverity.SEVERE: 15.0,
            DefectSeverity.MODERATE: 8.0,
            DefectSeverity.MINOR: 3.0,
        }
        return penalties.get(self.severity, 0.0)
    
    def to_dict(self) -> dict:
        """
        Convert defect to dictionary.
        
        Returns:
            Dictionary representation of the defect
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
            "confidence": self.confidence,
            "description": self.description,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None,
            "penalty_score": self.penalty_score,
        }



