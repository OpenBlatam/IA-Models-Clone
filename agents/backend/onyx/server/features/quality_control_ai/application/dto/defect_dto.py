"""
Defect DTO

Data transfer object for defect information.
"""

from dataclasses import dataclass
from typing import Optional, Dict
from datetime import datetime


@dataclass
class DefectDTO:
    """
    DTO for defect information.
    
    Attributes:
        id: Defect identifier
        type: Type of defect
        severity: Severity level
        location: Location dictionary with x, y, width, height
        confidence: Confidence score (0.0 to 1.0)
        description: Optional description
        penalty_score: Penalty score for quality calculation
        detected_at: Timestamp when defect was detected
    """
    
    id: str
    type: str
    severity: str
    location: Dict[str, int]  # {"x": int, "y": int, "width": int, "height": int}
    confidence: float
    description: Optional[str] = None
    penalty_score: float = 0.0
    detected_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """
        Convert defect DTO to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "id": self.id,
            "type": self.type,
            "severity": self.severity,
            "location": self.location,
            "confidence": self.confidence,
            "description": self.description,
            "penalty_score": self.penalty_score,
            "detected_at": self.detected_at.isoformat() if self.detected_at else None,
        }
    
    @classmethod
    def from_domain_entity(cls, defect) -> 'DefectDTO':
        """
        Create DTO from domain entity.
        
        Args:
            defect: Defect domain entity
        
        Returns:
            DefectDTO instance
        """
        return cls(
            id=defect.id,
            type=defect.type.value,
            severity=defect.severity.value,
            location={
                "x": defect.location.x,
                "y": defect.location.y,
                "width": defect.location.width,
                "height": defect.location.height,
            },
            confidence=defect.confidence,
            description=defect.description,
            penalty_score=defect.penalty_score,
            detected_at=defect.detected_at,
        )



