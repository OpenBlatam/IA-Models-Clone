"""
Detection Result Value Object

Immutable value object representing a detection result.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Any


@dataclass(frozen=True)
class DetectionResult:
    """
    Immutable value object representing a detection result.
    
    Attributes:
        detected: Whether something was detected
        confidence: Confidence score (0.0 to 1.0)
        location: Optional bounding box (x, y, width, height)
        class_name: Optional class name for the detection
        metadata: Optional additional metadata
    """
    
    detected: bool
    confidence: float = 0.0
    location: Optional[Tuple[int, int, int, int]] = None
    class_name: Optional[str] = None
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        """Validate detection result."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        if self.location is not None:
            if len(self.location) != 4:
                raise ValueError(f"Location must be a 4-tuple (x, y, width, height), got {self.location}")
            x, y, width, height = self.location
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValueError(f"Invalid location values: {self.location}")
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if detection has high confidence (>= 0.7)."""
        return self.confidence >= 0.7
    
    @property
    def is_medium_confidence(self) -> bool:
        """Check if detection has medium confidence (0.5-0.7)."""
        return 0.5 <= self.confidence < 0.7
    
    @property
    def is_low_confidence(self) -> bool:
        """Check if detection has low confidence (< 0.5)."""
        return self.confidence < 0.5
    
    def to_dict(self) -> dict:
        """
        Convert detection result to dictionary.
        
        Returns:
            Dictionary representation of the detection result
        """
        result = {
            "detected": self.detected,
            "confidence": self.confidence,
            "class_name": self.class_name,
            "is_high_confidence": self.is_high_confidence,
            "is_medium_confidence": self.is_medium_confidence,
            "is_low_confidence": self.is_low_confidence,
        }
        
        if self.location is not None:
            result["location"] = {
                "x": self.location[0],
                "y": self.location[1],
                "width": self.location[2],
                "height": self.location[3],
            }
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result



