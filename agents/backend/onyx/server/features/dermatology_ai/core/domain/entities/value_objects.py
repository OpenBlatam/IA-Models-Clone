"""
Value objects for domain entities.

Value objects are immutable objects that represent domain concepts
without identity. They are used to encapsulate related data and behavior.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class SkinMetrics:
    """
    Value object representing skin quality metrics.
    
    All scores range from 0.0 to 100.0, where higher values indicate
    better skin quality for that metric.
    """
    
    overall_score: float
    texture_score: float
    hydration_score: float
    elasticity_score: float
    pigmentation_score: float
    pore_size_score: float
    wrinkles_score: float
    redness_score: float
    dark_spots_score: float
    
    def to_dict(self) -> Dict[str, float]:
        """
        Convert metrics to dictionary format.
        
        Returns:
            Dictionary with all metric scores
        """
        return {
            "overall_score": self.overall_score,
            "texture_score": self.texture_score,
            "hydration_score": self.hydration_score,
            "elasticity_score": self.elasticity_score,
            "pigmentation_score": self.pigmentation_score,
            "pore_size_score": self.pore_size_score,
            "wrinkles_score": self.wrinkles_score,
            "redness_score": self.redness_score,
            "dark_spots_score": self.dark_spots_score,
        }


@dataclass(frozen=True)
class Condition:
    """
    Value object representing a detected skin condition.
    
    Contains information about a skin condition detected during analysis,
    including confidence level and severity.
    """
    
    name: str
    confidence: float
    severity: str
    description: Optional[str] = None


@dataclass(frozen=True)
class Recommendation:
    """
    Value object representing a product recommendation.
    
    Contains information about a recommended skincare product
    with priority, confidence, and usage instructions.
    """
    
    product_id: str
    product_name: str
    category: str
    priority: int
    reason: str
    confidence: float
    usage_frequency: Optional[str] = None










