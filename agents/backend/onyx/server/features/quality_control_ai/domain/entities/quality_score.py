"""
Quality Score Entity

Represents the quality score and status of an inspection.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


class QualityStatus(str, Enum):
    """Quality status based on score."""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"           # 75-89
    ACCEPTABLE = "acceptable"  # 60-74
    POOR = "poor"           # 40-59
    REJECTED = "rejected"   # 0-39


@dataclass
class QualityScore:
    """
    Quality score entity representing the overall quality assessment.
    
    Attributes:
        score: Quality score (0.0 to 100.0)
        status: Quality status based on score
        defects_count: Number of defects detected
        anomalies_count: Number of anomalies detected
        calculated_at: Timestamp when score was calculated
    """
    
    score: float
    defects_count: int = 0
    anomalies_count: int = 0
    calculated_at: datetime = None
    
    def __post_init__(self):
        """Initialize default values and validate."""
        if self.calculated_at is None:
            self.calculated_at = datetime.utcnow()
        
        # Validate score
        if not 0.0 <= self.score <= 100.0:
            raise ValueError(f"Quality score must be between 0.0 and 100.0, got {self.score}")
        
        # Validate counts
        if self.defects_count < 0:
            raise ValueError(f"Defects count cannot be negative, got {self.defects_count}")
        if self.anomalies_count < 0:
            raise ValueError(f"Anomalies count cannot be negative, got {self.anomalies_count}")
    
    @property
    def status(self) -> QualityStatus:
        """
        Get quality status based on score.
        
        Returns:
            Quality status
        """
        if self.score >= 90.0:
            return QualityStatus.EXCELLENT
        elif self.score >= 75.0:
            return QualityStatus.GOOD
        elif self.score >= 60.0:
            return QualityStatus.ACCEPTABLE
        elif self.score >= 40.0:
            return QualityStatus.POOR
        else:
            return QualityStatus.REJECTED
    
    @property
    def is_acceptable(self) -> bool:
        """
        Check if quality is acceptable (score >= 60).
        
        Returns:
            True if quality is acceptable
        """
        return self.score >= 60.0
    
    @property
    def recommendation(self) -> str:
        """
        Get recommendation based on quality status.
        
        Returns:
            Recommendation string
        """
        recommendations = {
            QualityStatus.EXCELLENT: "Product meets excellent quality standards. Approve.",
            QualityStatus.GOOD: "Product meets good quality standards. Approve.",
            QualityStatus.ACCEPTABLE: "Product meets minimum quality standards. Approve with caution.",
            QualityStatus.POOR: "Product quality is below standards. Requires review before approval.",
            QualityStatus.REJECTED: "Product quality is unacceptable. Reject.",
        }
        return recommendations.get(self.status, "Unknown quality status.")
    
    def to_dict(self) -> dict:
        """
        Convert quality score to dictionary.
        
        Returns:
            Dictionary representation of the quality score
        """
        return {
            "score": self.score,
            "status": self.status.value,
            "defects_count": self.defects_count,
            "anomalies_count": self.anomalies_count,
            "is_acceptable": self.is_acceptable,
            "recommendation": self.recommendation,
            "calculated_at": self.calculated_at.isoformat() if self.calculated_at else None,
        }



