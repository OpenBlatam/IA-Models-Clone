"""
Quality Metrics Value Object

Immutable value object representing aggregated quality metrics.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime


@dataclass(frozen=True)
class QualityMetrics:
    """
    Immutable value object representing aggregated quality metrics.
    
    Attributes:
        total_inspections: Total number of inspections
        average_quality_score: Average quality score
        defects_count: Total number of defects
        anomalies_count: Total number of anomalies
        rejection_rate: Percentage of rejected items (0.0 to 100.0)
        calculated_at: Timestamp when metrics were calculated
        period_start: Start of the period for these metrics
        period_end: End of the period for these metrics
    """
    
    total_inspections: int
    average_quality_score: float
    defects_count: int = 0
    anomalies_count: int = 0
    rejection_rate: float = 0.0
    calculated_at: Optional[datetime] = None
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate quality metrics."""
        if self.total_inspections < 0:
            raise ValueError(f"Total inspections cannot be negative, got {self.total_inspections}")
        if not 0.0 <= self.average_quality_score <= 100.0:
            raise ValueError(
                f"Average quality score must be between 0.0 and 100.0, "
                f"got {self.average_quality_score}"
            )
        if self.defects_count < 0:
            raise ValueError(f"Defects count cannot be negative, got {self.defects_count}")
        if self.anomalies_count < 0:
            raise ValueError(f"Anomalies count cannot be negative, got {self.anomalies_count}")
        if not 0.0 <= self.rejection_rate <= 100.0:
            raise ValueError(
                f"Rejection rate must be between 0.0 and 100.0, got {self.rejection_rate}"
            )
    
    @property
    def acceptance_rate(self) -> float:
        """Calculate acceptance rate (100.0 - rejection_rate)."""
        return 100.0 - self.rejection_rate
    
    @property
    def defects_per_inspection(self) -> float:
        """Calculate average defects per inspection."""
        if self.total_inspections == 0:
            return 0.0
        return self.defects_count / self.total_inspections
    
    @property
    def anomalies_per_inspection(self) -> float:
        """Calculate average anomalies per inspection."""
        if self.total_inspections == 0:
            return 0.0
        return self.anomalies_count / self.total_inspections
    
    def to_dict(self) -> dict:
        """
        Convert quality metrics to dictionary.
        
        Returns:
            Dictionary representation of the quality metrics
        """
        return {
            "total_inspections": self.total_inspections,
            "average_quality_score": self.average_quality_score,
            "defects_count": self.defects_count,
            "anomalies_count": self.anomalies_count,
            "rejection_rate": self.rejection_rate,
            "acceptance_rate": self.acceptance_rate,
            "defects_per_inspection": self.defects_per_inspection,
            "anomalies_per_inspection": self.anomalies_per_inspection,
            "calculated_at": self.calculated_at.isoformat() if self.calculated_at else None,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
        }



