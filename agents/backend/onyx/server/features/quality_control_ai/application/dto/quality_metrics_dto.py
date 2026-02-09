"""
Quality Metrics DTO

Data transfer object for quality metrics.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class QualityMetricsDTO:
    """
    DTO for quality metrics.
    
    Attributes:
        total_inspections: Total number of inspections
        average_quality_score: Average quality score
        defects_count: Total number of defects
        anomalies_count: Total number of anomalies
        rejection_rate: Percentage of rejected items
        acceptance_rate: Percentage of accepted items
        defects_per_inspection: Average defects per inspection
        anomalies_per_inspection: Average anomalies per inspection
        calculated_at: Timestamp when metrics were calculated
        period_start: Start of the period
        period_end: End of the period
    """
    
    total_inspections: int
    average_quality_score: float
    defects_count: int = 0
    anomalies_count: int = 0
    rejection_rate: float = 0.0
    acceptance_rate: float = 100.0
    defects_per_inspection: float = 0.0
    anomalies_per_inspection: float = 0.0
    calculated_at: Optional[datetime] = None
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """
        Convert quality metrics DTO to dictionary.
        
        Returns:
            Dictionary representation
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
    
    @classmethod
    def from_domain_value_object(cls, quality_metrics) -> 'QualityMetricsDTO':
        """
        Create DTO from domain value object.
        
        Args:
            quality_metrics: QualityMetrics domain value object
        
        Returns:
            QualityMetricsDTO instance
        """
        return cls(
            total_inspections=quality_metrics.total_inspections,
            average_quality_score=quality_metrics.average_quality_score,
            defects_count=quality_metrics.defects_count,
            anomalies_count=quality_metrics.anomalies_count,
            rejection_rate=quality_metrics.rejection_rate,
            acceptance_rate=quality_metrics.acceptance_rate,
            defects_per_inspection=quality_metrics.defects_per_inspection,
            anomalies_per_inspection=quality_metrics.anomalies_per_inspection,
            calculated_at=quality_metrics.calculated_at,
            period_start=quality_metrics.period_start,
            period_end=quality_metrics.period_end,
        )



