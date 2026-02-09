"""
Inspection Entity

Represents a complete inspection with all its results.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import uuid

from .defect import Defect
from .anomaly import Anomaly
from .quality_score import QualityScore
from ..value_objects import ImageMetadata


@dataclass
class Inspection:
    """
    Inspection entity representing a complete quality inspection.
    
    Attributes:
        id: Unique identifier for the inspection
        image_metadata: Metadata about the inspected image
        defects: List of detected defects
        anomalies: List of detected anomalies
        quality_score: Overall quality score
        created_at: Timestamp when inspection was created
        completed_at: Timestamp when inspection was completed
    """
    
    id: str
    image_metadata: ImageMetadata
    quality_score: QualityScore
    defects: List[Defect] = field(default_factory=list)
    anomalies: List[Anomaly] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize inspection ID if not provided."""
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def mark_completed(self):
        """Mark inspection as completed."""
        self.completed_at = datetime.utcnow()
    
    @property
    def is_completed(self) -> bool:
        """Check if inspection is completed."""
        return self.completed_at is not None
    
    @property
    def total_defects(self) -> int:
        """Get total number of defects."""
        return len(self.defects)
    
    @property
    def total_anomalies(self) -> int:
        """Get total number of anomalies."""
        return len(self.anomalies)
    
    @property
    def critical_defects(self) -> List[Defect]:
        """Get list of critical defects."""
        from .defect import DefectSeverity
        return [d for d in self.defects if d.severity == DefectSeverity.CRITICAL]
    
    def add_defect(self, defect: Defect):
        """Add a defect to the inspection."""
        self.defects.append(defect)
        # Update quality score
        self._recalculate_quality_score()
    
    def add_anomaly(self, anomaly: Anomaly):
        """Add an anomaly to the inspection."""
        self.anomalies.append(anomaly)
        # Update quality score
        self._recalculate_quality_score()
    
    def _recalculate_quality_score(self):
        """Recalculate quality score based on defects and anomalies."""
        # Start with perfect score
        score = 100.0
        
        # Subtract penalties from defects
        for defect in self.defects:
            score -= defect.penalty_score
        
        # Subtract penalties from anomalies
        for anomaly in self.anomalies:
            score -= anomaly.penalty_score
        
        # Ensure score doesn't go below 0
        score = max(0.0, score)
        
        # Update quality score
        self.quality_score = QualityScore(
            score=score,
            defects_count=self.total_defects,
            anomalies_count=self.total_anomalies,
        )
    
    def to_dict(self) -> dict:
        """
        Convert inspection to dictionary.
        
        Returns:
            Dictionary representation of the inspection
        """
        return {
            "id": self.id,
            "image_metadata": self.image_metadata.to_dict(),
            "quality_score": self.quality_score.to_dict(),
            "defects": [defect.to_dict() for defect in self.defects],
            "anomalies": [anomaly.to_dict() for anomaly in self.anomalies],
            "total_defects": self.total_defects,
            "total_anomalies": self.total_anomalies,
            "is_completed": self.is_completed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }



