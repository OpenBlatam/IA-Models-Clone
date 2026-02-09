"""
Analysis entity - Domain model for skin analysis results.

Represents a complete skin analysis with metrics, conditions, and status.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

from .enums import AnalysisStatus, SkinType
from .value_objects import SkinMetrics, Condition


@dataclass
class Analysis:
    """
    Analysis entity representing a skin analysis result.
    
    Contains all information about a skin analysis including metrics,
    detected conditions, skin type, and processing status.
    """
    
    id: str
    user_id: str
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    metrics: Optional[SkinMetrics] = None
    conditions: List[Condition] = field(default_factory=list)
    skin_type: Optional[SkinType] = None
    status: AnalysisStatus = AnalysisStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def mark_completed(
        self,
        metrics: SkinMetrics,
        conditions: List[Condition]
    ) -> None:
        """
        Mark analysis as completed with results.
        
        Args:
            metrics: Calculated skin metrics
            conditions: List of detected skin conditions
        """
        self.metrics = metrics
        self.conditions = conditions
        self.status = AnalysisStatus.COMPLETED
        self.completed_at = datetime.utcnow()
    
    def mark_failed(self) -> None:
        """Mark analysis as failed and set completion timestamp."""
        self.status = AnalysisStatus.FAILED
        self.completed_at = datetime.utcnow()
    
    def is_completed(self) -> bool:
        """
        Check if analysis is completed.
        
        Returns:
            True if analysis status is COMPLETED, False otherwise
        """
        return self.status == AnalysisStatus.COMPLETED









