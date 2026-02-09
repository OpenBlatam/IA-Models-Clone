"""
Inspection Service

Domain service for orchestrating inspections.
"""

from typing import List, Optional
from ..entities import Inspection, Defect, Anomaly, QualityScore
from ..value_objects import ImageMetadata
from ..exceptions import InspectionException, InvalidImageException
from .quality_assessment_service import QualityAssessmentService
from .defect_classification_service import DefectClassificationService


class InspectionService:
    """
    Service for orchestrating quality inspections.
    
    This service coordinates the inspection process:
    - Creating inspections
    - Adding defects and anomalies
    - Calculating quality scores
    - Completing inspections
    """
    
    def __init__(
        self,
        quality_assessment_service: Optional[QualityAssessmentService] = None,
        defect_classification_service: Optional[DefectClassificationService] = None
    ):
        """
        Initialize inspection service.
        
        Args:
            quality_assessment_service: Service for quality assessment
            defect_classification_service: Service for defect classification
        """
        self.quality_assessment_service = (
            quality_assessment_service or QualityAssessmentService()
        )
        self.defect_classification_service = (
            defect_classification_service or DefectClassificationService()
        )
    
    def create_inspection(
        self,
        image_metadata: ImageMetadata,
        inspection_id: Optional[str] = None
    ) -> Inspection:
        """
        Create a new inspection.
        
        Args:
            image_metadata: Metadata about the image being inspected
            inspection_id: Optional inspection ID (generated if not provided)
        
        Returns:
            New Inspection object
        """
        # Validate image metadata
        if image_metadata.width <= 0 or image_metadata.height <= 0:
            raise InvalidImageException("Invalid image dimensions")
        
        # Create initial quality score (perfect score)
        initial_quality_score = QualityScore(
            score=100.0,
            defects_count=0,
            anomalies_count=0,
        )
        
        # Create inspection
        inspection = Inspection(
            id=inspection_id or "",
            image_metadata=image_metadata,
            quality_score=initial_quality_score,
        )
        
        return inspection
    
    def add_defect_to_inspection(
        self,
        inspection: Inspection,
        defect: Defect
    ) -> Inspection:
        """
        Add a defect to an inspection and recalculate quality score.
        
        Args:
            inspection: Inspection to add defect to
            defect: Defect to add
        
        Returns:
            Updated inspection
        """
        # Validate defect
        is_valid, error_message = self.defect_classification_service.validate_defect(defect)
        if not is_valid:
            raise InspectionException(f"Invalid defect: {error_message}")
        
        # Add defect
        inspection.add_defect(defect)
        
        return inspection
    
    def add_anomaly_to_inspection(
        self,
        inspection: Inspection,
        anomaly: Anomaly
    ) -> Inspection:
        """
        Add an anomaly to an inspection and recalculate quality score.
        
        Args:
            inspection: Inspection to add anomaly to
            anomaly: Anomaly to add
        
        Returns:
            Updated inspection
        """
        # Add anomaly
        inspection.add_anomaly(anomaly)
        
        return inspection
    
    def complete_inspection(self, inspection: Inspection) -> Inspection:
        """
        Complete an inspection and finalize quality score.
        
        Args:
            inspection: Inspection to complete
        
        Returns:
            Completed inspection
        """
        # Recalculate quality score to ensure it's up to date
        inspection._recalculate_quality_score()
        
        # Mark as completed
        inspection.mark_completed()
        
        return inspection
    
    def get_inspection_summary(self, inspection: Inspection) -> dict:
        """
        Get a summary of an inspection.
        
        Args:
            inspection: Inspection to summarize
        
        Returns:
            Dictionary with inspection summary
        """
        return {
            "inspection_id": inspection.id,
            "quality_score": inspection.quality_score.score,
            "quality_status": inspection.quality_score.status.value,
            "total_defects": inspection.total_defects,
            "total_anomalies": inspection.total_anomalies,
            "critical_defects": len(inspection.critical_defects),
            "is_acceptable": inspection.quality_score.is_acceptable,
            "recommendation": inspection.quality_score.recommendation,
            "is_completed": inspection.is_completed,
        }



