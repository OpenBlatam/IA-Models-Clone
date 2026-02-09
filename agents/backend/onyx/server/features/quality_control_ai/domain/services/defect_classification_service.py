"""
Defect Classification Service

Domain service for classifying defects and determining their severity.
"""

from typing import Optional, Tuple
from ..entities import Defect, DefectType, DefectSeverity, DefectLocation


class DefectClassificationService:
    """
    Service for classifying defects and determining severity.
    
    This service implements business logic for:
    - Determining defect type based on characteristics
    - Calculating defect severity
    - Validating defect classifications
    """
    
    # Size thresholds for severity classification (in pixels)
    SIZE_THRESHOLDS = {
        DefectSeverity.CRITICAL: 5000,  # Large defects
        DefectSeverity.SEVERE: 2000,    # Medium-large defects
        DefectSeverity.MODERATE: 500,   # Medium defects
        DefectSeverity.MINOR: 0,        # Small defects
    }
    
    # Confidence thresholds for severity adjustment
    CONFIDENCE_THRESHOLDS = {
        DefectSeverity.CRITICAL: 0.9,
        DefectSeverity.SEVERE: 0.7,
        DefectSeverity.MODERATE: 0.5,
        DefectSeverity.MINOR: 0.3,
    }
    
    def classify_defect(
        self,
        defect_type: DefectType,
        location: DefectLocation,
        confidence: float,
        description: Optional[str] = None
    ) -> Defect:
        """
        Classify a defect and determine its severity.
        
        Args:
            defect_type: Type of defect
            location: Location of the defect
            confidence: Confidence score (0.0 to 1.0)
            description: Optional description
        
        Returns:
            Defect object with classified severity
        """
        # Determine severity based on size and type
        severity = self._determine_severity(location, defect_type, confidence)
        
        # Create defect
        import uuid
        defect = Defect(
            id=str(uuid.uuid4()),
            type=defect_type,
            severity=severity,
            location=location,
            confidence=confidence,
            description=description,
        )
        
        return defect
    
    def _determine_severity(
        self,
        location: DefectLocation,
        defect_type: DefectType,
        confidence: float
    ) -> DefectSeverity:
        """
        Determine defect severity based on size, type, and confidence.
        
        Args:
            location: Defect location
            defect_type: Type of defect
            confidence: Confidence score
        
        Returns:
            Defect severity
        """
        area = location.area
        
        # Some defect types are inherently more severe
        type_severity_boost = {
            DefectType.CRACK: 1,
            DefectType.MISSING_PART: 1,
            DefectType.DEFORMATION: 0.5,
        }
        
        severity_boost = type_severity_boost.get(defect_type, 0)
        adjusted_area = area * (1.0 + severity_boost)
        
        # Determine severity based on adjusted area
        if adjusted_area >= self.SIZE_THRESHOLDS[DefectSeverity.CRITICAL]:
            base_severity = DefectSeverity.CRITICAL
        elif adjusted_area >= self.SIZE_THRESHOLDS[DefectSeverity.SEVERE]:
            base_severity = DefectSeverity.SEVERE
        elif adjusted_area >= self.SIZE_THRESHOLDS[DefectSeverity.MODERATE]:
            base_severity = DefectSeverity.MODERATE
        else:
            base_severity = DefectSeverity.MINOR
        
        # Adjust severity based on confidence
        final_severity = self._adjust_severity_by_confidence(base_severity, confidence)
        
        return final_severity
    
    def _adjust_severity_by_confidence(
        self,
        base_severity: DefectSeverity,
        confidence: float
    ) -> DefectSeverity:
        """
        Adjust severity based on confidence score.
        
        Lower confidence may reduce severity, but never increases it.
        
        Args:
            base_severity: Base severity determined by size/type
            confidence: Confidence score
        
        Returns:
            Adjusted severity
        """
        # If confidence is very low, reduce severity by one level
        if confidence < 0.3:
            severity_order = [
                DefectSeverity.CRITICAL,
                DefectSeverity.SEVERE,
                DefectSeverity.MODERATE,
                DefectSeverity.MINOR,
            ]
            try:
                current_index = severity_order.index(base_severity)
                if current_index < len(severity_order) - 1:
                    return severity_order[current_index + 1]
            except ValueError:
                pass
        
        return base_severity
    
    def validate_defect(self, defect: Defect) -> Tuple[bool, Optional[str]]:
        """
        Validate a defect classification.
        
        Args:
            defect: Defect to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check confidence range
        if not 0.0 <= defect.confidence <= 1.0:
            return False, f"Invalid confidence: {defect.confidence}"
        
        # Check location validity
        if defect.location.width <= 0 or defect.location.height <= 0:
            return False, "Invalid defect location dimensions"
        
        # Check severity matches size
        expected_severity = self._determine_severity(
            defect.location,
            defect.type,
            defect.confidence
        )
        
        # Allow some flexibility (within one level)
        severity_order = [
            DefectSeverity.CRITICAL,
            DefectSeverity.SEVERE,
            DefectSeverity.MODERATE,
            DefectSeverity.MINOR,
        ]
        
        current_index = severity_order.index(defect.severity)
        expected_index = severity_order.index(expected_severity)
        
        if abs(current_index - expected_index) > 1:
            return False, f"Severity mismatch: expected {expected_severity}, got {defect.severity}"
        
        return True, None



