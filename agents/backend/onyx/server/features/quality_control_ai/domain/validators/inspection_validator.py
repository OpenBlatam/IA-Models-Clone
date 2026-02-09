"""
Inspection Validator

Validates inspection data and results.
"""

from typing import Tuple, Optional
from ..entities import Inspection, QualityScore
from ..exceptions import InspectionException


class InspectionValidator:
    """
    Validator for inspection entities.
    
    Validates inspection data, quality scores, and results.
    """
    
    def validate_inspection(
        self,
        inspection: Inspection
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate inspection entity.
        
        Args:
            inspection: Inspection to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate quality score
        is_valid, error = self.validate_quality_score(inspection.quality_score)
        if not is_valid:
            return False, f"Invalid quality score: {error}"
        
        # Validate defects count matches
        if len(inspection.defects) != inspection.quality_score.defects_count:
            return False, "Defects count mismatch"
        
        # Validate anomalies count matches
        if len(inspection.anomalies) != inspection.quality_score.anomalies_count:
            return False, "Anomalies count mismatch"
        
        # Validate image metadata
        if inspection.image_metadata.width <= 0 or inspection.image_metadata.height <= 0:
            return False, "Invalid image metadata dimensions"
        
        return True, None
    
    def validate_quality_score(
        self,
        quality_score: QualityScore
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate quality score.
        
        Args:
            quality_score: Quality score to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check score range
        if not 0.0 <= quality_score.score <= 100.0:
            return False, f"Quality score must be between 0.0 and 100.0, got {quality_score.score}"
        
        # Check counts
        if quality_score.defects_count < 0:
            return False, "Defects count cannot be negative"
        if quality_score.anomalies_count < 0:
            return False, "Anomalies count cannot be negative"
        
        return True, None
    
    def validate_inspection_request(
        self,
        image_data: any,
        image_format: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate inspection request.
        
        Args:
            image_data: Image data
            image_format: Image format
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate format
        from .image_validator import ImageValidator
        image_validator = ImageValidator()
        is_valid, error = image_validator.validate_format(image_format)
        if not is_valid:
            return False, error
        
        # Validate data based on format
        if image_format == "numpy":
            import numpy as np
            if not isinstance(image_data, np.ndarray):
                return False, "Image data must be numpy array for 'numpy' format"
            is_valid, error = image_validator.validate_image(image_data)
            if not is_valid:
                return False, error
        elif image_format == "bytes":
            if not isinstance(image_data, bytes):
                return False, "Image data must be bytes for 'bytes' format"
            if len(image_data) == 0:
                return False, "Image data is empty"
        elif image_format == "file_path":
            from pathlib import Path
            if not isinstance(image_data, str):
                return False, "Image data must be string (file path) for 'file_path' format"
            path = Path(image_data)
            if not path.exists():
                return False, f"Image file not found: {image_data}"
        elif image_format == "base64":
            if not isinstance(image_data, str):
                return False, "Image data must be string (base64) for 'base64' format"
            if len(image_data) == 0:
                return False, "Image data is empty"
        
        return True, None



