"""Domain validators."""

from typing import Optional, List
from PIL import Image

from api.schemas.visualization import SurgeryType
from core.exceptions import ImageValidationError, ValidationError
from core.constants import MIN_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION, DEFAULT_INTENSITY


class SurgeryValidator:
    """Validates surgery-related domain rules."""
    
    @staticmethod
    def validate_intensity(intensity: float) -> None:
        """
        Validate surgery intensity.
        
        Args:
            intensity: Intensity value (0.0 to 1.0)
            
        Raises:
            ValidationError: If intensity is invalid
        """
        if not isinstance(intensity, (int, float)):
            raise ValidationError("Intensity must be a number")
        
        if intensity < 0.0 or intensity > 1.0:
            raise ValidationError(
                f"Intensity must be between 0.0 and 1.0, got {intensity}"
            )
    
    @staticmethod
    def validate_surgery_type(surgery_type: SurgeryType) -> None:
        """
        Validate surgery type.
        
        Args:
            surgery_type: Surgery type to validate
            
        Raises:
            ValidationError: If surgery type is invalid
        """
        if not isinstance(surgery_type, SurgeryType):
            raise ValidationError(f"Invalid surgery type: {surgery_type}")
    
    @staticmethod
    def validate_target_areas(
        target_areas: Optional[List[str]],
        surgery_type: SurgeryType
    ) -> None:
        """
        Validate target areas for surgery type.
        
        Args:
            target_areas: List of target areas
            surgery_type: Surgery type
            
        Raises:
            ValidationError: If target areas are invalid
        """
        if target_areas is None:
            return
        
        if not isinstance(target_areas, list):
            raise ValidationError("Target areas must be a list")
        
        # Validate each area is a string
        for area in target_areas:
            if not isinstance(area, str):
                raise ValidationError(f"Target area must be a string: {area}")


class ImageDomainValidator:
    """Validates image-related domain rules."""
    
    @staticmethod
    def validate_image_dimensions(image: Image.Image) -> None:
        """
        Validate image dimensions.
        
        Args:
            image: PIL Image object
            
        Raises:
            ImageValidationError: If dimensions are invalid
        """
        if not image:
            raise ImageValidationError("Image is None or empty")
        
        width, height = image.size
        
        if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
            raise ImageValidationError(
                f"Image dimensions too small ({width}x{height}). "
                f"Minimum size: {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION} pixels"
            )
        
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            raise ImageValidationError(
                f"Image dimensions too large ({width}x{height}). "
                f"Maximum size: {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION} pixels"
            )
    
    @staticmethod
    def validate_image_format(image: Image.Image, supported_formats: List[str]) -> None:
        """
        Validate image format.
        
        Args:
            image: PIL Image object
            supported_formats: List of supported formats
            
        Raises:
            ImageValidationError: If format is invalid
        """
        if image.format and image.format.lower() not in supported_formats:
            raise ImageValidationError(
                f"Unsupported image format: {image.format}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )


class VisualizationRequestValidator:
    """Validates visualization requests."""
    
    def __init__(self):
        self.surgery_validator = SurgeryValidator()
        self.image_validator = ImageDomainValidator()
    
    def validate(
        self,
        surgery_type: SurgeryType,
        intensity: float,
        target_areas: Optional[List[str]] = None,
        image: Optional[Image.Image] = None,
        supported_formats: Optional[List[str]] = None
    ) -> None:
        """
        Validate visualization request.
        
        Args:
            surgery_type: Surgery type
            intensity: Intensity value
            target_areas: Target areas
            image: Image to validate
            supported_formats: Supported image formats
            
        Raises:
            ValidationError: If validation fails
        """
        self.surgery_validator.validate_surgery_type(surgery_type)
        self.surgery_validator.validate_intensity(intensity)
        self.surgery_validator.validate_target_areas(target_areas, surgery_type)
        
        if image and supported_formats:
            self.image_validator.validate_image_dimensions(image)
            self.image_validator.validate_image_format(image, supported_formats)

