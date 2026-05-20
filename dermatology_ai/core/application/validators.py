"""
Input validators for use cases
Centralized validation logic following Single Responsibility Principle
"""

from typing import Optional, Dict, Any
import logging

from .exceptions import ValidationError

logger = logging.getLogger(__name__)


class ImageValidator:
    """Validator for image data"""
    
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    MIN_IMAGE_SIZE = 1024  # 1KB
    ALLOWED_FORMATS = {'image/jpeg', 'image/jpg', 'image/png', 'image/webp'}
    
    @staticmethod
    def validate_image_data(image_data: bytes, filename: Optional[str] = None) -> None:
        """
        Validate image data
        
        Args:
            image_data: Image bytes
            filename: Optional filename for format detection
            
        Raises:
            ValidationError: If validation fails
        """
        if not image_data:
            raise ValidationError("Image data is required")
        
        if not isinstance(image_data, bytes):
            raise ValidationError("Image data must be bytes")
        
        size = len(image_data)
        if size < ImageValidator.MIN_IMAGE_SIZE:
            raise ValidationError(f"Image too small (minimum {ImageValidator.MIN_IMAGE_SIZE} bytes)")
        
        if size > ImageValidator.MAX_IMAGE_SIZE:
            raise ValidationError(f"Image too large (maximum {ImageValidator.MAX_IMAGE_SIZE} bytes)")
        
        # Basic format validation by magic bytes
        if not ImageValidator._is_valid_image_format(image_data):
            raise ValidationError("Invalid image format. Supported: JPEG, PNG, WebP")
    
    @staticmethod
    def _is_valid_image_format(image_data: bytes) -> bool:
        """Check if image data matches known formats"""
        if len(image_data) < 4:
            return False
        
        # JPEG: FF D8 FF
        if image_data[:3] == b'\xff\xd8\xff':
            return True
        
        # PNG: 89 50 4E 47
        if image_data[:4] == b'\x89PNG':
            return True
        
        # WebP: RIFF...WEBP
        if image_data[:4] == b'RIFF' and b'WEBP' in image_data[:12]:
            return True
        
        return False


class UserIdValidator:
    """Validator for user IDs"""
    
    @staticmethod
    def validate_user_id(user_id: str) -> None:
        """
        Validate user ID
        
        Args:
            user_id: User ID string
            
        Raises:
            ValidationError: If validation fails
        """
        if not user_id:
            raise ValidationError("User ID is required")
        
        if not isinstance(user_id, str):
            raise ValidationError("User ID must be a string")
        
        if len(user_id.strip()) == 0:
            raise ValidationError("User ID cannot be empty")
        
        if len(user_id) > 255:
            raise ValidationError("User ID too long (maximum 255 characters)")


class PaginationValidator:
    """Validator for pagination parameters"""
    
    MAX_LIMIT = 100
    MIN_LIMIT = 1
    MIN_OFFSET = 0
    
    @staticmethod
    def validate_pagination(limit: int, offset: int) -> None:
        """
        Validate pagination parameters
        
        Args:
            limit: Number of items per page
            offset: Number of items to skip
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(limit, int):
            raise ValidationError("Limit must be an integer")
        
        if limit < PaginationValidator.MIN_LIMIT:
            raise ValidationError(f"Limit must be at least {PaginationValidator.MIN_LIMIT}")
        
        if limit > PaginationValidator.MAX_LIMIT:
            raise ValidationError(f"Limit cannot exceed {PaginationValidator.MAX_LIMIT}")
        
        if not isinstance(offset, int):
            raise ValidationError("Offset must be an integer")
        
        if offset < PaginationValidator.MIN_OFFSET:
            raise ValidationError(f"Offset cannot be negative")


class MetadataValidator:
    """Validator for metadata dictionaries"""
    
    MAX_METADATA_SIZE = 10 * 1024  # 10KB
    MAX_KEY_LENGTH = 100
    MAX_VALUE_LENGTH = 1000
    
    @staticmethod
    def validate_metadata(metadata: Optional[Dict[str, Any]]) -> None:
        """
        Validate metadata dictionary
        
        Args:
            metadata: Metadata dictionary
            
        Raises:
            ValidationError: If validation fails
        """
        if metadata is None:
            return
        
        if not isinstance(metadata, dict):
            raise ValidationError("Metadata must be a dictionary")
        
        # Check total size
        import json
        try:
            metadata_str = json.dumps(metadata)
            if len(metadata_str.encode('utf-8')) > MetadataValidator.MAX_METADATA_SIZE:
                raise ValidationError(f"Metadata too large (maximum {MetadataValidator.MAX_METADATA_SIZE} bytes)")
        except (TypeError, ValueError) as e:
            raise ValidationError(f"Invalid metadata format: {e}")
        
        # Validate keys and values
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValidationError("Metadata keys must be strings")
            
            if len(key) > MetadataValidator.MAX_KEY_LENGTH:
                raise ValidationError(f"Metadata key too long (maximum {MetadataValidator.MAX_KEY_LENGTH} characters)")
            
            if isinstance(value, str) and len(value) > MetadataValidator.MAX_VALUE_LENGTH:
                raise ValidationError(f"Metadata value too long (maximum {MetadataValidator.MAX_VALUE_LENGTH} characters)")















