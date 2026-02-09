"""
Tests for Validators
Tests for all validation logic
"""

import pytest
from PIL import Image
import io
import json

from core.application.validators import (
    ImageValidator,
    UserIdValidator,
    PaginationValidator,
    MetadataValidator
)
from core.application.exceptions import ValidationError


class TestImageValidator:
    """Tests for ImageValidator"""
    
    def test_validate_valid_jpeg_image(self):
        """Test validation of valid JPEG image"""
        img = Image.new('RGB', (200, 200), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        image_data = img_bytes.read()
        
        # Should not raise
        ImageValidator.validate_image_data(image_data)
    
    def test_validate_valid_png_image(self):
        """Test validation of valid PNG image"""
        img = Image.new('RGB', (200, 200), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        image_data = img_bytes.read()
        
        # Should not raise
        ImageValidator.validate_image_data(image_data)
    
    def test_validate_empty_image(self):
        """Test validation of empty image"""
        with pytest.raises(ValidationError, match="Image data is required"):
            ImageValidator.validate_image_data(b"")
    
    def test_validate_none_image(self):
        """Test validation of None image"""
        with pytest.raises(ValidationError, match="Image data is required"):
            ImageValidator.validate_image_data(None)
    
    def test_validate_non_bytes_image(self):
        """Test validation of non-bytes image"""
        with pytest.raises(ValidationError, match="Image data must be bytes"):
            ImageValidator.validate_image_data("not bytes")
    
    def test_validate_too_small_image(self):
        """Test validation of too small image"""
        small_data = b"x" * 100  # Less than MIN_IMAGE_SIZE
        
        with pytest.raises(ValidationError, match="Image too small"):
            ImageValidator.validate_image_data(small_data)
    
    def test_validate_too_large_image(self):
        """Test validation of too large image"""
        large_data = b"x" * (ImageValidator.MAX_IMAGE_SIZE + 1)
        
        with pytest.raises(ValidationError, match="Image too large"):
            ImageValidator.validate_image_data(large_data)
    
    def test_validate_invalid_format(self):
        """Test validation of invalid image format"""
        invalid_data = b"not an image" * 1000  # Valid size but invalid format
        
        with pytest.raises(ValidationError, match="Invalid image format"):
            ImageValidator.validate_image_data(invalid_data)
    
    def test_is_valid_image_format_jpeg(self):
        """Test JPEG format detection"""
        jpeg_header = b'\xff\xd8\xff' + b'x' * 100
        assert ImageValidator._is_valid_image_format(jpeg_header) is True
    
    def test_is_valid_image_format_png(self):
        """Test PNG format detection"""
        png_header = b'\x89PNG' + b'x' * 100
        assert ImageValidator._is_valid_image_format(png_header) is True
    
    def test_is_valid_image_format_webp(self):
        """Test WebP format detection"""
        webp_header = b'RIFF' + b'x' * 4 + b'WEBP' + b'x' * 100
        assert ImageValidator._is_valid_image_format(webp_header) is True
    
    def test_is_valid_image_format_invalid(self):
        """Test invalid format detection"""
        invalid_header = b'INVALID' + b'x' * 100
        assert ImageValidator._is_valid_image_format(invalid_header) is False
    
    def test_is_valid_image_format_too_short(self):
        """Test format detection with too short data"""
        short_data = b'xx'
        assert ImageValidator._is_valid_image_format(short_data) is False


class TestUserIdValidator:
    """Tests for UserIdValidator"""
    
    def test_validate_valid_user_id(self):
        """Test validation of valid user ID"""
        # Should not raise
        UserIdValidator.validate_user_id("user-123")
        UserIdValidator.validate_user_id("test_user_456")
        UserIdValidator.validate_user_id("a" * 255)  # Max length
    
    def test_validate_empty_user_id(self):
        """Test validation of empty user ID"""
        with pytest.raises(ValidationError, match="User ID is required"):
            UserIdValidator.validate_user_id("")
    
    def test_validate_none_user_id(self):
        """Test validation of None user ID"""
        with pytest.raises(ValidationError, match="User ID is required"):
            UserIdValidator.validate_user_id(None)
    
    def test_validate_non_string_user_id(self):
        """Test validation of non-string user ID"""
        with pytest.raises(ValidationError, match="User ID must be a string"):
            UserIdValidator.validate_user_id(123)
    
    def test_validate_whitespace_only_user_id(self):
        """Test validation of whitespace-only user ID"""
        with pytest.raises(ValidationError, match="User ID cannot be empty"):
            UserIdValidator.validate_user_id("   ")
    
    def test_validate_too_long_user_id(self):
        """Test validation of too long user ID"""
        long_id = "a" * 256  # Exceeds max length
        
        with pytest.raises(ValidationError, match="User ID too long"):
            UserIdValidator.validate_user_id(long_id)


class TestPaginationValidator:
    """Tests for PaginationValidator"""
    
    def test_validate_valid_pagination(self):
        """Test validation of valid pagination"""
        # Should not raise
        PaginationValidator.validate_pagination(10, 0)
        PaginationValidator.validate_pagination(1, 0)  # Min limit
        PaginationValidator.validate_pagination(100, 0)  # Max limit
        PaginationValidator.validate_pagination(50, 100)  # Valid offset
    
    def test_validate_invalid_limit_type(self):
        """Test validation of non-integer limit"""
        with pytest.raises(ValidationError, match="Limit must be an integer"):
            PaginationValidator.validate_pagination("10", 0)
    
    def test_validate_limit_too_small(self):
        """Test validation of limit below minimum"""
        with pytest.raises(ValidationError, match="Limit must be at least"):
            PaginationValidator.validate_pagination(0, 0)
    
    def test_validate_limit_too_large(self):
        """Test validation of limit above maximum"""
        with pytest.raises(ValidationError, match="Limit cannot exceed"):
            PaginationValidator.validate_pagination(101, 0)
    
    def test_validate_invalid_offset_type(self):
        """Test validation of non-integer offset"""
        with pytest.raises(ValidationError, match="Offset must be an integer"):
            PaginationValidator.validate_pagination(10, "0")
    
    def test_validate_negative_offset(self):
        """Test validation of negative offset"""
        with pytest.raises(ValidationError, match="Offset cannot be negative"):
            PaginationValidator.validate_pagination(10, -1)


class TestMetadataValidator:
    """Tests for MetadataValidator"""
    
    def test_validate_valid_metadata(self):
        """Test validation of valid metadata"""
        metadata = {
            "filename": "test.jpg",
            "enhanced": True,
            "advanced_analysis": False
        }
        
        # Should not raise
        MetadataValidator.validate_metadata(metadata)
    
    def test_validate_none_metadata(self):
        """Test validation of None metadata"""
        # Should not raise (None is allowed)
        MetadataValidator.validate_metadata(None)
    
    def test_validate_empty_metadata(self):
        """Test validation of empty metadata"""
        # Should not raise
        MetadataValidator.validate_metadata({})
    
    def test_validate_non_dict_metadata(self):
        """Test validation of non-dict metadata"""
        with pytest.raises(ValidationError, match="Metadata must be a dictionary"):
            MetadataValidator.validate_metadata("not a dict")
    
    def test_validate_metadata_too_large(self):
        """Test validation of too large metadata"""
        # Create metadata that exceeds size limit
        large_value = "x" * (MetadataValidator.MAX_METADATA_SIZE + 1)
        metadata = {"large_field": large_value}
        
        with pytest.raises(ValidationError, match="Metadata too large"):
            MetadataValidator.validate_metadata(metadata)
    
    def test_validate_metadata_key_too_long(self):
        """Test validation of metadata key too long"""
        long_key = "a" * (MetadataValidator.MAX_KEY_LENGTH + 1)
        metadata = {long_key: "value"}
        
        with pytest.raises(ValidationError, match="Metadata key too long"):
            MetadataValidator.validate_metadata(metadata)
    
    def test_validate_metadata_value_too_long(self):
        """Test validation of metadata value too long"""
        long_value = "a" * (MetadataValidator.MAX_VALUE_LENGTH + 1)
        metadata = {"key": long_value}
        
        with pytest.raises(ValidationError, match="Metadata value too long"):
            MetadataValidator.validate_metadata(metadata)
    
    def test_validate_metadata_non_string_key(self):
        """Test validation of non-string metadata key"""
        metadata = {123: "value"}
        
        with pytest.raises(ValidationError, match="Metadata keys must be strings"):
            MetadataValidator.validate_metadata(metadata)
    
    def test_validate_metadata_complex_structure(self):
        """Test validation of complex metadata structure"""
        metadata = {
            "nested": {
                "level1": {
                    "level2": "value"
                }
            },
            "list": [1, 2, 3],
            "boolean": True,
            "number": 42
        }
        
        # Should not raise if within size limits
        MetadataValidator.validate_metadata(metadata)
    
    def test_validate_metadata_invalid_json(self):
        """Test validation of metadata that can't be serialized"""
        # Create metadata with non-serializable object
        class NonSerializable:
            pass
        
        metadata = {"key": NonSerializable()}
        
        with pytest.raises(ValidationError, match="Invalid metadata format"):
            MetadataValidator.validate_metadata(metadata)



