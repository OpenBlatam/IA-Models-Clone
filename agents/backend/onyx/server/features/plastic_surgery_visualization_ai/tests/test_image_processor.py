"""Tests for image processor."""

import pytest
from PIL import Image
import io

from core.services.image_processor import ImageProcessor
from core.exceptions import ImageValidationError, ImageProcessingError


@pytest.fixture
def processor():
    """Create image processor instance."""
    return ImageProcessor()


@pytest.fixture
def valid_image():
    """Create a valid test image."""
    img = Image.new('RGB', (500, 500), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.read()


@pytest.fixture
def small_image():
    """Create a small invalid image."""
    img = Image.new('RGB', (50, 50), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes.read()


@pytest.mark.asyncio
async def test_load_from_bytes_valid(processor, valid_image):
    """Test loading valid image from bytes."""
    image = await processor.load_from_bytes(valid_image)
    assert image is not None
    assert isinstance(image, Image.Image)
    assert image.size == (500, 500)


@pytest.mark.asyncio
async def test_load_from_bytes_small(processor, small_image):
    """Test loading small image (should fail validation)."""
    with pytest.raises(ImageValidationError):
        await processor.load_from_bytes(small_image)


@pytest.mark.asyncio
async def test_load_from_bytes_invalid():
    """Test loading invalid image data."""
    processor = ImageProcessor()
    invalid_data = b"not an image"
    
    with pytest.raises((ImageProcessingError, ImageValidationError)):
        await processor.load_from_bytes(invalid_data)


def test_validate_image_valid(processor):
    """Test validating a valid image."""
    image = Image.new('RGB', (500, 500), color='red')
    assert processor.validate_image(image) is True


def test_validate_image_small(processor):
    """Test validating a small image."""
    image = Image.new('RGB', (50, 50), color='red')
    with pytest.raises(ImageValidationError):
        processor.validate_image(image)


def test_validate_image_large(processor):
    """Test validating a large image."""
    image = Image.new('RGB', (15000, 15000), color='red')
    with pytest.raises(ImageValidationError):
        processor.validate_image(image)

