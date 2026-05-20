"""
Basic Tests
===========

Basic tests for upscaling functionality.
"""

import pytest
import asyncio
from PIL import Image
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a simple test image
    img_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def upscaling_service():
    """Create upscaling service instance."""
    from image_upscaling_ai.core.upscaling_service import UpscalingService
    from image_upscaling_ai.config.upscaling_config import UpscalingConfig
    
    config = UpscalingConfig(
        use_ai_enhancement=False,  # Disable for testing
        use_optimization_core=False
    )
    service = UpscalingService(config=config)
    service.initialize_model()
    return service


@pytest.mark.asyncio
async def test_basic_upscaling(upscaling_service, sample_image):
    """Test basic upscaling."""
    result = await upscaling_service.upscale_image(
        sample_image,
        scale_factor=2.0
    )
    
    assert result["success"] is True
    assert result["upscaled_size"][0] == sample_image.size[0] * 2
    assert result["upscaled_size"][1] == sample_image.size[1] * 2


@pytest.mark.asyncio
async def test_scale_factors(upscaling_service, sample_image):
    """Test different scale factors."""
    for scale in [1.5, 2.0, 4.0]:
        result = await upscaling_service.upscale_image(
            sample_image,
            scale_factor=scale
        )
        
        assert result["success"] is True
        expected_width = int(sample_image.size[0] * scale)
        expected_height = int(sample_image.size[1] * scale)
        assert result["upscaled_size"] == (expected_width, expected_height)


def test_image_utils():
    """Test image utilities."""
    from image_upscaling_ai.utils import ImageUtils
    
    # Create test image
    img = Image.new("RGB", (512, 512), color="red")
    
    # Test ensure_rgb
    rgb_img = ImageUtils.ensure_rgb(img)
    assert rgb_img.mode == "RGB"
    
    # Test get_image_info
    info = ImageUtils.get_image_info(img)
    assert info["size"] == (512, 512)
    assert info["mode"] == "RGB"
    
    # Test resize_maintain_aspect
    resized = ImageUtils.resize_maintain_aspect(img, (256, 256))
    assert resized.size[0] <= 256
    assert resized.size[1] <= 256


def test_config_validator():
    """Test configuration validator."""
    from image_upscaling_ai.utils import ConfigValidator
    
    # Valid config
    valid_config = {
        "default_scale_factor": 2.0,
        "quality_mode": "high",
        "use_realesrgan": False
    }
    result = ConfigValidator.validate_config(valid_config)
    assert result["valid"] is True
    
    # Invalid config
    invalid_config = {
        "default_scale_factor": 10.0,  # Too high
        "quality_mode": "invalid"
    }
    result = ConfigValidator.validate_config(invalid_config)
    assert result["valid"] is False
    assert len(result["issues"]) > 0


@pytest.mark.asyncio
async def test_quality_validation(sample_image):
    """Test quality validation."""
    from image_upscaling_ai.models import QualityValidator
    
    validator = QualityValidator(min_score=0.6)
    
    # Create upscaled version (simplified)
    upscaled = sample_image.resize(
        (sample_image.size[0] * 2, sample_image.size[1] * 2),
        Image.Resampling.LANCZOS
    )
    
    report = validator.validate(upscaled, sample_image, 2.0)
    
    assert report.overall_score >= 0.0
    assert report.overall_score <= 1.0
    assert "sharpness" in report.metrics


def test_smart_tiling():
    """Test smart tiling."""
    from image_upscaling_ai.models import SmartTiling
    
    tiler = SmartTiling(max_tile_size=512, overlap=32)
    
    # Test tile calculation
    tiles = tiler.calculate_tiles((2048, 2048), 4.0)
    assert len(tiles) > 0
    
    # Test should_tile
    should_tile = tiler.should_tile((4096, 4096), 4.0)
    assert should_tile is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


