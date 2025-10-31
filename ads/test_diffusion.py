from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import pytest
import asyncio
import torch
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List
from PIL import Image
import numpy as np
import base64
from io import BytesIO
from onyx.server.features.ads.diffusion_service import (
from typing import Any, List, Dict, Optional
import logging
"""
Comprehensive tests for the diffusion models service.
"""

    DiffusionService,
    DiffusionModelManager,
    ImageProcessor,
    GenerationParams,
    DiffusionConfig
)

class TestImageProcessor:
    """Test cases for ImageProcessor class."""
    
    def setup_method(self) -> Any:
        """Set up test fixtures."""
        self.processor = ImageProcessor()
        
        # Create test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
    
    def test_load_image(self) -> Any:
        """Test image loading from path."""
        with patch('PIL.Image.open') as mock_open:
            mock_open.return_value = self.test_image
            
            result = self.processor.load_image("test.jpg")
            
            assert result == self.test_image
            mock_open.assert_called_once_with("test.jpg")
    
    def test_load_image_from_url(self) -> Any:
        """Test image loading from URL."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.content = b"fake_image_data"
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            with patch('PIL.Image.open') as mock_open:
                mock_open.return_value = self.test_image
                
                result = self.processor.load_image_from_url("http://example.com/image.jpg")
                
                assert result == self.test_image
                mock_get.assert_called_once_with("http://example.com/image.jpg", timeout=30)
    
    def test_load_image_from_base64(self) -> Any:
        """Test image loading from base64 string."""
        # Create base64 string
        buffer = BytesIO()
        self.test_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        base64_string = f"data:image/png;base64,{img_str}"
        
        result = self.processor.load_image_from_base64(base64_string)
        
        assert isinstance(result, Image.Image)
        assert result.size == self.test_image.size
    
    def test_resize_image(self) -> Any:
        """Test image resizing."""
        result = self.processor.resize_image(self.test_image, 200, 200)
        
        assert result.size == (200, 200)
        assert isinstance(result, Image.Image)
    
    def test_create_mask(self) -> Any:
        """Test mask creation."""
        # Test center mask
        center_mask = self.processor.create_mask(self.test_image, "center")
        assert center_mask.mode == 'L'
        assert center_mask.size == self.test_image.size
        
        # Test random mask
        random_mask = self.processor.create_mask(self.test_image, "random")
        assert random_mask.mode == 'L'
        assert random_mask.size == self.test_image.size
        
        # Test full mask
        full_mask = self.processor.create_mask(self.test_image, "full")
        assert full_mask.mode == 'L'
        assert full_mask.size == self.test_image.size
    
    def test_apply_canny_edge_detection(self) -> Any:
        """Test Canny edge detection."""
        with patch('cv2.cvtColor') as mock_cvt:
            with patch('cv2.Canny') as mock_canny:
                mock_cvt.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
                mock_canny.return_value = np.zeros((100, 100), dtype=np.uint8)
                
                result = self.processor.apply_canny_edge_detection(self.test_image)
                
                assert isinstance(result, Image.Image)
                mock_cvt.assert_called_once()
                mock_canny.assert_called_once()
    
    def test_apply_depth_estimation(self) -> Any:
        """Test depth estimation."""
        with patch('cv2.GaussianBlur') as mock_blur:
            mock_blur.return_value = np.zeros((100, 100), dtype=np.uint8)
            
            result = self.processor.apply_depth_estimation(self.test_image)
            
            assert isinstance(result, Image.Image)
            mock_blur.assert_called_once()
    
    def test_save_image(self) -> Any:
        """Test image saving."""
        with patch.object(self.test_image, 'save') as mock_save:
            result = self.processor.save_image(self.test_image, "test_output.png")
            
            assert result == "test_output.png"
            mock_save.assert_called_once_with("test_output.png", format="PNG")
    
    def test_image_to_base64(self) -> Any:
        """Test image to base64 conversion."""
        result = self.processor.image_to_base64(self.test_image)
        
        assert result.startswith("data:image/png;base64,")
        assert len(result) > 0

class TestDiffusionModelManager:
    """Test cases for DiffusionModelManager class."""
    
    def setup_method(self) -> Any:
        """Set up test fixtures."""
        with patch('onyx.server.features.ads.diffusion_service.TokenizationService'):
            self.manager = DiffusionModelManager()
    
    def test_get_model_key(self) -> Optional[Dict[str, Any]]:
        """Test model key generation."""
        key = self.manager._get_model_key("test_model", "text2img")
        
        assert key == "diffusion:test_model:text2img"
    
    @patch('diffusers.StableDiffusionPipeline.from_pretrained')
    def test_get_text_to_image_pipeline(self, mock_from_pretrained) -> Optional[Dict[str, Any]]:
        """Test text-to-image pipeline loading."""
        mock_pipeline = Mock()
        mock_from_pretrained.return_value = mock_pipeline
        
        result = self.manager.get_text_to_image_pipeline("test_model")
        
        assert result == mock_pipeline
        mock_from_pretrained.assert_called_once()
    
    @patch('diffusers.StableDiffusionImg2ImgPipeline.from_pretrained')
    def test_get_image_to_image_pipeline(self, mock_from_pretrained) -> Optional[Dict[str, Any]]:
        """Test image-to-image pipeline loading."""
        mock_pipeline = Mock()
        mock_from_pretrained.return_value = mock_pipeline
        
        result = self.manager.get_image_to_image_pipeline("test_model")
        
        assert result == mock_pipeline
        mock_from_pretrained.assert_called_once()
    
    @patch('diffusers.StableDiffusionInpaintPipeline.from_pretrained')
    def test_get_inpaint_pipeline(self, mock_from_pretrained) -> Optional[Dict[str, Any]]:
        """Test inpainting pipeline loading."""
        mock_pipeline = Mock()
        mock_from_pretrained.return_value = mock_pipeline
        
        result = self.manager.get_inpaint_pipeline("test_model")
        
        assert result == mock_pipeline
        mock_from_pretrained.assert_called_once()
    
    @patch('diffusers.ControlNetModel.from_pretrained')
    @patch('diffusers.StableDiffusionControlNetPipeline.from_pretrained')
    def test_get_controlnet_pipeline(self, mock_controlnet_from_pretrained, mock_controlnet_model) -> Optional[Dict[str, Any]]:
        """Test ControlNet pipeline loading."""
        mock_controlnet = Mock()
        mock_pipeline = Mock()
        mock_controlnet_model.return_value = mock_controlnet
        mock_controlnet_from_pretrained.return_value = mock_pipeline
        
        result = self.manager.get_controlnet_pipeline("canny")
        
        assert result == mock_pipeline
        mock_controlnet_model.assert_called_once()
        mock_controlnet_from_pretrained.assert_called_once()

class TestDiffusionService:
    """Test cases for DiffusionService class."""
    
    def setup_method(self) -> Any:
        """Set up test fixtures."""
        with patch('onyx.server.features.ads.diffusion_service.DiffusionModelManager'):
            with patch('onyx.server.features.ads.diffusion_service.ImageProcessor'):
                self.service = DiffusionService()
                self.service.model_manager = Mock()
                self.service.image_processor = Mock()
    
    @pytest.mark.asyncio
    async def test_generate_text_to_image(self) -> Any:
        """Test text-to-image generation."""
        # Create test parameters
        params = GenerationParams(
            prompt="Test prompt",
            negative_prompt="Test negative",
            width=512,
            height=512,
            num_images=1,
            guidance_scale=7.5,
            num_inference_steps=50
        )
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value.images = [Image.new('RGB', (512, 512))]
        self.service.model_manager.get_text_to_image_pipeline.return_value = mock_pipeline
        
        # Mock Redis
        with patch.object(self.service, 'redis_client') as mock_redis:
            mock_redis.return_value.get.return_value = None
            
            result = await self.service.generate_text_to_image(params, "test_model")
            
            assert len(result) == 1
            assert isinstance(result[0], Image.Image)
            mock_pipeline.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_image_to_image(self) -> Any:
        """Test image-to-image generation."""
        # Create test parameters
        params = GenerationParams(
            prompt="Test prompt",
            width=512,
            height=512,
            num_images=1,
            guidance_scale=7.5,
            num_inference_steps=50
        )
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value.images = [Image.new('RGB', (512, 512))]
        self.service.model_manager.get_image_to_image_pipeline.return_value = mock_pipeline
        
        # Mock Redis
        with patch.object(self.service, 'redis_client') as mock_redis:
            mock_redis.return_value.get.return_value = None
            
            test_image = Image.new('RGB', (512, 512))
            result = await self.service.generate_image_to_image(
                init_image=test_image,
                params=params,
                model_name="test_model"
            )
            
            assert len(result) == 1
            assert isinstance(result[0], Image.Image)
            mock_pipeline.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_inpaint_image(self) -> Any:
        """Test image inpainting."""
        # Create test parameters
        params = GenerationParams(
            prompt="Test prompt",
            width=512,
            height=512,
            num_images=1,
            guidance_scale=7.5,
            num_inference_steps=50
        )
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value.images = [Image.new('RGB', (512, 512))]
        self.service.model_manager.get_inpaint_pipeline.return_value = mock_pipeline
        
        # Mock Redis
        with patch.object(self.service, 'redis_client') as mock_redis:
            mock_redis.return_value.get.return_value = None
            
            test_image = Image.new('RGB', (512, 512))
            test_mask = Image.new('L', (512, 512))
            
            result = await self.service.inpaint_image(
                image=test_image,
                mask=test_mask,
                params=params,
                model_name="test_model"
            )
            
            assert len(result) == 1
            assert isinstance(result[0], Image.Image)
            mock_pipeline.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_with_controlnet(self) -> Any:
        """Test ControlNet generation."""
        # Create test parameters
        params = GenerationParams(
            prompt="Test prompt",
            width=512,
            height=512,
            num_images=1,
            guidance_scale=7.5,
            num_inference_steps=50
        )
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value.images = [Image.new('RGB', (512, 512))]
        self.service.model_manager.get_controlnet_pipeline.return_value = mock_pipeline
        
        # Mock Redis
        with patch.object(self.service, 'redis_client') as mock_redis:
            mock_redis.return_value.get.return_value = None
            
            test_image = Image.new('RGB', (512, 512))
            
            result = await self.service.generate_with_controlnet(
                control_image=test_image,
                params=params,
                controlnet_type="canny"
            )
            
            assert len(result) == 1
            assert isinstance(result[0], Image.Image)
            mock_pipeline.assert_called_once()
    
    def test_get_cache_key(self) -> Optional[Dict[str, Any]]:
        """Test cache key generation."""
        params = GenerationParams(
            prompt="Test prompt",
            width=512,
            height=512
        )
        
        key = self.service._get_cache_key(params, "test_model", "text2img")
        
        assert isinstance(key, str)
        assert len(key) > 0
    
    def test_encode_images_for_cache(self) -> Any:
        """Test image encoding for cache."""
        test_images = [Image.new('RGB', (100, 100))]
        
        result = self.service._encode_images_for_cache(test_images)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_decode_cached_images(self) -> Any:
        """Test cached image decoding."""
        # Create test encoded data
        test_image = Image.new('RGB', (100, 100))
        encoded = self.service._encode_images_for_cache([test_image])
        
        result = self.service._decode_cached_images(encoded)
        
        assert len(result) == 1
        assert isinstance(result[0], Image.Image)
    
    @pytest.mark.asyncio
    async def test_get_generation_stats(self) -> Optional[Dict[str, Any]]:
        """Test generation statistics."""
        with patch.object(self.service, 'redis_client') as mock_redis:
            mock_redis.return_value.keys.return_value = ["key1", "key2", "key3"]
            mock_redis.return_value.get.side_effect = ["10", "5"]
            
            result = await self.service.get_generation_stats()
            
            assert 'total_cache_entries' in result
            assert 'cache_hits' in result
            assert 'cache_misses' in result
            assert 'cache_hit_rate' in result
            assert 'loaded_models' in result
            assert 'device' in result
    
    @pytest.mark.asyncio
    async def test_cleanup_cache(self) -> Any:
        """Test cache cleanup."""
        with patch.object(self.service, 'redis_client') as mock_redis:
            mock_redis.return_value.keys.return_value = ["key1", "key2"]
            mock_redis.return_value.ttl.side_effect = [3600, -1]  # One old, one new
            mock_redis.return_value.delete.return_value = 1
            
            result = await self.service.cleanup_cache(max_age_hours=24)
            
            assert 'deleted_entries' in result
            assert result['deleted_entries'] >= 0

class TestGenerationParams:
    """Test cases for GenerationParams dataclass."""
    
    def test_generation_params_creation(self) -> Any:
        """Test GenerationParams creation."""
        params = GenerationParams(
            prompt="Test prompt",
            negative_prompt="Test negative",
            width=512,
            height=512,
            num_images=1,
            guidance_scale=7.5,
            num_inference_steps=50,
            seed=42
        )
        
        assert params.prompt == "Test prompt"
        assert params.negative_prompt == "Test negative"
        assert params.width == 512
        assert params.height == 512
        assert params.num_images == 1
        assert params.guidance_scale == 7.5
        assert params.num_inference_steps == 50
        assert params.seed == 42
    
    def test_generation_params_defaults(self) -> Any:
        """Test GenerationParams default values."""
        params = GenerationParams(prompt="Test prompt")
        
        assert params.negative_prompt == ""
        assert params.width == 512
        assert params.height == 512
        assert params.num_images == 1
        assert params.guidance_scale == 7.5
        assert params.num_inference_steps == 50
        assert params.seed is None

class TestIntegration:
    """Integration tests for diffusion service."""
    
    @pytest.mark.asyncio
    async def test_full_generation_pipeline(self) -> Any:
        """Test complete generation pipeline."""
        # Mock all dependencies
        with patch('onyx.server.features.ads.diffusion_service.DiffusionModelManager'):
            with patch('onyx.server.features.ads.diffusion_service.ImageProcessor'):
                service = DiffusionService()
                service.model_manager = Mock()
                service.image_processor = Mock()
        
        # Create test parameters
        params = GenerationParams(
            prompt="Professional advertisement for premium coffee",
            negative_prompt="blurry, low quality",
            width=512,
            height=512,
            num_images=1,
            guidance_scale=7.5,
            num_inference_steps=50
        )
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value.images = [Image.new('RGB', (512, 512))]
        service.model_manager.get_text_to_image_pipeline.return_value = mock_pipeline
        
        # Mock Redis
        with patch.object(service, 'redis_client') as mock_redis:
            mock_redis.return_value.get.return_value = None
            
            # Test generation
            result = await service.generate_text_to_image(params, "test_model")
            
            assert len(result) == 1
            assert isinstance(result[0], Image.Image)
            mock_pipeline.assert_called_once()
    
    def test_image_processing_pipeline(self) -> Any:
        """Test complete image processing pipeline."""
        processor = ImageProcessor()
        
        # Create test image
        test_image = Image.new('RGB', (100, 100), color='blue')
        
        # Test resize
        resized = processor.resize_image(test_image, 200, 200)
        assert resized.size == (200, 200)
        
        # Test mask creation
        mask = processor.create_mask(resized, "center")
        assert mask.mode == 'L'
        assert mask.size == resized.size
        
        # Test base64 conversion
        base64_str = processor.image_to_base64(test_image)
        assert base64_str.startswith("data:image/png;base64,")
        
        # Test base64 decoding
        decoded = processor.load_image_from_base64(base64_str)
        assert isinstance(decoded, Image.Image)
        assert decoded.size == test_image.size

class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_generation_with_invalid_params(self) -> Any:
        """Test generation with invalid parameters."""
        service = DiffusionService()
        
        # Test with empty prompt
        params = GenerationParams(prompt="")
        
        with pytest.raises(Exception):
            await service.generate_text_to_image(params)
    
    @pytest.mark.asyncio
    async def test_pipeline_loading_error(self) -> Any:
        """Test pipeline loading error handling."""
        with patch('onyx.server.features.ads.diffusion_service.DiffusionModelManager'):
            service = DiffusionService()
            service.model_manager = Mock()
            service.model_manager.get_text_to_image_pipeline.side_effect = Exception("Model loading failed")
        
        params = GenerationParams(prompt="Test prompt")
        
        with pytest.raises(Exception):
            await service.generate_text_to_image(params)
    
    def test_image_loading_error(self) -> Any:
        """Test image loading error handling."""
        processor = ImageProcessor()
        
        with pytest.raises(Exception):
            processor.load_image("nonexistent.jpg")

match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 