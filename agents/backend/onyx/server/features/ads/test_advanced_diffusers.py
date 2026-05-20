from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import pytest
import asyncio
import torch
from unittest.mock import Mock, patch, AsyncMock
from PIL import Image
import numpy as np
from io import BytesIO
import base64
from onyx.server.features.ads.diffusion_service import (
from onyx.server.features.ads.diffusion_api import (
from typing import Any, List, Dict, Optional
import logging
"""
Comprehensive tests for advanced Diffusers features.
"""

    DiffusionService,
    DiffusionModelManager,
    DiffusionSchedulerFactory,
    GenerationParams
)
    LCMRequest,
    TCDRequest,
    CustomSchedulerRequest,
    AdvancedGenerationRequest
)

# Test data
SAMPLE_PROMPT = "A beautiful sunset over mountains"
SAMPLE_NEGATIVE_PROMPT = "blurry, low quality"
SAMPLE_IMAGE = Image.new('RGB', (512, 512), color='red')

def create_sample_image_base64():
    """Create a sample base64 encoded image."""
    buffer = BytesIO()
    SAMPLE_IMAGE.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

class TestDiffusionSchedulerFactory:
    """Test the DiffusionSchedulerFactory class."""
    
    def test_create_scheduler_basic(self) -> Any:
        """Test basic scheduler creation."""
        scheduler = DiffusionSchedulerFactory.create_scheduler("DDIM")
        assert scheduler is not None
        assert hasattr(scheduler, 'step')
    
    def test_create_scheduler_lcm(self) -> Any:
        """Test LCM scheduler creation with optimized defaults."""
        scheduler = DiffusionSchedulerFactory.create_scheduler("LCM")
        assert scheduler is not None
        # Check if LCM-specific defaults are set
        assert hasattr(scheduler, 'beta_start')
        assert hasattr(scheduler, 'beta_end')
    
    def test_create_scheduler_tcd(self) -> Any:
        """Test TCD scheduler creation with optimized defaults."""
        scheduler = DiffusionSchedulerFactory.create_scheduler("TCD")
        assert scheduler is not None
        # Check if TCD-specific defaults are set
        assert hasattr(scheduler, 'beta_start')
        assert hasattr(scheduler, 'beta_end')
    
    def test_create_scheduler_dpm_plus_plus(self) -> Any:
        """Test DPM++ scheduler creation with optimized defaults."""
        scheduler = DiffusionSchedulerFactory.create_scheduler("DPM++")
        assert scheduler is not None
        # Check if DPM++ specific defaults are set
        assert hasattr(scheduler, 'algorithm_type')
        assert hasattr(scheduler, 'solver_type')
    
    def test_get_optimal_scheduler_fast(self) -> Optional[Dict[str, Any]]:
        """Test optimal scheduler selection for fast generation."""
        scheduler = DiffusionSchedulerFactory.get_optimal_scheduler_for_task("text2img", "fast")
        assert scheduler == "LCM"
        
        scheduler = DiffusionSchedulerFactory.get_optimal_scheduler_for_task("img2img", "fast")
        assert scheduler == "TCD"
    
    def test_get_optimal_scheduler_high_quality(self) -> Optional[Dict[str, Any]]:
        """Test optimal scheduler selection for high quality."""
        scheduler = DiffusionSchedulerFactory.get_optimal_scheduler_for_task("text2img", "high")
        assert scheduler == "DPM++"
        
        scheduler = DiffusionSchedulerFactory.get_optimal_scheduler_for_task("img2img", "high")
        assert scheduler == "Heun"
    
    def test_get_optimal_scheduler_balanced(self) -> Optional[Dict[str, Any]]:
        """Test optimal scheduler selection for balanced generation."""
        scheduler = DiffusionSchedulerFactory.get_optimal_scheduler_for_task("text2img", "balanced")
        assert scheduler == "Euler"
        
        scheduler = DiffusionSchedulerFactory.get_optimal_scheduler_for_task("img2img", "balanced")
        assert scheduler == "DPM++"

class TestDiffusionModelManager:
    """Test the DiffusionModelManager class."""
    
    @pytest.fixture
    def model_manager(self) -> Any:
        """Create a model manager instance."""
        return DiffusionModelManager()
    
    @patch('torch.cuda.is_available')
    def test_get_text_to_image_pipeline_basic(self, mock_cuda, model_manager) -> Optional[Dict[str, Any]]:
        """Test basic text-to-image pipeline loading."""
        mock_cuda.return_value = False
        
        with patch('onyx.server.features.ads.diffusion_service.StableDiffusionPipeline.from_pretrained') as mock_load:
            mock_pipeline = Mock()
            mock_load.return_value = mock_pipeline
            
            pipeline = model_manager.get_text_to_image_pipeline("test-model")
            
            assert pipeline == mock_pipeline
            mock_load.assert_called_once()
    
    @patch('torch.cuda.is_available')
    def test_get_text_to_image_pipeline_sdxl(self, mock_cuda, model_manager) -> Optional[Dict[str, Any]]:
        """Test SDXL text-to-image pipeline loading."""
        mock_cuda.return_value = True
        
        with patch('onyx.server.features.ads.diffusion_service.StableDiffusionXLPipeline.from_pretrained') as mock_load:
            mock_pipeline = Mock()
            mock_load.return_value = mock_pipeline
            
            pipeline = model_manager.get_text_to_image_pipeline("stabilityai/stable-diffusion-xl-base-1.0")
            
            assert pipeline == mock_pipeline
            mock_load.assert_called_once()
    
    @patch('torch.cuda.is_available')
    def test_get_image_to_image_pipeline_sdxl(self, mock_cuda, model_manager) -> Optional[Dict[str, Any]]:
        """Test SDXL image-to-image pipeline loading."""
        mock_cuda.return_value = True
        
        with patch('onyx.server.features.ads.diffusion_service.StableDiffusionXLImg2ImgPipeline.from_pretrained') as mock_load:
            mock_pipeline = Mock()
            mock_load.return_value = mock_pipeline
            
            pipeline = model_manager.get_image_to_image_pipeline("stabilityai/stable-diffusion-xl-base-1.0")
            
            assert pipeline == mock_pipeline
            mock_load.assert_called_once()
    
    @patch('torch.cuda.is_available')
    def test_get_lcm_pipeline(self, mock_cuda, model_manager) -> Optional[Dict[str, Any]]:
        """Test LCM pipeline loading."""
        mock_cuda.return_value = True
        
        with patch('onyx.server.features.ads.diffusion_service.StableDiffusionPipeline.from_pretrained') as mock_load:
            mock_pipeline = Mock()
            mock_pipeline.scheduler = Mock()
            mock_load.return_value = mock_pipeline
            
            with patch('onyx.server.features.ads.diffusion_service.LCMScheduler.from_config') as mock_scheduler:
                mock_scheduler.return_value = Mock()
                
                pipeline = model_manager.get_lcm_pipeline("test-lcm-model")
                
                assert pipeline == mock_pipeline
                mock_load.assert_called_once()
                mock_scheduler.assert_called_once()
    
    @patch('torch.cuda.is_available')
    def test_get_tcd_pipeline(self, mock_cuda, model_manager) -> Optional[Dict[str, Any]]:
        """Test TCD pipeline loading."""
        mock_cuda.return_value = True
        
        with patch('onyx.server.features.ads.diffusion_service.StableDiffusionPipeline.from_pretrained') as mock_load:
            mock_pipeline = Mock()
            mock_pipeline.scheduler = Mock()
            mock_load.return_value = mock_pipeline
            
            with patch('onyx.server.features.ads.diffusion_service.TCDScheduler.from_config') as mock_scheduler:
                mock_scheduler.return_value = Mock()
                
                pipeline = model_manager.get_tcd_pipeline("test-tcd-model")
                
                assert pipeline == mock_pipeline
                mock_load.assert_called_once()
                mock_scheduler.assert_called_once()
    
    @patch('torch.cuda.is_available')
    def test_get_controlnet_pipeline(self, mock_cuda, model_manager) -> Optional[Dict[str, Any]]:
        """Test ControlNet pipeline loading."""
        mock_cuda.return_value = True
        
        with patch('onyx.server.features.ads.diffusion_service.ControlNetModel.from_pretrained') as mock_controlnet:
            with patch('onyx.server.features.ads.diffusion_service.StableDiffusionControlNetPipeline.from_pretrained') as mock_load:
                mock_controlnet_model = Mock()
                mock_controlnet.return_value = mock_controlnet_model
                
                mock_pipeline = Mock()
                mock_load.return_value = mock_pipeline
                
                pipeline = model_manager.get_controlnet_pipeline("canny")
                
                assert pipeline == mock_pipeline
                mock_controlnet.assert_called_once()
                mock_load.assert_called_once()

class TestDiffusionService:
    """Test the DiffusionService class."""
    
    @pytest.fixture
    def diffusion_service(self) -> Any:
        """Create a diffusion service instance."""
        return DiffusionService()
    
    @pytest.fixture
    def sample_params(self) -> Any:
        """Create sample generation parameters."""
        return GenerationParams(
            prompt=SAMPLE_PROMPT,
            negative_prompt=SAMPLE_NEGATIVE_PROMPT,
            width=512,
            height=512,
            num_images=1,
            guidance_scale=7.5,
            num_inference_steps=4,
            seed=42
        )
    
    @patch('onyx.server.features.ads.diffusion_service.aioredis.from_url')
    async def test_generate_with_lcm(self, mock_redis, diffusion_service, sample_params) -> Any:
        """Test LCM generation."""
        # Mock Redis
        mock_redis_client = AsyncMock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.get.return_value = None
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value.images = [SAMPLE_IMAGE]
        
        with patch.object(diffusion_service.model_manager, 'get_lcm_pipeline', return_value=mock_pipeline):
            images = await diffusion_service.generate_with_lcm(sample_params)
            
            assert len(images) == 1
            assert isinstance(images[0], Image.Image)
            mock_pipeline.assert_called_once()
    
    @patch('onyx.server.features.ads.diffusion_service.aioredis.from_url')
    async def test_generate_with_tcd(self, mock_redis, diffusion_service, sample_params) -> Any:
        """Test TCD generation."""
        # Mock Redis
        mock_redis_client = AsyncMock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.get.return_value = None
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value.images = [SAMPLE_IMAGE]
        
        with patch.object(diffusion_service.model_manager, 'get_tcd_pipeline', return_value=mock_pipeline):
            images = await diffusion_service.generate_with_tcd(sample_params)
            
            assert len(images) == 1
            assert isinstance(images[0], Image.Image)
            mock_pipeline.assert_called_once()
    
    @patch('onyx.server.features.ads.diffusion_service.aioredis.from_url')
    async def test_generate_with_custom_scheduler(self, mock_redis, diffusion_service, sample_params) -> Any:
        """Test custom scheduler generation."""
        # Mock Redis
        mock_redis_client = AsyncMock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.get.return_value = None
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value.images = [SAMPLE_IMAGE]
        
        with patch.object(diffusion_service.model_manager, 'get_text_to_image_pipeline', return_value=mock_pipeline):
            with patch('onyx.server.features.ads.diffusion_service.DiffusionSchedulerFactory.create_scheduler') as mock_scheduler:
                mock_scheduler.return_value = Mock()
                
                images = await diffusion_service.generate_with_custom_scheduler(
                    sample_params, "DPM++"
                )
                
                assert len(images) == 1
                assert isinstance(images[0], Image.Image)
                mock_pipeline.assert_called_once()
                mock_scheduler.assert_called_once_with("DPM++")
    
    @patch('onyx.server.features.ads.diffusion_service.aioredis.from_url')
    async def test_generate_with_advanced_options(self, mock_redis, diffusion_service, sample_params) -> Any:
        """Test advanced options generation."""
        # Mock Redis
        mock_redis_client = AsyncMock()
        mock_redis.return_value = mock_redis_client
        mock_redis_client.get.return_value = None
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.return_value.images = [SAMPLE_IMAGE]
        mock_pipeline.load_lora_weights = Mock()
        mock_pipeline.load_textual_inversion = Mock()
        
        with patch.object(diffusion_service.model_manager, 'get_text_to_image_pipeline', return_value=mock_pipeline):
            with patch('onyx.server.features.ads.diffusion_service.DiffusionSchedulerFactory.create_scheduler') as mock_scheduler:
                mock_scheduler.return_value = Mock()
                
                images = await diffusion_service.generate_with_advanced_options(
                    sample_params,
                    scheduler_type="DPM++",
                    use_lora=True,
                    lora_path="/path/to/lora",
                    use_textual_inversion=True,
                    textual_inversion_path="/path/to/ti"
                )
                
                assert len(images) == 1
                assert isinstance(images[0], Image.Image)
                mock_pipeline.assert_called_once()
                mock_pipeline.load_lora_weights.assert_called_once_with("/path/to/lora")
                mock_pipeline.load_textual_inversion.assert_called_once_with("/path/to/ti")
    
    def test_get_cache_key(self, diffusion_service, sample_params) -> Optional[Dict[str, Any]]:
        """Test cache key generation."""
        cache_key = diffusion_service._get_cache_key(sample_params, "test-model", "lcm")
        
        assert isinstance(cache_key, str)
        assert "test-model" in cache_key
        assert "lcm" in cache_key
        assert SAMPLE_PROMPT in cache_key
    
    def test_encode_decode_images(self, diffusion_service) -> Any:
        """Test image encoding and decoding for cache."""
        # Encode images
        encoded = diffusion_service._encode_images_for_cache([SAMPLE_IMAGE])
        assert isinstance(encoded, str)
        
        # Decode images
        decoded = diffusion_service._decode_cached_images(encoded)
        assert len(decoded) == 1
        assert isinstance(decoded[0], Image.Image)
        assert decoded[0].size == SAMPLE_IMAGE.size

class TestDiffusionAPI:
    """Test the diffusion API endpoints."""
    
    @pytest.fixture
    async def lcm_request(self) -> Any:
        """Create a sample LCM request."""
        return LCMRequest(
            prompt=SAMPLE_PROMPT,
            negative_prompt=SAMPLE_NEGATIVE_PROMPT,
            width=512,
            height=512,
            num_images=1,
            guidance_scale=7.5,
            num_inference_steps=4,
            model_name="SimianLuo/LCM_Dreamshaper_v7"
        )
    
    @pytest.fixture
    async def tcd_request(self) -> Any:
        """Create a sample TCD request."""
        return TCDRequest(
            prompt=SAMPLE_PROMPT,
            negative_prompt=SAMPLE_NEGATIVE_PROMPT,
            width=512,
            height=512,
            num_images=1,
            guidance_scale=7.5,
            num_inference_steps=1,
            model_name="h1t/TCD-SD15"
        )
    
    @pytest.fixture
    async def custom_scheduler_request(self) -> Any:
        """Create a sample custom scheduler request."""
        return CustomSchedulerRequest(
            prompt=SAMPLE_PROMPT,
            negative_prompt=SAMPLE_NEGATIVE_PROMPT,
            width=512,
            height=512,
            num_images=1,
            guidance_scale=7.5,
            num_inference_steps=50,
            scheduler_type="DPM++",
            model_name="runwayml/stable-diffusion-v1-5"
        )
    
    @pytest.fixture
    async def advanced_request(self) -> Any:
        """Create a sample advanced request."""
        return AdvancedGenerationRequest(
            prompt=SAMPLE_PROMPT,
            negative_prompt=SAMPLE_NEGATIVE_PROMPT,
            width=512,
            height=512,
            num_images=1,
            guidance_scale=7.5,
            num_inference_steps=50,
            model_name="runwayml/stable-diffusion-v1-5",
            scheduler_type="DPM++",
            use_lora=True,
            lora_path="/path/to/lora",
            use_textual_inversion=True,
            textual_inversion_path="/path/to/ti"
        )
    
    async def test_lcm_request_validation(self, lcm_request) -> Any:
        """Test LCM request validation."""
        assert lcm_request.prompt == SAMPLE_PROMPT
        assert lcm_request.num_inference_steps == 4
        assert lcm_request.model_name == "SimianLuo/LCM_Dreamshaper_v7"
    
    async def test_tcd_request_validation(self, tcd_request) -> Any:
        """Test TCD request validation."""
        assert tcd_request.prompt == SAMPLE_PROMPT
        assert tcd_request.num_inference_steps == 1
        assert tcd_request.model_name == "h1t/TCD-SD15"
    
    async def test_custom_scheduler_request_validation(self, custom_scheduler_request) -> Any:
        """Test custom scheduler request validation."""
        assert custom_scheduler_request.prompt == SAMPLE_PROMPT
        assert custom_scheduler_request.scheduler_type == "DPM++"
        assert custom_scheduler_request.model_name == "runwayml/stable-diffusion-v1-5"
    
    async def test_advanced_request_validation(self, advanced_request) -> Any:
        """Test advanced request validation."""
        assert advanced_request.prompt == SAMPLE_PROMPT
        assert advanced_request.use_lora is True
        assert advanced_request.use_textual_inversion is True
        assert advanced_request.lora_path == "/path/to/lora"
        assert advanced_request.textual_inversion_path == "/path/to/ti"

class TestIntegration:
    """Integration tests for the complete diffusion system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_lcm_generation(self) -> Any:
        """Test end-to-end LCM generation."""
        # This would test the complete flow from API to service to model
        # Implementation would depend on the actual API framework being used
        pass
    
    @pytest.mark.asyncio
    async def test_end_to_end_tcd_generation(self) -> Any:
        """Test end-to-end TCD generation."""
        # This would test the complete flow from API to service to model
        pass
    
    @pytest.mark.asyncio
    async def test_cache_integration(self) -> Any:
        """Test cache integration."""
        # This would test that caching works correctly across the system
        pass
    
    @pytest.mark.asyncio
    async def test_error_handling(self) -> Any:
        """Test error handling across the system."""
        # This would test error handling for various failure scenarios
        pass

# Performance tests
class TestPerformance:
    """Performance tests for the diffusion system."""
    
    @pytest.mark.asyncio
    async def test_lcm_speed(self) -> Any:
        """Test LCM generation speed."""
        # This would measure the actual speed of LCM generation
        pass
    
    @pytest.mark.asyncio
    async def test_tcd_speed(self) -> Any:
        """Test TCD generation speed."""
        # This would measure the actual speed of TCD generation
        pass
    
    @pytest.mark.asyncio
    async def test_memory_usage(self) -> Any:
        """Test memory usage of different models."""
        # This would measure memory usage of different models
        pass
    
    @pytest.mark.asyncio
    async def test_concurrent_generation(self) -> Any:
        """Test concurrent generation performance."""
        # This would test performance under concurrent load
        pass

match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 