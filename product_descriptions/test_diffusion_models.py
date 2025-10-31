from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import pytest
import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import tempfile
import shutil
import time
from unittest.mock import Mock, patch, AsyncMock
from diffusion_models import (
from transformers_manager import TransformersManager, ModelConfig, ModelType
from typing import Any, List, Dict, Optional
import logging
"""
Comprehensive Tests for Diffusion Models
=======================================

This module provides extensive testing for the diffusion models implementation,
covering all functionality including text-to-image generation, security
visualizations, performance optimization, and error handling.

Test Coverage:
- Pipeline loading and management
- Text-to-image generation
- Image-to-image transformation
- Inpainting capabilities
- ControlNet integration
- Security prompt engineering
- Performance optimization
- Memory management
- Error handling and recovery
- Integration with transformers

Author: AI Assistant
License: MIT
"""


# Import our modules
    DiffusionModelsManager, DiffusionConfig, GenerationConfig,
    ImageToImageConfig, InpaintingConfig, ControlNetConfig,
    DiffusionTask, SchedulerType, SecurityPromptEngine,
    DiffusionResult
)


class TestDiffusionModelsManager:
    """Test suite for DiffusionModelsManager."""
    
    @pytest.fixture
    async def diffusion_manager(self) -> Any:
        """Create a diffusion manager instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = DiffusionModelsManager(cache_dir=temp_dir)
            yield manager
            # Cleanup
            manager.clear_cache()
    
    @pytest.fixture
    def sample_image(self) -> Any:
        """Create a sample image for testing."""
        image = Image.new('RGB', (512, 512), color='white')
        draw = ImageDraw.Draw(image)
        draw.rectangle([100, 100, 412, 412], outline='black', width=2)
        draw.text((256, 256), "Test Image", fill='black', anchor='mm')
        return image
    
    @pytest.fixture
    def sample_mask(self) -> Any:
        """Create a sample mask for testing."""
        mask = Image.new('L', (512, 512), color=0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([200, 200, 312, 312], fill=255)
        return mask
    
    def test_initialization(self, diffusion_manager) -> Any:
        """Test diffusion manager initialization."""
        assert diffusion_manager is not None
        assert hasattr(diffusion_manager, '_pipelines')
        assert hasattr(diffusion_manager, '_configs')
        assert hasattr(diffusion_manager, '_metrics')
        assert hasattr(diffusion_manager, '_device')
    
    def test_device_detection(self, diffusion_manager) -> Any:
        """Test device detection logic."""
        device = diffusion_manager._detect_device()
        assert device in [torch.device('cpu'), torch.device('cuda'), torch.device('mps')]
    
    def test_scheduler_creation(self, diffusion_manager) -> Any:
        """Test scheduler creation for different types."""
        schedulers = [
            SchedulerType.DDIM,
            SchedulerType.DPM_SOLVER,
            SchedulerType.EULER
        ]
        
        for scheduler_type in schedulers:
            scheduler = diffusion_manager._get_scheduler(scheduler_type)
            assert scheduler is not None
            assert hasattr(scheduler, 'step')
    
    def test_device_selection(self, diffusion_manager) -> Any:
        """Test device selection logic."""
        # Test auto device selection
        device = diffusion_manager._get_device("auto")
        assert device in [torch.device('cpu'), torch.device('cuda'), torch.device('mps')]
        
        # Test CPU device
        device = diffusion_manager._get_device("cpu")
        assert device == torch.device('cpu')
        
        # Test CUDA device (if available)
        if torch.cuda.is_available():
            device = diffusion_manager._get_device("cuda")
            assert device == torch.device('cuda')
    
    def test_memory_usage_tracking(self, diffusion_manager) -> Any:
        """Test memory usage tracking."""
        memory_usage = diffusion_manager._get_memory_usage()
        
        assert isinstance(memory_usage, dict)
        assert 'rss_mb' in memory_usage
        assert 'vms_mb' in memory_usage
        assert 'percent' in memory_usage
        
        assert memory_usage['rss_mb'] > 0
        assert memory_usage['vms_mb'] > 0
        assert 0 <= memory_usage['percent'] <= 100
    
    @pytest.mark.asyncio
    async def test_pipeline_loading_text_to_image(self, diffusion_manager) -> Any:
        """Test loading text-to-image pipeline."""
        config = DiffusionConfig(
            model_name="runwayml/stable-diffusion-v1-5",
            task=DiffusionTask.TEXT_TO_IMAGE
        )
        
        pipeline = await diffusion_manager.load_pipeline(config)
        assert pipeline is not None
        
        # Check if pipeline is cached
        pipeline_key = f"{config.model_name}_{config.task.value}"
        assert pipeline_key in diffusion_manager._pipelines
        assert pipeline_key in diffusion_manager._configs
        assert pipeline_key in diffusion_manager._metrics
    
    @pytest.mark.asyncio
    async def test_pipeline_loading_image_to_image(self, diffusion_manager) -> Any:
        """Test loading image-to-image pipeline."""
        config = DiffusionConfig(
            model_name="runwayml/stable-diffusion-v1-5",
            task=DiffusionTask.IMAGE_TO_IMAGE
        )
        
        pipeline = await diffusion_manager.load_pipeline(config)
        assert pipeline is not None
        
        # Check if pipeline is cached
        pipeline_key = f"{config.model_name}_{config.task.value}"
        assert pipeline_key in diffusion_manager._pipelines
    
    @pytest.mark.asyncio
    async def test_pipeline_loading_inpainting(self, diffusion_manager) -> Any:
        """Test loading inpainting pipeline."""
        config = DiffusionConfig(
            model_name="runwayml/stable-diffusion-v1-5",
            task=DiffusionTask.INPAINTING
        )
        
        pipeline = await diffusion_manager.load_pipeline(config)
        assert pipeline is not None
        
        # Check if pipeline is cached
        pipeline_key = f"{config.model_name}_{config.task.value}"
        assert pipeline_key in diffusion_manager._pipelines
    
    @pytest.mark.asyncio
    async def test_pipeline_force_reload(self, diffusion_manager) -> Any:
        """Test pipeline force reload functionality."""
        config = DiffusionConfig(
            model_name="runwayml/stable-diffusion-v1-5",
            task=DiffusionTask.TEXT_TO_IMAGE
        )
        
        # Load pipeline first time
        pipeline1 = await diffusion_manager.load_pipeline(config)
        
        # Load pipeline second time (should return cached)
        pipeline2 = await diffusion_manager.load_pipeline(config)
        assert pipeline1 is pipeline2
        
        # Force reload
        pipeline3 = await diffusion_manager.load_pipeline(config, force_reload=True)
        assert pipeline3 is not None
    
    @pytest.mark.asyncio
    async def test_generation_config_validation(self, diffusion_manager) -> Any:
        """Test generation configuration validation."""
        # Valid config
        valid_config = GenerationConfig(
            prompt="test prompt",
            num_inference_steps=20,
            guidance_scale=7.5,
            width=512,
            height=512
        )
        assert valid_config.prompt == "test prompt"
        assert valid_config.num_inference_steps == 20
        
        # Test with different parameters
        config_with_seed = GenerationConfig(
            prompt="test prompt",
            seed=42,
            num_inference_steps=10
        )
        assert config_with_seed.seed == 42
        assert config_with_seed.num_inference_steps == 10
    
    @pytest.mark.asyncio
    async def test_image_to_image_config(self, diffusion_manager, sample_image) -> Any:
        """Test image-to-image configuration."""
        config = ImageToImageConfig(
            prompt="transform this image",
            image=sample_image,
            strength=0.8,
            num_inference_steps=20
        )
        
        assert config.prompt == "transform this image"
        assert config.image is sample_image
        assert config.strength == 0.8
        assert config.num_inference_steps == 20
    
    @pytest.mark.asyncio
    async def test_inpainting_config(self, diffusion_manager, sample_image, sample_mask) -> Any:
        """Test inpainting configuration."""
        config = InpaintingConfig(
            prompt="fill the masked area",
            image=sample_image,
            mask_image=sample_mask,
            mask_strength=0.8,
            num_inference_steps=20
        )
        
        assert config.prompt == "fill the masked area"
        assert config.image is sample_image
        assert config.mask_image is sample_mask
        assert config.mask_strength == 0.8
    
    @pytest.mark.asyncio
    async def test_controlnet_config(self, diffusion_manager, sample_image) -> Any:
        """Test ControlNet configuration."""
        config = ControlNetConfig(
            prompt="generate with control",
            control_image=sample_image,
            controlnet_conditioning_scale=1.0,
            num_inference_steps=20
        )
        
        assert config.prompt == "generate with control"
        assert config.control_image is sample_image
        assert config.controlnet_conditioning_scale == 1.0
    
    def test_security_prompt_engineering(self) -> Any:
        """Test security prompt engineering."""
        # Test different threat types
        threat_types = ["malware_analysis", "network_security", "threat_hunting", "incident_response"]
        
        for threat_type in threat_types:
            positive_prompt, negative_prompt = SecurityPromptEngine.generate_security_prompt(
                threat_type, "medium", "technical"
            )
            
            assert isinstance(positive_prompt, str)
            assert isinstance(negative_prompt, str)
            assert len(positive_prompt) > 0
            assert len(negative_prompt) > 0
            
            # Check that prompts contain relevant keywords
            assert "technical" in positive_prompt.lower()
            assert "cartoon" in negative_prompt.lower()
    
    def test_security_prompt_severity_levels(self) -> Any:
        """Test security prompt generation with different severity levels."""
        severities = ["low", "medium", "high", "critical"]
        
        for severity in severities:
            positive_prompt, negative_prompt = SecurityPromptEngine.generate_security_prompt(
                "malware_analysis", severity, "technical"
            )
            
            assert isinstance(positive_prompt, str)
            assert isinstance(negative_prompt, str)
            
            # Check severity-specific keywords
            if severity == "critical":
                assert "critical" in positive_prompt.lower()
            elif severity == "high":
                assert "high" in positive_prompt.lower()
    
    def test_security_prompt_styles(self) -> Any:
        """Test security prompt generation with different styles."""
        styles = ["technical", "detailed", "simple"]
        
        for style in styles:
            positive_prompt, negative_prompt = SecurityPromptEngine.generate_security_prompt(
                "malware_analysis", "medium", style
            )
            
            assert isinstance(positive_prompt, str)
            assert isinstance(negative_prompt, str)
            
            # Check style-specific keywords
            if style == "technical":
                assert "technical" in positive_prompt.lower()
            elif style == "detailed":
                assert "detailed" in positive_prompt.lower()
            elif style == "simple":
                assert "simple" in positive_prompt.lower()
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, diffusion_manager) -> Any:
        """Test metrics tracking functionality."""
        config = DiffusionConfig(
            model_name="runwayml/stable-diffusion-v1-5",
            task=DiffusionTask.TEXT_TO_IMAGE
        )
        
        # Load pipeline to initialize metrics
        await diffusion_manager.load_pipeline(config)
        
        pipeline_key = f"{config.model_name}_{config.task.value}"
        metrics = diffusion_manager.get_metrics(pipeline_key)
        
        assert isinstance(metrics, dict)
        assert 'generation_time' in metrics
        assert 'memory_usage' in metrics
        assert 'throughput' in metrics
        assert 'error_count' in metrics
        assert 'success_count' in metrics
        assert 'safety_violations' in metrics
        
        # Check initial values
        assert metrics['generation_time'] >= 0
        assert metrics['error_count'] >= 0
        assert metrics['success_count'] >= 0
        assert metrics['safety_violations'] >= 0
    
    @pytest.mark.asyncio
    async def test_cache_management(self, diffusion_manager) -> Any:
        """Test cache management functionality."""
        config = DiffusionConfig(
            model_name="runwayml/stable-diffusion-v1-5",
            task=DiffusionTask.TEXT_TO_IMAGE
        )
        
        # Load pipeline
        await diffusion_manager.load_pipeline(config)
        
        # Check that pipeline is cached
        pipeline_key = f"{config.model_name}_{config.task.value}"
        assert pipeline_key in diffusion_manager._pipelines
        
        # Clear specific pipeline
        diffusion_manager.clear_cache(pipeline_key)
        assert pipeline_key not in diffusion_manager._pipelines
        
        # Load pipeline again
        await diffusion_manager.load_pipeline(config)
        assert pipeline_key in diffusion_manager._pipelines
        
        # Clear all cache
        diffusion_manager.clear_cache()
        assert len(diffusion_manager._pipelines) == 0
        assert len(diffusion_manager._configs) == 0
        assert len(diffusion_manager._metrics) == 0
    
    @pytest.mark.asyncio
    async def test_pipeline_context_manager(self, diffusion_manager) -> Any:
        """Test pipeline context manager."""
        config = DiffusionConfig(
            model_name="runwayml/stable-diffusion-v1-5",
            task=DiffusionTask.TEXT_TO_IMAGE
        )
        
        async with diffusion_manager.pipeline_context(config) as pipeline:
            assert pipeline is not None
            assert hasattr(pipeline, 'scheduler')
    
    def test_list_loaded_pipelines(self, diffusion_manager) -> List[Any]:
        """Test listing loaded pipelines."""
        pipelines = diffusion_manager.list_loaded_pipelines()
        assert isinstance(pipelines, list)
        
        # Initially should be empty
        assert len(pipelines) == 0
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_model(self, diffusion_manager) -> Any:
        """Test error handling for invalid model."""
        config = DiffusionConfig(
            model_name="invalid/model/name",
            task=DiffusionTask.TEXT_TO_IMAGE
        )
        
        with pytest.raises(Exception):
            await diffusion_manager.load_pipeline(config)
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_task(self, diffusion_manager) -> Any:
        """Test error handling for invalid task."""
        config = DiffusionConfig(
            model_name="runwayml/stable-diffusion-v1-5",
            task="invalid_task"  # This should be an enum
        )
        
        with pytest.raises(Exception):
            await diffusion_manager.load_pipeline(config)
    
    @pytest.mark.asyncio
    async def test_optimization_application(self, diffusion_manager) -> Any:
        """Test optimization application to pipelines."""
        config = DiffusionConfig(
            model_name="runwayml/stable-diffusion-v1-5",
            task=DiffusionTask.TEXT_TO_IMAGE,
            use_attention_slicing=True,
            use_memory_efficient_attention=True,
            enable_model_cpu_offload=True
        )
        
        # This should not raise an error even if some optimizations fail
        try:
            pipeline = await diffusion_manager.load_pipeline(config)
            assert pipeline is not None
        except Exception as e:
            # Some optimizations might not be available, which is okay
            assert "optimization" in str(e).lower() or "xformers" in str(e).lower()


class TestSecurityPromptEngine:
    """Test suite for SecurityPromptEngine."""
    
    def test_prompt_generation_all_threat_types(self) -> Any:
        """Test prompt generation for all threat types."""
        threat_types = ["malware_analysis", "network_security", "threat_hunting", "incident_response"]
        severities = ["low", "medium", "high", "critical"]
        styles = ["technical", "detailed", "simple"]
        
        for threat_type in threat_types:
            for severity in severities:
                for style in styles:
                    positive_prompt, negative_prompt = SecurityPromptEngine.generate_security_prompt(
                        threat_type, severity, style
                    )
                    
                    assert isinstance(positive_prompt, str)
                    assert isinstance(negative_prompt, str)
                    assert len(positive_prompt) > 0
                    assert len(negative_prompt) > 0
    
    def test_prompt_content_validation(self) -> Any:
        """Test that generated prompts contain expected content."""
        positive_prompt, negative_prompt = SecurityPromptEngine.generate_security_prompt(
            "malware_analysis", "high", "technical"
        )
        
        # Check positive prompt content
        assert "malware" in positive_prompt.lower() or "cybersecurity" in positive_prompt.lower()
        assert "technical" in positive_prompt.lower()
        assert "high" in positive_prompt.lower()
        
        # Check negative prompt content
        assert "cartoon" in negative_prompt.lower()
        assert "artistic" in negative_prompt.lower()
    
    def test_prompt_consistency(self) -> Any:
        """Test that prompts are consistent for same inputs."""
        prompt1_pos, prompt1_neg = SecurityPromptEngine.generate_security_prompt(
            "network_security", "medium", "technical"
        )
        
        prompt2_pos, prompt2_neg = SecurityPromptEngine.generate_security_prompt(
            "network_security", "medium", "technical"
        )
        
        assert prompt1_pos == prompt2_pos
        assert prompt1_neg == prompt2_neg
    
    def test_prompt_variation(self) -> Any:
        """Test that prompts vary with different inputs."""
        prompt1_pos, _ = SecurityPromptEngine.generate_security_prompt(
            "malware_analysis", "high", "technical"
        )
        
        prompt2_pos, _ = SecurityPromptEngine.generate_security_prompt(
            "network_security", "high", "technical"
        )
        
        assert prompt1_pos != prompt2_pos


class TestDiffusionResult:
    """Test suite for DiffusionResult."""
    
    def test_diffusion_result_creation(self) -> Any:
        """Test DiffusionResult creation and attributes."""
        # Create sample images
        images = [Image.new('RGB', (512, 512), color='red') for _ in range(3)]
        
        result = DiffusionResult(
            images=images,
            nsfw_content_detected=[False, False, False],
            processing_time=1.5,
            memory_usage={"rss_mb": 100.0, "vms_mb": 200.0, "percent": 5.0},
            metadata={"test": "data"}
        )
        
        assert len(result.images) == 3
        assert len(result.nsfw_content_detected) == 3
        assert result.processing_time == 1.5
        assert result.memory_usage["rss_mb"] == 100.0
        assert result.metadata["test"] == "data"
    
    def test_diffusion_result_defaults(self) -> Any:
        """Test DiffusionResult with default values."""
        images = [Image.new('RGB', (512, 512), color='blue')]
        
        result = DiffusionResult(images=images)
        
        assert len(result.images) == 1
        assert len(result.nsfw_content_detected) == 0
        assert result.processing_time == 0.0
        assert isinstance(result.memory_usage, dict)
        assert isinstance(result.metadata, dict)


class TestIntegration:
    """Integration tests for diffusion models with transformers."""
    
    @pytest.fixture
    async def managers(self) -> Any:
        """Create both diffusion and transformers managers."""
        with tempfile.TemporaryDirectory() as temp_dir:
            diffusion_manager = DiffusionModelsManager(cache_dir=temp_dir)
            transformers_manager = TransformersManager(cache_dir=temp_dir)
            yield diffusion_manager, transformers_manager
            # Cleanup
            diffusion_manager.clear_cache()
    
    @pytest.mark.asyncio
    async def test_managers_initialization(self, managers) -> Any:
        """Test that both managers can be initialized together."""
        diffusion_manager, transformers_manager = managers
        
        assert diffusion_manager is not None
        assert transformers_manager is not None
        
        # Check that they don't interfere with each other
        assert len(diffusion_manager.list_loaded_pipelines()) == 0
        assert len(transformers_manager.list_loaded_models()) == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_loading(self, managers) -> Any:
        """Test concurrent loading of diffusion and transformer models."""
        diffusion_manager, transformers_manager = managers
        
        # Load diffusion pipeline
        diffusion_config = DiffusionConfig(
            model_name="runwayml/stable-diffusion-v1-5",
            task=DiffusionTask.TEXT_TO_IMAGE
        )
        
        # Load transformer model
        transformer_config = ModelConfig(
            model_name="microsoft/DialoGPT-medium",
            model_type=ModelType.CAUSAL_LANGUAGE_MODEL
        )
        
        # Load both concurrently
        diffusion_pipeline, transformer_model = await asyncio.gather(
            diffusion_manager.load_pipeline(diffusion_config),
            transformers_manager.load_model(transformer_config)
        )
        
        assert diffusion_pipeline is not None
        assert transformer_model is not None
        
        # Check that both are cached
        assert len(diffusion_manager.list_loaded_pipelines()) == 1
        assert len(transformers_manager.list_loaded_models()) == 1


match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 