from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import logging
import time
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
from diffusion_pipelines import (
from typing import Any, List, Dict, Optional
"""
Comprehensive Tests for Diffusion Pipelines
==========================================

This module provides extensive testing for the diffusion pipelines implementation,
including:

1. Unit tests for individual components
2. Integration tests for complete workflows
3. Performance benchmarks
4. Edge case testing
5. Error handling validation
6. Memory usage tests
7. Security considerations

Author: AI Assistant
License: MIT
"""



# Import our diffusion pipelines
    PipelineType, SchedulerType, PipelineConfig, GenerationConfig,
    DiffusionPipelineFactory, AdvancedPipelineManager,
    BaseDiffusionPipeline, StableDiffusionPipelineWrapper,
    StableDiffusionXLPipelineWrapper, StableDiffusionImg2ImgPipelineWrapper,
    StableDiffusionInpaintPipelineWrapper, StableDiffusionControlNetPipelineWrapper,
    create_pipeline, create_pipeline_manager,
    get_available_pipeline_types, get_available_scheduler_types
)

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class MockPipeline:
    """Mock pipeline for testing."""
    
    def __init__(self, config) -> Any:
        self.config = config
        self.scheduler = Mock()
        self.scheduler.config = {"beta_start": 0.0001, "beta_end": 0.02}
    
    def to(self, device) -> Any:
        return self
    
    def __call__(self, **kwargs) -> Any:
        # Mock generation result
        class MockResult:
            def __init__(self) -> Any:
                self.images = [Image.new('RGB', (512, 512), color='red')]
                self.nsfw_content_detected = [False]
        
        return MockResult()


class TestPipelineConfig(unittest.TestCase):
    """Test cases for pipeline configuration."""
    
    def test_pipeline_config_defaults(self) -> Any:
        """Test pipeline configuration defaults."""
        config = PipelineConfig()
        
        self.assertEqual(config.pipeline_type, PipelineType.STABLE_DIFFUSION)
        self.assertEqual(config.model_id, "runwayml/stable-diffusion-v1-5")
        self.assertEqual(config.scheduler_type, SchedulerType.DDIM)
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.torch_dtype, torch.float16)
        self.assertEqual(config.num_inference_steps, 50)
        self.assertEqual(config.guidance_scale, 7.5)
        self.assertEqual(config.height, 512)
        self.assertEqual(config.width, 512)
    
    def test_pipeline_config_custom(self) -> Any:
        """Test pipeline configuration with custom values."""
        config = PipelineConfig(
            pipeline_type=PipelineType.STABLE_DIFFUSION_XL,
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            scheduler_type=SchedulerType.DPM_SOLVER,
            device="cuda",
            torch_dtype=torch.float32,
            num_inference_steps=30,
            guidance_scale=10.0,
            height=768,
            width=768
        )
        
        self.assertEqual(config.pipeline_type, PipelineType.STABLE_DIFFUSION_XL)
        self.assertEqual(config.model_id, "stabilityai/stable-diffusion-xl-base-1.0")
        self.assertEqual(config.scheduler_type, SchedulerType.DPM_SOLVER)
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.torch_dtype, torch.float32)
        self.assertEqual(config.num_inference_steps, 30)
        self.assertEqual(config.guidance_scale, 10.0)
        self.assertEqual(config.height, 768)
        self.assertEqual(config.width, 768)


class TestGenerationConfig(unittest.TestCase):
    """Test cases for generation configuration."""
    
    def test_generation_config_defaults(self) -> Any:
        """Test generation configuration defaults."""
        config = GenerationConfig(prompt="test prompt")
        
        self.assertEqual(config.prompt, "test prompt")
        self.assertEqual(config.negative_prompt, "")
        self.assertEqual(config.num_inference_steps, 50)
        self.assertEqual(config.guidance_scale, 7.5)
        self.assertEqual(config.eta, 0.0)
        self.assertEqual(config.height, 512)
        self.assertEqual(config.width, 512)
        self.assertEqual(config.output_type, "pil")
        self.assertEqual(config.return_dict, True)
    
    def test_generation_config_custom(self) -> Any:
        """Test generation configuration with custom values."""
        config = GenerationConfig(
            prompt="test prompt",
            negative_prompt="bad quality",
            num_inference_steps=30,
            guidance_scale=10.0,
            eta=0.5,
            height=768,
            width=768,
            output_type="latent",
            return_dict=False
        )
        
        self.assertEqual(config.prompt, "test prompt")
        self.assertEqual(config.negative_prompt, "bad quality")
        self.assertEqual(config.num_inference_steps, 30)
        self.assertEqual(config.guidance_scale, 10.0)
        self.assertEqual(config.eta, 0.5)
        self.assertEqual(config.height, 768)
        self.assertEqual(config.width, 768)
        self.assertEqual(config.output_type, "latent")
        self.assertEqual(config.return_dict, False)


class TestBaseDiffusionPipeline(unittest.TestCase):
    """Test cases for base diffusion pipeline."""
    
    def setUp(self) -> Any:
        """Set up test fixtures."""
        self.config = PipelineConfig(
            pipeline_type=PipelineType.STABLE_DIFFUSION,
            model_id="runwayml/stable-diffusion-v1-5",
            scheduler_type=SchedulerType.DDIM
        )
    
    def test_device_detection_auto(self) -> Any:
        """Test automatic device detection."""
        config = PipelineConfig(device="auto")
        
        # Mock the device detection
        with patch('torch.cuda.is_available', return_value=True):
            pipeline = StableDiffusionPipelineWrapper(config)
            self.assertEqual(pipeline.device, torch.device("cuda"))
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=True):
                pipeline = StableDiffusionPipelineWrapper(config)
                self.assertEqual(pipeline.device, torch.device("mps"))
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                pipeline = StableDiffusionPipelineWrapper(config)
                self.assertEqual(pipeline.device, torch.device("cpu"))
    
    def test_device_detection_manual(self) -> Any:
        """Test manual device specification."""
        config = PipelineConfig(device="cpu")
        pipeline = StableDiffusionPipelineWrapper(config)
        self.assertEqual(pipeline.device, torch.device("cpu"))
    
    def test_memory_usage_tracking(self) -> Any:
        """Test memory usage tracking."""
        config = PipelineConfig(device="cpu")
        pipeline = StableDiffusionPipelineWrapper(config)
        
        memory_usage = pipeline._get_memory_usage()
        self.assertIsInstance(memory_usage, float)
        self.assertGreaterEqual(memory_usage, 0.0)
    
    def test_cleanup(self) -> Any:
        """Test pipeline cleanup."""
        config = PipelineConfig(device="cpu")
        pipeline = StableDiffusionPipelineWrapper(config)
        
        # Mock pipeline
        pipeline.pipeline = Mock()
        
        # Test cleanup
        pipeline.cleanup()
        self.assertIsNone(pipeline.pipeline)


class TestStableDiffusionPipelineWrapper(unittest.TestCase):
    """Test cases for StableDiffusionPipeline wrapper."""
    
    def setUp(self) -> Any:
        """Set up test fixtures."""
        self.config = PipelineConfig(
            pipeline_type=PipelineType.STABLE_DIFFUSION,
            model_id="runwayml/stable-diffusion-v1-5",
            scheduler_type=SchedulerType.DDIM,
            device="cpu"
        )
    
    @patch('diffusion_pipelines.StableDiffusionPipeline.from_pretrained')
    def test_create_pipeline(self, mock_from_pretrained) -> Any:
        """Test pipeline creation."""
        mock_pipeline = Mock()
        mock_from_pretrained.return_value = mock_pipeline
        
        wrapper = StableDiffusionPipelineWrapper(self.config)
        pipeline = wrapper._create_pipeline()
        
        mock_from_pretrained.assert_called_once_with(
            self.config.model_id,
            torch_dtype=self.config.torch_dtype,
            use_safetensors=self.config.use_safetensors,
            safety_checker=None,
            requires_safety_checking=self.config.requires_safety_checking,
        )
        self.assertEqual(pipeline, mock_pipeline)
    
    @patch('diffusion_pipelines.StableDiffusionPipeline.from_pretrained')
    def test_configure_pipeline(self, mock_from_pretrained) -> Any:
        """Test pipeline configuration."""
        mock_pipeline = Mock()
        mock_from_pretrained.return_value = mock_pipeline
        
        wrapper = StableDiffusionPipelineWrapper(self.config)
        wrapper.pipeline = mock_pipeline
        
        wrapper._configure_pipeline()
        
        # Check that optimization methods were called
        mock_pipeline.enable_attention_slicing.assert_called_once()
        mock_pipeline.enable_vae_slicing.assert_called_once()
    
    @patch('diffusion_pipelines.StableDiffusionPipeline.from_pretrained')
    def test_generate(self, mock_from_pretrained) -> Any:
        """Test image generation."""
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.images = [Image.new('RGB', (512, 512), color='red')]
        mock_result.nsfw_content_detected = [False]
        mock_pipeline.return_value = mock_result
        mock_from_pretrained.return_value = mock_pipeline
        
        wrapper = StableDiffusionPipelineWrapper(self.config)
        wrapper.load_pipeline()
        
        generation_config = GenerationConfig(
            prompt="test prompt",
            num_inference_steps=20,
            guidance_scale=7.5,
            height=512,
            width=512
        )
        
        result = wrapper.generate(generation_config)
        
        self.assertIsInstance(result.images, list)
        self.assertEqual(len(result.images), 1)
        self.assertIsInstance(result.processing_time, float)
        self.assertIsInstance(result.memory_usage, float)
        self.assertIsInstance(result.metadata, dict)


class TestStableDiffusionXLPipelineWrapper(unittest.TestCase):
    """Test cases for StableDiffusionXLPipeline wrapper."""
    
    def setUp(self) -> Any:
        """Set up test fixtures."""
        self.config = PipelineConfig(
            pipeline_type=PipelineType.STABLE_DIFFUSION_XL,
            model_id="stabilityai/stable-diffusion-xl-base-1.0",
            scheduler_type=SchedulerType.DPM_SOLVER,
            device="cpu"
        )
    
    @patch('diffusion_pipelines.StableDiffusionXLPipeline.from_pretrained')
    def test_create_pipeline(self, mock_from_pretrained) -> Any:
        """Test pipeline creation."""
        mock_pipeline = Mock()
        mock_from_pretrained.return_value = mock_pipeline
        
        wrapper = StableDiffusionXLPipelineWrapper(self.config)
        pipeline = wrapper._create_pipeline()
        
        mock_from_pretrained.assert_called_once_with(
            self.config.model_id,
            torch_dtype=self.config.torch_dtype,
            use_safetensors=self.config.use_safetensors,
            safety_checker=None,
            requires_safety_checking=self.config.requires_safety_checking,
        )
        self.assertEqual(pipeline, mock_pipeline)
    
    @patch('diffusion_pipelines.StableDiffusionXLPipeline.from_pretrained')
    def test_configure_pipeline(self, mock_from_pretrained) -> Any:
        """Test pipeline configuration."""
        mock_pipeline = Mock()
        mock_from_pretrained.return_value = mock_pipeline
        
        wrapper = StableDiffusionXLPipelineWrapper(self.config)
        wrapper.pipeline = mock_pipeline
        
        wrapper._configure_pipeline()
        
        # Check that optimization methods were called
        mock_pipeline.enable_attention_slicing.assert_called_once()
        mock_pipeline.enable_vae_slicing.assert_called_once()


class TestStableDiffusionImg2ImgPipelineWrapper(unittest.TestCase):
    """Test cases for StableDiffusionImg2ImgPipeline wrapper."""
    
    def setUp(self) -> Any:
        """Set up test fixtures."""
        self.config = PipelineConfig(
            pipeline_type=PipelineType.IMG2IMG,
            model_id="runwayml/stable-diffusion-v1-5",
            scheduler_type=SchedulerType.DDIM,
            device="cpu"
        )
    
    @patch('diffusion_pipelines.StableDiffusionImg2ImgPipeline.from_pretrained')
    def test_create_pipeline(self, mock_from_pretrained) -> Any:
        """Test pipeline creation."""
        mock_pipeline = Mock()
        mock_from_pretrained.return_value = mock_pipeline
        
        wrapper = StableDiffusionImg2ImgPipelineWrapper(self.config)
        pipeline = wrapper._create_pipeline()
        
        mock_from_pretrained.assert_called_once_with(
            self.config.model_id,
            torch_dtype=self.config.torch_dtype,
            use_safetensors=self.config.use_safetensors,
            safety_checker=None,
            requires_safety_checking=self.config.requires_safety_checking,
        )
        self.assertEqual(pipeline, mock_pipeline)
    
    @patch('diffusion_pipelines.StableDiffusionImg2ImgPipeline.from_pretrained')
    def test_generate_without_image(self, mock_from_pretrained) -> Any:
        """Test generation without image (should fail)."""
        mock_pipeline = Mock()
        mock_from_pretrained.return_value = mock_pipeline
        
        wrapper = StableDiffusionImg2ImgPipelineWrapper(self.config)
        wrapper.load_pipeline()
        
        generation_config = GenerationConfig(
            prompt="test prompt",
            num_inference_steps=20,
            guidance_scale=7.5,
            height=512,
            width=512
        )
        
        with self.assertRaises(ValueError):
            wrapper.generate(generation_config)


class TestStableDiffusionInpaintPipelineWrapper(unittest.TestCase):
    """Test cases for StableDiffusionInpaintPipeline wrapper."""
    
    def setUp(self) -> Any:
        """Set up test fixtures."""
        self.config = PipelineConfig(
            pipeline_type=PipelineType.INPAINT,
            model_id="runwayml/stable-diffusion-inpainting",
            scheduler_type=SchedulerType.DDIM,
            device="cpu"
        )
    
    @patch('diffusion_pipelines.StableDiffusionInpaintPipeline.from_pretrained')
    def test_create_pipeline(self, mock_from_pretrained) -> Any:
        """Test pipeline creation."""
        mock_pipeline = Mock()
        mock_from_pretrained.return_value = mock_pipeline
        
        wrapper = StableDiffusionInpaintPipelineWrapper(self.config)
        pipeline = wrapper._create_pipeline()
        
        mock_from_pretrained.assert_called_once_with(
            self.config.model_id,
            torch_dtype=self.config.torch_dtype,
            use_safetensors=self.config.use_safetensors,
            safety_checker=None,
            requires_safety_checking=self.config.requires_safety_checking,
        )
        self.assertEqual(pipeline, mock_pipeline)
    
    @patch('diffusion_pipelines.StableDiffusionInpaintPipeline.from_pretrained')
    def test_generate_without_image_and_mask(self, mock_from_pretrained) -> Any:
        """Test generation without image and mask (should fail)."""
        mock_pipeline = Mock()
        mock_from_pretrained.return_value = mock_pipeline
        
        wrapper = StableDiffusionInpaintPipelineWrapper(self.config)
        wrapper.load_pipeline()
        
        generation_config = GenerationConfig(
            prompt="test prompt",
            num_inference_steps=20,
            guidance_scale=7.5,
            height=512,
            width=512
        )
        
        with self.assertRaises(ValueError):
            wrapper.generate(generation_config)


class TestStableDiffusionControlNetPipelineWrapper(unittest.TestCase):
    """Test cases for StableDiffusionControlNetPipeline wrapper."""
    
    def setUp(self) -> Any:
        """Set up test fixtures."""
        self.config = PipelineConfig(
            pipeline_type=PipelineType.CONTROLNET,
            model_id="runwayml/stable-diffusion-v1-5",
            scheduler_type=SchedulerType.DDIM,
            device="cpu"
        )
    
    @patch('diffusion_pipelines.ControlNetModel.from_pretrained')
    @patch('diffusion_pipelines.StableDiffusionControlNetPipeline.from_pretrained')
    def test_create_pipeline(self, mock_pipeline_from_pretrained, mock_controlnet_from_pretrained) -> Any:
        """Test pipeline creation."""
        mock_controlnet = Mock()
        mock_pipeline = Mock()
        mock_controlnet_from_pretrained.return_value = mock_controlnet
        mock_pipeline_from_pretrained.return_value = mock_pipeline
        
        wrapper = StableDiffusionControlNetPipelineWrapper(self.config)
        pipeline = wrapper._create_pipeline()
        
        mock_controlnet_from_pretrained.assert_called_once()
        mock_pipeline_from_pretrained.assert_called_once()
        self.assertEqual(pipeline, mock_pipeline)


class TestDiffusionPipelineFactory(unittest.TestCase):
    """Test cases for diffusion pipeline factory."""
    
    def test_create_stable_diffusion_pipeline(self) -> Any:
        """Test creating StableDiffusionPipeline."""
        config = PipelineConfig(
            pipeline_type=PipelineType.STABLE_DIFFUSION,
            model_id="runwayml/stable-diffusion-v1-5"
        )
        
        pipeline = DiffusionPipelineFactory.create(config)
        self.assertIsInstance(pipeline, StableDiffusionPipelineWrapper)
    
    def test_create_stable_diffusion_xl_pipeline(self) -> Any:
        """Test creating StableDiffusionXLPipeline."""
        config = PipelineConfig(
            pipeline_type=PipelineType.STABLE_DIFFUSION_XL,
            model_id="stabilityai/stable-diffusion-xl-base-1.0"
        )
        
        pipeline = DiffusionPipelineFactory.create(config)
        self.assertIsInstance(pipeline, StableDiffusionXLPipelineWrapper)
    
    def test_create_img2img_pipeline(self) -> Any:
        """Test creating StableDiffusionImg2ImgPipeline."""
        config = PipelineConfig(
            pipeline_type=PipelineType.IMG2IMG,
            model_id="runwayml/stable-diffusion-v1-5"
        )
        
        pipeline = DiffusionPipelineFactory.create(config)
        self.assertIsInstance(pipeline, StableDiffusionImg2ImgPipelineWrapper)
    
    def test_create_inpaint_pipeline(self) -> Any:
        """Test creating StableDiffusionInpaintPipeline."""
        config = PipelineConfig(
            pipeline_type=PipelineType.INPAINT,
            model_id="runwayml/stable-diffusion-inpainting"
        )
        
        pipeline = DiffusionPipelineFactory.create(config)
        self.assertIsInstance(pipeline, StableDiffusionInpaintPipelineWrapper)
    
    def test_create_controlnet_pipeline(self) -> Any:
        """Test creating StableDiffusionControlNetPipeline."""
        config = PipelineConfig(
            pipeline_type=PipelineType.CONTROLNET,
            model_id="runwayml/stable-diffusion-v1-5"
        )
        
        pipeline = DiffusionPipelineFactory.create(config, controlnet_model_id="test/controlnet")
        self.assertIsInstance(pipeline, StableDiffusionControlNetPipelineWrapper)
    
    def test_create_invalid_pipeline_type(self) -> Any:
        """Test creating pipeline with invalid type."""
        config = PipelineConfig(
            pipeline_type="invalid",
            model_id="test/model"
        )
        
        with self.assertRaises(ValueError):
            DiffusionPipelineFactory.create(config)


class TestAdvancedPipelineManager(unittest.TestCase):
    """Test cases for advanced pipeline manager."""
    
    def setUp(self) -> Any:
        """Set up test fixtures."""
        self.manager = AdvancedPipelineManager()
    
    def test_initialization(self) -> Any:
        """Test manager initialization."""
        self.assertEqual(len(self.manager.pipelines), 0)
        self.assertIsNone(self.manager.active_pipeline)
    
    def test_add_pipeline(self) -> Any:
        """Test adding a pipeline."""
        config = PipelineConfig(
            pipeline_type=PipelineType.STABLE_DIFFUSION,
            model_id="runwayml/stable-diffusion-v1-5",
            device="cpu"
        )
        
        # Mock the pipeline creation
        with patch('diffusion_pipelines.DiffusionPipelineFactory.create') as mock_create:
            mock_pipeline = Mock()
            mock_pipeline.load_pipeline = Mock()
            mock_create.return_value = mock_pipeline
            
            name = self.manager.add_pipeline("test_pipeline", config)
            
            self.assertEqual(name, "test_pipeline")
            self.assertIn("test_pipeline", self.manager.pipelines)
            self.assertEqual(self.manager.active_pipeline, "test_pipeline")
    
    def test_remove_pipeline(self) -> Any:
        """Test removing a pipeline."""
        # Add a pipeline first
        config = PipelineConfig(
            pipeline_type=PipelineType.STABLE_DIFFUSION,
            model_id="runwayml/stable-diffusion-v1-5",
            device="cpu"
        )
        
        with patch('diffusion_pipelines.DiffusionPipelineFactory.create') as mock_create:
            mock_pipeline = Mock()
            mock_pipeline.load_pipeline = Mock()
            mock_create.return_value = mock_pipeline
            
            self.manager.add_pipeline("test_pipeline", config)
            
            # Remove the pipeline
            self.manager.remove_pipeline("test_pipeline")
            
            self.assertNotIn("test_pipeline", self.manager.pipelines)
            self.assertIsNone(self.manager.active_pipeline)
    
    def test_set_active_pipeline(self) -> Any:
        """Test setting active pipeline."""
        # Add a pipeline first
        config = PipelineConfig(
            pipeline_type=PipelineType.STABLE_DIFFUSION,
            model_id="runwayml/stable-diffusion-v1-5",
            device="cpu"
        )
        
        with patch('diffusion_pipelines.DiffusionPipelineFactory.create') as mock_create:
            mock_pipeline = Mock()
            mock_pipeline.load_pipeline = Mock()
            mock_create.return_value = mock_pipeline
            
            self.manager.add_pipeline("test_pipeline", config)
            
            # Set as active
            self.manager.set_active_pipeline("test_pipeline")
            self.assertEqual(self.manager.active_pipeline, "test_pipeline")
    
    def test_set_active_pipeline_not_found(self) -> Any:
        """Test setting active pipeline that doesn't exist."""
        with self.assertRaises(ValueError):
            self.manager.set_active_pipeline("nonexistent_pipeline")
    
    def test_get_pipeline_info(self) -> Optional[Dict[str, Any]]:
        """Test getting pipeline information."""
        # Add a pipeline first
        config = PipelineConfig(
            pipeline_type=PipelineType.STABLE_DIFFUSION,
            model_id="runwayml/stable-diffusion-v1-5",
            device="cpu"
        )
        
        with patch('diffusion_pipelines.DiffusionPipelineFactory.create') as mock_create:
            mock_pipeline = Mock()
            mock_pipeline.load_pipeline = Mock()
            mock_pipeline.config = config
            mock_pipeline.device = torch.device("cpu")
            mock_create.return_value = mock_pipeline
            
            self.manager.add_pipeline("test_pipeline", config)
            
            info = self.manager.get_pipeline_info()
            
            self.assertIn("test_pipeline", info)
            self.assertEqual(info["test_pipeline"]["type"], "stable_diffusion")
            self.assertEqual(info["test_pipeline"]["model_id"], "runwayml/stable-diffusion-v1-5")
            self.assertTrue(info["test_pipeline"]["is_active"])
    
    def test_cleanup_all(self) -> Any:
        """Test cleaning up all pipelines."""
        # Add a pipeline first
        config = PipelineConfig(
            pipeline_type=PipelineType.STABLE_DIFFUSION,
            model_id="runwayml/stable-diffusion-v1-5",
            device="cpu"
        )
        
        with patch('diffusion_pipelines.DiffusionPipelineFactory.create') as mock_create:
            mock_pipeline = Mock()
            mock_pipeline.load_pipeline = Mock()
            mock_pipeline.cleanup = Mock()
            mock_create.return_value = mock_pipeline
            
            self.manager.add_pipeline("test_pipeline", config)
            
            # Cleanup all
            self.manager.cleanup_all()
            
            self.assertEqual(len(self.manager.pipelines), 0)
            self.assertIsNone(self.manager.active_pipeline)
            mock_pipeline.cleanup.assert_called_once()


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_create_pipeline(self) -> Any:
        """Test create_pipeline utility function."""
        with patch('diffusion_pipelines.DiffusionPipelineFactory.create') as mock_create:
            mock_pipeline = Mock()
            mock_create.return_value = mock_pipeline
            
            pipeline = create_pipeline(PipelineType.STABLE_DIFFUSION, "test/model")
            
            mock_create.assert_called_once()
            self.assertEqual(pipeline, mock_pipeline)
    
    def test_create_pipeline_manager(self) -> Any:
        """Test create_pipeline_manager utility function."""
        manager = create_pipeline_manager()
        self.assertIsInstance(manager, AdvancedPipelineManager)
    
    def test_get_available_pipeline_types(self) -> Optional[Dict[str, Any]]:
        """Test get_available_pipeline_types utility function."""
        pipeline_types = get_available_pipeline_types()
        self.assertIsInstance(pipeline_types, list)
        self.assertIn(PipelineType.STABLE_DIFFUSION, pipeline_types)
        self.assertIn(PipelineType.STABLE_DIFFUSION_XL, pipeline_types)
        self.assertIn(PipelineType.IMG2IMG, pipeline_types)
        self.assertIn(PipelineType.INPAINT, pipeline_types)
        self.assertIn(PipelineType.CONTROLNET, pipeline_types)
    
    def test_get_available_scheduler_types(self) -> Optional[Dict[str, Any]]:
        """Test get_available_scheduler_types utility function."""
        scheduler_types = get_available_scheduler_types()
        self.assertIsInstance(scheduler_types, list)
        self.assertIn(SchedulerType.DDPM, scheduler_types)
        self.assertIn(SchedulerType.DDIM, scheduler_types)
        self.assertIn(SchedulerType.DPM_SOLVER, scheduler_types)
        self.assertIn(SchedulerType.EULER, scheduler_types)


class TestPerformanceAndMemory(unittest.TestCase):
    """Test cases for performance and memory usage."""
    
    def setUp(self) -> Any:
        """Set up test fixtures."""
        self.config = PipelineConfig(
            pipeline_type=PipelineType.STABLE_DIFFUSION,
            model_id="runwayml/stable-diffusion-v1-5",
            scheduler_type=SchedulerType.DDIM,
            device="cpu"
        )
    
    def test_memory_usage_tracking(self) -> Any:
        """Test memory usage tracking."""
        with patch('diffusion_pipelines.StableDiffusionPipeline.from_pretrained') as mock_from_pretrained:
            mock_pipeline = Mock()
            mock_from_pretrained.return_value = mock_pipeline
            
            wrapper = StableDiffusionPipelineWrapper(self.config)
            memory_usage = wrapper._get_memory_usage()
            
            self.assertIsInstance(memory_usage, float)
            self.assertGreaterEqual(memory_usage, 0.0)
    
    def test_processing_time_tracking(self) -> Any:
        """Test processing time tracking."""
        with patch('diffusion_pipelines.StableDiffusionPipeline.from_pretrained') as mock_from_pretrained:
            mock_pipeline = Mock()
            mock_result = Mock()
            mock_result.images = [Image.new('RGB', (512, 512), color='red')]
            mock_result.nsfw_content_detected = [False]
            mock_pipeline.return_value = mock_result
            mock_from_pretrained.return_value = mock_pipeline
            
            wrapper = StableDiffusionPipelineWrapper(self.config)
            wrapper.load_pipeline()
            
            generation_config = GenerationConfig(
                prompt="test prompt",
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512
            )
            
            result = wrapper.generate(generation_config)
            
            self.assertIsInstance(result.processing_time, float)
            self.assertGreaterEqual(result.processing_time, 0.0)


class TestEdgeCases(unittest.TestCase):
    """Test cases for edge cases and error conditions."""
    
    def setUp(self) -> Any:
        """Set up test fixtures."""
        self.config = PipelineConfig(
            pipeline_type=PipelineType.STABLE_DIFFUSION,
            model_id="runwayml/stable-diffusion-v1-5",
            scheduler_type=SchedulerType.DDIM,
            device="cpu"
        )
    
    def test_pipeline_not_loaded_error(self) -> Any:
        """Test error when pipeline is not loaded."""
        wrapper = StableDiffusionPipelineWrapper(self.config)
        
        generation_config = GenerationConfig(
            prompt="test prompt",
            num_inference_steps=20,
            guidance_scale=7.5,
            height=512,
            width=512
        )
        
        with self.assertRaises(RuntimeError):
            wrapper.generate(generation_config)
    
    def test_invalid_device(self) -> Any:
        """Test invalid device specification."""
        config = PipelineConfig(device="invalid_device")
        
        with self.assertRaises(Exception):
            wrapper = StableDiffusionPipelineWrapper(config)
    
    def test_empty_prompt(self) -> Any:
        """Test generation with empty prompt."""
        with patch('diffusion_pipelines.StableDiffusionPipeline.from_pretrained') as mock_from_pretrained:
            mock_pipeline = Mock()
            mock_result = Mock()
            mock_result.images = [Image.new('RGB', (512, 512), color='red')]
            mock_result.nsfw_content_detected = [False]
            mock_pipeline.return_value = mock_result
            mock_from_pretrained.return_value = mock_pipeline
            
            wrapper = StableDiffusionPipelineWrapper(self.config)
            wrapper.load_pipeline()
            
            generation_config = GenerationConfig(
                prompt="",
                num_inference_steps=20,
                guidance_scale=7.5,
                height=512,
                width=512
            )
            
            # This should work (empty prompt is valid)
            result = wrapper.generate(generation_config)
            self.assertIsInstance(result.images, list)
    
    def test_zero_inference_steps(self) -> Any:
        """Test generation with zero inference steps."""
        with patch('diffusion_pipelines.StableDiffusionPipeline.from_pretrained') as mock_from_pretrained:
            mock_pipeline = Mock()
            mock_result = Mock()
            mock_result.images = [Image.new('RGB', (512, 512), color='red')]
            mock_result.nsfw_content_detected = [False]
            mock_pipeline.return_value = mock_result
            mock_from_pretrained.return_value = mock_pipeline
            
            wrapper = StableDiffusionPipelineWrapper(self.config)
            wrapper.load_pipeline()
            
            generation_config = GenerationConfig(
                prompt="test prompt",
                num_inference_steps=0,
                guidance_scale=7.5,
                height=512,
                width=512
            )
            
            # This should work (zero steps might be handled by the underlying pipeline)
            result = wrapper.generate(generation_config)
            self.assertIsInstance(result.images, list)


def run_performance_benchmarks():
    """Run performance benchmarks."""
    logger.info("Running performance benchmarks...")
    
    # Test configurations
    configs = [
        {
            "name": "stable_diffusion",
            "config": PipelineConfig(
                pipeline_type=PipelineType.STABLE_DIFFUSION,
                model_id="runwayml/stable-diffusion-v1-5",
                scheduler_type=SchedulerType.DDIM,
                device="cpu"
            )
        },
        {
            "name": "stable_diffusion_xl",
            "config": PipelineConfig(
                pipeline_type=PipelineType.STABLE_DIFFUSION_XL,
                model_id="stabilityai/stable-diffusion-xl-base-1.0",
                scheduler_type=SchedulerType.DPM_SOLVER,
                device="cpu"
            )
        }
    ]
    
    results = {}
    
    for config_info in configs:
        name = config_info["name"]
        config = config_info["config"]
        
        logger.info(f"Benchmarking {name} pipeline...")
        
        try:
            with patch('diffusion_pipelines.StableDiffusionPipeline.from_pretrained') as mock_from_pretrained:
                mock_pipeline = Mock()
                mock_result = Mock()
                mock_result.images = [Image.new('RGB', (512, 512), color='red')]
                mock_result.nsfw_content_detected = [False]
                mock_pipeline.return_value = mock_result
                mock_from_pretrained.return_value = mock_pipeline
                
                pipeline = DiffusionPipelineFactory.create(config)
                pipeline.load_pipeline()
                
                generation_config = GenerationConfig(
                    prompt="test prompt",
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    height=512,
                    width=512
                )
                
                # Benchmark
                times = []
                for _ in range(5):
                    start_time = time.time()
                    result = pipeline.generate(generation_config)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                results[name] = {
                    "mean_time": np.mean(times),
                    "std_time": np.std(times),
                    "min_time": np.min(times),
                    "max_time": np.max(times)
                }
                
                pipeline.cleanup()
                
        except Exception as e:
            logger.error(f"Benchmark failed for {name}: {e}")
            results[name] = {"error": str(e)}
    
    # Print results
    logger.info("Performance benchmark results:")
    for name, result in results.items():
        if "error" not in result:
            logger.info(f"{name}: {result['mean_time']:.3f}s ± {result['std_time']:.3f}s")
        else:
            logger.info(f"{name}: FAILED - {result['error']}")
    
    return results


if __name__ == "__main__":
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run performance benchmarks
    run_performance_benchmarks() 