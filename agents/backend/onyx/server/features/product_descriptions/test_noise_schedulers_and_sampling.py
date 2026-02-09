from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
import time
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from noise_schedulers_and_sampling import (
            import psutil
from typing import Any, List, Dict, Optional
"""
Comprehensive Tests for Noise Schedulers and Sampling Methods
============================================================

This module provides extensive testing for the noise schedulers and sampling
methods implementation, including:

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



# Import our noise schedulers and sampling methods
    NoiseScheduleType, SamplingMethod, NoiseSchedulerConfig, SamplingConfig,
    NoiseSchedulerFactory, SamplerFactory, AdvancedSamplingManager,
    BaseNoiseScheduler, BaseSampler, LinearNoiseScheduler, CosineNoiseScheduler,
    QuadraticNoiseScheduler, SigmoidNoiseScheduler, ExponentialNoiseScheduler,
    CustomNoiseScheduler, DDPMSampler, DDIMSampler, DPMSolverSampler, EulerSampler,
    create_noise_scheduler, create_sampler, create_advanced_sampling_manager
)

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class MockDiffusionModel(nn.Module):
    """Mock diffusion model for testing."""
    
    def __init__(self, in_channels: int = 4, out_channels: int = 4, hidden_size: int = 64):
        
    """__init__ function."""
super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        
        # Simple architecture for testing
        self.conv_in = nn.Conv2d(in_channels, hidden_size, 3, padding=1)
        self.conv_out = nn.Conv2d(hidden_size, out_channels, 3, padding=1)
        
        # Timestep embedding
        self.time_embed = nn.Linear(1, hidden_size)
        
        # Text embedding projection
        self.text_proj = nn.Linear(768, hidden_size)
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Timestep embedding
        timestep_float = timestep.float().view(-1, 1)
        time_emb = self.time_embed(timestep_float)
        time_emb = time_emb.view(batch_size, -1, 1, 1)
        
        # Text embedding (if provided)
        if encoder_hidden_states is not None:
            text_emb = encoder_hidden_states.mean(dim=1)
            text_emb = self.text_proj(text_emb)
            text_emb = text_emb.view(batch_size, -1, 1, 1)
            combined_emb = time_emb + text_emb
        else:
            combined_emb = time_emb
        
        # Forward pass
        h = self.conv_in(x)
        h = h + combined_emb
        h = torch.relu(h)
        output = self.conv_out(h)
        
        return output


class TestNoiseSchedulers(unittest.TestCase):
    """Test cases for noise schedulers."""
    
    def setUp(self) -> Any:
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02
        )
    
    def test_linear_noise_scheduler(self) -> Any:
        """Test linear noise scheduler."""
        scheduler = LinearNoiseScheduler(self.config)
        
        # Check basic properties
        self.assertEqual(len(scheduler.betas), self.config.num_train_timesteps)
        self.assertEqual(len(scheduler.alphas), self.config.num_train_timesteps)
        self.assertEqual(len(scheduler.alphas_cumprod), self.config.num_train_timesteps)
        
        # Check beta range
        self.assertAlmostEqual(scheduler.betas[0].item(), self.config.beta_start, places=5)
        self.assertAlmostEqual(scheduler.betas[-1].item(), self.config.beta_end, places=5)
        
        # Check alpha properties
        self.assertTrue(torch.all(scheduler.alphas > 0))
        self.assertTrue(torch.all(scheduler.alphas < 1))
        self.assertTrue(torch.all(scheduler.alphas_cumprod > 0))
        self.assertTrue(torch.all(scheduler.alphas_cumprod <= 1))
        
        # Check monotonicity
        self.assertTrue(torch.all(scheduler.alphas_cumprod[:-1] >= scheduler.alphas_cumprod[1:]))
    
    def test_cosine_noise_scheduler(self) -> Any:
        """Test cosine noise scheduler."""
        scheduler = CosineNoiseScheduler(self.config)
        
        # Check basic properties
        self.assertEqual(len(scheduler.betas), self.config.num_train_timesteps)
        
        # Check beta range
        self.assertTrue(torch.all(scheduler.betas >= 0.0001))
        self.assertTrue(torch.all(scheduler.betas <= 0.9999))
        
        # Check alpha properties
        self.assertTrue(torch.all(scheduler.alphas > 0))
        self.assertTrue(torch.all(scheduler.alphas < 1))
        self.assertTrue(torch.all(scheduler.alphas_cumprod > 0))
        self.assertTrue(torch.all(scheduler.alphas_cumprod <= 1))
    
    def test_quadratic_noise_scheduler(self) -> Any:
        """Test quadratic noise scheduler."""
        scheduler = QuadraticNoiseScheduler(self.config)
        
        # Check basic properties
        self.assertEqual(len(scheduler.betas), self.config.num_train_timesteps)
        
        # Check beta range
        self.assertAlmostEqual(scheduler.betas[0].item(), self.config.beta_start, places=5)
        self.assertAlmostEqual(scheduler.betas[-1].item(), self.config.beta_end, places=5)
        
        # Check alpha properties
        self.assertTrue(torch.all(scheduler.alphas > 0))
        self.assertTrue(torch.all(scheduler.alphas < 1))
    
    def test_sigmoid_noise_scheduler(self) -> Any:
        """Test sigmoid noise scheduler."""
        scheduler = SigmoidNoiseScheduler(self.config)
        
        # Check basic properties
        self.assertEqual(len(scheduler.betas), self.config.num_train_timesteps)
        
        # Check beta range
        self.assertTrue(torch.all(scheduler.betas >= self.config.beta_start))
        self.assertTrue(torch.all(scheduler.betas <= self.config.beta_end))
        
        # Check alpha properties
        self.assertTrue(torch.all(scheduler.alphas > 0))
        self.assertTrue(torch.all(scheduler.alphas < 1))
    
    def test_exponential_noise_scheduler(self) -> Any:
        """Test exponential noise scheduler."""
        scheduler = ExponentialNoiseScheduler(self.config)
        
        # Check basic properties
        self.assertEqual(len(scheduler.betas), self.config.num_train_timesteps)
        
        # Check beta range
        self.assertAlmostEqual(scheduler.betas[0].item(), self.config.beta_start, places=5)
        self.assertAlmostEqual(scheduler.betas[-1].item(), self.config.beta_end, places=5)
        
        # Check alpha properties
        self.assertTrue(torch.all(scheduler.alphas > 0))
        self.assertTrue(torch.all(scheduler.alphas < 1))
    
    def test_custom_noise_scheduler(self) -> Any:
        """Test custom noise scheduler."""
        # Test with custom betas
        custom_betas = torch.linspace(0.001, 0.01, 100)
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            schedule_type=NoiseScheduleType.CUSTOM,
            custom_betas=custom_betas
        )
        scheduler = CustomNoiseScheduler(config)
        
        # Check that custom betas are used
        torch.testing.assert_close(scheduler.betas, custom_betas)
        
        # Test with custom alphas_cumprod
        custom_alphas_cumprod = torch.linspace(1.0, 0.1, 101)
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            schedule_type=NoiseScheduleType.CUSTOM,
            custom_alphas_cumprod=custom_alphas_cumprod
        )
        scheduler = CustomNoiseScheduler(config)
        
        # Check that alphas_cumprod matches
        torch.testing.assert_close(scheduler.alphas_cumprod, custom_alphas_cumprod[:-1])
    
    def test_custom_noise_scheduler_validation(self) -> Any:
        """Test custom noise scheduler validation."""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            schedule_type=NoiseScheduleType.CUSTOM
        )
        
        # Should raise error when neither custom_betas nor custom_alphas_cumprod is provided
        with self.assertRaises(ValueError):
            CustomNoiseScheduler(config)
    
    def test_noise_scheduler_factory(self) -> Any:
        """Test noise scheduler factory."""
        # Test all schedule types
        for schedule_type in NoiseScheduleType:
            if schedule_type == NoiseScheduleType.CUSTOM:
                # Skip custom for now
                continue
                
            config = NoiseSchedulerConfig(schedule_type=schedule_type)
            scheduler = NoiseSchedulerFactory.create(config)
            
            # Check that correct type is created
            expected_class = {
                NoiseScheduleType.LINEAR: LinearNoiseScheduler,
                NoiseScheduleType.COSINE: CosineNoiseScheduler,
                NoiseScheduleType.QUADRATIC: QuadraticNoiseScheduler,
                NoiseScheduleType.SIGMOID: SigmoidNoiseScheduler,
                NoiseScheduleType.EXPONENTIAL: ExponentialNoiseScheduler,
            }[schedule_type]
            
            self.assertIsInstance(scheduler, expected_class)
    
    def test_noise_scheduler_factory_invalid_type(self) -> Any:
        """Test noise scheduler factory with invalid type."""
        config = NoiseSchedulerConfig(schedule_type="invalid")
        
        with self.assertRaises(ValueError):
            NoiseSchedulerFactory.create(config)
    
    def test_get_schedule_info(self) -> Optional[Dict[str, Any]]:
        """Test getting schedule information."""
        scheduler = LinearNoiseScheduler(self.config)
        info = scheduler.get_schedule_info()
        
        # Check required keys
        required_keys = ["num_timesteps", "beta_start", "beta_end", "schedule_type", "betas_range", "alphas_cumprod_range"]
        for key in required_keys:
            self.assertIn(key, info)
        
        # Check values
        self.assertEqual(info["num_timesteps"], self.config.num_train_timesteps)
        self.assertEqual(info["beta_start"], self.config.beta_start)
        self.assertEqual(info["beta_end"], self.config.beta_end)
        self.assertEqual(info["schedule_type"], self.config.schedule_type.value)


class TestSamplers(unittest.TestCase):
    """Test cases for samplers."""
    
    def setUp(self) -> Any:
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scheduler_config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            schedule_type=NoiseScheduleType.LINEAR
        )
        self.sampling_config = SamplingConfig(
            method=SamplingMethod.DDPM,
            num_inference_steps=10,
            guidance_scale=7.5,
            eta=0.0
        )
        
        # Create scheduler and model
        self.scheduler = LinearNoiseScheduler(self.scheduler_config)
        self.model = MockDiffusionModel().to(self.device)
    
    def test_ddpm_sampler(self) -> Any:
        """Test DDPM sampler."""
        sampler = DDPMSampler(self.scheduler, self.sampling_config)
        
        # Create test data
        latents = torch.randn(1, 4, 32, 32, device=self.device)
        prompt_embeds = torch.randn(1, 77, 768, device=self.device)
        negative_prompt_embeds = torch.randn(1, 77, 768, device=self.device)
        
        # Test sampling
        result = sampler.sample(self.model, latents, prompt_embeds, negative_prompt_embeds)
        
        # Check result structure
        self.assertIsInstance(result.samples, torch.Tensor)
        self.assertEqual(result.samples.shape, latents.shape)
        self.assertIsInstance(result.latents, list)
        self.assertIsInstance(result.timesteps, list)
        self.assertIsInstance(result.processing_time, float)
        self.assertIsInstance(result.metadata, dict)
        
        # Check metadata
        self.assertEqual(result.metadata["method"], "ddpm")
        self.assertEqual(result.metadata["eta"], self.sampling_config.eta)
        
        # Check timesteps
        self.assertEqual(len(result.timesteps), self.sampling_config.num_inference_steps)
    
    def test_ddim_sampler(self) -> Any:
        """Test DDIM sampler."""
        config = SamplingConfig(
            method=SamplingMethod.DDIM,
            num_inference_steps=10,
            guidance_scale=7.5,
            eta=0.0
        )
        sampler = DDIMSampler(self.scheduler, config)
        
        # Create test data
        latents = torch.randn(1, 4, 32, 32, device=self.device)
        prompt_embeds = torch.randn(1, 77, 768, device=self.device)
        negative_prompt_embeds = torch.randn(1, 77, 768, device=self.device)
        
        # Test sampling
        result = sampler.sample(self.model, latents, prompt_embeds, negative_prompt_embeds)
        
        # Check result structure
        self.assertIsInstance(result.samples, torch.Tensor)
        self.assertEqual(result.samples.shape, latents.shape)
        self.assertEqual(result.metadata["method"], "ddim")
    
    def test_dpm_solver_sampler(self) -> Any:
        """Test DPM-Solver sampler."""
        config = SamplingConfig(
            method=SamplingMethod.DPM_SOLVER,
            num_inference_steps=10,
            guidance_scale=7.5,
            eta=0.0
        )
        sampler = DPMSolverSampler(self.scheduler, config)
        
        # Create test data
        latents = torch.randn(1, 4, 32, 32, device=self.device)
        prompt_embeds = torch.randn(1, 77, 768, device=self.device)
        negative_prompt_embeds = torch.randn(1, 77, 768, device=self.device)
        
        # Test sampling
        result = sampler.sample(self.model, latents, prompt_embeds, negative_prompt_embeds)
        
        # Check result structure
        self.assertIsInstance(result.samples, torch.Tensor)
        self.assertEqual(result.samples.shape, latents.shape)
        self.assertEqual(result.metadata["method"], "dpm_solver")
    
    def test_euler_sampler(self) -> Any:
        """Test Euler sampler."""
        config = SamplingConfig(
            method=SamplingMethod.EULER,
            num_inference_steps=10,
            guidance_scale=7.5,
            eta=0.0
        )
        sampler = EulerSampler(self.scheduler, config)
        
        # Create test data
        latents = torch.randn(1, 4, 32, 32, device=self.device)
        prompt_embeds = torch.randn(1, 77, 768, device=self.device)
        negative_prompt_embeds = torch.randn(1, 77, 768, device=self.device)
        
        # Test sampling
        result = sampler.sample(self.model, latents, prompt_embeds, negative_prompt_embeds)
        
        # Check result structure
        self.assertIsInstance(result.samples, torch.Tensor)
        self.assertEqual(result.samples.shape, latents.shape)
        self.assertEqual(result.metadata["method"], "euler")
    
    def test_sampler_factory(self) -> Any:
        """Test sampler factory."""
        # Test all supported methods
        methods = [SamplingMethod.DDPM, SamplingMethod.DDIM, SamplingMethod.DPM_SOLVER, SamplingMethod.EULER]
        
        for method in methods:
            config = SamplingConfig(method=method, num_inference_steps=10)
            sampler = SamplerFactory.create(self.scheduler, config)
            
            # Check that correct type is created
            expected_class = {
                SamplingMethod.DDPM: DDPMSampler,
                SamplingMethod.DDIM: DDIMSampler,
                SamplingMethod.DPM_SOLVER: DPMSolverSampler,
                SamplingMethod.EULER: EulerSampler,
            }[method]
            
            self.assertIsInstance(sampler, expected_class)
    
    def test_sampler_factory_invalid_method(self) -> Any:
        """Test sampler factory with invalid method."""
        config = SamplingConfig(method="invalid")
        
        with self.assertRaises(ValueError):
            SamplerFactory.create(self.scheduler, config)
    
    def test_classifier_free_guidance(self) -> Any:
        """Test classifier-free guidance."""
        sampler = DDPMSampler(self.scheduler, self.sampling_config)
        
        # Create test data
        latents = torch.randn(2, 4, 32, 32, device=self.device)
        timestep = torch.tensor([50], device=self.device)
        prompt_embeds = torch.randn(1, 77, 768, device=self.device)
        negative_prompt_embeds = torch.randn(1, 77, 768, device=self.device)
        
        # Test guidance
        result = sampler._apply_classifier_free_guidance(
            latents, timestep, prompt_embeds, negative_prompt_embeds
        )
        
        # Check result
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, latents.shape)


class TestAdvancedSamplingManager(unittest.TestCase):
    """Test cases for advanced sampling manager."""
    
    def setUp(self) -> Any:
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scheduler_config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            schedule_type=NoiseScheduleType.LINEAR
        )
        self.sampling_config = SamplingConfig(
            method=SamplingMethod.DDPM,
            num_inference_steps=10,
            guidance_scale=7.5,
            eta=0.0
        )
        
        self.manager = AdvancedSamplingManager(self.scheduler_config, self.sampling_config)
        self.model = MockDiffusionModel().to(self.device)
    
    def test_advanced_sampling_manager_initialization(self) -> Any:
        """Test advanced sampling manager initialization."""
        self.assertIsInstance(self.manager.scheduler, BaseNoiseScheduler)
        self.assertIsInstance(self.manager.sampler, BaseSampler)
        
        # Check configurations
        self.assertEqual(self.manager.scheduler_config, self.scheduler_config)
        self.assertEqual(self.manager.sampling_config, self.sampling_config)
    
    def test_advanced_sampling_manager_sample(self) -> Any:
        """Test advanced sampling manager sampling."""
        latents = torch.randn(1, 4, 32, 32, device=self.device)
        prompt_embeds = torch.randn(1, 77, 768, device=self.device)
        negative_prompt_embeds = torch.randn(1, 77, 768, device=self.device)
        
        result = self.manager.sample(
            self.model, latents, prompt_embeds, negative_prompt_embeds
        )
        
        # Check result
        self.assertIsInstance(result.samples, torch.Tensor)
        self.assertEqual(result.samples.shape, latents.shape)
    
    def test_advanced_sampling_manager_get_schedule_info(self) -> Optional[Dict[str, Any]]:
        """Test getting schedule information from manager."""
        info = self.manager.get_schedule_info()
        
        # Check required keys
        required_keys = ["num_timesteps", "beta_start", "beta_end", "schedule_type", "betas_range", "alphas_cumprod_range"]
        for key in required_keys:
            self.assertIn(key, info)
    
    def test_advanced_sampling_manager_compare_methods(self) -> Any:
        """Test comparing different sampling methods."""
        latents = torch.randn(1, 4, 32, 32, device=self.device)
        prompt_embeds = torch.randn(1, 77, 768, device=self.device)
        negative_prompt_embeds = torch.randn(1, 77, 768, device=self.device)
        
        methods = [SamplingMethod.DDPM, SamplingMethod.DDIM]
        results = self.manager.compare_sampling_methods(
            self.model, latents, prompt_embeds, negative_prompt_embeds, methods
        )
        
        # Check results
        self.assertIsInstance(results, dict)
        self.assertEqual(len(results), len(methods))
        
        for method in methods:
            self.assertIn(method.value, results)
            result = results[method.value]
            self.assertIsInstance(result.samples, torch.Tensor)
            self.assertEqual(result.samples.shape, latents.shape)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_create_noise_scheduler(self) -> Any:
        """Test create_noise_scheduler utility function."""
        scheduler = create_noise_scheduler(NoiseScheduleType.LINEAR)
        
        self.assertIsInstance(scheduler, LinearNoiseScheduler)
        
        # Test with custom parameters
        scheduler = create_noise_scheduler(
            NoiseScheduleType.COSINE,
            num_train_timesteps=200,
            beta_start=0.001,
            beta_end=0.01
        )
        
        self.assertIsInstance(scheduler, CosineNoiseScheduler)
        self.assertEqual(scheduler.config.num_train_timesteps, 200)
        self.assertEqual(scheduler.config.beta_start, 0.001)
        self.assertEqual(scheduler.config.beta_end, 0.01)
    
    def test_create_sampler(self) -> Any:
        """Test create_sampler utility function."""
        sampler = create_sampler(NoiseScheduleType.LINEAR, SamplingMethod.DDPM)
        
        self.assertIsInstance(sampler, DDPMSampler)
        self.assertIsInstance(sampler.scheduler, LinearNoiseScheduler)
        
        # Test with custom parameters
        sampler = create_sampler(
            NoiseScheduleType.COSINE,
            SamplingMethod.DDIM,
            num_inference_steps=20,
            guidance_scale=10.0
        )
        
        self.assertIsInstance(sampler, DDIMSampler)
        self.assertIsInstance(sampler.scheduler, CosineNoiseScheduler)
        self.assertEqual(sampler.config.num_inference_steps, 20)
        self.assertEqual(sampler.config.guidance_scale, 10.0)
    
    def test_create_advanced_sampling_manager(self) -> Any:
        """Test create_advanced_sampling_manager utility function."""
        manager = create_advanced_sampling_manager(
            NoiseScheduleType.LINEAR,
            SamplingMethod.DDPM
        )
        
        self.assertIsInstance(manager, AdvancedSamplingManager)
        self.assertIsInstance(manager.scheduler, LinearNoiseScheduler)
        self.assertIsInstance(manager.sampler, DDPMSampler)
        
        # Test with custom parameters
        manager = create_advanced_sampling_manager(
            NoiseScheduleType.COSINE,
            SamplingMethod.DDIM,
            num_inference_steps=20,
            guidance_scale=10.0
        )
        
        self.assertIsInstance(manager, AdvancedSamplingManager)
        self.assertIsInstance(manager.scheduler, CosineNoiseScheduler)
        self.assertIsInstance(manager.sampler, DDIMSampler)
        self.assertEqual(manager.sampling_config.num_inference_steps, 20)
        self.assertEqual(manager.sampling_config.guidance_scale, 10.0)


class TestPerformanceAndMemory(unittest.TestCase):
    """Test cases for performance and memory usage."""
    
    def setUp(self) -> Any:
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MockDiffusionModel().to(self.device)
    
    def test_memory_usage(self) -> Any:
        """Test memory usage during sampling."""
        # Create configurations
        scheduler_config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            schedule_type=NoiseScheduleType.LINEAR
        )
        sampling_config = SamplingConfig(
            method=SamplingMethod.DDPM,
            num_inference_steps=10
        )
        
        manager = AdvancedSamplingManager(scheduler_config, sampling_config)
        
        # Create test data
        latents = torch.randn(1, 4, 64, 64, device=self.device)
        prompt_embeds = torch.randn(1, 77, 768, device=self.device)
        negative_prompt_embeds = torch.randn(1, 77, 768, device=self.device)
        
        # Measure memory before
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_before = torch.cuda.memory_allocated()
        else:
            process = psutil.Process()
            memory_before = process.memory_info().rss
        
        # Run sampling
        result = manager.sample(
            self.model, latents, prompt_embeds, negative_prompt_embeds
        )
        
        # Measure memory after
        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated()
            memory_used = memory_after - memory_before
        else:
            process = psutil.Process()
            memory_after = process.memory_info().rss
            memory_used = memory_after - memory_before
        
        # Check that memory usage is reasonable (less than 1GB for this test)
        self.assertLess(memory_used, 1024 * 1024 * 1024)  # 1GB
    
    def test_sampling_speed(self) -> Any:
        """Test sampling speed."""
        # Create configurations
        scheduler_config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            schedule_type=NoiseScheduleType.LINEAR
        )
        sampling_config = SamplingConfig(
            method=SamplingMethod.DDPM,
            num_inference_steps=10
        )
        
        manager = AdvancedSamplingManager(scheduler_config, sampling_config)
        
        # Create test data
        latents = torch.randn(1, 4, 32, 32, device=self.device)
        prompt_embeds = torch.randn(1, 77, 768, device=self.device)
        negative_prompt_embeds = torch.randn(1, 77, 768, device=self.device)
        
        # Measure time
        start_time = time.time()
        result = manager.sample(
            self.model, latents, prompt_embeds, negative_prompt_embeds
        )
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Check that sampling completes in reasonable time (less than 30 seconds for this test)
        self.assertLess(total_time, 30.0)
        
        # Check that processing time is recorded
        self.assertGreater(result.processing_time, 0.0)
        self.assertLessEqual(result.processing_time, total_time)


class TestEdgeCases(unittest.TestCase):
    """Test cases for edge cases and error conditions."""
    
    def setUp(self) -> Any:
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MockDiffusionModel().to(self.device)
    
    def test_zero_timesteps(self) -> Any:
        """Test with zero timesteps."""
        config = NoiseSchedulerConfig(
            num_train_timesteps=0,
            schedule_type=NoiseScheduleType.LINEAR
        )
        
        with self.assertRaises(ValueError):
            LinearNoiseScheduler(config)
    
    def test_negative_timesteps(self) -> Any:
        """Test with negative timesteps."""
        config = NoiseSchedulerConfig(
            num_train_timesteps=-1,
            schedule_type=NoiseScheduleType.LINEAR
        )
        
        with self.assertRaises(ValueError):
            LinearNoiseScheduler(config)
    
    def test_invalid_beta_range(self) -> Any:
        """Test with invalid beta range."""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.1,
            beta_end=0.05,  # beta_end < beta_start
            schedule_type=NoiseScheduleType.LINEAR
        )
        
        # This should work but might produce unexpected results
        scheduler = LinearNoiseScheduler(config)
        self.assertIsInstance(scheduler, LinearNoiseScheduler)
    
    def test_large_timesteps(self) -> Any:
        """Test with large number of timesteps."""
        config = NoiseSchedulerConfig(
            num_train_timesteps=10000,
            schedule_type=NoiseScheduleType.LINEAR
        )
        
        # This should work without errors
        scheduler = LinearNoiseScheduler(config)
        self.assertEqual(len(scheduler.betas), 10000)
    
    def test_single_timestep(self) -> Any:
        """Test with single timestep."""
        config = NoiseSchedulerConfig(
            num_train_timesteps=1,
            schedule_type=NoiseScheduleType.LINEAR
        )
        
        scheduler = LinearNoiseScheduler(config)
        self.assertEqual(len(scheduler.betas), 1)
    
    def test_empty_latents(self) -> Any:
        """Test with empty latents."""
        scheduler_config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            schedule_type=NoiseScheduleType.LINEAR
        )
        sampling_config = SamplingConfig(
            method=SamplingMethod.DDPM,
            num_inference_steps=10
        )
        
        manager = AdvancedSamplingManager(scheduler_config, sampling_config)
        
        # Empty latents
        latents = torch.empty(0, 4, 32, 32, device=self.device)
        
        with self.assertRaises(RuntimeError):
            manager.sample(self.model, latents)
    
    def test_mismatched_shapes(self) -> Any:
        """Test with mismatched shapes."""
        scheduler_config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            schedule_type=NoiseScheduleType.LINEAR
        )
        sampling_config = SamplingConfig(
            method=SamplingMethod.DDPM,
            num_inference_steps=10
        )
        
        manager = AdvancedSamplingManager(scheduler_config, sampling_config)
        
        # Mismatched batch sizes
        latents = torch.randn(2, 4, 32, 32, device=self.device)
        prompt_embeds = torch.randn(1, 77, 768, device=self.device)
        
        with self.assertRaises(RuntimeError):
            manager.sample(self.model, latents, prompt_embeds)


def run_performance_benchmarks():
    """Run performance benchmarks."""
    logger.info("Running performance benchmarks...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MockDiffusionModel().to(device)
    
    # Test configurations
    configs = [
        {"method": SamplingMethod.DDPM, "steps": 20},
        {"method": SamplingMethod.DDIM, "steps": 20},
        {"method": SamplingMethod.DPM_SOLVER, "steps": 20},
        {"method": SamplingMethod.EULER, "steps": 20},
    ]
    
    results = {}
    
    for config in configs:
        logger.info(f"Benchmarking {config['method'].value} with {config['steps']} steps...")
        
        scheduler_config = NoiseSchedulerConfig(
            num_train_timesteps=1000,
            schedule_type=NoiseScheduleType.COSINE
        )
        sampling_config = SamplingConfig(
            method=config["method"],
            num_inference_steps=config["steps"]
        )
        
        manager = AdvancedSamplingManager(scheduler_config, sampling_config)
        
        # Create test data
        latents = torch.randn(1, 4, 64, 64, device=device)
        prompt_embeds = torch.randn(1, 77, 768, device=device)
        negative_prompt_embeds = torch.randn(1, 77, 768, device=device)
        
        # Warm up
        for _ in range(3):
            manager.sample(model, latents, prompt_embeds, negative_prompt_embeds)
        
        # Benchmark
        times = []
        for _ in range(10):
            start_time = time.time()
            result = manager.sample(model, latents, prompt_embeds, negative_prompt_embeds)
            end_time = time.time()
            times.append(end_time - start_time)
        
        results[config["method"].value] = {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times)
        }
    
    # Print results
    logger.info("Performance benchmark results:")
    for method, result in results.items():
        logger.info(f"{method}: {result['mean_time']:.3f}s ± {result['std_time']:.3f}s")
    
    return results


if __name__ == "__main__":
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run performance benchmarks
    run_performance_benchmarks() 