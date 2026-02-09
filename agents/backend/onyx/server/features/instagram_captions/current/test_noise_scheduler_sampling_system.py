"""
Comprehensive Test Suite for Noise Scheduler and Sampling System

This test suite covers:
- Unit tests for all components
- Integration tests for pipelines
- Performance tests for optimization
- Edge case testing
- Error handling validation
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import time
import gc
from typing import Dict, Any

# Import the system under test
from noise_scheduler_sampling_system import (
    AdvancedDiffusionSystem,
    NoiseSchedulerConfig,
    BetaSchedule,
    SamplingMethod,
    BaseNoiseScheduler,
    BaseSampler,
    DiffusionPipeline,
    NoiseSchedulerFactory,
    SamplerFactory,
    LinearNoiseScheduler,
    CosineNoiseScheduler,
    QuadraticNoiseScheduler,
    SigmoidNoiseScheduler,
    ExponentialNoiseScheduler,
    DDPMSampler,
    DDIMSampler,
    AncestralSampler,
    EulerSampler,
    HeunSampler
)

class TestNoiseSchedulerConfig(unittest.TestCase):
    """Test cases for NoiseSchedulerConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = NoiseSchedulerConfig()
        
        self.assertEqual(config.num_train_timesteps, 1000)
        self.assertEqual(config.beta_start, 0.0001)
        self.assertEqual(config.beta_end, 0.02)
        self.assertEqual(config.beta_schedule, BetaSchedule.LINEAR)
        self.assertTrue(config.clip_sample)
        self.assertEqual(config.prediction_type, "epsilon")
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = NoiseSchedulerConfig(
            num_train_timesteps=500,
            beta_start=0.001,
            beta_end=0.05,
            beta_schedule=BetaSchedule.COSINE,
            clip_sample=False
        )
        
        self.assertEqual(config.num_train_timesteps, 500)
        self.assertEqual(config.beta_start, 0.001)
        self.assertEqual(config.beta_end, 0.05)
        self.assertEqual(config.beta_schedule, BetaSchedule.COSINE)
        self.assertFalse(config.clip_sample)
    
    def test_all_beta_schedules(self):
        """Test all available beta schedules."""
        schedules = [
            BetaSchedule.LINEAR,
            BetaSchedule.COSINE,
            BetaSchedule.QUADRATIC,
            BetaSchedule.SIGMOID,
            BetaSchedule.EXPONENTIAL
        ]
        
        for schedule in schedules:
            config = NoiseSchedulerConfig(beta_schedule=schedule)
            self.assertEqual(config.beta_schedule, schedule)

class TestBaseNoiseScheduler(unittest.TestCase):
    """Test cases for BaseNoiseScheduler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.001,
            beta_end=0.02
        )
    
    def test_abstract_class(self):
        """Test that BaseNoiseScheduler is abstract."""
        with self.assertRaises(TypeError):
            BaseNoiseScheduler(self.config)
    
    def test_derived_class_initialization(self):
        """Test initialization of derived scheduler classes."""
        schedulers = [
            LinearNoiseScheduler(self.config),
            CosineNoiseScheduler(self.config),
            QuadraticNoiseScheduler(self.config),
            SigmoidNoiseScheduler(self.config),
            ExponentialNoiseScheduler(self.config)
        ]
        
        for scheduler in schedulers:
            self.assertIsInstance(scheduler, BaseNoiseScheduler)
            self.assertEqual(scheduler.num_train_timesteps, 100)
            self.assertEqual(scheduler.betas.shape[0], 100)
            self.assertEqual(scheduler.alphas.shape[0], 100)
            self.assertEqual(scheduler.alphas_cumprod.shape[0], 100)
    
    def test_beta_properties(self):
        """Test beta schedule properties."""
        scheduler = LinearNoiseScheduler(self.config)
        
        # Test beta values
        self.assertTrue(torch.all(scheduler.betas >= 0))
        self.assertTrue(torch.all(scheduler.betas <= 1))
        self.assertEqual(scheduler.betas[0], 0.001)
        self.assertEqual(scheduler.betas[-1], 0.02)
        
        # Test alpha values
        self.assertTrue(torch.all(scheduler.alphas >= 0))
        self.assertTrue(torch.all(scheduler.alphas <= 1))
        self.assertEqual(scheduler.alphas[0], 1.0 - 0.001)
        self.assertEqual(scheduler.alphas[-1], 1.0 - 0.02)
    
    def test_alphas_cumprod_properties(self):
        """Test cumulative alpha product properties."""
        scheduler = LinearNoiseScheduler(self.config)
        
        # Test alphas_cumprod is decreasing
        for i in range(1, len(scheduler.alphas_cumprod)):
            self.assertGreaterEqual(scheduler.alphas_cumprod[i-1], scheduler.alphas_cumprod[i])
        
        # Test alphas_cumprod_prev
        self.assertEqual(scheduler.alphas_cumprod_prev[0], 1.0)
        self.assertEqual(scheduler.alphas_cumprod_prev[1], scheduler.alphas_cumprod[0])
    
    def test_add_noise(self):
        """Test noise addition functionality."""
        scheduler = LinearNoiseScheduler(self.config)
        
        # Create test samples
        original_samples = torch.randn(2, 3, 32, 32)
        timesteps = torch.tensor([10, 50])
        
        # Add noise
        noisy_samples, noise = scheduler.add_noise(original_samples, timesteps)
        
        # Check shapes
        self.assertEqual(noisy_samples.shape, original_samples.shape)
        self.assertEqual(noise.shape, original_samples.shape)
        
        # Check that noise was actually added
        self.assertFalse(torch.allclose(noisy_samples, original_samples))
    
    def test_get_velocity(self):
        """Test velocity calculation."""
        scheduler = LinearNoiseScheduler(self.config)
        
        # Create test data
        sample = torch.randn(2, 3, 32, 32)
        noise = torch.randn(2, 3, 32, 32)
        timesteps = torch.tensor([10, 50])
        
        # Calculate velocity
        velocity = scheduler.get_velocity(sample, noise, timesteps)
        
        # Check shape
        self.assertEqual(velocity.shape, sample.shape)
        
        # Check that velocity is different from input
        self.assertFalse(torch.allclose(velocity, sample))
        self.assertFalse(torch.allclose(velocity, noise))

class TestSpecificNoiseSchedulers(unittest.TestCase):
    """Test cases for specific noise scheduler implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.001,
            beta_end=0.02
        )
    
    def test_linear_scheduler(self):
        """Test linear noise scheduler."""
        scheduler = LinearNoiseScheduler(self.config)
        
        # Test linear progression
        betas = scheduler.betas
        for i in range(1, len(betas)):
            self.assertGreater(betas[i], betas[i-1])
        
        # Test endpoints
        self.assertAlmostEqual(betas[0].item(), 0.001, places=6)
        self.assertAlmostEqual(betas[-1].item(), 0.02, places=6)
    
    def test_cosine_scheduler(self):
        """Test cosine noise scheduler."""
        scheduler = CosineNoiseScheduler(self.config)
        
        # Test cosine properties
        betas = scheduler.betas
        self.assertTrue(torch.all(betas >= 0.0001))
        self.assertTrue(torch.all(betas <= 0.9999))
        
        # Test smooth progression (no sharp jumps)
        beta_diffs = torch.diff(betas)
        self.assertTrue(torch.all(beta_diffs >= -0.1))  # Allow some variation
    
    def test_quadratic_scheduler(self):
        """Test quadratic noise scheduler."""
        scheduler = QuadraticNoiseScheduler(self.config)
        
        # Test quadratic progression
        betas = scheduler.betas
        self.assertAlmostEqual(betas[0].item(), 0.001, places=6)
        self.assertAlmostEqual(betas[-1].item(), 0.02, places=6)
        
        # Test that it's not linear
        linear_betas = torch.linspace(0.001, 0.02, 100)
        self.assertFalse(torch.allclose(betas, linear_betas))
    
    def test_sigmoid_scheduler(self):
        """Test sigmoid noise scheduler."""
        scheduler = SigmoidNoiseScheduler(self.config)
        
        # Test sigmoid properties
        betas = scheduler.betas
        self.assertAlmostEqual(betas[0].item(), 0.001, places=6)
        self.assertAlmostEqual(betas[-1].item(), 0.02, places=6)
        
        # Test gradual start and end
        early_beta_diff = betas[10] - betas[0]
        late_beta_diff = betas[-1] - betas[-11]
        self.assertGreater(early_beta_diff, 0)
        self.assertGreater(late_beta_diff, 0)
    
    def test_exponential_scheduler(self):
        """Test exponential noise scheduler."""
        scheduler = ExponentialNoiseScheduler(self.config)
        
        # Test exponential properties
        betas = scheduler.betas
        self.assertAlmostEqual(betas[0].item(), 0.001, places=6)
        self.assertAlmostEqual(betas[-1].item(), 0.02, places=6)
        
        # Test exponential growth
        for i in range(1, len(betas)):
            self.assertGreater(betas[i], betas[i-1])

class TestBaseSampler(unittest.TestCase):
    """Test cases for BaseSampler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.001,
            beta_end=0.02
        )
        self.scheduler = LinearNoiseScheduler(self.config)
    
    def test_abstract_class(self):
        """Test that BaseSampler is abstract."""
        with self.assertRaises(TypeError):
            BaseSampler(self.scheduler)
    
    def test_derived_class_initialization(self):
        """Test initialization of derived sampler classes."""
        samplers = [
            DDPMSampler(self.scheduler),
            DDIMSampler(self.scheduler),
            AncestralSampler(self.scheduler),
            EulerSampler(self.scheduler),
            HeunSampler(self.scheduler)
        ]
        
        for sampler in samplers:
            self.assertIsInstance(sampler, BaseSampler)
            self.assertEqual(sampler.scheduler, self.scheduler)
    
    def test_sampler_step_interface(self):
        """Test that all samplers implement the step method."""
        samplers = [
            DDPMSampler(self.scheduler),
            DDIMSampler(self.scheduler),
            AncestralSampler(self.scheduler),
            EulerSampler(self.scheduler),
            HeunSampler(self.scheduler)
        ]
        
        for sampler in samplers:
            # Test that step method exists and is callable
            self.assertTrue(hasattr(sampler, 'step'))
            self.assertTrue(callable(sampler.step))
            
            # Test step method signature
            import inspect
            sig = inspect.signature(sampler.step)
            params = list(sig.parameters.keys())
            self.assertIn('model_output', params)
            self.assertIn('timestep', params)
            self.assertIn('sample', params)

class TestSpecificSamplers(unittest.TestCase):
    """Test cases for specific sampler implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.001,
            beta_end=0.02
        )
        self.scheduler = LinearNoiseScheduler(self.config)
    
    def test_ddpm_sampler(self):
        """Test DDPM sampler."""
        sampler = DDPMSampler(self.scheduler)
        
        # Create test data
        model_output = torch.randn(2, 3, 32, 32)
        sample = torch.randn(2, 3, 32, 32)
        timestep = 50
        
        # Test step
        result = sampler.step(model_output, timestep, sample)
        
        # Check shape
        self.assertEqual(result.shape, sample.shape)
        
        # Check that result is different from input
        self.assertFalse(torch.allclose(result, sample))
    
    def test_ddim_sampler(self):
        """Test DDIM sampler."""
        # Test with eta = 0 (deterministic)
        sampler = DDIMSampler(self.scheduler, eta=0.0)
        
        model_output = torch.randn(2, 3, 32, 32)
        sample = torch.randn(2, 3, 32, 32)
        timestep = 50
        
        result = sampler.step(model_output, timestep, sample)
        self.assertEqual(result.shape, sample.shape)
        
        # Test with eta > 0 (stochastic)
        sampler_stochastic = DDIMSampler(self.scheduler, eta=0.5)
        result_stochastic = sampler_stochastic.step(model_output, timestep, sample)
        self.assertEqual(result_stochastic.shape, sample.shape)
    
    def test_ancestral_sampler(self):
        """Test ancestral sampler."""
        sampler = AncestralSampler(self.scheduler)
        
        model_output = torch.randn(2, 3, 32, 32)
        sample = torch.randn(2, 3, 32, 32)
        timestep = 50
        
        result = sampler.step(model_output, timestep, sample)
        self.assertEqual(result.shape, sample.shape)
    
    def test_euler_sampler(self):
        """Test Euler sampler."""
        sampler = EulerSampler(self.scheduler)
        
        model_output = torch.randn(2, 3, 32, 32)
        sample = torch.randn(2, 3, 32, 32)
        timestep = 50
        
        result = sampler.step(model_output, timestep, sample)
        self.assertEqual(result.shape, sample.shape)
    
    def test_heun_sampler(self):
        """Test Heun sampler."""
        sampler = HeunSampler(self.scheduler)
        
        model_output = torch.randn(2, 3, 32, 32)
        sample = torch.randn(2, 3, 32, 32)
        timestep = 50
        
        result = sampler.step(model_output, timestep, sample)
        self.assertEqual(result.shape, sample.shape)

class TestFactories(unittest.TestCase):
    """Test cases for factory classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.001,
            beta_end=0.02
        )
    
    def test_noise_scheduler_factory(self):
        """Test NoiseSchedulerFactory."""
        # Test all scheduler types
        scheduler_types = [
            BetaSchedule.LINEAR,
            BetaSchedule.COSINE,
            BetaSchedule.QUADRATIC,
            BetaSchedule.SIGMOID,
            BetaSchedule.EXPONENTIAL
        ]
        
        for schedule_type in scheduler_types:
            config = NoiseSchedulerConfig(beta_schedule=schedule_type)
            scheduler = NoiseSchedulerFactory.create_scheduler(config)
            self.assertIsInstance(scheduler, BaseNoiseScheduler)
        
        # Test invalid schedule type
        with self.assertRaises(ValueError):
            config = NoiseSchedulerConfig(beta_schedule=BetaSchedule.CUSTOM)
            NoiseSchedulerFactory.create_scheduler(config)
    
    def test_sampler_factory(self):
        """Test SamplerFactory."""
        scheduler = LinearNoiseScheduler(self.config)
        
        # Test all sampler types
        sampler_types = [
            SamplingMethod.DDPM,
            SamplingMethod.DDIM,
            SamplingMethod.ANCESTRAL,
            SamplingMethod.EULER,
            SamplingMethod.HEUN
        ]
        
        for sampler_type in sampler_types:
            sampler = SamplerFactory.create_sampler(sampler_type, scheduler)
            self.assertIsInstance(sampler, BaseSampler)
        
        # Test DDIM with eta parameter
        sampler = SamplerFactory.create_sampler(
            SamplingMethod.DDIM, 
            scheduler, 
            eta=0.5
        )
        self.assertIsInstance(sampler, DDIMSampler)
        self.assertEqual(sampler.eta, 0.5)
        
        # Test invalid sampler type
        with self.assertRaises(ValueError):
            SamplerFactory.create_sampler(SamplingMethod.PNDM, scheduler)

class TestDiffusionPipeline(unittest.TestCase):
    """Test cases for DiffusionPipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.001,
            beta_end=0.02
        )
        self.pipeline = DiffusionPipeline(self.config, SamplingMethod.DDPM)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline.scheduler)
        self.assertIsNotNone(self.pipeline.sampler)
        self.assertIsNotNone(self.pipeline.device)
    
    def test_add_noise(self):
        """Test noise addition in pipeline."""
        original_samples = torch.randn(2, 3, 32, 32)
        timesteps = torch.tensor([10, 50])
        
        noisy_samples, noise = self.pipeline.add_noise(original_samples, timesteps)
        
        self.assertEqual(noisy_samples.shape, original_samples.shape)
        self.assertEqual(noise.shape, original_samples.shape)
        self.assertFalse(torch.allclose(noisy_samples, original_samples))
    
    def test_get_velocity(self):
        """Test velocity calculation in pipeline."""
        sample = torch.randn(2, 3, 32, 32)
        noise = torch.randn(2, 3, 32, 32)
        timesteps = torch.tensor([10, 50])
        
        velocity = self.pipeline.get_velocity(sample, noise, timesteps)
        
        self.assertEqual(velocity.shape, sample.shape)
        self.assertFalse(torch.allclose(velocity, sample))
    
    def test_sample_with_dummy_model(self):
        """Test sampling with a dummy model."""
        # Create a simple dummy model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x, t, **kwargs):
                return torch.randn_like(x)
        
        model = DummyModel()
        shape = (1, 3, 32, 32)
        
        # Test sampling
        samples = self.pipeline.sample(
            model=model,
            shape=shape,
            num_inference_steps=10,
            classifier_free_guidance=False
        )
        
        self.assertEqual(samples.shape, shape)
    
    def test_sample_with_classifier_free_guidance(self):
        """Test sampling with classifier-free guidance."""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x, t, **kwargs):
                return torch.randn_like(x)
        
        model = DummyModel()
        shape = (1, 3, 32, 32)
        
        # Test with guidance
        samples = self.pipeline.sample(
            model=model,
            shape=shape,
            num_inference_steps=10,
            classifier_free_guidance=True,
            guidance_scale=7.5
        )
        
        self.assertEqual(samples.shape, shape)

class TestAdvancedDiffusionSystem(unittest.TestCase):
    """Test cases for AdvancedDiffusionSystem."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = AdvancedDiffusionSystem()
    
    def test_system_initialization(self):
        """Test system initialization."""
        self.assertEqual(len(self.system.pipelines), 0)
        self.assertEqual(len(self.system.configs), 0)
    
    def test_create_pipeline(self):
        """Test pipeline creation."""
        config = NoiseSchedulerConfig(
            beta_schedule=BetaSchedule.COSINE,
            num_train_timesteps=100
        )
        
        pipeline = self.system.create_pipeline(
            "test_pipeline",
            config,
            SamplingMethod.DDIM
        )
        
        self.assertIn("test_pipeline", self.system.pipelines)
        self.assertIn("test_pipeline", self.system.configs)
        self.assertIsInstance(pipeline, DiffusionPipeline)
    
    def test_get_pipeline(self):
        """Test pipeline retrieval."""
        config = NoiseSchedulerConfig()
        self.system.create_pipeline("test", config, SamplingMethod.DDPM)
        
        pipeline = self.system.get_pipeline("test")
        self.assertIsInstance(pipeline, DiffusionPipeline)
        
        # Test non-existent pipeline
        with self.assertRaises(ValueError):
            self.system.get_pipeline("non_existent")
    
    def test_list_pipelines(self):
        """Test pipeline listing."""
        config = NoiseSchedulerConfig()
        
        self.system.create_pipeline("pipeline1", config, SamplingMethod.DDPM)
        self.system.create_pipeline("pipeline2", config, SamplingMethod.DDIM)
        
        pipelines = self.system.list_pipelines()
        self.assertEqual(len(pipelines), 2)
        self.assertIn("pipeline1", pipelines)
        self.assertIn("pipeline2", pipelines)
    
    def test_compare_schedulers(self):
        """Test scheduler comparison."""
        # Create multiple pipelines
        configs = {
            'linear': NoiseSchedulerConfig(beta_schedule=BetaSchedule.LINEAR),
            'cosine': NoiseSchedulerConfig(beta_schedule=BetaSchedule.COSINE),
            'quadratic': NoiseSchedulerConfig(beta_schedule=BetaSchedule.QUADRATIC)
        }
        
        for name, config in configs.items():
            self.system.create_pipeline(name, config, SamplingMethod.DDPM)
        
        # Compare schedulers
        shape = (1, 3, 32, 32)
        results = self.system.compare_schedulers(shape, num_inference_steps=5)
        
        self.assertEqual(len(results), 3)
        for name, result in results.items():
            if result is not None:
                self.assertEqual(result.shape, shape)
    
    def test_config_file_operations(self):
        """Test configuration file operations."""
        # Create test configuration
        config = NoiseSchedulerConfig(
            beta_schedule=BetaSchedule.COSINE,
            num_train_timesteps=100
        )
        
        self.system.create_pipeline("test", config, SamplingMethod.DDPM)
        
        # Test save and load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save configuration
            self.system.save_config(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Create new system and load configuration
            new_system = AdvancedDiffusionSystem(temp_path)
            self.assertEqual(len(new_system.pipelines), 1)
            self.assertIn("test", new_system.pipelines)
            
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = AdvancedDiffusionSystem()
        
        # Create multiple pipelines with different configurations
        configs = {
            'linear_ddpm': NoiseSchedulerConfig(
                beta_schedule=BetaSchedule.LINEAR,
                num_train_timesteps=50
            ),
            'cosine_ddim': NoiseSchedulerConfig(
                beta_schedule=BetaSchedule.COSINE,
                num_train_timesteps=50
            ),
            'quadratic_ancestral': NoiseSchedulerConfig(
                beta_schedule=BetaSchedule.QUADRATIC,
                num_train_timesteps=50
            )
        }
        
        samplers = [
            SamplingMethod.DDPM,
            SamplingMethod.DDIM,
            SamplingMethod.ANCESTRAL
        ]
        
        for (name, config), sampler in zip(configs.items(), samplers):
            self.system.create_pipeline(name, config, sampler)
    
    def test_full_workflow(self):
        """Test complete workflow from configuration to sampling."""
        # Test that all pipelines are created
        self.assertEqual(len(self.system.pipelines), 3)
        
        # Test sampling with each pipeline
        shape = (1, 3, 16, 16)  # Small shape for fast testing
        
        for name in self.system.list_pipelines():
            pipeline = self.system.get_pipeline(name)
            
            # Create dummy model
            class DummyModel(nn.Module):
                def __init__(self):
                    super().__init__()
                
                def forward(self, x, t, **kwargs):
                    return torch.randn_like(x)
            
            model = DummyModel()
            
            # Test sampling
            samples = pipeline.sample(
                model=model,
                shape=shape,
                num_inference_steps=5,
                classifier_free_guidance=False
            )
            
            self.assertEqual(samples.shape, shape)
    
    def test_pipeline_comparison(self):
        """Test pipeline comparison functionality."""
        shape = (1, 3, 16, 16)
        
        # Compare all pipelines
        results = self.system.compare_schedulers(
            shape=shape,
            num_inference_steps=5
        )
        
        # Check that all pipelines completed successfully
        for name, result in results.items():
            self.assertIsNotNone(result, f"Pipeline {name} failed")
            self.assertEqual(result.shape, shape)

class TestPerformance(unittest.TestCase):
    """Performance tests for the system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = AdvancedDiffusionSystem()
        
        # Create test pipeline
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_schedule=BetaSchedule.COSINE
        )
        
        self.pipeline = self.system.create_pipeline(
            "performance_test",
            config,
            SamplingMethod.DDIM
        )
    
    def test_sampling_performance(self):
        """Test sampling performance."""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x, t, **kwargs):
                return torch.randn_like(x)
        
        model = DummyModel()
        shape = (1, 3, 64, 64)
        
        # Measure sampling time
        start_time = time.time()
        
        samples = self.pipeline.sample(
            model=model,
            shape=shape,
            num_inference_steps=20,
            classifier_free_guidance=False
        )
        
        end_time = time.time()
        sampling_time = end_time - start_time
        
        # Check that sampling completed in reasonable time
        self.assertLess(sampling_time, 10.0)  # Should complete in under 10 seconds
        self.assertEqual(samples.shape, shape)
    
    def test_memory_usage(self):
        """Test memory usage during sampling."""
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x, t, **kwargs):
                return torch.randn_like(x)
        
        model = DummyModel()
        shape = (1, 3, 32, 32)
        
        # Force garbage collection
        gc.collect()
        
        # Get initial memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        else:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss
        
        # Perform sampling
        samples = self.pipeline.sample(
            model=model,
            shape=shape,
            num_inference_steps=10,
            classifier_free_guidance=False
        )
        
        # Force garbage collection again
        gc.collect()
        
        # Get final memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
        else:
            process = psutil.Process()
            final_memory = process.memory_info().rss
        
        # Check that memory usage is reasonable
        memory_increase = final_memory - initial_memory
        if torch.cuda.is_available():
            # GPU memory should not increase significantly
            self.assertLess(memory_increase, 100 * 1024 * 1024)  # 100MB
        else:
            # CPU memory should not increase significantly
            self.assertLess(memory_increase, 50 * 1024 * 1024)  # 50MB

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.system = AdvancedDiffusionSystem()
    
    def test_invalid_timesteps(self):
        """Test behavior with invalid timesteps."""
        config = NoiseSchedulerConfig(
            num_train_timesteps=0  # Invalid
        )
        
        with self.assertRaises(Exception):
            LinearNoiseScheduler(config)
    
    def test_extreme_beta_values(self):
        """Test behavior with extreme beta values."""
        config = NoiseSchedulerConfig(
            beta_start=0.0,
            beta_end=1.0
        )
        
        # This should work but may produce warnings
        scheduler = LinearNoiseScheduler(config)
        self.assertIsInstance(scheduler, LinearNoiseScheduler)
    
    def test_empty_pipeline_list(self):
        """Test behavior with empty pipeline list."""
        pipelines = self.system.list_pipelines()
        self.assertEqual(len(pipelines), 0)
        
        # Test comparison with no pipelines
        results = self.system.compare_schedulers((1, 3, 32, 32))
        self.assertEqual(len(results), 0)
    
    def test_invalid_config_file(self):
        """Test behavior with invalid configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content")
            temp_path = f.name
        
        try:
            # This should handle the error gracefully
            system = AdvancedDiffusionSystem(temp_path)
            self.assertEqual(len(system.pipelines), 0)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

def run_performance_benchmark():
    """Run performance benchmark tests."""
    print("Running Performance Benchmarks...")
    
    # Test different configurations
    configs = [
        ("Linear DDPM", BetaSchedule.LINEAR, SamplingMethod.DDPM),
        ("Cosine DDIM", BetaSchedule.COSINE, SamplingMethod.DDIM),
        ("Quadratic Ancestral", BetaSchedule.QUADRATIC, SamplingMethod.ANCESTRAL)
    ]
    
    system = AdvancedDiffusionSystem()
    shape = (1, 3, 64, 64)
    
    for name, beta_schedule, sampling_method in configs:
        print(f"\nTesting {name}...")
        
        config = NoiseSchedulerConfig(
            beta_schedule=beta_schedule,
            num_train_timesteps=100
        )
        
        pipeline = system.create_pipeline(name, config, sampling_method)
        
        # Create dummy model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x, t, **kwargs):
                return torch.randn_like(x)
        
        model = DummyModel()
        
        # Benchmark
        start_time = time.time()
        samples = pipeline.sample(
            model=model,
            shape=shape,
            num_inference_steps=20,
            classifier_free_guidance=False
        )
        end_time = time.time()
        
        print(f"  Time: {end_time - start_time:.2f}s")
        print(f"  Shape: {samples.shape}")
        print(f"  Success: ✓")

if __name__ == "__main__":
    # Run unit tests
    print("Running Unit Tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run performance benchmarks
    print("\n" + "="*50)
    run_performance_benchmark()
    
    print("\n" + "="*50)
    print("All tests completed!")


