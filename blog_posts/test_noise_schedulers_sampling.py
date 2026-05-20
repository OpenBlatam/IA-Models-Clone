from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import numpy as np
from typing import Dict, Any, List, Tuple
import tempfile
import os
import warnings
import matplotlib.pyplot as plt
from noise_schedulers_sampling import (
        import shutil
        import shutil
        import shutil
        import shutil
        import shutil
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Comprehensive tests for Noise Schedulers and Sampling Methods
Tests all components including schedulers, sampling methods, analysis tools,
and practical implementations
"""


# Import noise schedulers and sampling methods
    NoiseSchedulerConfig, NoiseSchedulerBase, AdvancedNoiseScheduler,
    AdvancedSamplingMethods, NoiseSchedulerAnalyzer, NoiseSchedulerVisualizer,
    CustomNoiseScheduler
)

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")


class TestNoiseSchedulerConfig:
    """Test NoiseSchedulerConfig dataclass"""
    
    def test_noise_scheduler_config_defaults(self) -> Any:
        """Test default configuration values"""
        config = NoiseSchedulerConfig()
        
        # Basic scheduler parameters
        assert config.num_train_timesteps == 1000
        assert config.beta_start == 0.0001
        assert config.beta_end == 0.02
        assert config.beta_schedule == "linear"
        assert config.prediction_type == "epsilon"
        
        # Scheduler type
        assert config.scheduler_type == "ddpm"
        
        # Sampling parameters
        assert config.num_inference_steps == 50
        assert config.guidance_scale == 7.5
        assert config.eta == 0.0
        assert config.steps_offset == 1
        
        # Advanced sampling parameters
        assert config.use_karras_sigmas is False
        assert config.algorithm_type == "dpmsolver++"
        assert config.solver_type == "midpoint"
        assert config.lower_order_final is True
        assert config.timestep_spacing == "linspace"
        
        # Noise scheduling parameters
        assert config.clip_sample is True
        assert config.clip_sample_range == 1.0
        assert config.sample_max_value == 1.0
        assert config.thresholding is False
        assert config.dynamic_thresholding_ratio == 0.995
        
        # Advanced sampling methods
        assert config.use_classifier_free_guidance is True
        assert config.use_attention_slicing is False
        assert config.use_vae_slicing is False
        assert config.use_memory_efficient_attention is False
        assert config.use_xformers is False
        
        # Custom sampling parameters
        assert config.temperature == 1.0
        assert config.top_k is None
        assert config.top_p is None
        assert config.repetition_penalty == 1.0
        assert config.length_penalty == 1.0
        
        # Noise injection parameters
        assert config.noise_injection_strength == 0.0
        assert config.noise_injection_schedule == "constant"
        
        # Adaptive sampling parameters
        assert config.adaptive_sampling is False
        assert config.adaptive_threshold == 0.1
        assert config.adaptive_window_size == 10
    
    def test_noise_scheduler_config_custom(self) -> Any:
        """Test custom configuration values"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=500,
            beta_start=0.0002,
            beta_end=0.01,
            beta_schedule="cosine",
            prediction_type="v_prediction",
            scheduler_type="ddim",
            num_inference_steps=25,
            guidance_scale=10.0,
            eta=0.5,
            steps_offset=2,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver",
            solver_type="heun",
            lower_order_final=False,
            timestep_spacing="leading",
            clip_sample=False,
            clip_sample_range=2.0,
            sample_max_value=2.0,
            thresholding=True,
            dynamic_thresholding_ratio=0.99,
            use_classifier_free_guidance=False,
            use_attention_slicing=True,
            use_vae_slicing=True,
            use_memory_efficient_attention=True,
            use_xformers=True,
            temperature=0.8,
            top_k=10,
            top_p=0.9,
            repetition_penalty=1.1,
            length_penalty=1.2,
            noise_injection_strength=0.1,
            noise_injection_schedule="linear",
            adaptive_sampling=True,
            adaptive_threshold=0.2,
            adaptive_window_size=15
        )
        
        # Verify custom values
        assert config.num_train_timesteps == 500
        assert config.beta_start == 0.0002
        assert config.beta_end == 0.01
        assert config.beta_schedule == "cosine"
        assert config.prediction_type == "v_prediction"
        assert config.scheduler_type == "ddim"
        assert config.num_inference_steps == 25
        assert config.guidance_scale == 10.0
        assert config.eta == 0.5
        assert config.steps_offset == 2
        assert config.use_karras_sigmas is True
        assert config.algorithm_type == "dpmsolver"
        assert config.solver_type == "heun"
        assert config.lower_order_final is False
        assert config.timestep_spacing == "leading"
        assert config.clip_sample is False
        assert config.clip_sample_range == 2.0
        assert config.sample_max_value == 2.0
        assert config.thresholding is True
        assert config.dynamic_thresholding_ratio == 0.99
        assert config.use_classifier_free_guidance is False
        assert config.use_attention_slicing is True
        assert config.use_vae_slicing is True
        assert config.use_memory_efficient_attention is True
        assert config.use_xformers is True
        assert config.temperature == 0.8
        assert config.top_k == 10
        assert config.top_p == 0.9
        assert config.repetition_penalty == 1.1
        assert config.length_penalty == 1.2
        assert config.noise_injection_strength == 0.1
        assert config.noise_injection_schedule == "linear"
        assert config.adaptive_sampling is True
        assert config.adaptive_threshold == 0.2
        assert config.adaptive_window_size == 15


class TestNoiseSchedulerBase:
    """Test base noise scheduler functionality"""
    
    def test_beta_schedule_linear(self) -> Any:
        """Test linear beta schedule"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        # Create a concrete implementation for testing
        class TestScheduler(NoiseSchedulerBase):
            def step(self, model_output, timestep, sample, **kwargs) -> Any:
                return {'prev_sample': sample}
            
            def add_noise(self, original_samples, timesteps) -> Any:
                return original_samples
            
            def set_timesteps(self, num_inference_steps, device) -> Any:
                pass
        
        scheduler = TestScheduler(config)
        
        assert scheduler.betas.shape == (100,)
        assert torch.allclose(scheduler.betas[0], torch.tensor(0.0001))
        assert torch.allclose(scheduler.betas[-1], torch.tensor(0.02))
        assert torch.all(scheduler.betas >= 0)
        assert torch.all(scheduler.betas <= 1)
    
    def test_beta_schedule_cosine(self) -> Any:
        """Test cosine beta schedule"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="cosine"
        )
        
        class TestScheduler(NoiseSchedulerBase):
            def step(self, model_output, timestep, sample, **kwargs) -> Any:
                return {'prev_sample': sample}
            
            def add_noise(self, original_samples, timesteps) -> Any:
                return original_samples
            
            def set_timesteps(self, num_inference_steps, device) -> Any:
                pass
        
        scheduler = TestScheduler(config)
        
        assert scheduler.betas.shape == (100,)
        assert torch.all(scheduler.betas >= 0)
        assert torch.all(scheduler.betas <= 1)
        
        # Cosine schedule should be different from linear
        linear_config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        linear_scheduler = TestScheduler(linear_config)
        
        assert not torch.allclose(scheduler.betas, linear_scheduler.betas)
    
    def test_beta_schedule_quadratic(self) -> Any:
        """Test quadratic beta schedule"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="quadratic"
        )
        
        class TestScheduler(NoiseSchedulerBase):
            def step(self, model_output, timestep, sample, **kwargs) -> Any:
                return {'prev_sample': sample}
            
            def add_noise(self, original_samples, timesteps) -> Any:
                return original_samples
            
            def set_timesteps(self, num_inference_steps, device) -> Any:
                pass
        
        scheduler = TestScheduler(config)
        
        assert scheduler.betas.shape == (100,)
        assert torch.allclose(scheduler.betas[0], torch.tensor(0.0001))
        assert torch.allclose(scheduler.betas[-1], torch.tensor(0.02))
        assert torch.all(scheduler.betas >= 0)
        assert torch.all(scheduler.betas <= 1)
    
    def test_beta_schedule_sigmoid(self) -> Any:
        """Test sigmoid beta schedule"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="sigmoid"
        )
        
        class TestScheduler(NoiseSchedulerBase):
            def step(self, model_output, timestep, sample, **kwargs) -> Any:
                return {'prev_sample': sample}
            
            def add_noise(self, original_samples, timesteps) -> Any:
                return original_samples
            
            def set_timesteps(self, num_inference_steps, device) -> Any:
                pass
        
        scheduler = TestScheduler(config)
        
        assert scheduler.betas.shape == (100,)
        assert torch.all(scheduler.betas >= 0)
        assert torch.all(scheduler.betas <= 1)
    
    def test_beta_schedule_scaled_linear(self) -> Any:
        """Test scaled linear beta schedule"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="scaled_linear"
        )
        
        class TestScheduler(NoiseSchedulerBase):
            def step(self, model_output, timestep, sample, **kwargs) -> Any:
                return {'prev_sample': sample}
            
            def add_noise(self, original_samples, timesteps) -> Any:
                return original_samples
            
            def set_timesteps(self, num_inference_steps, device) -> Any:
                pass
        
        scheduler = TestScheduler(config)
        
        assert scheduler.betas.shape == (100,)
        assert torch.allclose(scheduler.betas[0], torch.tensor(0.0001))
        assert torch.allclose(scheduler.betas[-1], torch.tensor(0.02))
        assert torch.all(scheduler.betas >= 0)
        assert torch.all(scheduler.betas <= 1)
    
    def test_derived_quantities(self) -> Any:
        """Test derived quantities from beta schedule"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        class TestScheduler(NoiseSchedulerBase):
            def step(self, model_output, timestep, sample, **kwargs) -> Any:
                return {'prev_sample': sample}
            
            def add_noise(self, original_samples, timesteps) -> Any:
                return original_samples
            
            def set_timesteps(self, num_inference_steps, device) -> Any:
                pass
        
        scheduler = TestScheduler(config)
        
        # Test alphas
        assert torch.allclose(scheduler.alphas, 1.0 - scheduler.betas)
        assert torch.all(scheduler.alphas >= 0)
        assert torch.all(scheduler.alphas <= 1)
        
        # Test alphas_cumprod
        assert torch.allclose(scheduler.alphas_cumprod, torch.cumprod(scheduler.alphas, dim=0))
        assert torch.all(scheduler.alphas_cumprod >= 0)
        assert torch.all(scheduler.alphas_cumprod <= 1)
        assert scheduler.alphas_cumprod[0] == scheduler.alphas[0]
        assert scheduler.alphas_cumprod[-1] < scheduler.alphas_cumprod[0]  # Decreasing
        
        # Test sqrt_alphas_cumprod
        assert torch.allclose(scheduler.sqrt_alphas_cumprod, torch.sqrt(scheduler.alphas_cumprod))
        
        # Test sqrt_one_minus_alphas_cumprod
        assert torch.allclose(scheduler.sqrt_one_minus_alphas_cumprod, torch.sqrt(1.0 - scheduler.alphas_cumprod))


class TestAdvancedNoiseScheduler:
    """Test advanced noise scheduler"""
    
    def test_scheduler_initialization(self) -> Any:
        """Test scheduler initialization"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
        scheduler = AdvancedNoiseScheduler(config)
        
        assert scheduler.config == config
        assert scheduler.scheduler is not None
        assert scheduler.timesteps is None
    
    def test_scheduler_add_noise(self) -> Any:
        """Test scheduler add noise"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
        scheduler = AdvancedNoiseScheduler(config)
        
        original_samples = torch.randn(2, 3, 32, 32)
        timesteps = torch.randint(0, 100, (2,))
        
        noisy_samples = scheduler.add_noise(original_samples, timesteps)
        
        assert noisy_samples.shape == original_samples.shape
        assert not torch.isnan(noisy_samples).any()
        assert not torch.allclose(noisy_samples, original_samples)
    
    def test_scheduler_step(self) -> Any:
        """Test scheduler step"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
        scheduler = AdvancedNoiseScheduler(config)
        
        model_output = torch.randn(2, 3, 32, 32)
        timestep = 50
        sample = torch.randn(2, 3, 32, 32)
        
        result = scheduler.step(model_output, timestep, sample)
        
        assert 'prev_sample' in result
        assert result['prev_sample'].shape == sample.shape
        assert not torch.isnan(result['prev_sample']).any()
    
    def test_scheduler_set_timesteps(self) -> Any:
        """Test scheduler set timesteps"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
        scheduler = AdvancedNoiseScheduler(config)
        device = torch.device('cpu')
        
        scheduler.set_timesteps(20, device)
        
        assert hasattr(scheduler.scheduler, 'timesteps')
        assert len(scheduler.scheduler.timesteps) == 20
        assert scheduler.timesteps is not None
    
    def test_scheduler_scale_model_input(self) -> Any:
        """Test scheduler scale model input"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
        scheduler = AdvancedNoiseScheduler(config)
        
        sample = torch.randn(2, 3, 32, 32)
        timestep = 50
        
        scaled_sample = scheduler.scale_model_input(sample, timestep)
        
        assert scaled_sample.shape == sample.shape
        assert not torch.isnan(scaled_sample).any()
    
    def test_different_scheduler_types(self) -> Any:
        """Test different scheduler types"""
        scheduler_types = ["ddpm", "ddim", "pndm", "euler"]
        
        for scheduler_type in scheduler_types:
            config = NoiseSchedulerConfig(
                num_train_timesteps=100,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="linear",
                scheduler_type=scheduler_type
            )
            
            scheduler = AdvancedNoiseScheduler(config)
            
            assert scheduler.scheduler is not None
            assert scheduler.config.scheduler_type == scheduler_type


class TestAdvancedSamplingMethods:
    """Test advanced sampling methods"""
    
    def test_sampling_methods_initialization(self) -> Any:
        """Test sampling methods initialization"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
        sampling_methods = AdvancedSamplingMethods(config)
        
        assert sampling_methods.config == config
        assert sampling_methods.device is not None
    
    def test_standard_sampling(self) -> Any:
        """Test standard sampling"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
        sampling_methods = AdvancedSamplingMethods(config)
        scheduler = AdvancedNoiseScheduler(config)
        
        # Simple model for testing
        class SimpleModel(nn.Module):
            def __init__(self) -> Any:
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)
                self.time_embed = nn.Embedding(100, 3)
                
            def forward(self, x, t) -> Any:
                t_emb = self.time_embed(t).unsqueeze(-1).unsqueeze(-1)
                t_emb = t_emb.expand(-1, -1, x.shape[2], x.shape[3])
                return self.conv(x + t_emb)
        
        model = SimpleModel()
        latents = torch.randn(1, 3, 32, 32)
        
        samples = sampling_methods.standard_sampling(model, latents, scheduler, num_inference_steps=10)
        
        assert samples.shape == latents.shape
        assert not torch.isnan(samples).any()
    
    def test_temperature_sampling(self) -> Any:
        """Test temperature sampling"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
        sampling_methods = AdvancedSamplingMethods(config)
        scheduler = AdvancedNoiseScheduler(config)
        
        class SimpleModel(nn.Module):
            def __init__(self) -> Any:
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)
                self.time_embed = nn.Embedding(100, 3)
                
            def forward(self, x, t) -> Any:
                t_emb = self.time_embed(t).unsqueeze(-1).unsqueeze(-1)
                t_emb = t_emb.expand(-1, -1, x.shape[2], x.shape[3])
                return self.conv(x + t_emb)
        
        model = SimpleModel()
        latents = torch.randn(1, 3, 32, 32)
        
        samples = sampling_methods.temperature_sampling(model, latents, scheduler, temperature=0.8)
        
        assert samples.shape == latents.shape
        assert not torch.isnan(samples).any()
    
    def test_noise_injection_sampling(self) -> Any:
        """Test noise injection sampling"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
        sampling_methods = AdvancedSamplingMethods(config)
        scheduler = AdvancedNoiseScheduler(config)
        
        class SimpleModel(nn.Module):
            def __init__(self) -> Any:
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)
                self.time_embed = nn.Embedding(100, 3)
                
            def forward(self, x, t) -> Any:
                t_emb = self.time_embed(t).unsqueeze(-1).unsqueeze(-1)
                t_emb = t_emb.expand(-1, -1, x.shape[2], x.shape[3])
                return self.conv(x + t_emb)
        
        model = SimpleModel()
        latents = torch.randn(1, 3, 32, 32)
        
        samples = sampling_methods.noise_injection_sampling(model, latents, scheduler, injection_strength=0.1)
        
        assert samples.shape == latents.shape
        assert not torch.isnan(samples).any()
    
    def test_classifier_free_guidance_sampling(self) -> Any:
        """Test classifier-free guidance sampling"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
        sampling_methods = AdvancedSamplingMethods(config)
        scheduler = AdvancedNoiseScheduler(config)
        
        class SimpleModel(nn.Module):
            def __init__(self) -> Any:
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)
                self.time_embed = nn.Embedding(100, 3)
                
            def forward(self, x, t, embeddings=None) -> Any:
                t_emb = self.time_embed(t).unsqueeze(-1).unsqueeze(-1)
                t_emb = t_emb.expand(-1, -1, x.shape[2], x.shape[3])
                return self.conv(x + t_emb)
        
        model = SimpleModel()
        latents = torch.randn(1, 3, 32, 32)
        prompt_embeds = torch.randn(1, 77, 768)
        uncond_embeds = torch.randn(1, 77, 768)
        
        samples = sampling_methods.classifier_free_guidance_sampling(
            model, prompt_embeds, uncond_embeds, latents, scheduler, num_inference_steps=10
        )
        
        assert samples.shape == latents.shape
        assert not torch.isnan(samples).any()


class TestNoiseSchedulerAnalyzer:
    """Test noise scheduler analyzer"""
    
    def test_analyzer_initialization(self) -> Any:
        """Test analyzer initialization"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        analyzer = NoiseSchedulerAnalyzer(config)
        
        assert analyzer.config == config
    
    def test_analyze_scheduler_properties(self) -> Any:
        """Test scheduler properties analysis"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
        analyzer = NoiseSchedulerAnalyzer(config)
        scheduler = AdvancedNoiseScheduler(config)
        
        properties = analyzer.analyze_scheduler_properties(scheduler)
        
        assert 'timesteps' in properties
        assert 'num_timesteps' in properties
        assert 'scheduler_type' in properties
        assert 'beta_schedule' in properties
        assert 'prediction_type' in properties
        
        assert properties['scheduler_type'] == "ddpm"
        assert properties['beta_schedule'] == "linear"
        assert properties['prediction_type'] == "epsilon"
    
    def test_compare_schedulers(self) -> Any:
        """Test scheduler comparison"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        analyzer = NoiseSchedulerAnalyzer(config)
        scheduler_types = ["ddpm", "ddim", "pndm"]
        
        comparison = analyzer.compare_schedulers(scheduler_types)
        
        assert "ddpm" in comparison
        assert "ddim" in comparison
        assert "pndm" in comparison
        assert len(comparison) == 3
        
        for scheduler_type in scheduler_types:
            assert comparison[scheduler_type]['scheduler_type'] == scheduler_type
    
    def test_analyze_sampling_efficiency(self) -> Any:
        """Test sampling efficiency analysis"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
        analyzer = NoiseSchedulerAnalyzer(config)
        scheduler = AdvancedNoiseScheduler(config)
        
        class SimpleModel(nn.Module):
            def __init__(self) -> Any:
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)
                self.time_embed = nn.Embedding(100, 3)
                
            def forward(self, x, t) -> Any:
                t_emb = self.time_embed(t).unsqueeze(-1).unsqueeze(-1)
                t_emb = t_emb.expand(-1, -1, x.shape[2], x.shape[3])
                return self.conv(x + t_emb)
        
        model = SimpleModel()
        test_latents = torch.randn(1, 3, 32, 32)
        num_steps_list = [10, 20, 30]
        
        results = analyzer.analyze_sampling_efficiency(scheduler, model, test_latents, num_steps_list)
        
        assert 'num_steps' in results
        assert 'sampling_times' in results
        assert 'memory_usage' in results
        assert 'quality_scores' in results
        
        assert len(results['num_steps']) == 3
        assert len(results['sampling_times']) == 3
        assert len(results['memory_usage']) == 3
        assert len(results['quality_scores']) == 3


class TestNoiseSchedulerVisualizer:
    """Test noise scheduler visualizer"""
    
    def test_visualizer_initialization(self) -> Any:
        """Test visualizer initialization"""
        config = NoiseSchedulerConfig(
            save_path="test_visualizations"
        )
        
        visualizer = NoiseSchedulerVisualizer(config)
        
        assert visualizer.config == config
        assert visualizer.save_path == "test_visualizations"
        assert os.path.exists("test_visualizations")
        
        # Cleanup
        shutil.rmtree("test_visualizations")
    
    def test_visualize_beta_schedules(self) -> Any:
        """Test beta schedules visualization"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            save_path="test_visualizations"
        )
        
        visualizer = NoiseSchedulerVisualizer(config)
        scheduler_types = ["ddpm", "ddim", "pndm", "euler"]
        
        visualizer.visualize_beta_schedules(scheduler_types)
        
        assert os.path.exists(os.path.join("test_visualizations", "beta_schedules.png"))
        
        # Cleanup
        shutil.rmtree("test_visualizations")
    
    def test_visualize_alphas_cumprod(self) -> Any:
        """Test alphas cumprod visualization"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            save_path="test_visualizations"
        )
        
        visualizer = NoiseSchedulerVisualizer(config)
        scheduler_types = ["ddpm", "ddim", "pndm", "euler"]
        
        visualizer.visualize_alphas_cumprod(scheduler_types)
        
        assert os.path.exists(os.path.join("test_visualizations", "alphas_cumprod.png"))
        
        # Cleanup
        shutil.rmtree("test_visualizations")
    
    def test_visualize_sampling_comparison(self) -> Any:
        """Test sampling comparison visualization"""
        config = NoiseSchedulerConfig(
            save_path="test_visualizations"
        )
        
        visualizer = NoiseSchedulerVisualizer(config)
        
        results = {
            'num_steps': [10, 20, 30],
            'sampling_times': [100, 200, 300],
            'memory_usage': [1.0, 2.0, 3.0],
            'quality_scores': [0.8, 0.9, 0.95]
        }
        
        visualizer.visualize_sampling_comparison(results)
        
        assert os.path.exists(os.path.join("test_visualizations", "sampling_comparison.png"))
        
        # Cleanup
        shutil.rmtree("test_visualizations")


class TestCustomNoiseScheduler:
    """Test custom noise scheduler"""
    
    def test_custom_scheduler_initialization(self) -> Any:
        """Test custom scheduler initialization"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            temperature=0.8,
            noise_injection_strength=0.1
        )
        
        scheduler = CustomNoiseScheduler(config)
        
        assert scheduler.config == config
        assert scheduler.temperature == 0.8
        assert scheduler.noise_injection_strength == 0.1
        assert scheduler.betas.shape == (100,)
    
    def test_custom_scheduler_step(self) -> Any:
        """Test custom scheduler step"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            temperature=0.8,
            noise_injection_strength=0.1
        )
        
        scheduler = CustomNoiseScheduler(config)
        
        model_output = torch.randn(2, 3, 32, 32)
        timestep = 50
        sample = torch.randn(2, 3, 32, 32)
        
        result = scheduler.step(model_output, timestep, sample)
        
        assert 'prev_sample' in result
        assert result['prev_sample'].shape == sample.shape
        assert not torch.isnan(result['prev_sample']).any()
    
    def test_custom_scheduler_add_noise(self) -> Any:
        """Test custom scheduler add noise"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        scheduler = CustomNoiseScheduler(config)
        
        original_samples = torch.randn(2, 3, 32, 32)
        timesteps = torch.tensor([10, 20])
        
        noisy_samples = scheduler.add_noise(original_samples, timesteps)
        
        assert noisy_samples.shape == original_samples.shape
        assert not torch.isnan(noisy_samples).any()
        assert not torch.allclose(noisy_samples, original_samples)
    
    def test_custom_scheduler_set_timesteps(self) -> Any:
        """Test custom scheduler set timesteps"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        scheduler = CustomNoiseScheduler(config)
        device = torch.device('cpu')
        
        scheduler.set_timesteps(20, device)
        
        assert scheduler.timesteps is not None
        assert len(scheduler.timesteps) == 20
        assert scheduler.timesteps.device == device
    
    def test_custom_scheduler_scale_model_input(self) -> Any:
        """Test custom scheduler scale model input"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        scheduler = CustomNoiseScheduler(config)
        
        sample = torch.randn(2, 3, 32, 32)
        timestep = 50
        
        scaled_sample = scheduler.scale_model_input(sample, timestep)
        
        assert scaled_sample.shape == sample.shape
        assert torch.allclose(scaled_sample, sample)  # Custom scheduler doesn't scale


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_sampling(self) -> Any:
        """Test end-to-end sampling pipeline"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
        scheduler = AdvancedNoiseScheduler(config)
        sampling_methods = AdvancedSamplingMethods(config)
        
        class SimpleModel(nn.Module):
            def __init__(self) -> Any:
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)
                self.time_embed = nn.Embedding(100, 3)
                
            def forward(self, x, t) -> Any:
                t_emb = self.time_embed(t).unsqueeze(-1).unsqueeze(-1)
                t_emb = t_emb.expand(-1, -1, x.shape[2], x.shape[3])
                return self.conv(x + t_emb)
        
        model = SimpleModel()
        latents = torch.randn(1, 3, 32, 32)
        
        # Test different sampling methods
        samples_standard = sampling_methods.standard_sampling(model, latents, scheduler)
        samples_temp = sampling_methods.temperature_sampling(model, latents, scheduler, temperature=0.8)
        samples_noise = sampling_methods.noise_injection_sampling(model, latents, scheduler, injection_strength=0.1)
        
        assert samples_standard.shape == latents.shape
        assert samples_temp.shape == latents.shape
        assert samples_noise.shape == latents.shape
        
        assert not torch.isnan(samples_standard).any()
        assert not torch.isnan(samples_temp).any()
        assert not torch.isnan(samples_noise).any()
    
    def test_scheduler_comparison_pipeline(self) -> Any:
        """Test scheduler comparison pipeline"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        analyzer = NoiseSchedulerAnalyzer(config)
        visualizer = NoiseSchedulerVisualizer(config)
        
        # Compare schedulers
        scheduler_types = ["ddpm", "ddim", "pndm"]
        comparison = analyzer.compare_schedulers(scheduler_types)
        
        assert len(comparison) == 3
        for scheduler_type in scheduler_types:
            assert scheduler_type in comparison
        
        # Visualize
        visualizer.visualize_beta_schedules(scheduler_types)
        visualizer.visualize_alphas_cumprod(scheduler_types)
        
        assert os.path.exists(os.path.join(config.save_path, "beta_schedules.png"))
        assert os.path.exists(os.path.join(config.save_path, "alphas_cumprod.png"))
        
        # Cleanup
        shutil.rmtree(config.save_path)


class TestMathematicalCorrectness:
    """Test mathematical correctness of noise schedulers"""
    
    def test_beta_schedule_properties(self) -> Any:
        """Test properties of beta schedule"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        scheduler = CustomNoiseScheduler(config)
        
        # Betas should be in [0, 1]
        assert torch.all(scheduler.betas >= 0)
        assert torch.all(scheduler.betas <= 1)
        
        # Betas should be monotonically increasing
        assert torch.all(scheduler.betas[1:] >= scheduler.betas[:-1])
        
        # Alphas should be in [0, 1]
        assert torch.all(scheduler.alphas >= 0)
        assert torch.all(scheduler.alphas <= 1)
        
        # Alphas should be monotonically decreasing
        assert torch.all(scheduler.alphas[1:] <= scheduler.alphas[:-1])
        
        # Alphas cumprod should be monotonically decreasing
        assert torch.all(scheduler.alphas_cumprod[1:] <= scheduler.alphas_cumprod[:-1])
    
    def test_noise_addition_equation(self) -> Any:
        """Test that noise addition follows the correct equation"""
        config = NoiseSchedulerConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        scheduler = CustomNoiseScheduler(config)
        
        original_samples = torch.randn(2, 3, 32, 32)
        timesteps = torch.tensor([10, 20])
        
        noisy_samples = scheduler.add_noise(original_samples, timesteps)
        
        # Verify equation: x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε
        sqrt_alphas_cumprod_t = scheduler.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = scheduler.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        
        # Reconstruct noise
        noise = (noisy_samples - sqrt_alphas_cumprod_t * original_samples) / sqrt_one_minus_alphas_cumprod_t
        
        # Verify reconstruction
        expected_noisy_samples = sqrt_alphas_cumprod_t * original_samples + sqrt_one_minus_alphas_cumprod_t * noise
        
        assert torch.allclose(noisy_samples, expected_noisy_samples, atol=1e-6)


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"]) 