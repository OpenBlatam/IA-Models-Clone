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
from diffusion_processes import (
        import shutil
        import shutil
        import shutil
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Comprehensive tests for Forward and Reverse Diffusion Processes
Tests all components including mathematical correctness, forward/reverse processes,
schedulers, and practical implementations
"""


# Import diffusion processes
    DiffusionProcessConfig, DiffusionProcessBase, DDPMProcess, DDIMProcess,
    DiffusionProcessTrainer, DiffusionProcessVisualizer, AdvancedDiffusionScheduler,
    DiffusionProcessAnalyzer
)

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")


class TestDiffusionProcessConfig:
    """Test DiffusionProcessConfig dataclass"""
    
    def test_diffusion_config_defaults(self) -> Any:
        """Test default configuration values"""
        config = DiffusionProcessConfig()
        
        # Process parameters
        assert config.num_train_timesteps == 1000
        assert config.beta_start == 0.0001
        assert config.beta_end == 0.02
        assert config.beta_schedule == "linear"
        assert config.prediction_type == "epsilon"
        
        # Scheduler specific
        assert config.scheduler_type == "ddpm"
        assert config.clip_sample is True
        assert config.clip_sample_range == 1.0
        assert config.sample_max_value == 1.0
        assert config.thresholding is False
        assert config.dynamic_thresholding_ratio == 0.995
        
        # DDIM specific
        assert config.eta == 0.0
        assert config.steps_offset == 1
        
        # DPM-Solver specific
        assert config.algorithm_type == "dpmsolver++"
        assert config.solver_type == "midpoint"
        assert config.lower_order_final is True
        assert config.use_karras_sigmas is False
        assert config.timestep_spacing == "linspace"
        
        # Training
        assert config.loss_type == "l2"
        assert config.snr_gamma is None
        assert config.v_prediction is False
        
        # Visualization
        assert config.save_intermediate is False
        assert config.save_path == "diffusion_processes"
    
    def test_diffusion_config_custom(self) -> Any:
        """Test custom configuration values"""
        config = DiffusionProcessConfig(
            num_train_timesteps=500,
            beta_start=0.0002,
            beta_end=0.01,
            beta_schedule="cosine",
            prediction_type="v_prediction",
            scheduler_type="ddim",
            clip_sample=False,
            clip_sample_range=2.0,
            sample_max_value=2.0,
            thresholding=True,
            dynamic_thresholding_ratio=0.99,
            eta=0.5,
            steps_offset=2,
            algorithm_type="dpmsolver",
            solver_type="heun",
            lower_order_final=False,
            use_karras_sigmas=True,
            timestep_spacing="leading",
            loss_type="l1",
            snr_gamma=5.0,
            v_prediction=True,
            save_intermediate=True,
            save_path="custom_diffusion"
        )
        
        # Verify custom values
        assert config.num_train_timesteps == 500
        assert config.beta_start == 0.0002
        assert config.beta_end == 0.01
        assert config.beta_schedule == "cosine"
        assert config.prediction_type == "v_prediction"
        assert config.scheduler_type == "ddim"
        assert config.clip_sample is False
        assert config.clip_sample_range == 2.0
        assert config.sample_max_value == 2.0
        assert config.thresholding is True
        assert config.dynamic_thresholding_ratio == 0.99
        assert config.eta == 0.5
        assert config.steps_offset == 2
        assert config.algorithm_type == "dpmsolver"
        assert config.solver_type == "heun"
        assert config.lower_order_final is False
        assert config.use_karras_sigmas is True
        assert config.timestep_spacing == "leading"
        assert config.loss_type == "l1"
        assert config.snr_gamma == 5.0
        assert config.v_prediction is True
        assert config.save_intermediate is True
        assert config.save_path == "custom_diffusion"


class TestDiffusionProcessBase:
    """Test base diffusion process functionality"""
    
    def test_beta_schedule_linear(self) -> Any:
        """Test linear beta schedule"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        process = DDPMProcess(config)
        
        assert process.betas.shape == (100,)
        assert torch.allclose(process.betas[0], torch.tensor(0.0001))
        assert torch.allclose(process.betas[-1], torch.tensor(0.02))
        assert torch.all(process.betas >= 0)
        assert torch.all(process.betas <= 1)
    
    def test_beta_schedule_cosine(self) -> Any:
        """Test cosine beta schedule"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="cosine"
        )
        
        process = DDPMProcess(config)
        
        assert process.betas.shape == (100,)
        assert torch.all(process.betas >= 0)
        assert torch.all(process.betas <= 1)
        
        # Cosine schedule should be different from linear
        linear_config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        linear_process = DDPMProcess(linear_config)
        
        assert not torch.allclose(process.betas, linear_process.betas)
    
    def test_beta_schedule_quadratic(self) -> Any:
        """Test quadratic beta schedule"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="quadratic"
        )
        
        process = DDPMProcess(config)
        
        assert process.betas.shape == (100,)
        assert torch.allclose(process.betas[0], torch.tensor(0.0001))
        assert torch.allclose(process.betas[-1], torch.tensor(0.02))
        assert torch.all(process.betas >= 0)
        assert torch.all(process.betas <= 1)
    
    def test_beta_schedule_sigmoid(self) -> Any:
        """Test sigmoid beta schedule"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="sigmoid"
        )
        
        process = DDPMProcess(config)
        
        assert process.betas.shape == (100,)
        assert torch.all(process.betas >= 0)
        assert torch.all(process.betas <= 1)
    
    def test_derived_quantities(self) -> Any:
        """Test derived quantities from beta schedule"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        process = DDPMProcess(config)
        
        # Test alphas
        assert torch.allclose(process.alphas, 1.0 - process.betas)
        assert torch.all(process.alphas >= 0)
        assert torch.all(process.alphas <= 1)
        
        # Test alphas_cumprod
        assert torch.allclose(process.alphas_cumprod, torch.cumprod(process.alphas, dim=0))
        assert torch.all(process.alphas_cumprod >= 0)
        assert torch.all(process.alphas_cumprod <= 1)
        assert process.alphas_cumprod[0] == process.alphas[0]
        assert process.alphas_cumprod[-1] < process.alphas_cumprod[0]  # Decreasing
        
        # Test sqrt_alphas_cumprod
        assert torch.allclose(process.sqrt_alphas_cumprod, torch.sqrt(process.alphas_cumprod))
        
        # Test sqrt_one_minus_alphas_cumprod
        assert torch.allclose(process.sqrt_one_minus_alphas_cumprod, torch.sqrt(1.0 - process.alphas_cumprod))
    
    def test_posterior_quantities(self) -> Any:
        """Test posterior variance and mean coefficients"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        process = DDPMProcess(config)
        
        # Test posterior variance
        assert process.posterior_variance.shape == (100,)
        assert torch.all(process.posterior_variance >= 0)
        
        # Test posterior mean coefficients
        assert process.posterior_mean_coef1.shape == (100,)
        assert process.posterior_mean_coef2.shape == (100,)


class TestDDPMProcess:
    """Test DDPM diffusion process"""
    
    def test_ddpm_initialization(self) -> Any:
        """Test DDPM process initialization"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        process = DDPMProcess(config)
        
        assert process.config == config
        assert process.betas.shape == (100,)
        assert process.alphas_cumprod.shape == (100,)
    
    def test_forward_process_shape(self) -> Any:
        """Test forward process output shapes"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        process = DDPMProcess(config)
        
        batch_size = 4
        channels = 3
        height = 32
        width = 32
        
        x_0 = torch.randn(batch_size, channels, height, width)
        t = torch.randint(0, 100, (batch_size,))
        
        x_t, noise = process.forward_process(x_0, t)
        
        assert x_t.shape == x_0.shape
        assert noise.shape == x_0.shape
        assert not torch.isnan(x_t).any()
        assert not torch.isnan(noise).any()
    
    def test_forward_process_properties(self) -> Any:
        """Test forward process mathematical properties"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        process = DDPMProcess(config)
        
        batch_size = 2
        channels = 3
        height = 16
        width = 16
        
        x_0 = torch.randn(batch_size, channels, height, width)
        t = torch.tensor([0, 99])  # Test first and last timestep
        
        x_t, noise = process.forward_process(x_0, t)
        
        # At t=0, x_t should be close to x_0
        assert torch.allclose(x_t[0], x_0[0], atol=1e-5)
        
        # At t=99, x_t should be very noisy
        assert not torch.allclose(x_t[1], x_0[1], atol=1e-1)
    
    def test_reverse_process_shape(self) -> Any:
        """Test reverse process output shapes"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        
        process = DDPMProcess(config)
        
        batch_size = 4
        channels = 3
        height = 32
        width = 32
        
        x_t = torch.randn(batch_size, channels, height, width)
        t = torch.randint(1, 100, (batch_size,))  # Start from 1 for reverse process
        predicted_noise = torch.randn(batch_size, channels, height, width)
        
        x_prev = process.reverse_process(x_t, t, predicted_noise)
        
        assert x_prev.shape == x_t.shape
        assert not torch.isnan(x_prev).any()
    
    def test_reverse_process_epsilon_prediction(self) -> Any:
        """Test reverse process with epsilon prediction"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        
        process = DDPMProcess(config)
        
        batch_size = 2
        channels = 3
        height = 16
        width = 16
        
        x_t = torch.randn(batch_size, channels, height, width)
        t = torch.tensor([50, 75])
        predicted_noise = torch.randn(batch_size, channels, height, width)
        
        x_prev = process.reverse_process(x_t, t, predicted_noise)
        
        assert x_prev.shape == x_t.shape
        assert not torch.isnan(x_prev).any()
    
    def test_reverse_process_v_prediction(self) -> Any:
        """Test reverse process with v prediction"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="v_prediction"
        )
        
        process = DDPMProcess(config)
        
        batch_size = 2
        channels = 3
        height = 16
        width = 16
        
        x_t = torch.randn(batch_size, channels, height, width)
        t = torch.tensor([50, 75])
        predicted_noise = torch.randn(batch_size, channels, height, width)
        
        x_prev = process.reverse_process(x_t, t, predicted_noise)
        
        assert x_prev.shape == x_t.shape
        assert not torch.isnan(x_prev).any()
    
    def test_extract_into_tensor(self) -> Any:
        """Test extract_into_tensor utility function"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        process = DDPMProcess(config)
        
        a = torch.randn(100, 5)
        t = torch.tensor([10, 20, 30])
        x_shape = (3, 3, 32, 32)
        
        result = process.extract_into_tensor(a, t, x_shape)
        
        assert result.shape == (3, 1, 1, 1)
        assert torch.allclose(result[0], a[10])
        assert torch.allclose(result[1], a[20])
        assert torch.allclose(result[2], a[30])


class TestDDIMProcess:
    """Test DDIM diffusion process"""
    
    def test_ddim_initialization(self) -> Any:
        """Test DDIM process initialization"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddim",
            eta=0.0
        )
        
        process = DDIMProcess(config)
        
        assert process.config == config
        assert process.eta == 0.0
        assert process.betas.shape == (100,)
    
    def test_ddim_forward_process(self) -> Any:
        """Test DDIM forward process (same as DDPM)"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddim"
        )
        
        process = DDIMProcess(config)
        
        batch_size = 4
        channels = 3
        height = 32
        width = 32
        
        x_0 = torch.randn(batch_size, channels, height, width)
        t = torch.randint(0, 100, (batch_size,))
        
        x_t, noise = process.forward_process(x_0, t)
        
        assert x_t.shape == x_0.shape
        assert noise.shape == x_0.shape
        assert not torch.isnan(x_t).any()
        assert not torch.isnan(noise).any()
    
    def test_ddim_reverse_process_deterministic(self) -> Any:
        """Test DDIM reverse process with eta=0 (deterministic)"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddim",
            eta=0.0,
            prediction_type="epsilon"
        )
        
        process = DDIMProcess(config)
        
        batch_size = 2
        channels = 3
        height = 16
        width = 16
        
        x_t = torch.randn(batch_size, channels, height, width)
        t = torch.tensor([50, 75])
        predicted_noise = torch.randn(batch_size, channels, height, width)
        
        x_prev = process.reverse_process(x_t, t, predicted_noise)
        
        assert x_prev.shape == x_t.shape
        assert not torch.isnan(x_prev).any()
    
    def test_ddim_reverse_process_stochastic(self) -> Any:
        """Test DDIM reverse process with eta=1.0 (stochastic)"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddim",
            eta=1.0,
            prediction_type="epsilon"
        )
        
        process = DDIMProcess(config)
        
        batch_size = 2
        channels = 3
        height = 16
        width = 16
        
        x_t = torch.randn(batch_size, channels, height, width)
        t = torch.tensor([50, 75])
        predicted_noise = torch.randn(batch_size, channels, height, width)
        
        x_prev = process.reverse_process(x_t, t, predicted_noise)
        
        assert x_prev.shape == x_t.shape
        assert not torch.isnan(x_prev).any()


class TestDiffusionProcessTrainer:
    """Test diffusion process trainer"""
    
    def test_trainer_initialization(self) -> Any:
        """Test trainer initialization"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
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
        trainer = DiffusionProcessTrainer(model, config)
        
        assert trainer.model == model
        assert trainer.config == config
        assert isinstance(trainer.diffusion_process, DDPMProcess)
        assert isinstance(trainer.optimizer, optim.AdamW)
    
    def test_trainer_with_ddim(self) -> Any:
        """Test trainer with DDIM process"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddim"
        )
        
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
        trainer = DiffusionProcessTrainer(model, config)
        
        assert isinstance(trainer.diffusion_process, DDIMProcess)
    
    def test_train_step(self) -> Any:
        """Test training step"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
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
        trainer = DiffusionProcessTrainer(model, config)
        
        batch = {
            'images': torch.randn(4, 3, 32, 32)
        }
        
        metrics = trainer.train_step(batch)
        
        assert 'loss' in metrics
        assert metrics['loss'] > 0
        assert not np.isnan(metrics['loss'])
    
    def test_sample(self) -> Any:
        """Test sampling from trained model"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
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
        trainer = DiffusionProcessTrainer(model, config)
        
        shape = (2, 3, 32, 32)
        samples = trainer.sample(shape, num_steps=10)
        
        assert samples.shape == shape
        assert not torch.isnan(samples).any()


class TestAdvancedDiffusionScheduler:
    """Test advanced diffusion scheduler"""
    
    def test_scheduler_initialization(self) -> Any:
        """Test scheduler initialization"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
        scheduler = AdvancedDiffusionScheduler(config)
        
        assert scheduler.config == config
        assert scheduler.scheduler is not None
    
    def test_scheduler_add_noise(self) -> Any:
        """Test scheduler add noise"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
        scheduler = AdvancedDiffusionScheduler(config)
        
        original_samples = torch.randn(2, 3, 32, 32)
        timesteps = torch.randint(0, 100, (2,))
        
        noisy_samples = scheduler.add_noise(original_samples, timesteps)
        
        assert noisy_samples.shape == original_samples.shape
        assert not torch.isnan(noisy_samples).any()
        assert not torch.allclose(noisy_samples, original_samples)
    
    def test_scheduler_step(self) -> Any:
        """Test scheduler step"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
        scheduler = AdvancedDiffusionScheduler(config)
        
        model_output = torch.randn(2, 3, 32, 32)
        timestep = 50
        sample = torch.randn(2, 3, 32, 32)
        
        result = scheduler.step(model_output, timestep, sample)
        
        assert 'prev_sample' in result
        assert result['prev_sample'].shape == sample.shape
        assert not torch.isnan(result['prev_sample']).any()
    
    def test_scheduler_set_timesteps(self) -> Any:
        """Test scheduler set timesteps"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
        scheduler = AdvancedDiffusionScheduler(config)
        device = torch.device('cpu')
        
        scheduler.set_timesteps(20, device)
        
        assert hasattr(scheduler.scheduler, 'timesteps')
        assert len(scheduler.scheduler.timesteps) == 20
    
    def test_scheduler_scale_model_input(self) -> Any:
        """Test scheduler scale model input"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
        scheduler = AdvancedDiffusionScheduler(config)
        
        sample = torch.randn(2, 3, 32, 32)
        timestep = 50
        
        scaled_sample = scheduler.scale_model_input(sample, timestep)
        
        assert scaled_sample.shape == sample.shape
        assert not torch.isnan(scaled_sample).any()


class TestDiffusionProcessAnalyzer:
    """Test diffusion process analyzer"""
    
    def test_analyzer_initialization(self) -> Any:
        """Test analyzer initialization"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        analyzer = DiffusionProcessAnalyzer(config)
        
        assert analyzer.config == config
    
    def test_analyze_noise_schedule(self) -> Any:
        """Test noise schedule analysis"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        analyzer = DiffusionProcessAnalyzer(config)
        analysis = analyzer.analyze_noise_schedule()
        
        assert 'snr' in analysis
        assert 'noise_level' in analysis
        assert 'info_content' in analysis
        assert 'alphas_cumprod' in analysis
        assert 'betas' in analysis
        
        assert analysis['snr'].shape == (100,)
        assert analysis['noise_level'].shape == (100,)
        assert analysis['info_content'].shape == (100,)
        assert analysis['alphas_cumprod'].shape == (100,)
        assert analysis['betas'].shape == (100,)
        
        # SNR should be decreasing
        assert analysis['snr'][0] > analysis['snr'][-1]
        
        # Noise level should be increasing
        assert analysis['noise_level'][0] < analysis['noise_level'][-1]
    
    def test_compare_schedulers(self) -> Any:
        """Test scheduler comparison"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        analyzer = DiffusionProcessAnalyzer(config)
        schedulers = analyzer.compare_schedulers(["ddpm", "ddim"])
        
        assert "ddpm" in schedulers
        assert "ddim" in schedulers
        assert len(schedulers) == 2


class TestDiffusionProcessVisualizer:
    """Test diffusion process visualizer"""
    
    def test_visualizer_initialization(self) -> Any:
        """Test visualizer initialization"""
        config = DiffusionProcessConfig(
            save_path="test_diffusion"
        )
        
        visualizer = DiffusionProcessVisualizer(config)
        
        assert visualizer.config == config
        assert visualizer.save_path == "test_diffusion"
        assert os.path.exists("test_diffusion")
        
        # Cleanup
        shutil.rmtree("test_diffusion")
    
    def test_visualize_beta_schedule(self) -> Any:
        """Test beta schedule visualization"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            save_path="test_diffusion"
        )
        
        visualizer = DiffusionProcessVisualizer(config)
        visualizer.visualize_beta_schedule()
        
        assert os.path.exists(os.path.join("test_diffusion", "beta_schedule.png"))
        
        # Cleanup
        shutil.rmtree("test_diffusion")
    
    def test_visualize_alphas_cumprod(self) -> Any:
        """Test alphas cumprod visualization"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            save_path="test_diffusion"
        )
        
        visualizer = DiffusionProcessVisualizer(config)
        visualizer.visualize_alphas_cumprod()
        
        assert os.path.exists(os.path.join("test_diffusion", "alphas_cumprod.png"))
        
        # Cleanup
        shutil.rmtree("test_diffusion")


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_ddpm(self) -> Any:
        """Test end-to-end DDPM process"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddpm"
        )
        
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
        trainer = DiffusionProcessTrainer(model, config)
        
        # Training loop
        for epoch in range(3):
            batch = {'images': torch.randn(4, 3, 32, 32)}
            metrics = trainer.train_step(batch)
            
            assert metrics['loss'] > 0
            assert not np.isnan(metrics['loss'])
        
        # Sampling
        samples = trainer.sample((2, 3, 32, 32), num_steps=10)
        assert samples.shape == (2, 3, 32, 32)
        assert not torch.isnan(samples).any()
    
    def test_end_to_end_ddim(self) -> Any:
        """Test end-to-end DDIM process"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            scheduler_type="ddim",
            eta=0.0
        )
        
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
        trainer = DiffusionProcessTrainer(model, config)
        
        # Training loop
        for epoch in range(3):
            batch = {'images': torch.randn(4, 3, 32, 32)}
            metrics = trainer.train_step(batch)
            
            assert metrics['loss'] > 0
            assert not np.isnan(metrics['loss'])
        
        # Sampling
        samples = trainer.sample((2, 3, 32, 32), num_steps=10)
        assert samples.shape == (2, 3, 32, 32)
        assert not torch.isnan(samples).any()


class TestMathematicalCorrectness:
    """Test mathematical correctness of diffusion processes"""
    
    def test_forward_process_equation(self) -> Any:
        """Test that forward process follows the correct equation"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        process = DDPMProcess(config)
        
        batch_size = 2
        channels = 3
        height = 16
        width = 16
        
        x_0 = torch.randn(batch_size, channels, height, width)
        t = torch.tensor([50, 75])
        
        x_t, noise = process.forward_process(x_0, t)
        
        # Verify equation: x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε
        sqrt_alphas_cumprod_t = process.extract_into_tensor(process.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = process.extract_into_tensor(
            process.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )
        
        expected_x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        assert torch.allclose(x_t, expected_x_t, atol=1e-6)
    
    def test_reverse_process_equation(self) -> Any:
        """Test that reverse process follows the correct equation"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        
        process = DDPMProcess(config)
        
        batch_size = 2
        channels = 3
        height = 16
        width = 16
        
        x_t = torch.randn(batch_size, channels, height, width)
        t = torch.tensor([50, 75])
        predicted_noise = torch.randn(batch_size, channels, height, width)
        
        x_prev = process.reverse_process(x_t, t, predicted_noise)
        
        # Verify that x_prev has the correct shape and properties
        assert x_prev.shape == x_t.shape
        assert not torch.isnan(x_prev).any()
    
    def test_beta_schedule_properties(self) -> Any:
        """Test properties of beta schedule"""
        config = DiffusionProcessConfig(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
        
        process = DDPMProcess(config)
        
        # Betas should be in [0, 1]
        assert torch.all(process.betas >= 0)
        assert torch.all(process.betas <= 1)
        
        # Betas should be monotonically increasing
        assert torch.all(process.betas[1:] >= process.betas[:-1])
        
        # Alphas should be in [0, 1]
        assert torch.all(process.alphas >= 0)
        assert torch.all(process.alphas <= 1)
        
        # Alphas should be monotonically decreasing
        assert torch.all(process.alphas[1:] <= process.alphas[:-1])
        
        # Alphas cumprod should be monotonically decreasing
        assert torch.all(process.alphas_cumprod[1:] <= process.alphas_cumprod[:-1])


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"]) 