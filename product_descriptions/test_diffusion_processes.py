from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import tempfile
import shutil
import time
from unittest.mock import Mock, patch, AsyncMock
from diffusion_processes import (
from typing import Any, List, Dict, Optional
import logging
"""
Comprehensive Tests for Forward and Reverse Diffusion Processes
==============================================================

This module provides extensive testing for the diffusion processes implementation,
covering mathematical foundations, forward and reverse processes, custom schedules,
and security applications.

Test Coverage:
- Mathematical foundations and noise schedules
- Forward diffusion process
- Reverse diffusion process
- Custom noise schedules
- Step-by-step processes
- Security applications
- Performance analysis
- Error handling and edge cases

Author: AI Assistant
License: MIT
"""


# Import our diffusion processes
    DiffusionProcesses, SecurityDiffusionProcesses, DiffusionConfig,
    DiffusionSchedule, NoiseSchedule, DiffusionVisualizer,
    ForwardDiffusionResult, ReverseDiffusionResult
)


class TestDiffusionConfig:
    """Test suite for DiffusionConfig."""
    
    def test_default_configuration(self) -> Any:
        """Test default configuration values."""
        config = DiffusionConfig()
        
        assert config.num_timesteps == 1000
        assert config.beta_start == 0.0001
        assert config.beta_end == 0.02
        assert config.schedule == DiffusionSchedule.LINEAR
        assert config.noise_schedule == NoiseSchedule.LINEAR
        assert config.device == "auto"
        assert config.torch_dtype == torch.float32
        assert config.prediction_type == "epsilon"
        assert config.clip_sample is False
    
    def test_custom_configuration(self) -> Any:
        """Test custom configuration values."""
        config = DiffusionConfig(
            num_timesteps=500,
            beta_start=0.001,
            beta_end=0.05,
            schedule=DiffusionSchedule.COSINE,
            noise_schedule=NoiseSchedule.COSINE,
            device="cuda",
            torch_dtype=torch.float16,
            prediction_type="v_prediction",
            clip_sample=True,
            clip_sample_range=2.0,
            sample_max_value=2.0,
            timestep_spacing="trailing",
            rescale_betas_zero_snr=True
        )
        
        assert config.num_timesteps == 500
        assert config.beta_start == 0.001
        assert config.beta_end == 0.05
        assert config.schedule == DiffusionSchedule.COSINE
        assert config.noise_schedule == NoiseSchedule.COSINE
        assert config.device == "cuda"
        assert config.torch_dtype == torch.float16
        assert config.prediction_type == "v_prediction"
        assert config.clip_sample is True
        assert config.clip_sample_range == 2.0
        assert config.sample_max_value == 2.0
        assert config.timestep_spacing == "trailing"
        assert config.rescale_betas_zero_snr is True


class TestDiffusionProcesses:
    """Test suite for DiffusionProcesses."""
    
    @pytest.fixture
    def diffusion_process(self) -> Any:
        """Create a diffusion process instance for testing."""
        config = DiffusionConfig(
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            schedule=DiffusionSchedule.LINEAR
        )
        return DiffusionProcesses(config)
    
    @pytest.fixture
    def test_image(self) -> Any:
        """Create a test image tensor."""
        return torch.randn(1, 3, 32, 32)  # Batch, Channels, Height, Width
    
    def test_initialization(self, diffusion_process) -> Any:
        """Test diffusion process initialization."""
        assert diffusion_process is not None
        assert hasattr(diffusion_process, 'config')
        assert hasattr(diffusion_process, 'device')
        assert hasattr(diffusion_process, 'betas')
        assert hasattr(diffusion_process, 'alphas')
        assert hasattr(diffusion_process, 'alphas_cumprod')
        assert hasattr(diffusion_process, 'variance')
        assert hasattr(diffusion_process, 'log_variance')
    
    def test_device_detection(self, diffusion_process) -> Any:
        """Test device detection logic."""
        device = diffusion_process._detect_device()
        assert device in [torch.device('cpu'), torch.device('cuda'), torch.device('mps')]
    
    def test_noise_schedule_linear(self) -> Any:
        """Test linear noise schedule."""
        config = DiffusionConfig(
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            schedule=DiffusionSchedule.LINEAR
        )
        
        diffusion_process = DiffusionProcesses(config)
        
        betas = diffusion_process.betas
        assert betas.shape == (100,)
        assert betas[0].item() == pytest.approx(0.0001, rel=1e-4)
        assert betas[-1].item() == pytest.approx(0.02, rel=1e-4)
        assert torch.all(betas >= 0)
        assert torch.all(betas <= 1)
    
    def test_noise_schedule_cosine(self) -> Any:
        """Test cosine noise schedule."""
        config = DiffusionConfig(
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            schedule=DiffusionSchedule.COSINE
        )
        
        diffusion_process = DiffusionProcesses(config)
        
        betas = diffusion_process.betas
        assert betas.shape == (100,)
        assert torch.all(betas >= 0)
        assert torch.all(betas <= 1)
        
        # Cosine schedule should be non-linear
        linear_betas = torch.linspace(0.0001, 0.02, 100)
        assert not torch.allclose(betas, linear_betas, atol=1e-6)
    
    def test_noise_schedule_quadratic(self) -> Any:
        """Test quadratic noise schedule."""
        config = DiffusionConfig(
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            schedule=DiffusionSchedule.QUADRATIC
        )
        
        diffusion_process = DiffusionProcesses(config)
        
        betas = diffusion_process.betas
        assert betas.shape == (100,)
        assert torch.all(betas >= 0)
        assert torch.all(betas <= 1)
    
    def test_noise_schedule_sigmoid(self) -> Any:
        """Test sigmoid noise schedule."""
        config = DiffusionConfig(
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            schedule=DiffusionSchedule.SIGMOID
        )
        
        diffusion_process = DiffusionProcesses(config)
        
        betas = diffusion_process.betas
        assert betas.shape == (100,)
        assert torch.all(betas >= 0)
        assert torch.all(betas <= 1)
    
    def test_invalid_schedule(self) -> Any:
        """Test error handling for invalid schedule."""
        config = DiffusionConfig(
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            schedule="invalid_schedule"  # This should be an enum
        )
        
        with pytest.raises(Exception):
            DiffusionProcesses(config)
    
    def test_forward_diffusion_basic(self, diffusion_process, test_image) -> Any:
        """Test basic forward diffusion process."""
        t = 50
        result = diffusion_process.forward_diffusion(test_image, t)
        
        assert isinstance(result, ForwardDiffusionResult)
        assert result.noisy_image.shape == test_image.shape
        assert result.noise.shape == test_image.shape
        assert result.timestep == t
        assert 0 <= result.alpha_bar <= 1
        assert 0 <= result.beta <= 1
        assert result.processing_time >= 0
    
    def test_forward_diffusion_tensor_timestep(self, diffusion_process, test_image) -> Any:
        """Test forward diffusion with tensor timestep."""
        t = torch.tensor([50], device=diffusion_process.device)
        result = diffusion_process.forward_diffusion(test_image, t)
        
        assert isinstance(result, ForwardDiffusionResult)
        assert result.noisy_image.shape == test_image.shape
        assert result.timestep == [50]  # Should be a list for tensor input
    
    def test_forward_diffusion_custom_noise(self, diffusion_process, test_image) -> Any:
        """Test forward diffusion with custom noise."""
        t = 50
        custom_noise = torch.randn_like(test_image)
        
        result = diffusion_process.forward_diffusion(test_image, t, noise=custom_noise)
        
        assert isinstance(result, ForwardDiffusionResult)
        assert torch.allclose(result.noise, custom_noise)
    
    def test_forward_diffusion_mathematical_properties(self, diffusion_process, test_image) -> Any:
        """Test mathematical properties of forward diffusion."""
        t = 50
        result = diffusion_process.forward_diffusion(test_image, t)
        
        # Check that x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        sqrt_alpha_bar = torch.sqrt(torch.tensor(result.alpha_bar))
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - result.alpha_bar)
        
        expected_noisy = sqrt_alpha_bar * test_image + sqrt_one_minus_alpha_bar * result.noise
        assert torch.allclose(result.noisy_image, expected_noisy, atol=1e-6)
    
    def test_reverse_diffusion_step_basic(self, diffusion_process, test_image) -> Any:
        """Test basic reverse diffusion step."""
        # First add noise
        forward_result = diffusion_process.forward_diffusion(test_image, 50)
        x_t = forward_result.noisy_image
        
        # Predict noise (simulate model prediction)
        predicted_noise = torch.randn_like(x_t)
        
        result = diffusion_process.reverse_diffusion_step(x_t, 50, predicted_noise)
        
        assert isinstance(result, ReverseDiffusionResult)
        assert result.denoised_image.shape == x_t.shape
        assert result.predicted_noise.shape == x_t.shape
        assert result.timestep == 50
        assert 0 <= result.alpha <= 1
        assert 0 <= result.beta <= 1
        assert result.processing_time >= 0
    
    def test_reverse_diffusion_epsilon_prediction(self, diffusion_process, test_image) -> Any:
        """Test reverse diffusion with epsilon prediction."""
        config = DiffusionConfig(
            num_timesteps=100,
            prediction_type="epsilon"
        )
        diffusion_process = DiffusionProcesses(config)
        
        forward_result = diffusion_process.forward_diffusion(test_image, 50)
        x_t = forward_result.noisy_image
        
        # Use actual noise as prediction
        predicted_noise = forward_result.noise
        
        result = diffusion_process.reverse_diffusion_step(x_t, 50, predicted_noise)
        
        # Should recover original image approximately
        assert torch.allclose(result.denoised_image, test_image, atol=0.1)
    
    def test_reverse_diffusion_v_prediction(self, diffusion_process, test_image) -> Any:
        """Test reverse diffusion with v prediction."""
        config = DiffusionConfig(
            num_timesteps=100,
            prediction_type="v_prediction"
        )
        diffusion_process = DiffusionProcesses(config)
        
        forward_result = diffusion_process.forward_diffusion(test_image, 50)
        x_t = forward_result.noisy_image
        
        # Predict v (velocity)
        predicted_v = torch.randn_like(x_t)
        
        result = diffusion_process.reverse_diffusion_step(x_t, 50, predicted_v)
        
        assert isinstance(result, ReverseDiffusionResult)
        assert result.denoised_image.shape == x_t.shape
    
    def test_reverse_diffusion_sample_prediction(self, diffusion_process, test_image) -> Any:
        """Test reverse diffusion with sample prediction."""
        config = DiffusionConfig(
            num_timesteps=100,
            prediction_type="sample"
        )
        diffusion_process = DiffusionProcesses(config)
        
        forward_result = diffusion_process.forward_diffusion(test_image, 50)
        x_t = forward_result.noisy_image
        
        # Predict x_0 directly
        predicted_x0 = torch.randn_like(x_t)
        
        result = diffusion_process.reverse_diffusion_step(x_t, 50, predicted_x0)
        
        assert isinstance(result, ReverseDiffusionResult)
        assert result.denoised_image.shape == x_t.shape
    
    def test_reverse_diffusion_invalid_prediction_type(self, diffusion_process, test_image) -> Any:
        """Test error handling for invalid prediction type."""
        config = DiffusionConfig(
            num_timesteps=100,
            prediction_type="invalid_type"
        )
        diffusion_process = DiffusionProcesses(config)
        
        forward_result = diffusion_process.forward_diffusion(test_image, 50)
        x_t = forward_result.noisy_image
        predicted_noise = torch.randn_like(x_t)
        
        with pytest.raises(ValueError):
            diffusion_process.reverse_diffusion_step(x_t, 50, predicted_noise)
    
    def test_reverse_diffusion_stochastic(self, diffusion_process, test_image) -> Any:
        """Test stochastic reverse diffusion."""
        forward_result = diffusion_process.forward_diffusion(test_image, 50)
        x_t = forward_result.noisy_image
        predicted_noise = torch.randn_like(x_t)
        
        # Deterministic
        result_det = diffusion_process.reverse_diffusion_step(x_t, 50, predicted_noise, eta=0.0)
        
        # Stochastic
        result_stoch = diffusion_process.reverse_diffusion_step(x_t, 50, predicted_noise, eta=1.0)
        
        # Results should be different for stochastic sampling
        assert not torch.allclose(result_det.denoised_image, result_stoch.denoised_image)
    
    def test_sample_trajectory(self, diffusion_process, test_image) -> Any:
        """Test sampling forward diffusion trajectory."""
        trajectory = diffusion_process.sample_trajectory(test_image, num_steps=5)
        
        assert isinstance(trajectory, list)
        assert len(trajectory) == 5
        
        for i, result in enumerate(trajectory):
            assert isinstance(result, ForwardDiffusionResult)
            assert result.noisy_image.shape == test_image.shape
            assert result.timestep >= 0
    
    def test_denoise_trajectory(self, diffusion_process, test_image) -> Any:
        """Test sampling reverse diffusion trajectory."""
        # Create noisy image
        forward_result = diffusion_process.forward_diffusion(test_image, 99)
        x_T = forward_result.noisy_image
        
        # Simple noise predictor
        def noise_predictor(x_t: torch.Tensor, t: int) -> torch.Tensor:
            return torch.randn_like(x_t)
        
        trajectory = diffusion_process.denoise_trajectory(x_T, noise_predictor, num_steps=5)
        
        assert isinstance(trajectory, list)
        assert len(trajectory) == 5
        
        for i, result in enumerate(trajectory):
            assert isinstance(result, ReverseDiffusionResult)
            assert result.denoised_image.shape == x_T.shape
            assert result.timestep >= 0
    
    def test_get_noise_schedule_info(self, diffusion_process) -> Optional[Dict[str, Any]]:
        """Test getting noise schedule information."""
        info = diffusion_process.get_noise_schedule_info()
        
        assert isinstance(info, dict)
        assert "num_timesteps" in info
        assert "beta_start" in info
        assert "beta_end" in info
        assert "schedule" in info
        assert "betas" in info
        assert "alphas" in info
        assert "alphas_cumprod" in info
        assert "variance" in info
        
        assert info["num_timesteps"] == 100
        assert info["beta_start"] == 0.0001
        assert info["beta_end"] == 0.02
        assert info["schedule"] == "linear"
        assert len(info["betas"]) == 100
        assert len(info["alphas"]) == 100
        assert len(info["alphas_cumprod"]) == 100
        assert len(info["variance"]) == 100


class TestSecurityDiffusionProcesses:
    """Test suite for SecurityDiffusionProcesses."""
    
    @pytest.fixture
    def security_diffusion(self) -> Any:
        """Create a security diffusion process instance for testing."""
        config = DiffusionConfig(
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            schedule=DiffusionSchedule.LINEAR
        )
        return SecurityDiffusionProcesses(config)
    
    @pytest.fixture
    def test_image(self) -> Any:
        """Create a test image tensor."""
        return torch.randn(1, 3, 32, 32)
    
    def test_initialization(self, security_diffusion) -> Any:
        """Test security diffusion process initialization."""
        assert isinstance(security_diffusion, SecurityDiffusionProcesses)
        assert hasattr(security_diffusion, 'security_noise_scale')
        assert hasattr(security_diffusion, 'privacy_preserving')
        assert security_diffusion.security_noise_scale == 1.0
        assert security_diffusion.privacy_preserving is False
    
    def test_forward_diffusion_with_privacy(self, security_diffusion, test_image) -> Any:
        """Test privacy-preserving forward diffusion."""
        # Test different privacy levels
        for privacy_level in [0.0, 0.5, 1.0]:
            result = security_diffusion.forward_diffusion_with_privacy(
                test_image, t=50, privacy_level=privacy_level
            )
            
            assert isinstance(result, ForwardDiffusionResult)
            assert result.noisy_image.shape == test_image.shape
            
            # Higher privacy should result in more noise
            if privacy_level > 0:
                noise_norm = torch.norm(result.noise).item()
                assert noise_norm > 0
    
    def test_security_aware_denoising(self, security_diffusion, test_image) -> Any:
        """Test security-aware denoising."""
        # Create noisy image
        forward_result = security_diffusion.forward_diffusion(test_image, 50)
        x_t = forward_result.noisy_image
        
        # Test different security thresholds
        for threshold in [0.1, 0.5, 1.0]:
            predicted_noise = torch.randn_like(x_t) * 0.5
            
            result = security_diffusion.security_aware_denoising(
                x_t, t=50, predicted_noise=predicted_noise, security_threshold=threshold
            )
            
            assert isinstance(result, ReverseDiffusionResult)
            assert result.denoised_image.shape == x_t.shape
            
            # Check that noise magnitude is limited by threshold
            noise_magnitude = torch.norm(result.predicted_noise).item()
            if noise_magnitude > threshold:
                # Should be scaled down
                assert noise_magnitude <= threshold * 1.1  # Allow small numerical errors


class TestDiffusionVisualizer:
    """Test suite for DiffusionVisualizer."""
    
    @pytest.fixture
    def visualizer(self) -> Any:
        """Create a visualizer instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            return DiffusionVisualizer(temp_dir)
    
    @pytest.fixture
    def diffusion_process(self) -> Any:
        """Create a diffusion process for testing."""
        config = DiffusionConfig(
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            schedule=DiffusionSchedule.LINEAR
        )
        return DiffusionProcesses(config)
    
    def test_initialization(self, visualizer) -> Any:
        """Test visualizer initialization."""
        assert visualizer is not None
        assert hasattr(visualizer, 'output_dir')
        assert visualizer.output_dir.exists()
    
    def test_plot_noise_schedule(self, visualizer, diffusion_process) -> Any:
        """Test plotting noise schedule."""
        # This test might fail if matplotlib is not available
        try:
            visualizer.plot_noise_schedule(diffusion_process, "test_schedule.png")
            
            # Check if file was created
            output_file = visualizer.output_dir / "test_schedule.png"
            assert output_file.exists()
            
        except ImportError:
            pytest.skip("matplotlib not available")
    
    def test_plot_diffusion_trajectory(self, visualizer, diffusion_process) -> Any:
        """Test plotting diffusion trajectory."""
        # Create test trajectory
        test_image = torch.randn(1, 3, 32, 32)
        trajectory = diffusion_process.sample_trajectory(test_image, num_steps=3)
        
        try:
            visualizer.plot_diffusion_trajectory(trajectory, "test_trajectory.png")
            
            # Check if file was created
            output_file = visualizer.output_dir / "test_trajectory.png"
            assert output_file.exists()
            
        except ImportError:
            pytest.skip("matplotlib not available")
    
    def test_plot_denoising_trajectory(self, visualizer, diffusion_process) -> Any:
        """Test plotting denoising trajectory."""
        # Create test trajectory
        test_image = torch.randn(1, 3, 32, 32)
        x_T = diffusion_process.forward_diffusion(test_image, 99).noisy_image
        
        def noise_predictor(x_t: torch.Tensor, t: int) -> torch.Tensor:
            return torch.randn_like(x_t)
        
        trajectory = diffusion_process.denoise_trajectory(x_T, noise_predictor, num_steps=3)
        
        try:
            visualizer.plot_denoising_trajectory(trajectory, "test_denoising.png")
            
            # Check if file was created
            output_file = visualizer.output_dir / "test_denoising.png"
            assert output_file.exists()
            
        except ImportError:
            pytest.skip("matplotlib not available")


class TestForwardDiffusionResult:
    """Test suite for ForwardDiffusionResult."""
    
    def test_forward_diffusion_result_creation(self) -> Any:
        """Test ForwardDiffusionResult creation."""
        noisy_image = torch.randn(1, 3, 32, 32)
        noise = torch.randn(1, 3, 32, 32)
        
        result = ForwardDiffusionResult(
            noisy_image=noisy_image,
            noise=noise,
            timestep=50,
            alpha=0.95,
            beta=0.05,
            alpha_bar=0.8,
            processing_time=0.1
        )
        
        assert result.noisy_image is noisy_image
        assert result.noise is noise
        assert result.timestep == 50
        assert result.alpha == 0.95
        assert result.beta == 0.05
        assert result.alpha_bar == 0.8
        assert result.processing_time == 0.1
    
    def test_forward_diffusion_result_defaults(self) -> Any:
        """Test ForwardDiffusionResult with default values."""
        noisy_image = torch.randn(1, 3, 32, 32)
        noise = torch.randn(1, 3, 32, 32)
        
        result = ForwardDiffusionResult(
            noisy_image=noisy_image,
            noise=noise,
            timestep=50,
            alpha=0.95,
            beta=0.05,
            alpha_bar=0.8
        )
        
        assert result.processing_time == 0.0


class TestReverseDiffusionResult:
    """Test suite for ReverseDiffusionResult."""
    
    def test_reverse_diffusion_result_creation(self) -> Any:
        """Test ReverseDiffusionResult creation."""
        denoised_image = torch.randn(1, 3, 32, 32)
        predicted_noise = torch.randn(1, 3, 32, 32)
        
        result = ReverseDiffusionResult(
            denoised_image=denoised_image,
            predicted_noise=predicted_noise,
            timestep=50,
            alpha=0.95,
            beta=0.05,
            alpha_bar=0.8,
            processing_time=0.1
        )
        
        assert result.denoised_image is denoised_image
        assert result.predicted_noise is predicted_noise
        assert result.timestep == 50
        assert result.alpha == 0.95
        assert result.beta == 0.05
        assert result.alpha_bar == 0.8
        assert result.processing_time == 0.1
    
    def test_reverse_diffusion_result_defaults(self) -> Any:
        """Test ReverseDiffusionResult with default values."""
        denoised_image = torch.randn(1, 3, 32, 32)
        predicted_noise = torch.randn(1, 3, 32, 32)
        
        result = ReverseDiffusionResult(
            denoised_image=denoised_image,
            predicted_noise=predicted_noise,
            timestep=50,
            alpha=0.95,
            beta=0.05,
            alpha_bar=0.8
        )
        
        assert result.processing_time == 0.0


class TestIntegration:
    """Integration tests for diffusion processes."""
    
    @pytest.fixture
    def diffusion_process(self) -> Any:
        """Create a diffusion process for integration testing."""
        config = DiffusionConfig(
            num_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            schedule=DiffusionSchedule.LINEAR
        )
        return DiffusionProcesses(config)
    
    def test_end_to_end_diffusion(self, diffusion_process) -> Any:
        """Test end-to-end forward and reverse diffusion."""
        # Create test image
        test_image = torch.randn(1, 3, 32, 32)
        
        # Forward diffusion
        forward_result = diffusion_process.forward_diffusion(test_image, 50)
        x_t = forward_result.noisy_image
        
        # Reverse diffusion (using actual noise as prediction)
        reverse_result = diffusion_process.reverse_diffusion_step(
            x_t, 50, forward_result.noise
        )
        
        # Should recover original image approximately
        assert torch.allclose(reverse_result.denoised_image, test_image, atol=0.1)
    
    def test_trajectory_consistency(self, diffusion_process) -> Any:
        """Test consistency of forward and reverse trajectories."""
        test_image = torch.randn(1, 3, 32, 32)
        
        # Forward trajectory
        forward_trajectory = diffusion_process.sample_trajectory(test_image, num_steps=5)
        
        # Reverse trajectory
        x_T = forward_trajectory[-1].noisy_image
        
        def noise_predictor(x_t: torch.Tensor, t: int) -> torch.Tensor:
            return torch.randn_like(x_t)
        
        reverse_trajectory = diffusion_process.denoise_trajectory(x_T, noise_predictor, num_steps=5)
        
        # Check that trajectories have correct lengths
        assert len(forward_trajectory) == 5
        assert len(reverse_trajectory) == 5
        
        # Check that timesteps are consistent
        for i, (fwd, rev) in enumerate(zip(forward_trajectory, reversed(reverse_trajectory))):
            assert fwd.timestep >= rev.timestep


match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 