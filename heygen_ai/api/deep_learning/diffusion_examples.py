from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
            from .diffusion_models import DiffusionConfig, DDPM
            from .diffusion_models import DiffusionConfig, DDIM
            from .diffusion_models import DiffusionConfig, NoiseScheduler
            from .diffusion_models import UNet, SinusoidalPositionEmbeddings
            from .diffusion_models import DiffusionConfig, DDPM
            from .diffusion_models import DiffusionConfig, DDPM, DDIM
            from .diffusion_models import DiffusionConfig, DDPM
            from .diffusion_models import DiffusionConfig, NoiseScheduler
            from .diffusion_models import UNet, UNetBlock, SinusoidalPositionEmbeddings
            from .diffusion_models import DiffusionConfig, DDPM, DDIM
            import psutil
            import gc
            from .diffusion_models import DiffusionConfig, DDPM, DDIM
            from .diffusion_models import DiffusionConfig, NoiseScheduler
            from .diffusion_models import DiffusionConfig, DDPM
            from .diffusion_models import DiffusionConfig, DDPM
from typing import Any, List, Dict, Optional
import asyncio
"""
Diffusion Models Examples for HeyGen AI.

Comprehensive examples demonstrating usage of diffusion models including DDPM, DDIM,
and advanced variants following PEP 8 style guidelines.
"""


logger = logging.getLogger(__name__)


class DiffusionExamples:
    """Examples of diffusion model usage."""

    @staticmethod
    def ddpm_basic_example():
        """Basic DDPM example."""
        
        try:
            
            # Create configuration
            config = DiffusionConfig(
                num_timesteps=1000,
                beta_start=1e-4,
                beta_end=0.02,
                beta_schedule="linear",
                prediction_type="epsilon",
                loss_type="mse",
                guidance_scale=7.5,
                classifier_free_guidance=True,
                clip_denoised=True,
                use_ema=True,
                ema_decay=0.9999
            )
            
            # Create DDPM model
            ddpm_model = DDPM(config)
            
            # Sample data
            batch_size = 4
            image_size = 64
            channels = 3
            
            # Generate samples
            samples = ddpm_model.sample(
                batch_size=batch_size,
                image_size=image_size,
                channels=channels,
                guidance_scale=7.5
            )
            
            logger.info(f"Generated {batch_size} samples with shape {samples.shape}")
            logger.info(f"Sample statistics - Mean: {samples.mean():.4f}, Std: {samples.std():.4f}")
            logger.info(f"Sample range - Min: {samples.min():.4f}, Max: {samples.max():.4f}")
            
            return ddpm_model, samples
            
        except ImportError as e:
            logger.error(f"Diffusion models module not available: {e}")
            return None, None

    @staticmethod
    def ddim_basic_example():
        """Basic DDIM example."""
        
        try:
            
            # Create configuration
            config = DiffusionConfig(
                num_timesteps=1000,
                beta_start=1e-4,
                beta_end=0.02,
                beta_schedule="cosine",
                prediction_type="epsilon",
                loss_type="mse",
                guidance_scale=7.5,
                classifier_free_guidance=True,
                clip_denoised=True,
                use_ema=False  # DDIM typically doesn't use EMA
            )
            
            # Create DDIM model
            ddim_model = DDIM(config)
            
            # Sample data
            batch_size = 4
            image_size = 64
            channels = 3
            
            # Generate samples
            samples = ddim_model.sample(
                batch_size=batch_size,
                image_size=image_size,
                channels=channels,
                guidance_scale=7.5
            )
            
            logger.info(f"Generated {batch_size} DDIM samples with shape {samples.shape}")
            logger.info(f"Sample statistics - Mean: {samples.mean():.4f}, Std: {samples.std():.4f}")
            logger.info(f"Sample range - Min: {samples.min():.4f}, Max: {samples.max():.4f}")
            
            return ddim_model, samples
            
        except ImportError as e:
            logger.error(f"Diffusion models module not available: {e}")
            return None, None

    @staticmethod
    def noise_scheduler_example():
        """Noise scheduler example."""
        
        try:
            
            # Create configuration
            config = DiffusionConfig(
                num_timesteps=1000,
                beta_start=1e-4,
                beta_end=0.02,
                beta_schedule="linear"
            )
            
            # Create noise scheduler
            scheduler = NoiseScheduler(config)
            
            # Test noise addition
            batch_size = 2
            image_size = 32
            channels = 3
            
            x_start = torch.randn(batch_size, channels, image_size, image_size)
            timesteps = torch.randint(0, config.num_timesteps, (batch_size,))
            
            # Add noise
            noisy_x, noise = scheduler.add_noise(x_start, timesteps)
            
            # Remove noise
            denoised_x = scheduler.remove_noise(noisy_x, noise, timesteps)
            
            logger.info(f"Original shape: {x_start.shape}")
            logger.info(f"Noisy shape: {noisy_x.shape}")
            logger.info(f"Denoised shape: {denoised_x.shape}")
            logger.info(f"Reconstruction error: {F.mse_loss(x_start, denoised_x):.6f}")
            
            return scheduler, x_start, noisy_x, denoised_x
            
        except ImportError as e:
            logger.error(f"Diffusion models module not available: {e}")
            return None, None, None, None

    @staticmethod
    def unet_model_example():
        """UNet model example."""
        
        try:
            
            # Create UNet model
            unet_model = UNet(
                in_channels=3,
                out_channels=3,
                model_channels=64,
                num_res_blocks=2,
                attention_resolutions=(16, 8),
                dropout=0.1,
                channel_mult=(1, 2, 4),
                num_heads=8
            )
            
            # Create time embeddings
            time_embed = SinusoidalPositionEmbeddings(64)
            
            # Test forward pass
            batch_size = 2
            image_size = 32
            channels = 3
            
            x = torch.randn(batch_size, channels, image_size, image_size)
            timesteps = torch.randint(0, 1000, (batch_size,))
            
            # Forward pass
            output = unet_model(x, timesteps)
            
            logger.info(f"Input shape: {x.shape}")
            logger.info(f"Output shape: {output.shape}")
            logger.info(f"Model parameters: {sum(p.numel() for p in unet_model.parameters()):,}")
            
            return unet_model, x, output
            
        except ImportError as e:
            logger.error(f"Diffusion models module not available: {e}")
            return None, None, None

    @staticmethod
    def training_example():
        """Training example."""
        
        try:
            
            # Create configuration
            config = DiffusionConfig(
                num_timesteps=1000,
                beta_start=1e-4,
                beta_end=0.02,
                beta_schedule="linear",
                prediction_type="epsilon",
                loss_type="mse",
                use_ema=True,
                ema_decay=0.9999
            )
            
            # Create DDPM model
            ddpm_model = DDPM(config)
            
            # Create optimizer
            optimizer = torch.optim.AdamW(ddpm_model.model.parameters(), lr=1e-4)
            
            # Sample training data
            batch_size = 4
            image_size = 32
            channels = 3
            
            # Training loop
            num_epochs = 5
            losses = []
            
            for epoch in range(num_epochs):
                epoch_losses = []
                
                for step in range(10):  # 10 steps per epoch
                    # Generate random training data
                    batch = torch.randn(batch_size, channels, image_size, image_size)
                    
                    # Training step
                    metrics = ddpm_model.training_step(batch, optimizer)
                    loss = metrics["loss"]
                    epoch_losses.append(loss)
                    
                    if step % 5 == 0:
                        logger.info(f"Epoch {epoch}, Step {step}, Loss: {loss:.6f}")
                
                avg_loss = np.mean(epoch_losses)
                losses.append(avg_loss)
                logger.info(f"Epoch {epoch} average loss: {avg_loss:.6f}")
            
            logger.info(f"Training completed. Final loss: {losses[-1]:.6f}")
            
            return ddpm_model, losses
            
        except ImportError as e:
            logger.error(f"Diffusion models module not available: {e}")
            return None, None

    @staticmethod
    def sampling_comparison_example():
        """Compare different sampling methods."""
        
        try:
            
            # Create configurations
            ddpm_config = DiffusionConfig(
                num_timesteps=1000,
                beta_start=1e-4,
                beta_end=0.02,
                beta_schedule="linear",
                prediction_type="epsilon",
                loss_type="mse",
                use_ema=True,
                ema_decay=0.9999
            )
            
            ddim_config = DiffusionConfig(
                num_timesteps=1000,
                beta_start=1e-4,
                beta_end=0.02,
                beta_schedule="cosine",
                prediction_type="epsilon",
                loss_type="mse",
                use_ema=False
            )
            
            # Create models
            ddpm_model = DDPM(ddpm_config)
            ddim_model = DDIM(ddim_config)
            
            # Generate samples
            batch_size = 2
            image_size = 32
            channels = 3
            
            # DDPM samples
            ddpm_samples = ddpm_model.sample(
                batch_size=batch_size,
                image_size=image_size,
                channels=channels,
                guidance_scale=7.5
            )
            
            # DDIM samples
            ddim_samples = ddim_model.sample(
                batch_size=batch_size,
                image_size=image_size,
                channels=channels,
                guidance_scale=7.5
            )
            
            logger.info(f"DDPM samples shape: {ddpm_samples.shape}")
            logger.info(f"DDIM samples shape: {dddim_samples.shape}")
            logger.info(f"DDPM statistics - Mean: {ddpm_samples.mean():.4f}, Std: {ddpm_samples.std():.4f}")
            logger.info(f"DDIM statistics - Mean: {ddim_samples.mean():.4f}, Std: {ddim_samples.std():.4f}")
            
            return ddpm_model, ddim_model, ddpm_samples, ddim_samples
            
        except ImportError as e:
            logger.error(f"Diffusion models module not available: {e}")
            return None, None, None, None

    @staticmethod
    def guidance_example():
        """Classifier-free guidance example."""
        
        try:
            
            # Create configuration with guidance
            config = DiffusionConfig(
                num_timesteps=1000,
                beta_start=1e-4,
                beta_end=0.02,
                beta_schedule="linear",
                prediction_type="epsilon",
                loss_type="mse",
                guidance_scale=7.5,
                classifier_free_guidance=True,
                use_ema=True,
                ema_decay=0.9999
            )
            
            # Create DDPM model
            ddpm_model = DDPM(config)
            
            # Generate samples with different guidance scales
            batch_size = 2
            image_size = 32
            channels = 3
            
            guidance_scales = [1.0, 3.0, 7.5, 15.0]
            samples_dict = {}
            
            for guidance_scale in guidance_scales:
                samples = ddpm_model.sample(
                    batch_size=batch_size,
                    image_size=image_size,
                    channels=channels,
                    guidance_scale=guidance_scale
                )
                samples_dict[guidance_scale] = samples
                
                logger.info(f"Guidance scale {guidance_scale}:")
                logger.info(f"  Shape: {samples.shape}")
                logger.info(f"  Mean: {samples.mean():.4f}")
                logger.info(f"  Std: {samples.std():.4f}")
                logger.info(f"  Range: [{samples.min():.4f}, {samples.max():.4f}]")
            
            return ddpm_model, samples_dict
            
        except ImportError as e:
            logger.error(f"Diffusion models module not available: {e}")
            return None, None

    @staticmethod
    def beta_schedule_comparison_example():
        """Compare different beta schedules."""
        
        try:
            
            # Create configurations with different beta schedules
            configs = {
                "linear": DiffusionConfig(
                    num_timesteps=1000,
                    beta_start=1e-4,
                    beta_end=0.02,
                    beta_schedule="linear"
                ),
                "cosine": DiffusionConfig(
                    num_timesteps=1000,
                    beta_start=1e-4,
                    beta_end=0.02,
                    beta_schedule="cosine"
                ),
                "sigmoid": DiffusionConfig(
                    num_timesteps=1000,
                    beta_start=1e-4,
                    beta_end=0.02,
                    beta_schedule="sigmoid"
                )
            }
            
            schedulers = {}
            beta_curves = {}
            
            for name, config in configs.items():
                scheduler = NoiseScheduler(config)
                schedulers[name] = scheduler
                beta_curves[name] = scheduler.betas.cpu().numpy()
                
                logger.info(f"{name.capitalize()} beta schedule:")
                logger.info(f"  Beta range: [{scheduler.betas.min():.6f}, {scheduler.betas.max():.6f}]")
                logger.info(f"  Alpha cumprod range: [{scheduler.alphas_cumprod.min():.6f}, {scheduler.alphas_cumprod.max():.6f}]")
            
            return schedulers, beta_curves
            
        except ImportError as e:
            logger.error(f"Diffusion models module not available: {e}")
            return None, None

    @staticmethod
    def custom_unet_example():
        """Custom UNet architecture example."""
        
        try:
            
            # Create custom UNet with different parameters
            custom_unet = UNet(
                in_channels=3,
                out_channels=3,
                model_channels=96,
                num_res_blocks=3,
                attention_resolutions=(16, 8, 4),
                dropout=0.15,
                channel_mult=(1, 2, 4, 8),
                num_heads=12,
                use_spatial_transformer=True,
                transformer_depth=2,
                context_dim=768,
                use_checkpoint=True,
                use_fp16=False,
                use_scale_shift_norm=True,
                resblock_updown=True,
                use_new_attention_order=True
            )
            
            # Test forward pass
            batch_size = 2
            image_size = 64
            channels = 3
            
            x = torch.randn(batch_size, channels, image_size, image_size)
            timesteps = torch.randint(0, 1000, (batch_size,))
            context = torch.randn(batch_size, 77, 768)  # Text context
            
            # Forward pass
            output = custom_unet(x, timesteps, context)
            
            logger.info(f"Custom UNet input shape: {x.shape}")
            logger.info(f"Custom UNet output shape: {output.shape}")
            logger.info(f"Custom UNet parameters: {sum(p.numel() for p in custom_unet.parameters()):,}")
            
            return custom_unet, x, output
            
        except ImportError as e:
            logger.error(f"Diffusion models module not available: {e}")
            return None, None, None

    @staticmethod
    def performance_benchmark_example():
        """Performance benchmark example."""
        
        try:
            
            # Create configurations
            ddpm_config = DiffusionConfig(
                num_timesteps=1000,
                beta_start=1e-4,
                beta_end=0.02,
                beta_schedule="linear",
                prediction_type="epsilon",
                loss_type="mse",
                use_ema=True,
                ema_decay=0.9999
            )
            
            ddim_config = DiffusionConfig(
                num_timesteps=1000,
                beta_start=1e-4,
                beta_end=0.02,
                beta_schedule="cosine",
                prediction_type="epsilon",
                loss_type="mse",
                use_ema=False
            )
            
            # Create models
            ddpm_model = DDPM(ddpm_config)
            ddim_model = DDIM(ddim_config)
            
            # Benchmark parameters
            batch_size = 1
            image_size = 64
            channels = 3
            num_runs = 5
            
            # Benchmark DDPM
            ddpm_times = []
            for _ in range(num_runs):
                start_time = time.time()
                samples = ddpm_model.sample(
                    batch_size=batch_size,
                    image_size=image_size,
                    channels=channels,
                    guidance_scale=7.5
                )
                end_time = time.time()
                ddpm_times.append(end_time - start_time)
            
            # Benchmark DDIM
            ddim_times = []
            for _ in range(num_runs):
                start_time = time.time()
                samples = ddim_model.sample(
                    batch_size=batch_size,
                    image_size=image_size,
                    channels=channels,
                    guidance_scale=7.5
                )
                end_time = time.time()
                ddim_times.append(end_time - start_time)
            
            logger.info("Performance Benchmark Results:")
            logger.info(f"DDPM - Average time: {np.mean(ddpm_times):.4f}s ± {np.std(ddpm_times):.4f}s")
            logger.info(f"DDIM - Average time: {np.mean(ddim_times):.4f}s ± {np.std(ddim_times):.4f}s")
            logger.info(f"Speedup: {np.mean(ddpm_times) / np.mean(ddim_times):.2f}x")
            
            return ddpm_model, ddim_model, ddpm_times, ddim_times
            
        except ImportError as e:
            logger.error(f"Diffusion models module not available: {e}")
            return None, None, None, None

    @staticmethod
    def memory_usage_example():
        """Memory usage analysis example."""
        
        try:
            
            
            # Create configurations
            ddpm_config = DiffusionConfig(
                num_timesteps=1000,
                beta_start=1e-4,
                beta_end=0.02,
                beta_schedule="linear",
                prediction_type="epsilon",
                loss_type="mse",
                use_ema=True,
                ema_decay=0.9999
            )
            
            ddim_config = DiffusionConfig(
                num_timesteps=1000,
                beta_start=1e-4,
                beta_end=0.02,
                beta_schedule="cosine",
                prediction_type="epsilon",
                loss_type="mse",
                use_ema=False
            )
            
            # Memory analysis
            process = psutil.Process()
            
            # DDPM memory usage
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            ddpm_model = DDPM(ddpm_config)
            ddpm_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Generate samples
            samples = ddpm_model.sample(batch_size=1, image_size=64, channels=3)
            ddpm_sample_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Clean up
            del ddpm_model, samples
            gc.collect()
            
            # DDIM memory usage
            ddim_model = DDIM(ddim_config)
            ddim_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Generate samples
            samples = ddim_model.sample(batch_size=1, image_size=64, channels=3)
            ddim_sample_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            logger.info("Memory Usage Analysis:")
            logger.info(f"Initial memory: {initial_memory:.2f} MB")
            logger.info(f"DDPM model memory: {ddpm_memory:.2f} MB (+{ddpm_memory - initial_memory:.2f} MB)")
            logger.info(f"DDPM sampling memory: {ddpm_sample_memory:.2f} MB (+{ddpm_sample_memory - ddpm_memory:.2f} MB)")
            logger.info(f"DDIM model memory: {ddim_memory:.2f} MB (+{ddim_memory - initial_memory:.2f} MB)")
            logger.info(f"DDIM sampling memory: {ddim_sample_memory:.2f} MB (+{ddim_sample_memory - ddim_memory:.2f} MB)")
            
            return ddpm_model, ddim_model, {
                "initial_memory": initial_memory,
                "ddpm_memory": ddpm_memory,
                "ddpm_sample_memory": ddpm_sample_memory,
                "ddim_memory": ddim_memory,
                "ddim_sample_memory": ddim_sample_memory
            }
            
        except ImportError as e:
            logger.error(f"Required modules not available: {e}")
            return None, None, None


class VisualizationExamples:
    """Examples of diffusion model visualization."""

    @staticmethod
    def plot_beta_schedules():
        """Plot different beta schedules."""
        
        try:
            
            # Create configurations
            configs = {
                "linear": DiffusionConfig(beta_schedule="linear"),
                "cosine": DiffusionConfig(beta_schedule="cosine"),
                "sigmoid": DiffusionConfig(beta_schedule="sigmoid")
            }
            
            plt.figure(figsize=(15, 5))
            
            for i, (name, config) in enumerate(configs.items()):
                scheduler = NoiseScheduler(config)
                
                plt.subplot(1, 3, i + 1)
                plt.plot(scheduler.betas.cpu().numpy(), label="Beta")
                plt.plot(scheduler.alphas_cumprod.cpu().numpy(), label="Alpha cumprod")
                plt.title(f"{name.capitalize()} Schedule")
                plt.xlabel("Timestep")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig("beta_schedules.png", dpi=300, bbox_inches="tight")
            plt.close()
            
            logger.info("Beta schedules plot saved as 'beta_schedules.png'")
            
        except ImportError as e:
            logger.error(f"Diffusion models module not available: {e}")

    @staticmethod
    def plot_sampling_process():
        """Plot sampling process."""
        
        try:
            
            # Create model
            config = DiffusionConfig(num_timesteps=100)
            ddpm_model = DDPM(config)
            
            # Generate samples at different timesteps
            batch_size = 1
            image_size = 32
            channels = 3
            
            # Start from noise
            x = torch.randn(batch_size, channels, image_size, image_size)
            
            # Sample at different timesteps
            timesteps = [99, 75, 50, 25, 0]
            samples = []
            
            for t in timesteps:
                timestep_tensor = torch.full((batch_size,), t, dtype=torch.long)
                model_output = ddpm_model.model(x, timestep_tensor)
                x = ddpm_model.sampling_step(x, timestep_tensor, model_output)
                samples.append(x.cpu().numpy())
            
            # Plot samples
            plt.figure(figsize=(15, 3))
            for i, sample in enumerate(samples):
                plt.subplot(1, len(samples), i + 1)
                img = sample[0].transpose(1, 2, 0)
                img = (img - img.min()) / (img.max() - img.min())
                plt.imshow(img)
                plt.title(f"Timestep {timesteps[i]}")
                plt.axis("off")
            
            plt.tight_layout()
            plt.savefig("sampling_process.png", dpi=300, bbox_inches="tight")
            plt.close()
            
            logger.info("Sampling process plot saved as 'sampling_process.png'")
            
        except ImportError as e:
            logger.error(f"Diffusion models module not available: {e}")

    @staticmethod
    def plot_guidance_comparison():
        """Plot guidance comparison."""
        
        try:
            
            # Create model
            config = DiffusionConfig(
                num_timesteps=100,
                guidance_scale=7.5,
                classifier_free_guidance=True
            )
            ddpm_model = DDPM(config)
            
            # Generate samples with different guidance scales
            guidance_scales = [1.0, 3.0, 7.5, 15.0]
            samples = []
            
            for guidance_scale in guidance_scales:
                sample = ddpm_model.sample(
                    batch_size=1,
                    image_size=32,
                    channels=3,
                    guidance_scale=guidance_scale
                )
                samples.append(sample.cpu().numpy())
            
            # Plot samples
            plt.figure(figsize=(15, 4))
            for i, sample in enumerate(samples):
                plt.subplot(1, len(samples), i + 1)
                img = sample[0].transpose(1, 2, 0)
                img = (img - img.min()) / (img.max() - img.min())
                plt.imshow(img)
                plt.title(f"Guidance: {guidance_scales[i]}")
                plt.axis("off")
            
            plt.tight_layout()
            plt.savefig("guidance_comparison.png", dpi=300, bbox_inches="tight")
            plt.close()
            
            logger.info("Guidance comparison plot saved as 'guidance_comparison.png'")
            
        except ImportError as e:
            logger.error(f"Diffusion models module not available: {e}")


def run_diffusion_examples():
    """Run all diffusion examples."""
    
    logger.info("Running Diffusion Models Examples")
    logger.info("=" * 60)
    
    # Basic examples
    logger.info("\n1. DDPM Basic Example:")
    ddpm_model, ddpm_samples = DiffusionExamples.ddpm_basic_example()
    
    logger.info("\n2. DDIM Basic Example:")
    ddim_model, ddim_samples = DiffusionExamples.ddim_basic_example()
    
    logger.info("\n3. Noise Scheduler Example:")
    scheduler, x_start, noisy_x, denoised_x = DiffusionExamples.noise_scheduler_example()
    
    logger.info("\n4. UNet Model Example:")
    unet_model, unet_x, unet_output = DiffusionExamples.unet_model_example()
    
    logger.info("\n5. Training Example:")
    trained_model, training_losses = DiffusionExamples.training_example()
    
    logger.info("\n6. Sampling Comparison Example:")
    ddpm_comp, ddim_comp, ddpm_comp_samples, ddim_comp_samples = DiffusionExamples.sampling_comparison_example()
    
    logger.info("\n7. Guidance Example:")
    guidance_model, guidance_samples = DiffusionExamples.guidance_example()
    
    logger.info("\n8. Beta Schedule Comparison Example:")
    beta_schedulers, beta_curves = DiffusionExamples.beta_schedule_comparison_example()
    
    logger.info("\n9. Custom UNet Example:")
    custom_unet, custom_x, custom_output = DiffusionExamples.custom_unet_example()
    
    logger.info("\n10. Performance Benchmark Example:")
    perf_ddpm, perf_ddim, perf_ddpm_times, perf_ddim_times = DiffusionExamples.performance_benchmark_example()
    
    logger.info("\n11. Memory Usage Example:")
    mem_ddpm, mem_ddim, memory_usage = DiffusionExamples.memory_usage_example()
    
    # Visualization examples
    logger.info("\n12. Plot Beta Schedules:")
    VisualizationExamples.plot_beta_schedules()
    
    logger.info("\n13. Plot Sampling Process:")
    VisualizationExamples.plot_sampling_process()
    
    logger.info("\n14. Plot Guidance Comparison:")
    VisualizationExamples.plot_guidance_comparison()
    
    logger.info("\nAll diffusion examples completed successfully!")
    
    return {
        "models": {
            "ddpm_model": ddpm_model,
            "ddim_model": ddim_model,
            "unet_model": unet_model,
            "trained_model": trained_model,
            "guidance_model": guidance_model,
            "custom_unet": custom_unet,
            "perf_ddpm": perf_ddpm,
            "perf_ddim": perf_ddim,
            "mem_ddpm": mem_ddpm,
            "mem_ddim": mem_ddim
        },
        "schedulers": {
            "scheduler": scheduler,
            "beta_schedulers": beta_schedulers
        },
        "samples": {
            "ddpm_samples": ddpm_samples,
            "ddim_samples": ddim_samples,
            "ddpm_comp_samples": ddpm_comp_samples,
            "ddim_comp_samples": ddim_comp_samples,
            "guidance_samples": guidance_samples
        },
        "data": {
            "x_start": x_start,
            "noisy_x": noisy_x,
            "denoised_x": denoised_x,
            "unet_x": unet_x,
            "unet_output": unet_output,
            "custom_x": custom_x,
            "custom_output": custom_output
        },
        "metrics": {
            "training_losses": training_losses,
            "beta_curves": beta_curves,
            "perf_ddpm_times": perf_ddpm_times,
            "perf_ddim_times": perf_ddim_times,
            "memory_usage": memory_usage
        }
    }


if __name__ == "__main__":
    # Run examples
    examples = run_diffusion_examples()
    logger.info("Diffusion Models Examples completed!") 