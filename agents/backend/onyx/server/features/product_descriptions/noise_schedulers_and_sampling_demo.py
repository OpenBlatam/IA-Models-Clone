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

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import psutil
import gc
from noise_schedulers_and_sampling import (
from typing import Any, List, Dict, Optional
"""
Advanced Noise Schedulers and Sampling Methods Demo
==================================================

This demo showcases the comprehensive noise schedulers and sampling methods
implementation, including:

1. Different noise schedules (Linear, Cosine, Quadratic, Sigmoid, Exponential)
2. Various sampling methods (DDPM, DDIM, DPM-Solver, Euler)
3. Performance comparisons
4. Visualizations of noise schedules
5. Practical examples with different configurations
6. Security considerations

Author: AI Assistant
License: MIT
"""



# Import our noise schedulers and sampling methods
    NoiseScheduleType, SamplingMethod, NoiseSchedulerConfig, SamplingConfig,
    NoiseSchedulerFactory, SamplerFactory, AdvancedSamplingManager,
    create_noise_scheduler, create_sampler, create_advanced_sampling_manager
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockDiffusionModel(nn.Module):
    """
    Mock diffusion model for demonstration purposes.
    
    This is a simple UNet-like model that can be used to test the sampling methods.
    """
    
    def __init__(self, in_channels: int = 4, out_channels: int = 4, hidden_size: int = 128):
        
    """__init__ function."""
super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        
        # Simple UNet-like architecture
        self.conv_in = nn.Conv2d(in_channels, hidden_size, 3, padding=1)
        self.conv_mid = nn.Conv2d(hidden_size, hidden_size, 3, padding=1)
        self.conv_out = nn.Conv2d(hidden_size, out_channels, 3, padding=1)
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Text embedding projection (for classifier-free guidance)
        self.text_proj = nn.Linear(768, hidden_size)  # Assuming CLIP embeddings
        
        logger.info(f"MockDiffusionModel initialized with {in_channels}->{out_channels} channels")
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the mock diffusion model.
        
        Args:
            x: Input tensor [B, C, H, W]
            timestep: Timestep tensor [B]
            encoder_hidden_states: Optional text embeddings [B, seq_len, hidden_dim]
            
        Returns:
            Predicted noise [B, C, H, W]
        """
        batch_size = x.shape[0]
        
        # Timestep embedding
        timestep_float = timestep.float().view(-1, 1)
        time_emb = self.time_embed(timestep_float)  # [B, hidden_size]
        time_emb = time_emb.view(batch_size, -1, 1, 1)  # [B, hidden_size, 1, 1]
        
        # Text embedding (if provided)
        if encoder_hidden_states is not None:
            # Use mean pooling for simplicity
            text_emb = encoder_hidden_states.mean(dim=1)  # [B, hidden_dim]
            text_emb = self.text_proj(text_emb)  # [B, hidden_size]
            text_emb = text_emb.view(batch_size, -1, 1, 1)  # [B, hidden_size, 1, 1]
            
            # Combine time and text embeddings
            combined_emb = time_emb + text_emb
        else:
            combined_emb = time_emb
        
        # Forward pass through UNet
        h = self.conv_in(x)
        h = h + combined_emb
        h = torch.relu(h)
        
        h = self.conv_mid(h)
        h = torch.relu(h)
        
        output = self.conv_out(h)
        
        return output


class NoiseSchedulersAndSamplingDemo:
    """
    Comprehensive demo for noise schedulers and sampling methods.
    
    This class provides demonstrations of:
    - Different noise schedules
    - Various sampling methods
    - Performance comparisons
    - Visualizations
    - Practical examples
    """
    
    def __init__(self, output_dir: str = "noise_schedulers_demo_output"):
        
    """__init__ function."""
self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create mock model
        self.model = MockDiffusionModel().to(self.device)
        
        # Default configurations
        self.default_scheduler_config = NoiseSchedulerConfig(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02
        )
        
        self.default_sampling_config = SamplingConfig(
            num_inference_steps=50,
            guidance_scale=7.5,
            eta=0.0
        )
        
        logger.info("NoiseSchedulersAndSamplingDemo initialized")
    
    def demo_noise_schedules(self) -> Dict[str, Dict]:
        """
        Demonstrate different noise schedules.
        
        Returns:
            Dictionary with schedule information for each type
        """
        logger.info("Demonstrating different noise schedules...")
        
        schedule_types = [
            NoiseScheduleType.LINEAR,
            NoiseScheduleType.COSINE,
            NoiseScheduleType.QUADRATIC,
            NoiseScheduleType.SIGMOID,
            NoiseScheduleType.EXPONENTIAL
        ]
        
        schedule_info = {}
        
        for schedule_type in schedule_types:
            logger.info(f"Creating {schedule_type.value} noise schedule...")
            
            config = NoiseSchedulerConfig(
                schedule_type=schedule_type,
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02
            )
            
            scheduler = NoiseSchedulerFactory.create(config)
            info = scheduler.get_schedule_info()
            schedule_info[schedule_type.value] = info
            
            logger.info(f"{schedule_type.value}: {info}")
        
        return schedule_info
    
    def visualize_noise_schedules(self, schedule_info: Dict[str, Dict]):
        """Visualize different noise schedules."""
        logger.info("Creating noise schedule visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        schedule_types = list(schedule_info.keys())
        
        for i, schedule_type in enumerate(schedule_types):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Create scheduler to get actual values
            config = NoiseSchedulerConfig(
                schedule_type=NoiseScheduleType(schedule_type),
                num_train_timesteps=1000,
                beta_start=0.0001,
                beta_end=0.02
            )
            scheduler = NoiseSchedulerFactory.create(config)
            
            # Plot beta schedule
            timesteps = torch.arange(1000)
            ax.plot(timesteps, scheduler.betas.cpu().numpy(), label='Beta', linewidth=2)
            ax.plot(timesteps, scheduler.alphas_cumprod.cpu().numpy(), label='Alpha_cumprod', linewidth=2)
            
            ax.set_title(f'{schedule_type.upper()} Schedule')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove extra subplot if needed
        if len(schedule_types) < len(axes):
            axes[-1].remove()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "noise_schedules_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved noise schedule visualization to {self.output_dir / 'noise_schedules_comparison.png'}")
    
    def demo_sampling_methods(self, latents: torch.Tensor, prompt_embeds: Optional[torch.Tensor] = None) -> Dict[str, Dict]:
        """
        Demonstrate different sampling methods.
        
        Args:
            latents: Initial latents for sampling
            prompt_embeds: Optional prompt embeddings for guidance
            
        Returns:
            Dictionary with sampling results for each method
        """
        logger.info("Demonstrating different sampling methods...")
        
        sampling_methods = [
            SamplingMethod.DDPM,
            SamplingMethod.DDIM,
            SamplingMethod.DPM_SOLVER,
            SamplingMethod.EULER
        ]
        
        results = {}
        
        for method in sampling_methods:
            logger.info(f"Testing {method.value} sampling method...")
            
            # Create sampling manager
            manager = create_advanced_sampling_manager(
                schedule_type=NoiseScheduleType.COSINE,
                method=method,
                num_inference_steps=20,  # Reduced for faster demo
                guidance_scale=7.5,
                eta=0.0
            )
            
            # Sample
            start_time = time.time()
            result = manager.sample(
                model=self.model,
                latents=latents.clone(),
                prompt_embeds=prompt_embeds
            )
            end_time = time.time()
            
            # Store results
            results[method.value] = {
                "result": result,
                "total_time": end_time - start_time,
                "avg_step_time": result.processing_time / len(result.timesteps) if result.timesteps else 0,
                "num_steps": len(result.timesteps),
                "final_sample_shape": result.samples.shape
            }
            
            logger.info(f"{method.value}: {results[method.value]['total_time']:.3f}s, {results[method.value]['num_steps']} steps")
        
        return results
    
    def compare_sampling_performance(self, latents: torch.Tensor, prompt_embeds: Optional[torch.Tensor] = None):
        """Compare performance of different sampling methods."""
        logger.info("Comparing sampling method performance...")
        
        # Test different step counts
        step_counts = [10, 20, 50]
        methods = [SamplingMethod.DDPM, SamplingMethod.DDIM, SamplingMethod.DPM_SOLVER, SamplingMethod.EULER]
        
        performance_data = {}
        
        for method in methods:
            performance_data[method.value] = {}
            
            for steps in step_counts:
                logger.info(f"Testing {method.value} with {steps} steps...")
                
                manager = create_advanced_sampling_manager(
                    schedule_type=NoiseScheduleType.COSINE,
                    method=method,
                    num_inference_steps=steps,
                    guidance_scale=7.5,
                    eta=0.0
                )
                
                start_time = time.time()
                result = manager.sample(
                    model=self.model,
                    latents=latents.clone(),
                    prompt_embeds=prompt_embeds
                )
                end_time = time.time()
                
                performance_data[method.value][steps] = {
                    "total_time": end_time - start_time,
                    "avg_step_time": result.processing_time / steps,
                    "memory_usage": self._get_memory_usage()
                }
        
        # Visualize performance comparison
        self._plot_performance_comparison(performance_data)
        
        return performance_data
    
    def _plot_performance_comparison(self, performance_data: Dict):
        """Plot performance comparison of sampling methods."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        methods = list(performance_data.keys())
        step_counts = list(performance_data[methods[0]].keys())
        
        # Total time comparison
        ax1 = axes[0]
        for method in methods:
            times = [performance_data[method][steps]["total_time"] for steps in step_counts]
            ax1.plot(step_counts, times, marker='o', label=method.upper(), linewidth=2)
        
        ax1.set_title("Total Sampling Time")
        ax1.set_xlabel("Number of Steps")
        ax1.set_ylabel("Time (seconds)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Average step time comparison
        ax2 = axes[1]
        for method in methods:
            avg_times = [performance_data[method][steps]["avg_step_time"] for steps in step_counts]
            ax2.plot(step_counts, avg_times, marker='o', label=method.upper(), linewidth=2)
        
        ax2.set_title("Average Step Time")
        ax2.set_xlabel("Number of Steps")
        ax2.set_ylabel("Time per Step (seconds)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Memory usage comparison
        ax3 = axes[2]
        for method in methods:
            memory_usage = [performance_data[method][steps]["memory_usage"] for steps in step_counts]
            ax3.plot(step_counts, memory_usage, marker='o', label=method.upper(), linewidth=2)
        
        ax3.set_title("Memory Usage")
        ax3.set_xlabel("Number of Steps")
        ax3.set_ylabel("Memory (MB)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "sampling_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved performance comparison to {self.output_dir / 'sampling_performance_comparison.png'}")
    
    def demo_guidance_techniques(self, latents: torch.Tensor, prompt_embeds: torch.Tensor, negative_prompt_embeds: torch.Tensor):
        """Demonstrate different guidance techniques."""
        logger.info("Demonstrating guidance techniques...")
        
        guidance_scales = [0.0, 3.0, 7.5, 15.0]
        results = {}
        
        for guidance_scale in guidance_scales:
            logger.info(f"Testing guidance scale: {guidance_scale}")
            
            manager = create_advanced_sampling_manager(
                schedule_type=NoiseScheduleType.COSINE,
                method=SamplingMethod.DDIM,
                num_inference_steps=20,
                guidance_scale=guidance_scale,
                eta=0.0
            )
            
            start_time = time.time()
            result = manager.sample(
                model=self.model,
                latents=latents.clone(),
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds
            )
            end_time = time.time()
            
            results[guidance_scale] = {
                "result": result,
                "total_time": end_time - start_time,
                "guidance_scale": guidance_scale
            }
        
        return results
    
    def demo_adaptive_sampling(self, latents: torch.Tensor, prompt_embeds: Optional[torch.Tensor] = None):
        """Demonstrate adaptive sampling techniques."""
        logger.info("Demonstrating adaptive sampling...")
        
        # Test different eta values (stochasticity)
        eta_values = [0.0, 0.5, 1.0]
        results = {}
        
        for eta in eta_values:
            logger.info(f"Testing eta value: {eta}")
            
            manager = create_advanced_sampling_manager(
                schedule_type=NoiseScheduleType.COSINE,
                method=SamplingMethod.DDPM,  # DDPM supports eta
                num_inference_steps=20,
                guidance_scale=7.5,
                eta=eta
            )
            
            start_time = time.time()
            result = manager.sample(
                model=self.model,
                latents=latents.clone(),
                prompt_embeds=prompt_embeds
            )
            end_time = time.time()
            
            results[eta] = {
                "result": result,
                "total_time": end_time - start_time,
                "eta": eta
            }
        
        return results
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
    
    def run_comprehensive_demo(self) -> Any:
        """Run the comprehensive demo."""
        logger.info("Starting comprehensive noise schedulers and sampling demo...")
        
        # Create sample data
        batch_size = 1
        channels = 4
        height = 64
        width = 64
        
        latents = torch.randn(batch_size, channels, height, width, device=self.device)
        
        # Create mock prompt embeddings
        prompt_embeds = torch.randn(batch_size, 77, 768, device=self.device)  # CLIP-like embeddings
        negative_prompt_embeds = torch.randn(batch_size, 77, 768, device=self.device)
        
        # 1. Demo noise schedules
        logger.info("\n" + "="*50)
        logger.info("1. DEMONSTRATING NOISE SCHEDULES")
        logger.info("="*50)
        
        schedule_info = self.demo_noise_schedules()
        self.visualize_noise_schedules(schedule_info)
        
        # 2. Demo sampling methods
        logger.info("\n" + "="*50)
        logger.info("2. DEMONSTRATING SAMPLING METHODS")
        logger.info("="*50)
        
        sampling_results = self.demo_sampling_methods(latents, prompt_embeds)
        
        # 3. Performance comparison
        logger.info("\n" + "="*50)
        logger.info("3. PERFORMANCE COMPARISON")
        logger.info("="*50)
        
        performance_data = self.compare_sampling_performance(latents, prompt_embeds)
        
        # 4. Guidance techniques
        logger.info("\n" + "="*50)
        logger.info("4. GUIDANCE TECHNIQUES")
        logger.info("="*50)
        
        guidance_results = self.demo_guidance_techniques(latents, prompt_embeds, negative_prompt_embeds)
        
        # 5. Adaptive sampling
        logger.info("\n" + "="*50)
        logger.info("5. ADAPTIVE SAMPLING")
        logger.info("="*50)
        
        adaptive_results = self.demo_adaptive_sampling(latents, prompt_embeds)
        
        # Generate summary report
        self._generate_summary_report(schedule_info, sampling_results, performance_data, guidance_results, adaptive_results)
        
        logger.info("\n" + "="*50)
        logger.info("DEMO COMPLETED SUCCESSFULLY!")
        logger.info(f"Output saved to: {self.output_dir}")
        logger.info("="*50)
    
    def _generate_summary_report(self, schedule_info, sampling_results, performance_data, guidance_results, adaptive_results) -> Any:
        """Generate a comprehensive summary report."""
        report_path = self.output_dir / "demo_summary_report.txt"
        
        with open(report_path, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("NOISE SCHEDULERS AND SAMPLING METHODS DEMO SUMMARY\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("=" * 60 + "\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Noise schedules summary
            f.write("1. NOISE SCHEDULES\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("-" * 20 + "\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            for schedule_type, info in schedule_info.items():
                f.write(f"{schedule_type.upper()}:\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"  - Beta range: {info['betas_range']}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"  - Alpha_cumprod range: {info['alphas_cumprod_range']}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"  - Timesteps: {info['num_timesteps']}\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Sampling methods summary
            f.write("2. SAMPLING METHODS\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("-" * 20 + "\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            for method, result in sampling_results.items():
                f.write(f"{method.upper()}:\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"  - Total time: {result['total_time']:.3f}s\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"  - Steps: {result['num_steps']}\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"  - Avg step time: {result['avg_step_time']:.6f}s\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"  - Sample shape: {result['final_sample_shape']}\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Performance summary
            f.write("3. PERFORMANCE COMPARISON\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("-" * 25 + "\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            for method in performance_data.keys():
                f.write(f"{method.upper()}:\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                for steps, data in performance_data[method].items():
                    f.write(f"  - {steps} steps: {data['total_time']:.3f}s, {data['memory_usage']:.1f}MB\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write("\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Guidance summary
            f.write("4. GUIDANCE TECHNIQUES\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("-" * 22 + "\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            for scale, result in guidance_results.items():
                f.write(f"Guidance scale {scale}:\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"  - Total time: {result['total_time']:.3f}s\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"  - Steps: {len(result['result'].timesteps)}\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            
            # Adaptive sampling summary
            f.write("5. ADAPTIVE SAMPLING\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            f.write("-" * 20 + "\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            for eta, result in adaptive_results.items():
                f.write(f"Eta {eta}:\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"  - Total time: {result['total_time']:.3f}s\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                f.write(f"  - Steps: {len(result['result'].timesteps)}\n\n")
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        
        logger.info(f"Summary report saved to: {report_path}")


async def main():
    """Main demo function."""
    logger.info("Starting Noise Schedulers and Sampling Methods Demo")
    
    # Create demo instance
    demo = NoiseSchedulersAndSamplingDemo()
    
    # Run comprehensive demo
    demo.run_comprehensive_demo()
    
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main()) 