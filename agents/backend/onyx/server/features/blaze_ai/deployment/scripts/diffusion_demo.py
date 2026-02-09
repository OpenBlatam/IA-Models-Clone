#!/usr/bin/env python3
"""
Diffusion Processes Demonstration Script
Shows forward and reverse diffusion processes step-by-step
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleDiffusionDemo:
    """Simple demonstration of diffusion processes"""
    
    def __init__(self, num_timesteps: int = 100):
        self.num_timesteps = num_timesteps
        
        # Create beta schedule (noise schedule)
        self.betas = torch.linspace(0.0001, 0.02, num_timesteps)
        
        # Precompute values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        logger.info(f"Initialized diffusion demo with {num_timesteps} timesteps")
        logger.info(f"Beta range: {self.betas[0]:.6f} to {self.betas[-1]:.6f}")
    
    def forward_process(self, x_0: torch.Tensor, t: int) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0)
        
        This adds noise to the original image according to the schedule.
        The amount of noise increases with timestep t.
        
        Args:
            x_0: Original image [B, C, H, W]
            t: Timestep (0 to num_timesteps-1)
            
        Returns:
            x_t: Noisy image at timestep t
        """
        # Get noise schedule values for timestep t
        alpha_t = self.alphas_cumprod[t]
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Generate random noise
        noise = torch.randn_like(x_0)
        
        # Forward process equation: x_t = sqrt(α_t) * x_0 + sqrt(1 - α_t) * ε
        x_t = sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise
        
        logger.info(f"Forward step {t}: α_t={alpha_t:.4f}, noise_scale={sqrt_one_minus_alpha_t:.4f}")
        
        return x_t
    
    def reverse_process_step(self, x_t: torch.Tensor, t: int, predicted_noise: torch.Tensor) -> torch.Tensor:
        """
        Single reverse diffusion step: p(x_{t-1} | x_t)
        
        This removes noise from the image using the predicted noise.
        In practice, the predicted noise comes from a neural network.
        
        Args:
            x_t: Noisy image at timestep t
            t: Current timestep
            predicted_noise: Noise predicted by the model
            
        Returns:
            x_{t-1}: Less noisy image at timestep t-1
        """
        # Get noise schedule values
        alpha_t = self.alphas_cumprod[t]
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reverse process equation: x_{t-1} = (x_t - sqrt(1 - α_t) * ε_pred) / sqrt(α_t)
        x_prev = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
        
        logger.info(f"Reverse step {t}: denoised using predicted noise")
        
        return x_prev
    
    def demonstrate_forward_process(self, x_0: torch.Tensor, num_steps: int = 10) -> List[torch.Tensor]:
        """
        Demonstrate the forward diffusion process step by step
        
        Args:
            x_0: Original image
            num_steps: Number of steps to show
            
        Returns:
            List of images showing progressive noise addition
        """
        logger.info("Demonstrating FORWARD diffusion process...")
        logger.info("This shows how an image gradually becomes pure noise")
        
        # Select timesteps to show
        timesteps = torch.linspace(0, self.num_timesteps - 1, num_steps).long()
        
        images = [x_0.clone()]
        
        for i, t in enumerate(timesteps):
            if i == 0:
                continue  # Skip first step (already have x_0)
            
            # Add noise according to timestep t
            x_t = self.forward_process(x_0, t)
            images.append(x_t)
            
            # Calculate noise level
            noise_level = 1.0 - self.alphas_cumprod[t]
            logger.info(f"Step {i}: t={t}, noise_level={noise_level:.3f}")
        
        return images
    
    def demonstrate_reverse_process(self, x_noisy: torch.Tensor, num_steps: int = 10) -> List[torch.Tensor]:
        """
        Demonstrate the reverse diffusion process step by step
        
        Args:
            x_noisy: Noisy image to start with
            num_steps: Number of steps to show
            
        Returns:
            List of images showing progressive noise removal
        """
        logger.info("Demonstrating REVERSE diffusion process...")
        logger.info("This shows how noise is gradually removed to reveal the image")
        
        # Select timesteps to show (in reverse order)
        timesteps = torch.linspace(self.num_timesteps - 1, 0, num_steps).long()
        
        images = [x_noisy.clone()]
        x_current = x_noisy
        
        for i, t in enumerate(timesteps):
            if i == 0:
                continue  # Skip first step (already have x_noisy)
            
            # For demonstration, we'll use the actual noise that was added
            # In practice, this would be predicted by a neural network
            if i == 1:
                # Use the last noisy image to estimate noise
                estimated_noise = x_current - x_noisy
            else:
                # Use random noise for demonstration
                estimated_noise = torch.randn_like(x_current)
            
            # Remove noise according to timestep t
            x_prev = self.reverse_process_step(x_current, t, estimated_noise)
            images.append(x_prev)
            x_current = x_prev
            
            # Calculate remaining noise level
            remaining_noise = 1.0 - self.alphas_cumprod[t]
            logger.info(f"Step {i}: t={t}, remaining_noise={remaining_noise:.3f}")
        
        return images


def create_sample_image(size: int = 32, channels: int = 3) -> torch.Tensor:
    """Create a simple sample image for demonstration"""
    # Create a simple pattern (circles and lines)
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Create circular pattern
    radius = 0.3
    circle = (X**2 + Y**2) < radius**2
    
    # Create line pattern
    line = torch.abs(X + Y) < 0.1
    
    # Combine patterns
    pattern = circle.float() + line.float()
    pattern = torch.clamp(pattern, 0, 1)
    
    # Expand to batch and channels
    image = pattern.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    image = image.expand(1, channels, size, size)  # [1, C, H, W]
    
    return image


def visualize_diffusion_process(images: List[torch.Tensor], title: str, save_path: str = None):
    """Visualize the diffusion process step by step"""
    if not images:
        logger.warning("No images to visualize")
        return
    
    # Convert to numpy and normalize
    images_np = []
    for img in images:
        img_np = img.detach().cpu().numpy()[0]  # Remove batch dimension
        # Normalize to [0, 1] for visualization
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        images_np.append(img_np)
    
    # Create visualization grid
    n_images = len(images_np)
    cols = min(5, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, img in enumerate(images_np):
        row = i // cols
        col = i % cols
        
        if img.shape[0] == 3:  # RGB
            axes[row, col].imshow(np.transpose(img, (1, 2, 0)))
        else:  # Grayscale
            axes[row, col].imshow(img[0], cmap='gray')
        
        axes[row, col].set_title(f"Step {i}")
        axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(n_images, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    
    plt.show()


def demonstrate_noise_schedule():
    """Demonstrate the noise schedule (beta values)"""
    logger.info("Demonstrating noise schedule...")
    
    # Create different beta schedules
    num_timesteps = 1000
    
    # Linear schedule
    betas_linear = torch.linspace(0.0001, 0.02, num_timesteps)
    alphas_linear = 1.0 - betas_linear
    alphas_cumprod_linear = torch.cumprod(alphas_linear, dim=0)
    
    # Cosine schedule (improved DDPM)
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps)
    alphas_cumprod_cosine = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
    alphas_cumprod_cosine = alphas_cumprod_cosine / alphas_cumprod_cosine[0]
    betas_cosine = 1 - (alphas_cumprod_cosine[1:] / alphas_cumprod_cosine[:-1])
    betas_cosine = torch.clip(betas_cosine, 0.0001, 0.9999)
    
    # Plot schedules
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Beta schedules
    axes[0, 0].plot(betas_linear.cpu().numpy(), label='Linear', linewidth=2)
    axes[0, 0].plot(betas_cosine.cpu().numpy(), label='Cosine', linewidth=2)
    axes[0, 0].set_title("Beta Schedule (Noise Level)")
    axes[0, 0].set_xlabel("Timestep")
    axes[0, 0].set_ylabel("Beta")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Alpha schedules
    axes[0, 1].plot(alphas_linear.cpu().numpy(), label='Linear', linewidth=2)
    axes[0, 1].plot(alphas_cumprod_cosine.cpu().numpy(), label='Cosine', linewidth=2)
    axes[0, 1].set_title("Cumulative Alpha Schedule")
    axes[0, 1].set_xlabel("Timestep")
    axes[0, 1].set_ylabel("Cumulative Alpha")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Noise level over time
    noise_level_linear = 1.0 - alphas_cumprod_linear
    noise_level_cosine = 1.0 - alphas_cumprod_cosine
    
    axes[1, 0].plot(noise_level_linear.cpu().numpy(), label='Linear', linewidth=2)
    axes[1, 0].plot(noise_level_cosine.cpu().numpy(), label='Cosine', linewidth=2)
    axes[1, 0].set_title("Noise Level Over Time")
    axes[1, 0].set_xlabel("Timestep")
    axes[1, 0].set_ylabel("Noise Level")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Step-by-step noise addition
    step_noise_linear = torch.sqrt(1.0 - alphas_cumprod_linear)
    step_noise_cosine = torch.sqrt(1.0 - alphas_cumprod_cosine)
    
    axes[1, 1].plot(step_noise_linear.cpu().numpy(), label='Linear', linewidth=2)
    axes[1, 1].plot(step_noise_cosine.cpu().numpy(), label='Cosine', linewidth=2)
    axes[1, 1].set_title("Step Noise Scale")
    axes[1, 1].set_xlabel("Timestep")
    axes[1, 1].set_ylabel("Noise Scale")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig("./noise_schedule_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("Noise schedule visualization completed")


def main():
    """Main demonstration function"""
    logger.info("🚀 Starting Diffusion Processes Demonstration")
    logger.info("=" * 60)
    
    # 1. Demonstrate noise schedule
    logger.info("\n📊 STEP 1: Understanding the Noise Schedule")
    demonstrate_noise_schedule()
    
    # 2. Create sample image
    logger.info("\n🖼️  STEP 2: Creating Sample Image")
    image_size = 64
    sample_image = create_sample_image(size=image_size, channels=3)
    logger.info(f"Created sample image with shape: {sample_image.shape}")
    
    # 3. Initialize diffusion demo
    logger.info("\n⚙️  STEP 3: Initializing Diffusion Process")
    diffusion_demo = SimpleDiffusionDemo(num_timesteps=100)
    
    # 4. Demonstrate forward process
    logger.info("\n➡️  STEP 4: Forward Diffusion Process")
    logger.info("Adding noise gradually to the image...")
    forward_images = diffusion_demo.demonstrate_forward_process(sample_image, num_steps=10)
    
    # Visualize forward process
    visualize_diffusion_process(
        forward_images, 
        "Forward Diffusion: Image → Noise",
        save_path="./forward_diffusion_demo.png"
    )
    
    # 5. Demonstrate reverse process
    logger.info("\n⬅️  STEP 5: Reverse Diffusion Process")
    logger.info("Removing noise gradually from the image...")
    reverse_images = diffusion_demo.demonstrate_reverse_process(
        forward_images[-1],  # Start with the noisiest image
        num_steps=10
    )
    
    # Visualize reverse process
    visualize_diffusion_process(
        reverse_images, 
        "Reverse Diffusion: Noise → Image",
        save_path="./reverse_diffusion_demo.png"
    )
    
    # 6. Summary and key insights
    logger.info("\n📋 SUMMARY: Key Insights About Diffusion Processes")
    logger.info("=" * 60)
    
    logger.info("🔍 FORWARD PROCESS (Training):")
    logger.info("   • Gradually adds noise to images according to a schedule")
    logger.info("   • The noise schedule (betas) determines how much noise to add at each step")
    logger.info("   • At the end, the image becomes pure random noise")
    logger.info("   • This process is deterministic given the noise")
    
    logger.info("\n🔍 REVERSE PROCESS (Inference):")
    logger.info("   • Starts with pure random noise")
    logger.info("   • Gradually removes noise step by step")
    logger.info("   • Each step uses a neural network to predict the noise")
    logger.info("   • The predicted noise is used to denoise the image")
    
    logger.info("\n🔍 KEY MATHEMATICAL CONCEPTS:")
    logger.info("   • q(x_t | x_0): Forward process distribution")
    logger.info("   • p(x_{t-1} | x_t): Reverse process distribution")
    logger.info("   • α_t: Noise schedule parameter (1 - β_t)")
    logger.info("   • ε: Random noise added during forward process")
    logger.info("   • ε_θ: Noise predicted by neural network during reverse process")
    
    logger.info("\n🔍 WHY THIS WORKS:")
    logger.info("   • The forward process is designed to be reversible")
    logger.info("   • The noise schedule is carefully chosen to allow gradual denoising")
    logger.info("   • The neural network learns to predict noise by observing noisy images")
    logger.info("   • During inference, we can generate new images by starting with noise")
    
    logger.info("\n✅ Demonstration completed successfully!")
    logger.info("Check the generated visualization files:")
    logger.info("   • noise_schedule_comparison.png")
    logger.info("   • forward_diffusion_demo.png")
    logger.info("   • reverse_diffusion_demo.png")


if __name__ == "__main__":
    main()
