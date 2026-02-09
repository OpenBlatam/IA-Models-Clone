"""
GAN Components for TruthGPT API
==============================

TensorFlow-like GAN implementation with improved structure and maintainability.

Refactored to:
- Extract magic numbers to named constants
- Create helper functions for layer construction
- Improve code organization and separation of concerns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

# Default hyperparameters
DEFAULT_LATENT_DIM = 100
DEFAULT_IMG_CHANNELS = 3
DEFAULT_IMG_SIZE = 64
DEFAULT_HIDDEN_DIM = 64

# Activation parameters
LEAKY_RELU_SLOPE = 0.2
CONV_KERNEL_SIZE = 3
CONV_STRIDE = 1
CONV_PADDING = 1
DISC_CONV_KERNEL_SIZE = 4
DISC_CONV_STRIDE = 2
DISC_CONV_PADDING = 1

# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def _create_generator_block(
    in_channels: int,
    out_channels: int,
    use_batch_norm: bool = True,
    use_upsample: bool = False
) -> nn.Module:
    """
    Create a generator block with optional upsampling and batch normalization.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        use_batch_norm: Whether to use batch normalization
        use_upsample: Whether to use upsampling
        
    Returns:
        Sequential block of layers
    """
    layers = []
    
    if use_upsample:
        layers.append(nn.Upsample(scale_factor=2))
    
    layers.append(
        nn.Conv2d(in_channels, out_channels, CONV_KERNEL_SIZE, 
                  stride=CONV_STRIDE, padding=CONV_PADDING)
    )
    
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    
    layers.append(nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True))
    
    return nn.Sequential(*layers)


def _create_discriminator_block(
    in_channels: int,
    out_channels: int,
    use_batch_norm: bool = True
) -> nn.Module:
    """
    Create a discriminator block with optional batch normalization.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        use_batch_norm: Whether to use batch normalization
        
    Returns:
        Sequential block of layers
    """
    layers = [
        nn.Conv2d(in_channels, out_channels, DISC_CONV_KERNEL_SIZE,
                  stride=DISC_CONV_STRIDE, padding=DISC_CONV_PADDING)
    ]
    
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    
    layers.append(nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True))
    
    return nn.Sequential(*layers)


# ════════════════════════════════════════════════════════════════════════════
# GENERATOR
# ════════════════════════════════════════════════════════════════════════════

class Generator(nn.Module):
    """
    Generator for GAN.
    
    Similar to tf.keras.models.Sequential, this class
    implements a generator network for GANs.
    
    Architecture:
    - Linear layer to project noise to feature maps
    - Series of upsampling and convolutional blocks
    - Final convolutional layer with Tanh activation
    """
    
    def __init__(self, 
                 latent_dim: int = DEFAULT_LATENT_DIM,
                 img_channels: int = DEFAULT_IMG_CHANNELS,
                 img_size: int = DEFAULT_IMG_SIZE,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 name: Optional[str] = None):
        """
        Initialize Generator.
        
        Args:
            latent_dim: Dimension of latent space
            img_channels: Number of image channels
            img_size: Size of generated images
            hidden_dim: Hidden dimension
            name: Optional name for the model
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        self.name = name or "generator"
        
        # Calculate the initial size after reshaping
        self.init_size = img_size // 4
        
        # Linear layer to project noise to feature maps
        self.l1 = nn.Linear(
            latent_dim, 
            hidden_dim * 8 * self.init_size * self.init_size
        )
        
        # Convolutional blocks using helper function
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(hidden_dim * 8),
            _create_generator_block(hidden_dim * 8, hidden_dim * 4, use_upsample=True),
            _create_generator_block(hidden_dim * 4, hidden_dim * 2, use_upsample=True),
            _create_generator_block(hidden_dim * 2, hidden_dim, use_upsample=False),
            nn.Conv2d(hidden_dim, img_channels, CONV_KERNEL_SIZE,
                     stride=CONV_STRIDE, padding=CONV_PADDING),
            nn.Tanh()
        )
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            noise: Input noise tensor of shape (batch_size, latent_dim)
            
        Returns:
            Generated images of shape (batch_size, img_channels, img_size, img_size)
        """
        out = self.l1(noise)
        out = out.view(out.shape[0], self.hidden_dim * 8, 
                      self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
    
    def __repr__(self) -> str:
        return (f"Generator(latent_dim={self.latent_dim}, "
                f"img_channels={self.img_channels}, img_size={self.img_size})")


# ════════════════════════════════════════════════════════════════════════════
# DISCRIMINATOR
# ════════════════════════════════════════════════════════════════════════════

class Discriminator(nn.Module):
    """
    Discriminator for GAN.
    
    Similar to tf.keras.models.Sequential, this class
    implements a discriminator network for GANs.
    
    Architecture:
    - Series of convolutional blocks with downsampling
    - Fully connected layer with sigmoid activation
    """
    
    def __init__(self, 
                 img_channels: int = DEFAULT_IMG_CHANNELS,
                 img_size: int = DEFAULT_IMG_SIZE,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 name: Optional[str] = None):
        """
        Initialize Discriminator.
        
        Args:
            img_channels: Number of image channels
            img_size: Size of input images
            hidden_dim: Hidden dimension
            name: Optional name for the model
        """
        super().__init__()
        
        self.img_channels = img_channels
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        self.name = name or "discriminator"
        
        # Convolutional blocks using helper function
        self.conv_blocks = nn.Sequential(
            # First block without batch norm
            nn.Conv2d(img_channels, hidden_dim, DISC_CONV_KERNEL_SIZE,
                     stride=DISC_CONV_STRIDE, padding=DISC_CONV_PADDING),
            nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True),
            
            # Subsequent blocks with batch norm
            _create_discriminator_block(hidden_dim, hidden_dim * 2),
            _create_discriminator_block(hidden_dim * 2, hidden_dim * 4),
            _create_discriminator_block(hidden_dim * 4, hidden_dim * 8),
        )
        
        # Calculate the size after convolutions
        final_size = img_size // 16
        self.adv_layer = nn.Sequential(
            nn.Linear(hidden_dim * 8 * final_size * final_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            img: Input images of shape (batch_size, img_channels, img_size, img_size)
            
        Returns:
            Discriminator output of shape (batch_size, 1)
        """
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
    
    def __repr__(self) -> str:
        return (f"Discriminator(img_channels={self.img_channels}, "
                f"img_size={self.img_size})")


# ════════════════════════════════════════════════════════════════════════════
# GAN MODEL
# ════════════════════════════════════════════════════════════════════════════

class GAN(nn.Module):
    """
    Complete GAN model.
    
    Similar to tf.keras.models.Model, this class
    implements a complete GAN with generator and discriminator.
    
    Provides methods for:
    - Generating samples
    - Training generator and discriminator separately
    - Forward pass through generator
    """
    
    def __init__(self, 
                 latent_dim: int = DEFAULT_LATENT_DIM,
                 img_channels: int = DEFAULT_IMG_CHANNELS,
                 img_size: int = DEFAULT_IMG_SIZE,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 name: Optional[str] = None):
        """
        Initialize GAN.
        
        Args:
            latent_dim: Dimension of latent space
            img_channels: Number of image channels
            img_size: Size of images
            hidden_dim: Hidden dimension
            name: Optional name for the model
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        self.name = name or "gan"
        
        # Create generator and discriminator
        self.generator = Generator(
            latent_dim=latent_dim,
            img_channels=img_channels,
            img_size=img_size,
            hidden_dim=hidden_dim
        )
        
        self.discriminator = Discriminator(
            img_channels=img_channels,
            img_size=img_size,
            hidden_dim=hidden_dim
        )
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through generator.
        
        Args:
            noise: Input noise tensor of shape (batch_size, latent_dim)
            
        Returns:
            Generated images of shape (batch_size, img_channels, img_size, img_size)
        """
        return self.generator(noise)
    
    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate samples from random noise.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate on
            
        Returns:
            Generated samples of shape (num_samples, img_channels, img_size, img_size)
        """
        noise = torch.randn(num_samples, self.latent_dim, device=device)
        return self.generator(noise)
    
    def discriminate(self, img: torch.Tensor) -> torch.Tensor:
        """
        Discriminate images (real or fake).
        
        Args:
            img: Input images of shape (batch_size, img_channels, img_size, img_size)
            
        Returns:
            Discriminator output of shape (batch_size, 1)
        """
        return self.discriminator(img)
    
    def train_generator(self, 
                       optimizer_g: torch.optim.Optimizer,
                       batch_size: int,
                       device: torch.device) -> float:
        """
        Train generator for one step.
        
        Args:
            optimizer_g: Generator optimizer
            batch_size: Batch size
            device: Device to train on
            
        Returns:
            Generator loss value
        """
        # Generate fake images
        noise = torch.randn(batch_size, self.latent_dim, device=device)
        fake_images = self.generator(noise)
        
        # Discriminate fake images
        fake_validity = self.discriminator(fake_images)
        
        # Generator loss (want discriminator to think fake images are real)
        real_labels = torch.ones(batch_size, 1, device=device)
        g_loss = self.adversarial_loss(fake_validity, real_labels)
        
        # Backward pass
        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()
        
        return g_loss.item()
    
    def train_discriminator(self, 
                           optimizer_d: torch.optim.Optimizer,
                           real_images: torch.Tensor,
                           batch_size: int,
                           device: torch.device) -> Tuple[float, float, float]:
        """
        Train discriminator for one step.
        
        Args:
            optimizer_d: Discriminator optimizer
            real_images: Real images tensor
            batch_size: Batch size
            device: Device to train on
            
        Returns:
            Tuple of (discriminator loss, real loss, fake loss)
        """
        # Real images
        real_validity = self.discriminator(real_images)
        real_labels = torch.ones(real_images.size(0), 1, device=device)
        real_loss = self.adversarial_loss(real_validity, real_labels)
        
        # Fake images
        noise = torch.randn(batch_size, self.latent_dim, device=device)
        fake_images = self.generator(noise).detach()
        fake_validity = self.discriminator(fake_images)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        fake_loss = self.adversarial_loss(fake_validity, fake_labels)
        
        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2
        
        # Backward pass
        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()
        
        return d_loss.item(), real_loss.item(), fake_loss.item()
    
    def __repr__(self) -> str:
        return (f"GAN(latent_dim={self.latent_dim}, "
                f"img_channels={self.img_channels}, img_size={self.img_size})")
