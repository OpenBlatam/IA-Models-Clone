"""
GAN Components for TruthGPT API
==============================

TensorFlow-like GAN implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class Generator(nn.Module):
    """
    Generator for GAN.
    
    Similar to tf.keras.models.Sequential, this class
    implements a generator network for GANs.
    """
    
    def __init__(self, 
                 latent_dim: int = 100,
                 img_channels: int = 3,
                 img_size: int = 64,
                 hidden_dim: int = 64,
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
        self.l1 = nn.Linear(latent_dim, hidden_dim * 8 * self.init_size * self.init_size)
        
        # Convolutional layers
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(hidden_dim * 8),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_dim * 8, hidden_dim * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, img_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            noise: Input noise tensor
            
        Returns:
            Generated images
        """
        out = self.l1(noise)
        out = out.view(out.shape[0], self.hidden_dim * 8, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
    
    def __repr__(self):
        return f"Generator(latent_dim={self.latent_dim}, img_channels={self.img_channels}, img_size={self.img_size})"


class Discriminator(nn.Module):
    """
    Discriminator for GAN.
    
    Similar to tf.keras.models.Sequential, this class
    implements a discriminator network for GANs.
    """
    
    def __init__(self, 
                 img_channels: int = 3,
                 img_size: int = 64,
                 hidden_dim: int = 64,
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
        
        # Convolutional layers
        self.conv_blocks = nn.Sequential(
            # Input: (img_channels, img_size, img_size)
            nn.Conv2d(img_channels, hidden_dim, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Calculate the size after convolutions
        self.adv_layer = nn.Sequential(
            nn.Linear(hidden_dim * 8 * (img_size // 16) * (img_size // 16), 1),
            nn.Sigmoid()
        )
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            img: Input images
            
        Returns:
            Discriminator output
        """
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
    
    def __repr__(self):
        return f"Discriminator(img_channels={self.img_channels}, img_size={self.img_size})"


class GAN(nn.Module):
    """
    Complete GAN model.
    
    Similar to tf.keras.models.Model, this class
    implements a complete GAN with generator and discriminator.
    """
    
    def __init__(self, 
                 latent_dim: int = 100,
                 img_channels: int = 3,
                 img_size: int = 64,
                 hidden_dim: int = 64,
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
        Forward pass.
        
        Args:
            noise: Input noise tensor
            
        Returns:
            Generated images
        """
        return self.generator(noise)
    
    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate samples.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate on
            
        Returns:
            Generated samples
        """
        noise = torch.randn(num_samples, self.latent_dim, device=device)
        return self.generator(noise)
    
    def discriminate(self, img: torch.Tensor) -> torch.Tensor:
        """
        Discriminate images.
        
        Args:
            img: Input images
            
        Returns:
            Discriminator output
        """
        return self.discriminator(img)
    
    def train_generator(self, 
                       optimizer_g: torch.optim.Optimizer,
                       batch_size: int,
                       device: torch.device) -> float:
        """
        Train generator.
        
        Args:
            optimizer_g: Generator optimizer
            batch_size: Batch size
            device: Device to train on
            
        Returns:
            Generator loss
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
                           device: torch.device) -> Tuple[float, float]:
        """
        Train discriminator.
        
        Args:
            optimizer_d: Discriminator optimizer
            real_images: Real images
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
    
    def __repr__(self):
        return f"GAN(latent_dim={self.latent_dim}, img_channels={self.img_channels}, img_size={self.img_size})"


