"""
PyTorch Autoencoder Model for Anomaly Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class EncoderBlock(nn.Module):
    """Encoder block with convolution and batch normalization"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block with transposed convolution and batch normalization"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2):
        super(DecoderBlock, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            output_padding=stride - 1
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class AnomalyAutoencoder(nn.Module):
    """
    Autoencoder for anomaly detection in quality control
    
    Architecture:
    - Encoder: Convolutional layers with decreasing spatial dimensions
    - Bottleneck: Latent representation
    - Decoder: Transposed convolutions to reconstruct input
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_dim: int = 128,
        encoder_channels: Tuple[int, ...] = (64, 128, 256, 512),
        input_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize autoencoder
        
        Args:
            input_channels: Number of input channels (3 for RGB, 1 for grayscale)
            latent_dim: Dimension of latent representation
            encoder_channels: Tuple of channel sizes for encoder
            input_size: Input image size (height, width)
        """
        super(AnomalyAutoencoder, self).__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.input_size = input_size
        
        # Encoder
        encoder_layers = []
        in_ch = input_channels
        for out_ch in encoder_channels:
            encoder_layers.append(EncoderBlock(in_ch, out_ch))
            in_ch = out_ch
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate flattened size after encoder
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, *input_size)
            dummy_output = self.encoder(dummy_input)
            flattened_size = dummy_output.numel() // dummy_output.size(0)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, flattened_size),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        decoder_layers = []
        decoder_channels = list(reversed(encoder_channels))
        for i, out_ch in enumerate(decoder_channels[:-1]):
            in_ch = decoder_channels[i]
            decoder_layers.append(DecoderBlock(in_ch, out_ch))
        
        # Final decoder layer to output channels
        decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    decoder_channels[-1], input_channels,
                    kernel_size=3, stride=2, padding=1, output_padding=1
                ),
                nn.Tanh()  # Normalize to [-1, 1]
            )
        )
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"AnomalyAutoencoder initialized: input_channels={input_channels}, "
                   f"latent_dim={latent_dim}, input_size={input_size}")
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Latent representation [B, latent_dim]
        """
        encoded = self.encoder(x)
        encoded_flat = encoded.view(encoded.size(0), -1)
        latent = self.bottleneck[0](encoded_flat)  # Flatten
        latent = self.bottleneck[1](latent)  # Linear to latent_dim
        latent = self.bottleneck[2](latent)  # ReLU
        return latent
    
    def decode(self, latent: torch.Tensor, target_shape: Optional[Tuple[int, int, int, int]] = None) -> torch.Tensor:
        """
        Decode latent representation to reconstruction
        
        Args:
            latent: Latent representation [B, latent_dim]
            target_shape: Target shape [B, C, H, W] for reshaping
            
        Returns:
            Reconstructed image [B, C, H, W]
        """
        # Get through bottleneck decoder part
        x = self.bottleneck[3](latent)  # Linear from latent_dim
        x = self.bottleneck[4](x)  # ReLU
        
        # Reshape to feature map
        if target_shape is None:
            # Estimate shape (this is approximate)
            batch_size = latent.size(0)
            # Calculate approximate spatial dimensions
            h, w = self.input_size
            for _ in range(len(self.encoder)):
                h, w = h // 2, w // 2
            channels = self.encoder[-1][0].out_channels
            x = x.view(batch_size, channels, h, w)
        else:
            x = x.view(target_shape)
        
        # Decode
        reconstructed = self.decoder(x)
        return reconstructed
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Tuple of (reconstructed, latent)
        """
        # Encode
        encoded = self.encoder(x)
        encoded_shape = encoded.shape
        
        # Bottleneck
        encoded_flat = encoded.view(encoded.size(0), -1)
        x = self.bottleneck(encoded_flat)
        
        # Reshape for decoder
        x = x.view(encoded_shape)
        
        # Decode
        reconstructed = self.decoder(x)
        
        # Get latent for return
        latent = self.encode(x)
        
        return reconstructed, latent
    
    def compute_reconstruction_error(self, x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        """
        Compute reconstruction error (MSE loss)
        
        Args:
            x: Input tensor [B, C, H, W]
            reduction: Reduction method ('mean', 'sum', 'none')
            
        Returns:
            Reconstruction error
        """
        reconstructed, _ = self.forward(x)
        
        # Normalize inputs to [-1, 1] if needed
        if x.max() > 1.0:
            x = x / 255.0 * 2.0 - 1.0
        
        error = F.mse_loss(reconstructed, x, reduction='none')
        
        if reduction == 'mean':
            return error.mean()
        elif reduction == 'sum':
            return error.sum()
        else:
            return error


def create_autoencoder(
    input_channels: int = 3,
    latent_dim: int = 128,
    input_size: Tuple[int, int] = (224, 224),
    device: Optional[torch.device] = None
) -> AnomalyAutoencoder:
    """
    Factory function to create autoencoder
    
    Args:
        input_channels: Number of input channels
        latent_dim: Latent dimension
        input_size: Input image size
        device: Device to place model on
        
    Returns:
        Initialized autoencoder model
    """
    model = AnomalyAutoencoder(
        input_channels=input_channels,
        latent_dim=latent_dim,
        input_size=input_size
    )
    
    if device is not None:
        model = model.to(device)
    
    return model

