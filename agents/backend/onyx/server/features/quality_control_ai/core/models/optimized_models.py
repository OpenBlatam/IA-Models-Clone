"""
Optimized Models for Fast Inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    import torch.jit
    JIT_AVAILABLE = True
except ImportError:
    JIT_AVAILABLE = False


class FastAutoencoder(nn.Module):
    """Lightweight autoencoder optimized for speed"""
    
    def __init__(self, input_channels=3, latent_dim=64, input_size=(224, 224)):
        super().__init__()
        self.input_size = input_size
        
        # Lightweight encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 2, 1),  # 112x112
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),  # 56x56
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),  # 28x28
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # 4x4
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.ReLU(inplace=True)
        )
        
        # Lightweight decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8x8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16x16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, input_channels, 4, 2, 1),  # 32x32
            nn.Tanh()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        b, c, h, w = encoded.shape
        flat = encoded.view(b, -1)
        latent = self.bottleneck(flat)
        decoded = latent.view(b, c, h, w)
        recon = self.decoder(decoded)
        return recon, latent[:, :self.bottleneck[1].out_features // 2]


class QuantizedModel:
    """Wrapper for quantized models"""
    
    @staticmethod
    def quantize_model(model: nn.Module, device='cpu'):
        """Quantize model for faster inference"""
        model.eval()
        try:
            # Dynamic quantization
            quantized = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d, nn.ConvTranspose2d}, dtype=torch.qint8
            )
            return quantized
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return model
    
    @staticmethod
    def fuse_model(model: nn.Module):
        """Fuse conv-bn-relu for faster inference"""
        try:
            torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']], inplace=True)
        except:
            pass
        return model


def create_fast_autoencoder(input_channels=3, latent_dim=64, device=None):
    """Create optimized fast autoencoder"""
    model = FastAutoencoder(input_channels, latent_dim)
    if device:
        model = model.to(device)
    model.eval()
    return model


def optimize_for_inference(model: nn.Module, example_input=None):
    """Optimize model for inference"""
    model.eval()
    
    # JIT compilation if available
    if JIT_AVAILABLE and example_input is not None:
        try:
            with torch.no_grad():
                traced = torch.jit.trace(model, example_input)
                traced = torch.jit.optimize_for_inference(traced)
                logger.info("Model optimized with JIT")
                return traced
        except Exception as e:
            logger.warning(f"JIT optimization failed: {e}")
    
    return model

