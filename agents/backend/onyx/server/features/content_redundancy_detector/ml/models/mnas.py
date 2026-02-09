"""
Mobile Neural Architecture Search (MNAS) Implementation
Following PyTorch best practices for neural architecture search
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
import random
import numpy as np

from .base import BaseModel
from .mobilenet import MobileNetV2, MobileNetV3, InvertedResidual

logger = logging.getLogger(__name__)


class SearchSpace:
    """
    Defines the search space for MNAS
    Different configurations of MobileNet blocks
    """
    
    def __init__(self):
        # Kernel sizes
        self.kernel_sizes = [3, 5, 7]
        
        # Expansion ratios
        self.expansion_ratios = [3, 4, 6]
        
        # Channel widths
        self.channel_widths = [16, 24, 32, 40, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256]
        
        # Number of blocks
        self.num_blocks = [1, 2, 3, 4]
        
        # Strides
        self.strides = [1, 2]
    
    def sample_config(self) -> Dict[str, Any]:
        """Sample a random configuration from search space"""
        return {
            "kernel_size": random.choice(self.kernel_sizes),
            "expansion_ratio": random.choice(self.expansion_ratios),
            "channel_width": random.choice(self.channel_widths),
            "num_blocks": random.choice(self.num_blocks),
            "stride": random.choice(self.strides),
        }
    
    def get_all_configs(self, max_configs: int = 100) -> List[Dict[str, Any]]:
        """Get all possible configurations (limited)"""
        configs = []
        for ks in self.kernel_sizes:
            for er in self.expansion_ratios:
                for cw in self.channel_widths[:8]:  # Limit channel widths
                    for nb in self.num_blocks:
                        for s in self.strides:
                            configs.append({
                                "kernel_size": ks,
                                "expansion_ratio": er,
                                "channel_width": cw,
                                "num_blocks": nb,
                                "stride": s,
                            })
                            if len(configs) >= max_configs:
                                return configs
        return configs


class MNASBlock(nn.Module):
    """
    MNAS Searchable Block
    Can be configured with different kernel sizes and expansion ratios
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        expansion_ratio: int = 6,
        stride: int = 1,
        norm_layer: Optional[callable] = None,
    ):
        super(MNASBlock, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        hidden_dim = int(in_channels * expansion_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels
        
        layers = []
        if expansion_ratio != 1:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                    norm_layer(hidden_dim),
                    nn.ReLU6(inplace=True)
                )
            )
        
        # Depthwise convolution with configurable kernel size
        padding = (kernel_size - 1) // 2
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    padding,
                    groups=hidden_dim,
                    bias=False
                ),
                norm_layer(hidden_dim),
                nn.ReLU6(inplace=True)
            )
        )
        
        # Pointwise linear
        layers.append(
            nn.Sequential(
                nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                norm_layer(out_channels)
            )
        )
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MNASNet(nn.Module):
    """
    MNASNet Architecture
    Neural architecture search optimized MobileNet
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
        config: Optional[Dict[str, Any]] = None,
        width_mult: float = 1.0,
        norm_layer: Optional[callable] = None,
    ):
        """
        Initialize MNASNet
        
        Args:
            num_classes: Number of output classes
            config: Architecture configuration (if None, uses default)
            width_mult: Width multiplier
            norm_layer: Normalization layer
        """
        super(MNASNet, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        if config is None:
            # Default MNASNet-A1 configuration
            config = {
                "blocks": [
                    {"kernel_size": 3, "expansion": 3, "channels": 16, "num_blocks": 1, "stride": 1},
                    {"kernel_size": 3, "expansion": 3, "channels": 24, "num_blocks": 2, "stride": 2},
                    {"kernel_size": 5, "expansion": 3, "channels": 40, "num_blocks": 3, "stride": 2},
                    {"kernel_size": 5, "expansion": 6, "channels": 80, "num_blocks": 4, "stride": 2},
                    {"kernel_size": 3, "expansion": 6, "channels": 96, "num_blocks": 2, "stride": 1},
                    {"kernel_size": 5, "expansion": 6, "channels": 192, "num_blocks": 4, "stride": 2},
                    {"kernel_size": 5, "expansion": 6, "channels": 320, "num_blocks": 1, "stride": 1},
                ]
            }
        
        # First layer
        input_channel = 32
        features = [
            nn.Sequential(
                nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
                norm_layer(input_channel),
                nn.ReLU6(inplace=True)
            )
        ]
        
        # Build blocks from config
        for block_config in config["blocks"]:
            kernel_size = block_config["kernel_size"]
            expansion = block_config["expansion"]
            channels = int(block_config["channels"] * width_mult)
            num_blocks = block_config["num_blocks"]
            stride = block_config["stride"]
            
            for i in range(num_blocks):
                block_stride = stride if i == 0 else 1
                features.append(
                    MNASBlock(
                        input_channel,
                        channels,
                        kernel_size=kernel_size,
                        expansion_ratio=expansion,
                        stride=block_stride,
                        norm_layer=norm_layer
                    )
                )
                input_channel = channels
        
        # Last layer
        last_channel = int(1280 * width_mult)
        features.append(
            nn.Sequential(
                nn.Conv2d(input_channel, last_channel, 1, 1, 0, bias=False),
                norm_layer(last_channel),
                nn.ReLU6(inplace=True)
            )
        )
        
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MNASModel(BaseModel):
    """
    MNAS Model Wrapper
    Implements neural architecture search for MobileNet variants
    """
    
    def __init__(
        self,
        model_name: str = "mnasnet",
        num_classes: int = 1000,
        width_mult: float = 1.0,
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize MNAS model
        
        Args:
            model_name: Model identifier
            num_classes: Number of output classes
            width_mult: Width multiplier
            device: PyTorch device
            use_mixed_precision: Use mixed precision
            config: Architecture configuration
        """
        super().__init__(model_name, device, use_mixed_precision)
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.config = config
        self.search_space = SearchSpace()
    
    async def load(self) -> None:
        """Load MNAS model"""
        if self.is_loaded:
            return
        
        try:
            logger.info(f"Loading MNAS model: {self.model_name}")
            
            self.model = MNASNet(
                num_classes=self.num_classes,
                config=self.config,
                width_mult=self.width_mult
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.is_loaded = True
            logger.info(f"Successfully loaded MNAS model")
        except Exception as e:
            logger.error(f"Error loading MNAS model: {e}", exc_info=True)
            raise
    
    async def predict(
        self,
        inputs: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, Any]:
        """
        Run inference on input tensor
        
        Args:
            inputs: Input tensor (B, C, H, W)
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with predictions and optionally features
        """
        if not self.is_loaded:
            await self.load()
        
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(f"Inputs must be torch.Tensor, got {type(inputs)}")
        
        inputs = inputs.to(self.device)
        
        try:
            with torch.no_grad():
                if self.use_mixed_precision and self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        if return_features:
                            features = self.model.features(inputs)
                            features = F.adaptive_avg_pool2d(features, (1, 1))
                            features = torch.flatten(features, 1)
                else:
                    outputs = self.model(inputs)
                    if return_features:
                        features = self.model.features(inputs)
                        features = F.adaptive_avg_pool2d(features, (1, 1))
                        features = torch.flatten(features, 1)
            
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            result = {
                "logits": outputs.cpu().numpy().tolist(),
                "probabilities": probs.cpu().numpy().tolist(),
                "predictions": preds.cpu().numpy().tolist(),
            }
            
            if return_features:
                result["features"] = features.cpu().numpy().tolist()
            
            return result
        except Exception as e:
            logger.error(f"Error in MNAS prediction: {e}", exc_info=True)
            raise
    
    def search_architecture(
        self,
        num_samples: int = 10,
        latency_weight: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Search for optimal architecture configurations
        
        Args:
            num_samples: Number of configurations to sample
            latency_weight: Weight for latency in objective function
            
        Returns:
            List of configurations with estimated metrics
        """
        configs = []
        for _ in range(num_samples):
            config = self.search_space.sample_config()
            
            # Estimate model size and latency (simplified)
            estimated_params = (
                config["channel_width"] * config["num_blocks"] * 1000
            )  # Rough estimate
            estimated_latency = (
                config["kernel_size"] * config["num_blocks"] * 0.1
            )  # Rough estimate in ms
            
            configs.append({
                "config": config,
                "estimated_parameters": estimated_params,
                "estimated_latency": estimated_latency,
                "score": 1.0 / (estimated_latency * latency_weight + estimated_params * (1 - latency_weight)),
            })
        
        # Sort by score
        configs.sort(key=lambda x: x["score"], reverse=True)
        return configs
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_type": "mnas",
            "num_classes": self.num_classes,
            "width_mult": self.width_mult,
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "config": self.config,
        }



