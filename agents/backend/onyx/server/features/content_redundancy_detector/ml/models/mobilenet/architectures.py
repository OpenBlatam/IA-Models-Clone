"""
MobileNet Architectures
Clean, modular architecture definitions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from .blocks import InvertedResidual, ConvBNReLU, SEBlock, HardSwish
from .utils import _make_divisible, initialize_weights
from .config import MobileNetV2Config, MobileNetV3Config


class MobileNetV2(nn.Module):
    """
    MobileNetV2 Architecture
    Efficient CNN architecture for mobile and edge devices
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        config: Optional[MobileNetV2Config] = None,
        norm_layer: Optional[callable] = None,
    ):
        """
        Initialize MobileNetV2
        
        Args:
            num_classes: Number of output classes
            width_mult: Width multiplier
            config: MobileNetV2 configuration
            norm_layer: Normalization layer factory
        """
        super(MobileNetV2, self).__init__()
        
        if config is None:
            config = MobileNetV2Config()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        input_channel = 32
        last_channel = 1280
        
        # Building first layer
        input_channel = _make_divisible(input_channel * width_mult, config.round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), config.round_nearest)
        
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        
        # Building inverted residual blocks
        for t, c, n, s in config.inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, config.round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer
                    )
                )
                input_channel = output_channel
        
        # Building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
        
        # Initialize weights
        initialize_weights(self)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MobileNetV3(nn.Module):
    """
    MobileNetV3 Architecture
    Improved version with better accuracy-efficiency trade-off
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        variant: str = "large",
        config: Optional[MobileNetV3Config] = None,
        norm_layer: Optional[callable] = None,
    ):
        """
        Initialize MobileNetV3
        
        Args:
            num_classes: Number of output classes
            width_mult: Width multiplier
            variant: 'large' or 'small'
            config: MobileNetV3 configuration
            norm_layer: Normalization layer factory
        """
        super(MobileNetV3, self).__init__()
        
        if config is None:
            config = MobileNetV3Config()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        input_channel = 16
        last_channel = 1280
        
        if config.reduced_tail:
            last_channel = _make_divisible(last_channel * 0.75, 8)
        
        # Select configuration based on variant
        if variant == "large":
            inverted_residual_setting = config.large_config
        else:
            inverted_residual_setting = config.small_config
        
        # Build features
        features = []
        firstconv_output_channel = _make_divisible(input_channel * width_mult, 8)
        features.append(ConvBNReLU(3, firstconv_output_channel, stride=2, norm_layer=norm_layer))
        
        input_channel = firstconv_output_channel
        for k, t, c, use_se, use_hs, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            
            # Select activation
            activation = HardSwish() if use_hs else nn.ReLU6(inplace=True)
            
            # Build block
            block = InvertedResidual(
                input_channel,
                output_channel,
                s,
                expand_ratio=t,
                norm_layer=norm_layer,
                activation=activation
            )
            
            # Add SE block if needed
            if use_se:
                se = SEBlock(exp_size)
                # Insert SE block after expansion
                # This is simplified - in full implementation, SE would be integrated into block
                features.append(block)
            else:
                features.append(block)
            
            input_channel = output_channel
        
        # Building last several layers
        lastconv_input_channel = input_channel
        lastconv_output_channel = _make_divisible(6 * lastconv_input_channel, 8)
        features.append(ConvBNReLU(
            lastconv_input_channel,
            lastconv_output_channel,
            kernel_size=1,
            norm_layer=norm_layer,
            activation=HardSwish()
        ))
        
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channel, last_channel),
            HardSwish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, num_classes),
        )
        
        # Initialize weights
        initialize_weights(self)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



