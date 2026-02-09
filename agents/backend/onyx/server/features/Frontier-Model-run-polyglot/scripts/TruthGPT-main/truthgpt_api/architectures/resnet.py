"""
ResNet Architecture for TruthGPT API
===================================

TensorFlow-like ResNet implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Callable


class BasicBlock(nn.Module):
    """Basic ResNet block."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """Bottleneck ResNet block."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 4)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels // 4, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels // 4)
        self.conv3 = nn.Conv2d(out_channels // 4, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    """
    ResNet architecture.
    
    Similar to tf.keras.applications.ResNet, this class
    implements the ResNet architecture.
    """
    
    def __init__(self, 
                 block: Callable,
                 layers: List[int],
                 num_classes: int = 1000,
                 input_channels: int = 3,
                 name: Optional[str] = None):
        """
        Initialize ResNet.
        
        Args:
            block: Block type (BasicBlock or Bottleneck)
            layers: Number of blocks in each layer
            num_classes: Number of output classes
            input_channels: Number of input channels
            name: Optional name for the model
        """
        super().__init__()
        
        self.block = block
        self.layers = layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.name = name or f"resnet_{sum(layers)}"
        
        # Initial layers
        self.conv1 = nn.Conv2d(input_channels, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block: Callable, channels: int, blocks: int, 
                   stride: int = 1) -> nn.Sequential:
        """Make a ResNet layer."""
        downsample = None
        if stride != 1 or channels != 64:
            downsample = nn.Sequential(
                nn.Conv2d(64, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels)
            )
        
        layers = []
        layers.append(block(64, channels, stride, downsample))
        
        for _ in range(1, blocks):
            layers.append(block(channels, channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def __repr__(self):
        return f"ResNet(block={self.block.__name__}, layers={self.layers}, num_classes={self.num_classes})"


def ResNet18(num_classes: int = 1000, input_channels: int = 3) -> ResNet:
    """ResNet-18 model."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, input_channels, "resnet18")


def ResNet34(num_classes: int = 1000, input_channels: int = 3) -> ResNet:
    """ResNet-34 model."""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, input_channels, "resnet34")


def ResNet50(num_classes: int = 1000, input_channels: int = 3) -> ResNet:
    """ResNet-50 model."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, input_channels, "resnet50")


def ResNet101(num_classes: int = 1000, input_channels: int = 3) -> ResNet:
    """ResNet-101 model."""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, input_channels, "resnet101")


def ResNet152(num_classes: int = 1000, input_channels: int = 3) -> ResNet:
    """ResNet-152 model."""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, input_channels, "resnet152")









