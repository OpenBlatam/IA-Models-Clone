"""
PyTorch Models for Dermatology AI
Implements custom nn.Module classes following best practices
Refactored with base classes and common utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

# Import base classes
import sys
from pathlib import Path
_parent = Path(__file__).parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

try:
    from ml.common.base_classes import BaseModel, DeviceAwareMixin
    from ml.common.utils import count_parameters, get_model_size
    BASE_CLASSES_AVAILABLE = True
except ImportError:
    BASE_CLASSES_AVAILABLE = False
    # Fallback to models.base
    try:
        from models.base import BaseModel
        BASE_CLASSES_AVAILABLE = True
    except ImportError:
        BASE_CLASSES_AVAILABLE = False

logger = logging.getLogger(__name__)


class SkinAnalysisCNN(BaseModel if BASE_CLASSES_AVAILABLE else nn.Module):
    """
    Custom CNN for skin analysis
    Implements proper weight initialization and normalization
    Refactored to inherit from BaseModel
    """
    
    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        dropout_rate: float = 0.5,
        name: str = "SkinAnalysisCNN"
    ):
        if BASE_CLASSES_AVAILABLE:
            super(SkinAnalysisCNN, self).__init__(name=name)
        else:
            super(SkinAnalysisCNN, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Initialize weights
        if BASE_CLASSES_AVAILABLE:
            self.initialize_weights()
        else:
            self._initialize_weights()
    
    def _initialize_weights(self):
        """Proper weight initialization (fallback if BaseModel not available)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class SkinQualityRegressor(nn.Module):
    """
    Multi-task regressor for skin quality metrics
    Predicts multiple quality scores simultaneously
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_metrics: int = 8,
        hidden_dim: int = 512
    ):
        super(SkinQualityRegressor, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Task-specific heads
        self.heads = nn.ModuleDict({
            'overall_score': nn.Sequential(
                nn.Linear(256, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ),
            'texture_score': nn.Sequential(
                nn.Linear(256, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ),
            'hydration_score': nn.Sequential(
                nn.Linear(256, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ),
            'elasticity_score': nn.Sequential(
                nn.Linear(256, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ),
            'pigmentation_score': nn.Sequential(
                nn.Linear(256, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ),
            'pore_size_score': nn.Sequential(
                nn.Linear(256, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ),
            'wrinkles_score': nn.Sequential(
                nn.Linear(256, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ),
            'redness_score': nn.Sequential(
                nn.Linear(256, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        })
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning all metric predictions"""
        # Extract features
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        # Predict all metrics
        predictions = {}
        for metric_name, head in self.heads.items():
            predictions[metric_name] = head(features).squeeze(-1) * 100  # Scale to 0-100
        
        return predictions


class ConditionClassifier(nn.Module):
    """
    Multi-label classifier for skin conditions
    Uses sigmoid activation for multi-label classification
    """
    
    def __init__(
        self,
        num_conditions: int = 6,
        input_channels: int = 3,
        backbone: str = 'resnet18'
    ):
        super(ConditionClassifier, self).__init__()
        
        # Use pre-trained backbone
        try:
            import torchvision.models as models
            
            if backbone == 'resnet18':
                backbone_model = models.resnet18(pretrained=True)
                self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])
                feature_dim = 512
            elif backbone == 'resnet50':
                backbone_model = models.resnet50(pretrained=True)
                self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])
                feature_dim = 2048
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")
        except ImportError:
            logger.warning("torchvision not available, using simple CNN")
            self.backbone = nn.Sequential(
                nn.Conv2d(input_channels, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1)
            )
            feature_dim = 128
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_conditions),
            nn.Sigmoid()  # Multi-label classification
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)


class AttentionModule(nn.Module):
    """
    Self-attention module for improved feature extraction
    """
    
    def __init__(self, channels: int):
        super(AttentionModule, self).__init__()
        self.channels = channels
        
        self.query_conv = nn.Conv2d(channels, channels // 8, 1)
        self.key_conv = nn.Conv2d(channels, channels // 8, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention"""
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        proj_query = self.query_conv(x).view(
            batch_size, -1, height * width
        ).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, height * width)
        proj_value = self.value_conv(x).view(batch_size, -1, height * width)
        
        # Attention
        attention = self.softmax(torch.bmm(proj_query, proj_key))
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Residual connection
        out = self.gamma * out + x
        return out


class EnhancedSkinAnalyzer(nn.Module):
    """
    Enhanced skin analyzer with attention mechanism
    Combines CNN features with attention for better analysis
    """
    
    def __init__(
        self,
        num_conditions: int = 6,
        num_metrics: int = 8,
        input_channels: int = 3
    ):
        super(EnhancedSkinAnalyzer, self).__init__()
        
        # Feature extraction with attention
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.attention1 = AttentionModule(64)
        
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.attention2 = AttentionModule(128)
        
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Multi-task heads
        self.condition_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_conditions),
            nn.Sigmoid()
        )
        
        self.metric_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_metrics),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Feature extraction with attention
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.attention1(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.attention2(x)
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Multi-task predictions
        conditions = self.condition_head(x)
        metrics = self.metric_head(x) * 100  # Scale to 0-100
        
        return {
            'conditions': conditions,
            'metrics': metrics
        }

