"""
MobileNet Configuration
Configuration management for MobileNet architectures
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


class MobileNetVariant(Enum):
    """MobileNet architecture variants"""
    MOBILENET_V1 = "mobilenet_v1"
    MOBILENET_V2 = "mobilenet_v2"
    MOBILENET_V3_SMALL = "mobilenet_v3_small"
    MOBILENET_V3_LARGE = "mobilenet_v3_large"


@dataclass
class MobileNetConfig:
    """Configuration for MobileNet model"""
    variant: MobileNetVariant = MobileNetVariant.MOBILENET_V2
    num_classes: int = 1000
    width_mult: float = 1.0
    pretrained: bool = False
    dropout: float = 0.2
    round_nearest: int = 8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "variant": self.variant.value,
            "num_classes": self.num_classes,
            "width_mult": self.width_mult,
            "pretrained": self.pretrained,
            "dropout": self.dropout,
            "round_nearest": self.round_nearest,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MobileNetConfig":
        """Create config from dictionary"""
        variant_str = config_dict.get("variant", "mobilenet_v2")
        variant = MobileNetVariant(variant_str)
        
        return cls(
            variant=variant,
            num_classes=config_dict.get("num_classes", 1000),
            width_mult=config_dict.get("width_mult", 1.0),
            pretrained=config_dict.get("pretrained", False),
            dropout=config_dict.get("dropout", 0.2),
            round_nearest=config_dict.get("round_nearest", 8),
        )


@dataclass
class MobileNetV2Config:
    """MobileNetV2 specific configuration"""
    inverted_residual_setting: Optional[List] = None
    round_nearest: int = 8
    
    def __post_init__(self):
        """Set default inverted residual setting if not provided"""
        if self.inverted_residual_setting is None:
            # t, c, n, s
            self.inverted_residual_setting = [
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]


@dataclass
class MobileNetV3Config:
    """MobileNetV3 specific configuration"""
    reduced_tail: bool = False
    dilated: bool = False
    
    # MobileNetV3-Large configuration
    large_config: List = None
    
    # MobileNetV3-Small configuration
    small_config: List = None
    
    def __post_init__(self):
        """Set default configurations if not provided"""
        if self.large_config is None:
            # k, t, c, SE, HS, s
            self.large_config = [
                [3, 1, 16, 0, 0, 1],
                [3, 4, 24, 0, 0, 2],
                [3, 3, 24, 0, 0, 1],
                [5, 3, 40, 1, 0, 2],
                [5, 3, 40, 1, 0, 1],
                [5, 3, 40, 1, 0, 1],
                [3, 6, 80, 0, 1, 2],
                [3, 2.5, 80, 0, 1, 1],
                [3, 2.3, 80, 0, 1, 1],
                [3, 2.3, 80, 0, 1, 1],
                [3, 6, 112, 1, 1, 1],
                [3, 6, 112, 1, 1, 1],
                [5, 6, 160, 1, 1, 2],
                [5, 6, 160, 1, 1, 1],
                [5, 6, 160, 1, 1, 1],
            ]
        
        if self.small_config is None:
            # k, t, c, SE, HS, s
            self.small_config = [
                [3, 1, 16, 1, 0, 2],
                [3, 4.5, 24, 0, 0, 2],
                [3, 3.67, 24, 0, 0, 1],
                [5, 4, 40, 1, 1, 2],
                [5, 6, 40, 1, 1, 1],
                [5, 6, 40, 1, 1, 1],
                [5, 3, 48, 1, 1, 1],
                [5, 3, 48, 1, 1, 1],
                [5, 6, 96, 1, 1, 2],
                [5, 6, 96, 1, 1, 1],
                [5, 6, 96, 1, 1, 1],
            ]


@dataclass
class TrainingConfig:
    """Training configuration"""
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 50
    weight_decay: float = 0.0001
    momentum: float = 0.9
    gradient_clip: Optional[float] = 1.0
    early_stopping_patience: Optional[int] = 5
    lr_scheduler_step: int = 30
    lr_scheduler_gamma: float = 0.1
    use_mixed_precision: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "gradient_clip": self.gradient_clip,
            "early_stopping_patience": self.early_stopping_patience,
            "lr_scheduler_step": self.lr_scheduler_step,
            "lr_scheduler_gamma": self.lr_scheduler_gamma,
            "use_mixed_precision": self.use_mixed_precision,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }



