"""
MobileNet Factory
Factory pattern for creating MobileNet models
"""

import torch
import logging
from typing import Optional, Dict, Any

from .architectures import MobileNetV2, MobileNetV3
from .config import MobileNetConfig, MobileNetVariant, MobileNetV2Config, MobileNetV3Config
from .utils import get_device

logger = logging.getLogger(__name__)


class MobileNetFactory:
    """
    Factory for creating MobileNet models
    """
    
    @staticmethod
    def create_model(
        config: MobileNetConfig,
        device: Optional[torch.device] = None,
    ) -> torch.nn.Module:
        """
        Create MobileNet model from configuration
        
        Args:
            config: MobileNet configuration
            device: Target device
            
        Returns:
            MobileNet model
        """
        if device is None:
            device = get_device()
        
        model = None
        
        if config.variant == MobileNetVariant.MOBILENET_V2:
            model = MobileNetV2(
                num_classes=config.num_classes,
                width_mult=config.width_mult,
            )
        elif config.variant == MobileNetVariant.MOBILENET_V3_LARGE:
            model = MobileNetV3(
                num_classes=config.num_classes,
                width_mult=config.width_mult,
                variant="large",
            )
        elif config.variant == MobileNetVariant.MOBILENET_V3_SMALL:
            model = MobileNetV3(
                num_classes=config.num_classes,
                width_mult=config.width_mult,
                variant="small",
            )
        else:
            raise ValueError(f"Unsupported MobileNet variant: {config.variant}")
        
        # Move to device
        model = model.to(device)
        
        # Load pretrained weights if requested
        if config.pretrained:
            MobileNetFactory._load_pretrained(model, config.variant, device)
        
        logger.info(f"Created {config.variant.value} model on {device}")
        return model
    
    @staticmethod
    def _load_pretrained(
        model: torch.nn.Module,
        variant: MobileNetVariant,
        device: torch.device,
    ) -> None:
        """
        Load pretrained weights
        
        Args:
            model: Model to load weights into
            variant: Model variant
            device: Target device
        """
        try:
            import torchvision.models as models
            
            if variant == MobileNetVariant.MOBILENET_V2:
                pretrained_model = models.mobilenet_v2(pretrained=True)
            elif variant == MobileNetVariant.MOBILENET_V3_LARGE:
                pretrained_model = models.mobilenet_v3_large(pretrained=True)
            elif variant == MobileNetVariant.MOBILENET_V3_SMALL:
                pretrained_model = models.mobilenet_v3_small(pretrained=True)
            else:
                logger.warning(f"No pretrained weights available for {variant}")
                return
            
            # Load state dict (with flexible matching)
            model_dict = model.state_dict()
            pretrained_dict = pretrained_model.state_dict()
            
            # Filter out incompatible keys
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }
            
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            logger.info(f"Loaded pretrained weights for {variant.value}")
            
        except ImportError:
            logger.warning("torchvision not available, skipping pretrained weights")
        except Exception as e:
            logger.warning(f"Could not load pretrained weights: {e}")
    
    @staticmethod
    def create_from_dict(
        config_dict: Dict[str, Any],
        device: Optional[torch.device] = None,
    ) -> torch.nn.Module:
        """
        Create model from dictionary configuration
        
        Args:
            config_dict: Configuration dictionary
            device: Target device
            
        Returns:
            MobileNet model
        """
        config = MobileNetConfig.from_dict(config_dict)
        return MobileNetFactory.create_model(config, device)
    
    @staticmethod
    def get_default_config(variant: MobileNetVariant) -> MobileNetConfig:
        """
        Get default configuration for variant
        
        Args:
            variant: Model variant
            
        Returns:
            Default configuration
        """
        return MobileNetConfig(variant=variant)



