"""
MobileNet Architecture Implementation
Following PyTorch best practices with proper nn.Module structure, GPU support, and mixed precision
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

from .base import BaseModel

logger = logging.getLogger(__name__)


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    """
    Convolution-BatchNorm-ReLU block
    Following MobileNet architecture patterns
    """
    
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[callable] = None,
    ):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False
            ),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    """
    Inverted Residual Block
    Core building block of MobileNetV2/V3
    """
    
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[callable] = None,
    ):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        
        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        
        layers.extend([
            # dw
            ConvBNReLU(
                hidden_dim,
                hidden_dim,
                stride=stride,
                groups=hidden_dim,
                norm_layer=norm_layer
            ),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2 Architecture
    Efficient CNN architecture for mobile and edge devices
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[list] = None,
        round_nearest: int = 8,
        block: Optional[callable] = None,
        norm_layer: Optional[callable] = None,
    ):
        """
        MobileNet V2 main class
        
        Args:
            num_classes: Number of classes
            width_mult: Width multiplier - thin vs wide MobileNet
            inverted_residual_setting: Network structure
            round_nearest: Round the number of channels in each layer to be a multiple of this number
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()
        
        if block is None:
            block = InvertedResidual
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        input_channel = 32
        last_channel = 1280
        
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        
        # Only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                "inverted_residual_setting should be non-empty "
                "or a 4-element list, got {}".format(inverted_residual_setting)
            )
        
        # Building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]
        
        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer)
                )
                input_channel = output_channel
        
        # Building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        
        # Make it nn.Sequential
        self.features = nn.Sequential(*features)
        
        # Building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
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
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have the forward logic.
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


class MobileNetV3(nn.Module):
    """
    MobileNetV3 Architecture
    Improved version with better accuracy-efficiency trade-off
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        reduced_tail: bool = False,
        dilated: bool = False,
        norm_layer: Optional[callable] = None,
    ):
        super(MobileNetV3, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # Building first layer
        input_channel = 16
        last_channel = 1280
        
        if reduced_tail:
            last_channel = _make_divisible(last_channel * 0.75, 8)
        
        # MobileNetV3-Large configuration
        inverted_residual_setting = [
            # k, t, c, SE, HS, s
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
        
        # Build features
        features = []
        firstconv_output_channel = _make_divisible(input_channel * width_mult, 8)
        features.append(ConvBNReLU(3, firstconv_output_channel, stride=2, norm_layer=norm_layer))
        
        input_channel = firstconv_output_channel
        for k, t, c, use_se, use_hs, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            features.append(
                InvertedResidual(
                    input_channel,
                    output_channel,
                    s,
                    expand_ratio=t,
                    norm_layer=norm_layer
                )
            )
            input_channel = output_channel
        
        # Building last several layers
        lastconv_input_channel = input_channel
        lastconv_output_channel = _make_divisible(6 * lastconv_input_channel, 8)
        features.append(ConvBNReLU(lastconv_input_channel, lastconv_output_channel, kernel_size=1, norm_layer=norm_layer))
        
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channel, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(last_channel, num_classes),
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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MobileNetModel(BaseModel):
    """
    MobileNet Model Wrapper
    Implements BaseModel interface with proper GPU support and mixed precision
    """
    
    def __init__(
        self,
        model_name: str = "mobilenet_v2",
        num_classes: int = 1000,
        width_mult: float = 1.0,
        device: Optional[torch.device] = None,
        use_mixed_precision: bool = False,
        pretrained: bool = False,
    ):
        """
        Initialize MobileNet model
        
        Args:
            model_name: Model variant ('mobilenet_v2' or 'mobilenet_v3')
            num_classes: Number of output classes
            width_mult: Width multiplier for model scaling
            device: PyTorch device
            use_mixed_precision: Use mixed precision
            pretrained: Load pretrained weights
        """
        super().__init__(model_name, device, use_mixed_precision)
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.pretrained = pretrained
        self.model_variant = model_name
    
    async def load(self) -> None:
        """Load MobileNet model"""
        if self.is_loaded:
            return
        
        try:
            logger.info(f"Loading MobileNet model: {self.model_variant}")
            
            if self.model_variant == "mobilenet_v2":
                self.model = MobileNetV2(
                    num_classes=self.num_classes,
                    width_mult=self.width_mult
                )
            elif self.model_variant == "mobilenet_v3":
                self.model = MobileNetV3(
                    num_classes=self.num_classes,
                    width_mult=self.width_mult
                )
            else:
                raise ValueError(f"Unsupported MobileNet variant: {self.model_variant}")
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Load pretrained weights if available
            if self.pretrained:
                try:
                    # Try to load from torchvision
                    import torchvision.models as models
                    if self.model_variant == "mobilenet_v2":
                        pretrained_model = models.mobilenet_v2(pretrained=True)
                        self.model.load_state_dict(pretrained_model.state_dict(), strict=False)
                    elif self.model_variant == "mobilenet_v3":
                        pretrained_model = models.mobilenet_v3_large(pretrained=True)
                        self.model.load_state_dict(pretrained_model.state_dict(), strict=False)
                    logger.info("Loaded pretrained weights")
                except Exception as e:
                    logger.warning(f"Could not load pretrained weights: {e}")
            
            self.model.eval()
            self.is_loaded = True
            logger.info(f"Successfully loaded {self.model_variant}")
        except Exception as e:
            logger.error(f"Error loading MobileNet model: {e}", exc_info=True)
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
        
        # Move inputs to device
        inputs = inputs.to(self.device)
        
        try:
            with torch.no_grad():
                if self.use_mixed_precision and self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        if return_features:
                            # Extract features before classifier
                            features = self.model.features(inputs)
                            features = self.model.avgpool(features) if hasattr(self.model, 'avgpool') else F.adaptive_avg_pool2d(features, (1, 1))
                            features = torch.flatten(features, 1)
                else:
                    outputs = self.model(inputs)
                    if return_features:
                        features = self.model.features(inputs)
                        features = self.model.avgpool(features) if hasattr(self.model, 'avgpool') else F.adaptive_avg_pool2d(features, (1, 1))
                        features = torch.flatten(features, 1)
            
            # Apply softmax for probabilities
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
            logger.error(f"Error in MobileNet prediction: {e}", exc_info=True)
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_variant": self.model_variant,
            "num_classes": self.num_classes,
            "width_mult": self.width_mult,
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "pretrained": self.pretrained,
        }



