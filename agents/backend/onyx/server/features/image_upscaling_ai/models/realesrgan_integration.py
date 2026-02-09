"""
Real-ESRGAN Integration
=======================

Integration with Real-ESRGAN for high-quality image upscaling.
Based on: https://github.com/xinntao/Real-ESRGAN
"""

import logging
import os
from typing import Optional, Dict, Any, Union, Tuple
from pathlib import Path
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Try to import Real-ESRGAN
REALESRGAN_AVAILABLE = False
REALESRGAN_MODEL = None

try:
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from realesrgan.archs.srvgg_arch import SRVGGNetCompact
    REALESRGAN_AVAILABLE = True
    logger.info("Real-ESRGAN library available")
except ImportError:
    try:
        # Alternative import path
        from realesrgan import RealESRGANer
        REALESRGAN_AVAILABLE = True
        logger.info("Real-ESRGAN library available (alternative path)")
    except ImportError:
        logger.warning(
            "Real-ESRGAN not available. Install with: "
            "pip install realesrgan basicsr"
        )


class RealESRGANWrapper:
    """
    Wrapper for Real-ESRGAN model.
    
    Supports multiple pre-trained models:
    - RealESRGAN_x4plus: 4x upscaling
    - RealESRGAN_x4plus_anime_6B: 4x for anime
    - RealESRNet_x4plus: 4x (no GAN)
    - RealESRGAN_x2plus: 2x upscaling
    """
    
    # Available models
    AVAILABLE_MODELS = {
        "RealESRGAN_x4plus": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            "scale": 4,
            "description": "General 4x upscaling model"
        },
        "RealESRGAN_x4plus_anime_6B": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            "scale": 4,
            "description": "4x upscaling optimized for anime"
        },
        "RealESRNet_x4plus": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
            "scale": 4,
            "description": "4x upscaling without GAN (faster)"
        },
        "RealESRGAN_x2plus": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            "scale": 2,
            "description": "2x upscaling model"
        },
    }
    
    def __init__(
        self,
        model_name: str = "RealESRGAN_x4plus",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        scale: Optional[int] = None,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
        half: bool = True,
    ):
        """
        Initialize Real-ESRGAN wrapper.
        
        Args:
            model_name: Model name (default: RealESRGAN_x4plus)
            model_path: Path to model file (auto-download if None)
            device: Device to use (cuda/cpu, auto-detect if None)
            scale: Model scale (auto-detect from model if None)
            tile: Tile size for processing large images (0 = no tiling)
            tile_pad: Padding for tiles
            pre_pad: Pre-padding
            half: Use half precision (float16) for faster processing
        """
        if not REALESRGAN_AVAILABLE:
            raise ImportError(
                "Real-ESRGAN not available. Install with: pip install realesrgan basicsr"
            )
        
        self.model_name = model_name
        if REALESRGAN_AVAILABLE and torch is not None:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device or "cpu"
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.half = half and self.device == "cuda"
        
        # Get model info
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(self.AVAILABLE_MODELS.keys())}"
            )
        
        model_info = self.AVAILABLE_MODELS[model_name]
        self.scale = scale or model_info["scale"]
        self.model_url = model_info["url"]
        
        # Determine model path
        if model_path is None:
            # Use default model directory
            model_dir = Path.home() / ".cache" / "realesrgan" / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{model_name}.pth"
        
        self.model_path = Path(model_path)
        
        # Initialize model
        self.upsampler = None
        self._load_model()
        
        logger.info(
            f"Real-ESRGAN initialized: {model_name} (scale: {self.scale}x, device: {self.device})"
        )
    
    def _load_model(self) -> None:
        """Load Real-ESRGAN model."""
        try:
            # Check if model exists
            if not self.model_path.exists():
                logger.warning(
                    f"Model not found at {self.model_path}. "
                    f"Please download from: {self.model_url}"
                )
                raise FileNotFoundError(
                    f"Model not found: {self.model_path}. "
                    f"Download from: {self.model_url}"
                )
            
            # Determine model architecture based on name
            if "x2plus" in self.model_name:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                               num_block=23, num_grow_ch=32, scale=2)
            elif "anime" in self.model_name:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                               num_block=6, num_grow_ch=32, scale=4)
            elif "RealESRNet" in self.model_name:
                # RealESRNet (no GAN)
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                               num_block=23, num_grow_ch=32, scale=4)
            else:
                # Default RealESRGAN_x4plus
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                               num_block=23, num_grow_ch=32, scale=4)
            
            # Initialize upsampler
            self.upsampler = RealESRGANer(
                scale=self.scale,
                model_path=str(self.model_path),
                model=model,
                tile=self.tile,
                tile_pad=self.tile_pad,
                pre_pad=self.pre_pad,
                half=self.half,
                device=torch.device(self.device)
            )
            
            logger.info(f"Real-ESRGAN model loaded: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading Real-ESRGAN model: {e}")
            raise RuntimeError(f"Failed to load Real-ESRGAN model: {e}")
    
    def upscale(
        self,
        image: Union[Image.Image, np.ndarray],
        outscale: Optional[float] = None
    ) -> Image.Image:
        """
        Upscale image using Real-ESRGAN.
        
        Args:
            image: Input image (PIL Image or numpy array)
            outscale: Output scale (if different from model scale, will use multiple passes)
            
        Returns:
            Upscaled PIL Image
        """
        if self.upsampler is None:
            raise RuntimeError("Real-ESRGAN model not loaded")
        
        # Convert to numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image.convert("RGB"))
        else:
            img_array = image
        
        # Determine target scale
        target_scale = outscale or self.scale
        
        # If target scale matches model scale, single pass
        if target_scale == self.scale:
            upscaled_array, _ = self.upsampler.enhance(img_array, outscale=target_scale)
        else:
            # Multiple passes for different scales
            current_image = img_array
            remaining_scale = target_scale
            
            while remaining_scale > 1.0:
                if remaining_scale >= self.scale:
                    pass_scale = self.scale
                else:
                    # For scales < model scale, use model scale and resize
                    pass_scale = self.scale
                
                current_image, _ = self.upsampler.enhance(current_image, outscale=pass_scale)
                remaining_scale /= pass_scale
                
                # If we need to downscale
                if remaining_scale < 1.0:
                    from PIL import Image
                    pil_img = Image.fromarray(current_image)
                    target_size = (
                        int(pil_img.width * remaining_scale),
                        int(pil_img.height * remaining_scale)
                    )
                    current_image = np.array(pil_img.resize(target_size, Image.Resampling.LANCZOS))
                    remaining_scale = 1.0
        
        # Convert back to PIL Image
        upscaled_image = Image.fromarray(upscaled_array)
        
        return upscaled_image
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "scale": self.scale,
            "device": self.device,
            "model_path": str(self.model_path),
            "tile": self.tile,
            "half_precision": self.half,
            "available": self.upsampler is not None,
        }
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """List all available models."""
        return cls.AVAILABLE_MODELS.copy()
    
    @classmethod
    def download_model(
        cls,
        model_name: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Download model from GitHub releases.
        
        Args:
            model_name: Model name to download
            output_path: Output path (default: ~/.cache/realesrgan/models/)
            
        Returns:
            Path to downloaded model
        """
        import urllib.request
        
        if model_name not in cls.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_info = cls.AVAILABLE_MODELS[model_name]
        model_url = model_info["url"]
        
        if output_path is None:
            model_dir = Path.home() / ".cache" / "realesrgan" / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            output_path = model_dir / f"{model_name}.pth"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.exists():
            logger.info(f"Model already exists: {output_path}")
            return str(output_path)
        
        logger.info(f"Downloading {model_name} from {model_url}...")
        logger.info(f"Saving to {output_path}")
        
        try:
            urllib.request.urlretrieve(model_url, output_path)
            logger.info(f"Model downloaded successfully: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise RuntimeError(f"Failed to download model: {e}")


class RealESRGANUpscaler:
    """
    High-level interface for Real-ESRGAN upscaling.
    
    Automatically selects appropriate model based on scale factor.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        auto_download: bool = False,
        prefer_anime_model: bool = False,
    ):
        """
        Initialize Real-ESRGAN upscaler.
        
        Args:
            device: Device to use (cuda/cpu)
            auto_download: Automatically download models if not found
            prefer_anime_model: Prefer anime-optimized model for 4x
        """
        self.device = device
        self.auto_download = auto_download
        self.prefer_anime_model = prefer_anime_model
        self.models = {}
    
    def _get_model_for_scale(
        self,
        scale_factor: float
    ) -> RealESRGANWrapper:
        """
        Get appropriate model for scale factor.
        
        Args:
            scale_factor: Desired scale factor
            
        Returns:
            RealESRGANWrapper instance
        """
        # Determine best model
        if scale_factor <= 2.0:
            model_name = "RealESRGAN_x2plus"
        elif scale_factor <= 4.0:
            if self.prefer_anime_model:
                model_name = "RealESRGAN_x4plus_anime_6B"
            else:
                model_name = "RealESRGAN_x4plus"
        else:
            # For > 4x, use 4x model with multiple passes
            model_name = "RealESRGAN_x4plus"
        
        # Check if model is already loaded
        if model_name in self.models:
            return self.models[model_name]
        
        # Try to load model
        try:
            model = RealESRGANWrapper(
                model_name=model_name,
                device=self.device,
            )
            self.models[model_name] = model
            return model
        except FileNotFoundError:
            if self.auto_download:
                logger.info(f"Auto-downloading model: {model_name}")
                model_path = RealESRGANWrapper.download_model(model_name)
                model = RealESRGANWrapper(
                    model_name=model_name,
                    model_path=model_path,
                    device=self.device,
                )
                self.models[model_name] = model
                return model
            else:
                raise FileNotFoundError(
                    f"Model {model_name} not found. Set auto_download=True or download manually."
                )
    
    def upscale(
        self,
        image: Union[Image.Image, str, Path],
        scale_factor: float
    ) -> Image.Image:
        """
        Upscale image using Real-ESRGAN.
        
        Args:
            image: Input image
            scale_factor: Scale factor
            
        Returns:
            Upscaled image
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Get appropriate model
        model = self._get_model_for_scale(scale_factor)
        
        # Upscale
        upscaled = model.upscale(pil_image, outscale=scale_factor)
        
        return upscaled
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available models."""
        return RealESRGANWrapper.list_available_models()

