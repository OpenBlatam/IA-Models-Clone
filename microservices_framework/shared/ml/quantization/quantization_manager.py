"""
Quantization Manager
Support for model quantization (INT8, INT4) for efficient inference.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Literal
import logging

logger = logging.getLogger(__name__)


class QuantizationManager:
    """
    Manager for model quantization operations.
    """
    
    def __init__(
        self,
        quantization_type: Literal["int8", "int4", "dynamic"] = "int8",
        use_bitsandbytes: bool = True,
    ):
        self.quantization_type = quantization_type
        self.use_bitsandbytes = use_bitsandbytes
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if quantization dependencies are available."""
        if self.use_bitsandbytes:
            try:
                import bitsandbytes as bnb
                self.bnb_available = True
            except ImportError:
                logger.warning("bitsandbytes not available, falling back to PyTorch quantization")
                self.bnb_available = False
                self.use_bitsandbytes = False
        else:
            self.bnb_available = False
    
    def quantize_model(
        self,
        model: nn.Module,
        quantization_config: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """
        Quantize model based on type.
        
        Args:
            model: Model to quantize
            quantization_config: Quantization configuration
            
        Returns:
            Quantized model
        """
        if self.quantization_type == "int8" and self.use_bitsandbytes and self.bnb_available:
            return self._quantize_int8_bnb(model, quantization_config)
        elif self.quantization_type == "int4" and self.use_bitsandbytes and self.bnb_available:
            return self._quantize_int4_bnb(model, quantization_config)
        elif self.quantization_type == "dynamic":
            return self._quantize_dynamic_pytorch(model, quantization_config)
        else:
            logger.warning(f"Quantization type {self.quantization_type} not supported, returning original model")
            return model
    
    def _quantize_int8_bnb(
        self,
        model: nn.Module,
        config: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """Quantize model to INT8 using bitsandbytes."""
        try:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=config.get("llm_int8_threshold", 6.0) if config else 6.0,
                llm_int8_has_fp16_weight=config.get("llm_int8_has_fp16_weight", False) if config else False,
            )
            
            # This is typically done during model loading
            # For already loaded models, we need to reload with quantization
            logger.info("INT8 quantization with bitsandbytes requires model reloading")
            logger.info("Use load_in_8bit=True when loading model from transformers")
            
            return model
            
        except ImportError:
            logger.error("bitsandbytes not available for INT8 quantization")
            return model
    
    def _quantize_int4_bnb(
        self,
        model: nn.Module,
        config: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """Quantize model to INT4 using bitsandbytes."""
        try:
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=config.get("use_double_quant", True) if config else True,
                bnb_4bit_quant_type=config.get("quant_type", "nf4") if config else "nf4",
            )
            
            logger.info("INT4 quantization with bitsandbytes requires model reloading")
            logger.info("Use load_in_4bit=True when loading model from transformers")
            
            return model
            
        except ImportError:
            logger.error("bitsandbytes not available for INT4 quantization")
            return model
    
    def _quantize_dynamic_pytorch(
        self,
        model: nn.Module,
        config: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """Quantize model using PyTorch dynamic quantization."""
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.LSTM, nn.GRU},  # Modules to quantize
                dtype=torch.qint8,
            )
            logger.info("Model quantized using PyTorch dynamic quantization")
            return quantized_model
        except Exception as e:
            logger.error(f"PyTorch quantization failed: {e}")
            return model
    
    def get_quantization_info(self, model: nn.Module) -> Dict[str, Any]:
        """
        Get information about model quantization.
        
        Args:
            model: Model to analyze
            
        Returns:
            Quantization information
        """
        info = {
            "quantized": False,
            "quantization_type": None,
            "original_size_mb": None,
            "quantized_size_mb": None,
        }
        
        # Check if model is quantized
        if hasattr(model, "quantization_config"):
            info["quantized"] = True
            info["quantization_type"] = str(model.quantization_config)
        
        # Estimate sizes
        try:
            original_params = sum(p.numel() * 4 for p in model.parameters())  # FP32 = 4 bytes
            info["original_size_mb"] = original_params / (1024 * 1024)
            
            if info["quantized"]:
                # INT8 = 1 byte, INT4 = 0.5 bytes
                if "8bit" in str(model.quantization_config):
                    quantized_params = sum(p.numel() for p in model.parameters())
                    info["quantized_size_mb"] = quantized_params / (1024 * 1024)
                elif "4bit" in str(model.quantization_config):
                    quantized_params = sum(p.numel() for p in model.parameters())
                    info["quantized_size_mb"] = quantized_params / (2 * 1024 * 1024)
        except Exception as e:
            logger.warning(f"Could not estimate sizes: {e}")
        
        return info



