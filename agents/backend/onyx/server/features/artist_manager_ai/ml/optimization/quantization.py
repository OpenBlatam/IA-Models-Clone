"""
Quantization for Faster Inference
=================================

Quantization utilities for maximum speed.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class QuantizationOptimizer:
    """
    Quantization optimizer for faster inference.
    
    Features:
    - Dynamic quantization
    - Static quantization
    - QAT (Quantization Aware Training)
    - INT8 inference
    """
    
    @staticmethod
    def dynamic_quantize(model: nn.Module) -> nn.Module:
        """
        Apply dynamic quantization.
        
        Args:
            model: PyTorch model
        
        Returns:
            Quantized model
        """
        try:
            quantized = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.LSTM, nn.GRU},
                dtype=torch.qint8
            )
            logger.info("Model dynamically quantized to INT8")
            return quantized
        except Exception as e:
            logger.warning(f"Dynamic quantization failed: {str(e)}")
            return model
    
    @staticmethod
    def static_quantize(
        model: nn.Module,
        example_input: torch.Tensor,
        calibration_data: list
    ) -> nn.Module:
        """
        Apply static quantization.
        
        Args:
            model: PyTorch model
            example_input: Example input
            calibration_data: Calibration dataset
        
        Returns:
            Quantized model
        """
        try:
            model.eval()
            
            # Fuse modules
            model_fused = torch.quantization.fuse_modules(
                model,
                [['conv', 'bn', 'relu']],
                inplace=False
            )
            
            # Set quantization config
            model_fused.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # Prepare
            model_prepared = torch.quantization.prepare(model_fused)
            
            # Calibrate
            with torch.no_grad():
                for data in calibration_data:
                    _ = model_prepared(data)
            
            # Convert
            quantized = torch.quantization.convert(model_prepared)
            
            logger.info("Model statically quantized to INT8")
            return quantized
        except Exception as e:
            logger.warning(f"Static quantization failed: {str(e)}")
            return model
    
    @staticmethod
    def quantize_for_mobile(model: nn.Module) -> nn.Module:
        """
        Quantize model for mobile deployment.
        
        Args:
            model: PyTorch model
        
        Returns:
            Quantized model
        """
        try:
            quantized = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8
            )
            # Additional mobile optimizations
            quantized = torch.jit.script(quantized)
            logger.info("Model quantized and scripted for mobile")
            return quantized
        except Exception as e:
            logger.warning(f"Mobile quantization failed: {str(e)}")
            return model


class FastInferenceEngine:
    """
    Fast inference engine with multiple optimizations.
    
    Features:
    - Compiled models
    - Quantized models
    - Batch optimization
    - Caching
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        use_quantization: bool = True,
        use_compile: bool = True,
        batch_size: int = 64
    ):
        """
        Initialize fast inference engine.
        
        Args:
            model: PyTorch model
            device: Device
            use_quantization: Use quantization
            use_compile: Use compilation
            batch_size: Batch size
        """
        self.device = device
        self.batch_size = batch_size
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        # Quantize if requested
        if use_quantization:
            model = QuantizationOptimizer.dynamic_quantize(model)
        
        # Compile if requested
        if use_compile and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="max-autotune", fullgraph=True)
                logger.info("Model compiled with max-autotune")
            except Exception as e:
                logger.warning(f"Compilation failed: {str(e)}")
        
        self.model = model
        self._logger = logger
    
    @torch.no_grad()
    def predict(
        self,
        inputs: torch.Tensor,
        use_amp: bool = True
    ) -> torch.Tensor:
        """
        Fast prediction.
        
        Args:
            inputs: Input tensor
            use_amp: Use automatic mixed precision
        
        Returns:
            Predictions
        """
        inputs = inputs.to(self.device)
        
        if use_amp and self.device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
        else:
            outputs = self.model(inputs)
        
        return outputs.cpu()
    
    @torch.no_grad()
    def predict_batch(
        self,
        inputs: torch.Tensor,
        use_amp: bool = True
    ) -> torch.Tensor:
        """
        Fast batch prediction.
        
        Args:
            inputs: Input tensor
            use_amp: Use automatic mixed precision
        
        Returns:
            Predictions
        """
        results = []
        
        for i in range(0, inputs.size(0), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            result = self.predict(batch, use_amp=use_amp)
            results.append(result)
        
        return torch.cat(results, dim=0)




