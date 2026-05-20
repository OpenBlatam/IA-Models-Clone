"""
Integrated Pipeline
Complete end-to-end pipeline with all features integrated
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
import logging

from ..validation.validators import InputValidator, ModelValidator
from ..monitoring.health_check import ModelHealthMonitor
from ..errors.error_handler import safe_inference, handle_errors
from ..optimization.ultra_fast_inference import UltraFastInference
from ..optimization.memory_optimizer import optimize_model_memory

logger = logging.getLogger(__name__)


class IntegratedPipeline:
    """
    Integrated pipeline with validation, monitoring, error handling, and optimization
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        enable_validation: bool = True,
        enable_monitoring: bool = True,
        enable_optimization: bool = True
    ):
        """
        Initialize integrated pipeline
        
        Args:
            model: PyTorch model
            device: Device to use
            enable_validation: Enable input validation
            enable_monitoring: Enable health monitoring
            enable_optimization: Enable optimizations
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        
        # Enable features
        self.enable_validation = enable_validation
        self.enable_monitoring = enable_monitoring
        self.enable_optimization = enable_optimization
        
        # Setup components
        if self.enable_validation:
            self.input_validator = InputValidator()
            self.model_validator = ModelValidator()
        
        if self.enable_monitoring:
            self.model_monitor = ModelHealthMonitor(self.model)
        
        if self.enable_optimization:
            optimize_model_memory(self.model)
            self.inference_engine = UltraFastInference(self.model, self.device)
        else:
            self.inference_engine = None
    
    @safe_inference
    @handle_errors(default_return=None)
    def predict(
        self,
        inputs: torch.Tensor,
        validate: bool = True,
        record_inference: bool = True
    ) -> Optional[torch.Tensor]:
        """
        Predict with full pipeline
        
        Args:
            inputs: Input tensor
            validate: Validate inputs
            record_inference: Record for monitoring
            
        Returns:
            Predictions
        """
        import time
        
        # Validation
        if validate and self.enable_validation:
            is_valid, error = self.input_validator.validate_tensor(
                inputs,
                device=self.device,
                check_nan=True,
                check_inf=True
            )
            if not is_valid:
                logger.error(f"Input validation failed: {error}")
                return None
        
        # Inference
        start_time = time.perf_counter()
        
        if self.inference_engine:
            outputs = self.inference_engine.predict(inputs)
        else:
            with torch.inference_mode():
                if self.device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)
        
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Output validation
        if validate and self.enable_validation:
            is_valid, error = self.model_validator.validate_output(
                outputs,
                check_nan=True,
                check_inf=True
            )
            if not is_valid:
                logger.error(f"Output validation failed: {error}")
                return None
        
        # Monitoring
        if record_inference and self.enable_monitoring:
            self.model_monitor.record_inference(inference_time, success=True)
        
        return outputs
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get complete health status
        
        Returns:
            Health status dictionary
        """
        status = {
            "model_optimized": self.enable_optimization,
            "validation_enabled": self.enable_validation,
            "monitoring_enabled": self.enable_monitoring
        }
        
        if self.enable_monitoring:
            status["model_health"] = self.model_monitor.check_model_health()
        
        if self.enable_validation:
            status["model_state"] = self.model_validator.validate_model_state(self.model)
        
        return status
    
    def predict_batch(
        self,
        inputs: List[torch.Tensor],
        batch_size: int = 32,
        validate: bool = True
    ) -> List[torch.Tensor]:
        """
        Batch prediction
        
        Args:
            inputs: List of input tensors
            batch_size: Batch size
            validate: Validate inputs
            
        Returns:
            List of predictions
        """
        results = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_tensor = torch.stack(batch).to(self.device, non_blocking=True)
            
            output = self.predict(batch_tensor, validate=validate, record_inference=False)
            if output is not None:
                results.extend(output.cpu().split(1))
        
        # Record batch inference
        if self.enable_monitoring:
            import time
            self.model_monitor.record_inference(
                time.perf_counter() * 1000,
                success=len(results) == len(inputs)
            )
        
        return results


def create_integrated_pipeline(
    model: nn.Module,
    device: Optional[torch.device] = None,
    **kwargs
) -> IntegratedPipeline:
    """
    Factory for integrated pipeline
    
    Args:
        model: PyTorch model
        device: Device to use
        **kwargs: Additional options
        
    Returns:
        Integrated pipeline
    """
    return IntegratedPipeline(model, device, **kwargs)













