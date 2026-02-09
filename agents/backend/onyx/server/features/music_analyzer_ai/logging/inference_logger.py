"""
Inference Logger
Specialized logger for inference
"""

from typing import Dict, Any, Optional
import logging


class InferenceLogger:
    """Logger for inference"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def log_inference_start(self, input_info: Dict[str, Any]):
        """Log inference start"""
        self.logger.info(f"Inference started: {input_info}")
    
    def log_inference_end(
        self,
        output_info: Dict[str, Any],
        inference_time: float
    ):
        """Log inference end"""
        self.logger.info(
            f"Inference completed in {inference_time:.3f}s: {output_info}"
        )
    
    def log_prediction(self, prediction: Dict[str, Any]):
        """Log prediction"""
        self.logger.debug(f"Prediction: {prediction}")
    
    def log_error(self, error: Exception):
        """Log inference error"""
        self.logger.error(f"Inference error: {str(error)}", exc_info=True)



