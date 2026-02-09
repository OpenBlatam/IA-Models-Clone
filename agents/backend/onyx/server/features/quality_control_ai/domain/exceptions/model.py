"""
ML Model-related exceptions.
"""

from .base import QualityControlException


class ModelException(QualityControlException):
    """Exception raised for ML model-related errors."""
    pass


class ModelLoadException(ModelException):
    """Exception raised when a model fails to load."""
    
    def __init__(self, model_name: str, reason: str):
        super().__init__(
            message=f"Failed to load model '{model_name}': {reason}",
            error_code="MODEL_LOAD_FAILED",
            details={"model_name": model_name, "reason": reason}
        )


class ModelInferenceException(ModelException):
    """Exception raised when model inference fails."""
    
    def __init__(self, model_name: str, reason: str):
        super().__init__(
            message=f"Model inference failed for '{model_name}': {reason}",
            error_code="MODEL_INFERENCE_FAILED",
            details={"model_name": model_name, "reason": reason}
        )


class ModelTrainingException(ModelException):
    """Exception raised when model training fails."""
    
    def __init__(self, model_name: str, reason: str):
        super().__init__(
            message=f"Model training failed for '{model_name}': {reason}",
            error_code="MODEL_TRAINING_FAILED",
            details={"model_name": model_name, "reason": reason}
        )



