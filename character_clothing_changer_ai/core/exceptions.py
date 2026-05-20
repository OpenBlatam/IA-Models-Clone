"""
Custom Exceptions Module
========================
Custom exception classes for the Character Clothing Changer AI.
"""


class ClothingChangerError(Exception):
    """Base exception for all Clothing Changer errors."""
    
    def __init__(self, message: str, code: str = None, details: dict = None):
        self.message = message
        self.code = code or "CLOTHING_CHANGER_ERROR"
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self):
        """Convert exception to dictionary."""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details
        }


class ModelError(ClothingChangerError):
    """Exception for model-related errors."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, code="MODEL_ERROR", details=details)


class ModelNotInitializedError(ModelError):
    """Exception when model is not initialized."""
    
    def __init__(self):
        super().__init__(
            "Model not initialized. Please wait for initialization to complete.",
            code="MODEL_NOT_INITIALIZED"
        )


class ModelLoadError(ModelError):
    """Exception when model fails to load."""
    
    def __init__(self, model_id: str, reason: str = None):
        details = {"model_id": model_id}
        if reason:
            details["reason"] = reason
        super().__init__(
            f"Failed to load model: {model_id}",
            code="MODEL_LOAD_ERROR",
            details=details
        )


class ValidationError(ClothingChangerError):
    """Exception for validation errors."""
    
    def __init__(self, message: str, field: str = None, details: dict = None):
        if field:
            details = details or {}
            details["field"] = field
        super().__init__(message, code="VALIDATION_ERROR", details=details)


class ImageValidationError(ValidationError):
    """Exception for image validation errors."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, code="IMAGE_VALIDATION_ERROR", details=details)


class TextValidationError(ValidationError):
    """Exception for text validation errors."""
    
    def __init__(self, message: str, field: str = None, details: dict = None):
        super().__init__(message, code="TEXT_VALIDATION_ERROR", field=field, details=details)


class ParameterValidationError(ValidationError):
    """Exception for parameter validation errors."""
    
    def __init__(self, message: str, parameter: str = None, details: dict = None):
        if parameter:
            details = details or {}
            details["parameter"] = parameter
        super().__init__(message, code="PARAMETER_VALIDATION_ERROR", details=details)


class ProcessingError(ClothingChangerError):
    """Exception for image processing errors."""
    
    def __init__(self, message: str, step: str = None, details: dict = None):
        if step:
            details = details or {}
            details["step"] = step
        super().__init__(message, code="PROCESSING_ERROR", details=details)


class TensorGenerationError(ClothingChangerError):
    """Exception for tensor generation errors."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message, code="TENSOR_GENERATION_ERROR", details=details)


class APIError(ClothingChangerError):
    """Exception for API-related errors."""
    
    def __init__(self, message: str, status_code: int = None, details: dict = None):
        if status_code:
            details = details or {}
            details["status_code"] = status_code
        super().__init__(message, code="API_ERROR", details=details)


class ConfigurationError(ClothingChangerError):
    """Exception for configuration errors."""
    
    def __init__(self, message: str, config_key: str = None, details: dict = None):
        if config_key:
            details = details or {}
            details["config_key"] = config_key
        super().__init__(message, code="CONFIGURATION_ERROR", details=details)

