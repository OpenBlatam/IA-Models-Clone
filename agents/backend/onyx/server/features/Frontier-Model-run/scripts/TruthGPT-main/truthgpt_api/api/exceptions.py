"""
Custom Exceptions
=================

Custom exception classes for the TruthGPT API.
"""

from fastapi import HTTPException, status


class TruthGPTAPIException(HTTPException):
    """Base exception for TruthGPT API errors."""
    
    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)


class ModelNotFoundError(TruthGPTAPIException):
    """Raised when a model is not found."""
    
    def __init__(self, model_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}"
        )


class ModelNotCompiledError(TruthGPTAPIException):
    """Raised when trying to use a model that hasn't been compiled."""
    
    def __init__(self, model_id: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model {model_id} must be compiled before this operation"
        )


class InvalidLayerTypeError(TruthGPTAPIException):
    """Raised when an invalid layer type is provided."""
    
    def __init__(self, layer_type: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown layer type: {layer_type}"
        )


class InvalidOptimizerError(TruthGPTAPIException):
    """Raised when an invalid optimizer is provided."""
    
    def __init__(self, optimizer: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown optimizer: {optimizer}"
        )


class InvalidLossError(TruthGPTAPIException):
    """Raised when an invalid loss function is provided."""
    
    def __init__(self, loss: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown loss function: {loss}"
        )


class ModelFileNotFoundError(TruthGPTAPIException):
    """Raised when a model file is not found."""
    
    def __init__(self, filepath: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model file not found: {filepath}"
        )

