"""
API Schemas
==========

Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Dict, Any, Union

from .constants import MAX_LAYERS, MAX_MODEL_NAME_LENGTH


class LayerConfig(BaseModel):
    """Configuration for a neural network layer."""
    type: str = Field(..., description="Layer type (e.g., 'dense', 'conv2d', 'lstm')")
    params: Dict[str, Any] = Field(default_factory=dict, description="Layer parameters")
    
    @validator('type')
    def validate_type(cls, v):
        """Validate layer type."""
        valid_types = [
            'dense', 'conv2d', 'lstm', 'gru', 'dropout', 'batchnormalization',
            'maxpooling2d', 'averagepooling2d', 'flatten', 'reshape', 'embedding'
        ]
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid layer type: {v}. Valid types: {', '.join(valid_types)}")
        return v.lower()


class CreateModelRequest(BaseModel):
    """Request schema for creating a new model."""
    layers: List[LayerConfig] = Field(..., min_items=1, description="List of layer configurations")
    name: Optional[str] = Field(None, max_length=MAX_MODEL_NAME_LENGTH, description="Optional model name")
    
    @validator('layers')
    def validate_layers(cls, v):
        """Validate layers list."""
        if not v:
            raise ValueError("At least one layer is required")
        if len(v) > MAX_LAYERS:
            raise ValueError(f"Maximum {MAX_LAYERS} layers allowed")
        return v


class CompileModelRequest(BaseModel):
    """Request schema for compiling a model."""
    optimizer: str = Field(..., description="Optimizer type (e.g., 'adam', 'sgd', 'rmsprop')")
    optimizer_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optimizer parameters")
    loss: str = Field(..., description="Loss function type")
    loss_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Loss function parameters")
    metrics: Optional[List[str]] = Field(default_factory=list, description="List of metrics to track")
    
    @validator('optimizer')
    def validate_optimizer(cls, v):
        """Validate optimizer type."""
        valid_optimizers = ['adam', 'sgd', 'rmsprop', 'adagrad', 'adamw']
        if v.lower() not in valid_optimizers:
            raise ValueError(f"Invalid optimizer: {v}. Valid optimizers: {', '.join(valid_optimizers)}")
        return v.lower()
    
    @validator('loss')
    def validate_loss(cls, v):
        """Validate loss function type."""
        valid_losses = [
            'sparsecategoricalcrossentropy', 'categoricalcrossentropy',
            'binarycrossentropy', 'meansquarederror', 'mse',
            'meanabsoluteerror', 'mae'
        ]
        if v.lower() not in valid_losses:
            raise ValueError(f"Invalid loss function: {v}. Valid losses: {', '.join(valid_losses)}")
        return v.lower()


class TrainRequest(BaseModel):
    """Request schema for training a model."""
    x_train: List[List[float]] = Field(..., description="Training input data")
    y_train: List[Union[int, float]] = Field(..., description="Training target data")
    epochs: int = Field(1, ge=1, le=10000, description="Number of training epochs")
    batch_size: int = Field(32, ge=1, le=10000, description="Batch size for training")
    validation_data: Optional[Dict[str, List[List[float]]]] = Field(None, description="Validation data")
    verbose: int = Field(1, ge=0, le=2, description="Verbosity level (0, 1, or 2)")
    
    @validator('x_train', 'y_train')
    def validate_data_not_empty(cls, v):
        """Validate data is not empty."""
        if not v:
            raise ValueError("Data cannot be empty")
        return v
    
    @root_validator
    def validate_data_lengths(cls, values):
        """Validate that x_train and y_train have matching lengths."""
        x_train = values.get('x_train', [])
        y_train = values.get('y_train', [])
        if len(x_train) != len(y_train):
            raise ValueError(f"x_train and y_train must have the same length. Got {len(x_train)} and {len(y_train)}")
        return values


class EvaluateRequest(BaseModel):
    """Request schema for evaluating a model."""
    x_test: List[List[float]] = Field(..., description="Test input data")
    y_test: List[Union[int, float]] = Field(..., description="Test target data")
    verbose: int = Field(0, ge=0, le=2, description="Verbosity level (0, 1, or 2)")
    
    @validator('x_test', 'y_test')
    def validate_data_not_empty(cls, v):
        """Validate data is not empty."""
        if not v:
            raise ValueError("Data cannot be empty")
        return v
    
    @root_validator
    def validate_data_lengths(cls, values):
        """Validate that x_test and y_test have matching lengths."""
        x_test = values.get('x_test', [])
        y_test = values.get('y_test', [])
        if len(x_test) != len(y_test):
            raise ValueError(f"x_test and y_test must have the same length. Got {len(x_test)} and {len(y_test)}")
        return values


class PredictRequest(BaseModel):
    """Request schema for making predictions."""
    x: List[List[float]] = Field(..., description="Input data for predictions")
    verbose: int = Field(0, ge=0, le=2, description="Verbosity level (0, 1, or 2)")
    
    @validator('x')
    def validate_data_not_empty(cls, v):
        """Validate data is not empty."""
        if not v:
            raise ValueError("Input data cannot be empty")
        return v


class ModelInfo(BaseModel):
    """Model information response."""
    model_id: str
    name: str
    compiled: bool


class ModelListResponse(BaseModel):
    """Response schema for listing models."""
    models: List[ModelInfo]
    count: int


class ModelResponse(BaseModel):
    """Base response schema for model operations."""
    model_id: str
    status: str
    message: str


class CreateModelResponse(ModelResponse):
    """Response schema for model creation."""
    name: str


class TrainResponse(ModelResponse):
    """Response schema for model training."""
    history: Dict[str, Any]


class EvaluateResponse(ModelResponse):
    """Response schema for model evaluation."""
    results: Dict[str, Any]


class PredictResponse(ModelResponse):
    """Response schema for predictions."""
    predictions: List[Any]


class SaveModelResponse(ModelResponse):
    """Response schema for saving a model."""
    filepath: str


class LoadModelResponse(ModelResponse):
    """Response schema for loading a model."""
    filepath: str


class BatchDeleteRequest(BaseModel):
    """Request schema for batch delete operation."""
    model_ids: List[str] = Field(..., min_items=1, max_items=100, description="List of model IDs to delete")


class BatchDeleteResponse(BaseModel):
    """Response schema for batch delete operation."""
    deleted: List[str] = Field(..., description="List of successfully deleted model IDs")
    failed: List[Dict[str, str]] = Field(default_factory=list, description="List of failed deletions with errors")
    total_requested: int
    total_deleted: int
    total_failed: int


class BatchPredictRequest(BaseModel):
    """Request schema for batch predictions."""
    model_ids: List[str] = Field(..., min_items=1, max_items=10, description="List of model IDs")
    x: List[List[float]] = Field(..., description="Input data for predictions")
    verbose: int = Field(0, ge=0, le=2, description="Verbosity level")
    
    @validator('x')
    def validate_data_not_empty(cls, v):
        """Validate data is not empty."""
        if not v:
            raise ValueError("Input data cannot be empty")
        return v


class BatchPredictResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: Dict[str, List[Any]] = Field(..., description="Predictions by model ID")
    failed: List[Dict[str, str]] = Field(default_factory=list, description="Failed predictions with errors")
    total_models: int
    successful: int
    failed_count: int

