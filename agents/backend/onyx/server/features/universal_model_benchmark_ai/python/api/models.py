"""
API Models - Pydantic models for REST API.

This module defines all request and response models
used by the REST API endpoints.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ════════════════════════════════════════════════════════════════════════════════
# REQUEST MODELS
# ════════════════════════════════════════════════════════════════════════════════

class BenchmarkRequest(BaseModel):
    """Request model for running a benchmark."""
    model_name: str = Field(..., description="Name of the model to benchmark")
    benchmark_name: str = Field(..., description="Name of the benchmark to run")
    config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional benchmark configuration"
    )


class ExperimentRequest(BaseModel):
    """Request model for creating an experiment."""
    name: str = Field(..., description="Experiment name")
    description: str = Field(default="", description="Experiment description")
    model_name: str = Field(..., description="Model name")
    benchmark_name: str = Field(..., description="Benchmark name")
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Hyperparameters for the experiment"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization"
    )


class ModelRegisterRequest(BaseModel):
    """Request model for registering a model."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    description: str = Field(default="", description="Model description")
    architecture: str = Field(default="", description="Model architecture")
    parameters: int = Field(default=0, description="Number of parameters")
    path: str = Field(..., description="Path to model")
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization"
    )


class WebhookRequest(BaseModel):
    """Request model for registering a webhook."""
    url: str = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="Events to subscribe to")
    secret: Optional[str] = Field(
        default=None,
        description="Optional webhook secret"
    )


# ════════════════════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ════════════════════════════════════════════════════════════════════════════════

class SuccessResponse(BaseModel):
    """Standard success response."""
    success: bool = True
    message: str = "Operation completed successfully"
    data: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class BenchmarkResponse(BaseModel):
    """Response model for benchmark results."""
    benchmark_name: str
    model_name: str
    accuracy: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput: float
    total_samples: int
    correct_samples: int
    timestamp: str


class ExperimentResponse(BaseModel):
    """Response model for experiment."""
    id: str
    name: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    results: Optional[Dict[str, Any]] = None


class ModelResponse(BaseModel):
    """Response model for model information."""
    name: str
    version: str
    description: str
    architecture: str
    parameters: int
    path: str
    tags: List[str]
    registered_at: str


__all__ = [
    # Requests
    "BenchmarkRequest",
    "ExperimentRequest",
    "ModelRegisterRequest",
    "WebhookRequest",
    # Responses
    "SuccessResponse",
    "ErrorResponse",
    "BenchmarkResponse",
    "ExperimentResponse",
    "ModelResponse",
]












