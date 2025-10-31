"""
ML Types and Definitions
========================

Type definitions for machine learning components.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import uuid

class ModelType(Enum):
    """Machine learning model types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    TIME_SERIES = "time_series"
    RECOMMENDATION = "recommendation"
    ANOMALY_DETECTION = "anomaly_detection"
    CUSTOM = "custom"

class TrainingStatus(Enum):
    """Training job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class InferenceStatus(Enum):
    """Inference status."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"

class FeatureType(Enum):
    """Feature data types."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    IMAGE = "image"
    TIME_SERIES = "time_series"
    EMBEDDING = "embedding"
    JSON = "json"

@dataclass
class ModelMetadata:
    """Model metadata information."""
    name: str
    version: str
    model_type: ModelType
    description: str
    author: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    size_bytes: Optional[int] = None
    checksum: Optional[str] = None

@dataclass
class TrainingConfig:
    """Training configuration."""
    model_type: ModelType
    algorithm: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    data_config: Dict[str, Any] = field(default_factory=dict)
    validation_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    callbacks: List[str] = field(default_factory=list)
    early_stopping: Dict[str, Any] = field(default_factory=dict)
    checkpointing: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InferenceConfig:
    """Inference configuration."""
    model_id: str
    batch_size: int = 1
    timeout_seconds: int = 30
    max_retries: int = 3
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    postprocessing: Dict[str, Any] = field(default_factory=dict)
    caching: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FeatureDefinition:
    """Feature definition."""
    name: str
    feature_type: FeatureType
    description: str
    required: bool = True
    default_value: Optional[Any] = None
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DatasetInfo:
    """Dataset information."""
    name: str
    version: str
    description: str
    size: int
    features: List[FeatureDefinition]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelPerformance:
    """Model performance metrics."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class ModelDrift:
    """Model drift detection results."""
    drift_detected: bool
    drift_score: float
    feature_drifts: Dict[str, float] = field(default_factory=dict)
    data_drift: Optional[float] = None
    concept_drift: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class ABIExplanation:
    """AI/ML model explanation."""
    feature_importance: Dict[str, float] = field(default_factory=dict)
    shap_values: Optional[Dict[str, Any]] = None
    lime_explanation: Optional[Dict[str, Any]] = None
    attention_weights: Optional[Dict[str, Any]] = None
    decision_path: Optional[List[str]] = None
    confidence_scores: Dict[str, float] = field(default_factory=dict)

@dataclass
class ModelBias:
    """Model bias analysis."""
    bias_detected: bool
    bias_score: float
    protected_attributes: List[str] = field(default_factory=list)
    bias_metrics: Dict[str, float] = field(default_factory=dict)
    fairness_constraints: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class ExperimentConfig:
    """ML experiment configuration."""
    name: str
    description: str
    objective: str
    metrics: List[str]
    hyperparameter_space: Dict[str, Any] = field(default_factory=dict)
    search_strategy: str = "random"
    max_trials: int = 100
    timeout_minutes: int = 60
    early_stopping: Dict[str, Any] = field(default_factory=dict)
    parallel_trials: int = 1

@dataclass
class ExperimentResult:
    """ML experiment result."""
    experiment_id: str
    best_trial_id: str
    best_score: float
    best_params: Dict[str, Any]
    total_trials: int
    completed_trials: int
    duration_minutes: float
    status: str
    trials: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ModelDeployment:
    """Model deployment configuration."""
    deployment_id: str
    model_id: str
    environment: str
    replicas: int = 1
    resources: Dict[str, Any] = field(default_factory=dict)
    scaling_config: Dict[str, Any] = field(default_factory=dict)
    health_check: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"

@dataclass
class ModelServing:
    """Model serving configuration."""
    serving_id: str
    model_id: str
    endpoint_url: str
    protocol: str = "http"
    authentication: Dict[str, Any] = field(default_factory=dict)
    rate_limiting: Dict[str, Any] = field(default_factory=dict)
    load_balancing: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "active"
