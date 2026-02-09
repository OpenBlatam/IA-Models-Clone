"""
Pydantic schemas for multi-model API
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum


class ModelType(str, Enum):
    """Supported AI model types"""
    COMPOSER_1 = "composer_1"
    SONNET_45 = "sonnet_4.5"
    GPT_51_CODEX = "gpt_5.1_codex"
    GPT_51 = "gpt_5.1"
    GPT_51_CODEX_MINI = "gpt_5.1_codex_mini"
    HAIKU_45 = "haiku_4.5"
    GROK_CODE = "grok_code"
    OPENROUTER_GPT4 = "openrouter/gpt-4"
    OPENROUTER_GPT35 = "openrouter/gpt-3.5-turbo"
    OPENROUTER_CLAUDE_OPUS = "openrouter/claude-3-opus"
    OPENROUTER_CLAUDE_SONNET = "openrouter/claude-3-sonnet"
    OPENROUTER_CLAUDE_HAIKU = "openrouter/claude-3-haiku"
    OPENROUTER_GEMINI_PRO = "openrouter/gemini-pro"
    OPENROUTER_LLAMA3_70B = "openrouter/llama-3-70b"
    OPENROUTER_MISTRAL_LARGE = "openrouter/mistral-large"


class ModelConfig(BaseModel):
    """Configuration for a single model"""
    model_type: ModelType = Field(..., description="Type of AI model")
    multiplier: int = Field(default=1, ge=1, le=5, description="Multiplier/weight for this model (1-5)")
    is_enabled: bool = Field(default=True, description="Whether this model is active")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Temperature setting")
    max_tokens: Optional[int] = Field(default=None, ge=1, le=100000, description="Max tokens for response")
    custom_params: Optional[Dict[str, Any]] = Field(default=None, description="Custom model parameters")
    openrouter_model: Optional[str] = Field(default=None, description="OpenRouter model identifier (e.g., 'openai/gpt-4')")


class MultiModelRequest(BaseModel):
    """Request for multi-model API"""
    prompt: str = Field(..., min_length=1, max_length=50000, description="Input prompt for models")
    models: List[ModelConfig] = Field(..., min_items=1, max_items=5, description="List of models to use (max 5)")
    strategy: Literal["parallel", "sequential", "consensus"] = Field(
        default="parallel",
        description="Execution strategy: parallel (all at once), sequential (one by one), consensus (vote)"
    )
    consensus_method: Optional[Literal["majority", "weighted", "similarity", "average", "best"]] = Field(
        default="majority",
        description="Consensus method for consensus strategy"
    )
    cache_enabled: bool = Field(default=True, description="Enable response caching")
    cache_ttl: Optional[int] = Field(default=3600, ge=0, le=86400, description="Cache TTL in seconds")
    timeout: Optional[float] = Field(default=None, ge=1.0, le=300.0, description="Request timeout in seconds")
    allow_partial_success: bool = Field(default=True, description="Allow partial success if some models fail")
    min_successful_models: Optional[int] = Field(default=None, ge=1, description="Minimum number of successful models required")
    
    @validator('models')
    def validate_models_count(cls, v):
        if len(v) > 5:
            raise ValueError("Maximum 5 models allowed")
        if not any(model.is_enabled for model in v):
            raise ValueError("At least one model must be enabled")
        return v
    
    @validator('models')
    def validate_unique_models(cls, v):
        model_types = [model.model_type for model in v if model.is_enabled]
        if len(model_types) != len(set(model_types)):
            raise ValueError("Duplicate model types are not allowed")
        return v
    
    @validator('min_successful_models')
    def validate_min_successful_models(cls, v, values):
        if v is not None:
            enabled_count = sum(1 for model in values.get('models', []) if model.is_enabled)
            if v > enabled_count:
                raise ValueError(f"min_successful_models ({v}) cannot exceed number of enabled models ({enabled_count})")
        return v
    
    @validator('prompt')
    def validate_prompt_not_empty(cls, v):
        if not v:
            raise ValueError("Prompt cannot be empty")
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("Prompt cannot be whitespace only")
        return v_stripped


class ModelResponse(BaseModel):
    """Response from a single model"""
    model_type: ModelType
    response: str
    tokens_used: Optional[int] = None
    latency_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None


class MultiModelResponse(BaseModel):
    """Response from multi-model API"""
    request_id: str
    prompt: str
    strategy: str
    responses: List[ModelResponse]
    aggregated_response: Optional[str] = None
    total_tokens: Optional[int] = None
    total_latency_ms: Optional[float] = None
    cache_hit: bool = False
    timestamp: str
    success_count: int = Field(default=0, description="Number of successful model responses")
    failure_count: int = Field(default=0, description="Number of failed model responses")
    partial_success: bool = Field(default=False, description="Whether some models succeeded and some failed")


class ModelStatus(BaseModel):
    """Status of a model"""
    model_type: ModelType
    is_available: bool
    is_enabled: bool
    multiplier: int
    last_used: Optional[str] = None
    success_rate: Optional[float] = None
    avg_latency_ms: Optional[float] = None


class ModelsListResponse(BaseModel):
    """List of available models"""
    models: List[ModelStatus]
    total_available: int
    total_enabled: int


class BatchMultiModelRequest(BaseModel):
    """Batch request for multiple multi-model API calls"""
    requests: List[MultiModelRequest] = Field(..., min_items=1, max_items=10, description="List of requests (max 10)")
    stop_on_first_error: bool = Field(default=False, description="Stop processing if one request fails")


class BatchMultiModelResponse(BaseModel):
    """Batch response from multi-model API"""
    request_id: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    responses: List[MultiModelResponse]
    timestamp: str

