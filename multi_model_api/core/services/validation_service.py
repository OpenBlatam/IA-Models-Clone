"""
Validation service for Multi-Model API
Centralized validation logic
"""

import logging
from typing import List, Optional
from ...api.schemas import MultiModelRequest, ModelConfig
from ...api.exceptions import ValidationException

logger = logging.getLogger(__name__)


class ValidationService:
    """Service for validating requests and configurations"""
    
    # Strategy validation
    VALID_STRATEGIES = frozenset({"parallel", "sequential", "consensus"})
    VALID_CONSENSUS_METHODS = frozenset({
        "majority", "weighted", "similarity", "average", "best"
    })
    
    # Model limits
    MAX_MODELS = 10
    MIN_MODELS = 1
    
    # Prompt limits
    MIN_PROMPT_LENGTH = 1
    MAX_PROMPT_LENGTH = 100000
    
    # Timeout limits (seconds)
    MIN_TIMEOUT = 1.0
    MAX_TIMEOUT = 300.0
    DEFAULT_TIMEOUT = 30.0
    
    # Model parameter limits
    MIN_TEMPERATURE = 0.0
    MAX_TEMPERATURE = 2.0
    MIN_MAX_TOKENS = 1
    MAX_MAX_TOKENS = 100000
    MIN_MULTIPLIER = 0
    
    @classmethod
    def validate_request(cls, request: MultiModelRequest) -> None:
        """
        Validate multi-model request
        
        Args:
            request: Request to validate
            
        Raises:
            ValidationException: If validation fails
        """
        # Validate prompt
        if not request.prompt or not isinstance(request.prompt, str):
            raise ValidationException(
                message="Prompt must be a non-empty string",
                field="prompt"
            )
        
        prompt_length = len(request.prompt.strip())
        if prompt_length < cls.MIN_PROMPT_LENGTH:
            raise ValidationException(
                message=f"Prompt must be at least {cls.MIN_PROMPT_LENGTH} character(s)",
                field="prompt"
            )
        
        if prompt_length > cls.MAX_PROMPT_LENGTH:
            raise ValidationException(
                message=f"Prompt must not exceed {cls.MAX_PROMPT_LENGTH} characters",
                field="prompt"
            )
        
        # Validate models
        if not request.models:
            raise ValidationException(
                message="At least one model must be specified",
                field="models"
            )
        
        if len(request.models) > cls.MAX_MODELS:
            raise ValidationException(
                message=f"Maximum {cls.MAX_MODELS} models allowed per request (got {len(request.models)})",
                field="models"
            )
        
        if len(request.models) < cls.MIN_MODELS:
            raise ValidationException(
                message=f"At least {cls.MIN_MODELS} model must be specified",
                field="models"
            )
        
        enabled_models = [m for m in request.models if m.is_enabled]
        if not enabled_models:
            raise ValidationException(
                message="At least one model must be enabled",
                field="models"
            )
        
        # Validate strategy
        if request.strategy not in cls.VALID_STRATEGIES:
            raise ValidationException(
                message=f"Invalid strategy: {request.strategy}. Must be one of {cls.VALID_STRATEGIES}",
                field="strategy"
            )
        
        # Validate consensus method if strategy is consensus
        if request.strategy == "consensus":
            consensus_method = request.consensus_method or "majority"
            if consensus_method not in cls.VALID_CONSENSUS_METHODS:
                raise ValidationException(
                    message=f"Invalid consensus method: {consensus_method}. Must be one of {cls.VALID_CONSENSUS_METHODS}",
                    field="consensus_method"
                )
        
        # Validate timeout
        if request.timeout is not None:
            if request.timeout < cls.MIN_TIMEOUT:
                raise ValidationException(
                    message=f"Timeout must be at least {cls.MIN_TIMEOUT} seconds",
                    field="timeout"
                )
            if request.timeout > cls.MAX_TIMEOUT:
                raise ValidationException(
                    message=f"Timeout must not exceed {cls.MAX_TIMEOUT} seconds",
                    field="timeout"
                )
        
        # Validate model configurations
        for i, model in enumerate(request.models):
            cls._validate_model_config(model, index=i)
    
    @classmethod
    def _validate_model_config(cls, model: ModelConfig, index: int) -> None:
        """
        Validate individual model configuration
        
        Args:
            model: Model configuration to validate
            index: Index of model in request.models list
            
        Raises:
            ValidationException: If validation fails
        """
        if model.temperature is not None:
            if not (cls.MIN_TEMPERATURE <= model.temperature <= cls.MAX_TEMPERATURE):
                raise ValidationException(
                    message=f"Temperature must be between {cls.MIN_TEMPERATURE} and {cls.MAX_TEMPERATURE} (model {index})",
                    field=f"models[{index}].temperature"
                )
        
        if model.max_tokens is not None:
            if model.max_tokens < cls.MIN_MAX_TOKENS:
                raise ValidationException(
                    message=f"max_tokens must be at least {cls.MIN_MAX_TOKENS} (model {index})",
                    field=f"models[{index}].max_tokens"
                )
            if model.max_tokens > cls.MAX_MAX_TOKENS:
                raise ValidationException(
                    message=f"max_tokens must not exceed {cls.MAX_MAX_TOKENS} (model {index})",
                    field=f"models[{index}].max_tokens"
                )
        
        if model.multiplier is not None:
            if model.multiplier < cls.MIN_MULTIPLIER:
                raise ValidationException(
                    message=f"Multiplier must be non-negative (model {index})",
                    field=f"models[{index}].multiplier"
                )




