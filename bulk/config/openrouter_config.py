"""
OpenRouter Configuration for BUL System
=======================================

Configuration for OpenRouter integration with LangChain for AI model access.
Supports multiple models and provides fallback mechanisms.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class ModelProvider(Enum):
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"

@dataclass
class ModelConfig:
    """Configuration for a specific AI model."""
    name: str
    provider: ModelProvider
    max_tokens: int
    temperature: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    cost_per_token: float
    is_available: bool = True

class OpenRouterConfig:
    """OpenRouter configuration manager."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.default_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://blatam-academy.com",
            "X-Title": "BUL Business Unlimited"
        }
        
        # Model configurations
        self.models = self._initialize_models()
        self.default_model = "openai/gpt-4o"
        self.fallback_models = [
            "anthropic/claude-3.5-sonnet",
            "google/gemini-pro-1.5",
            "openai/gpt-4-turbo"
        ]
        
    def _initialize_models(self) -> Dict[str, ModelConfig]:
        """Initialize available models with their configurations."""
        return {
            "openai/gpt-4o": ModelConfig(
                name="openai/gpt-4o",
                provider=ModelProvider.OPENROUTER,
                max_tokens=4096,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                cost_per_token=0.00003
            ),
            "anthropic/claude-3.5-sonnet": ModelConfig(
                name="anthropic/claude-3.5-sonnet",
                provider=ModelProvider.OPENROUTER,
                max_tokens=4096,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                cost_per_token=0.000015
            ),
            "google/gemini-pro-1.5": ModelConfig(
                name="google/gemini-pro-1.5",
                provider=ModelProvider.OPENROUTER,
                max_tokens=8192,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                cost_per_token=0.0000125
            ),
            "openai/gpt-4-turbo": ModelConfig(
                name="openai/gpt-4-turbo",
                provider=ModelProvider.OPENROUTER,
                max_tokens=4096,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                cost_per_token=0.00003
            ),
            "meta-llama/llama-3.1-70b-instruct": ModelConfig(
                name="meta-llama/llama-3.1-70b-instruct",
                provider=ModelProvider.OPENROUTER,
                max_tokens=4096,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                cost_per_token=0.0000009
            )
        }
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return self.models.get(model_name)
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return [name for name, config in self.models.items() if config.is_available]
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        return self.default_headers.copy()
    
    def get_base_url(self) -> str:
        """Get base URL for API requests."""
        return self.base_url
    
    def is_configured(self) -> bool:
        """Check if OpenRouter is properly configured."""
        return bool(self.api_key)
    
    def get_model_for_task(self, task_type: str) -> str:
        """Get the best model for a specific task type."""
        task_model_mapping = {
            "document_generation": "openai/gpt-4o",
            "analysis": "anthropic/claude-3.5-sonnet",
            "creative_writing": "google/gemini-pro-1.5",
            "code_generation": "openai/gpt-4o",
            "data_processing": "meta-llama/llama-3.1-70b-instruct"
        }
        return task_model_mapping.get(task_type, self.default_model)

