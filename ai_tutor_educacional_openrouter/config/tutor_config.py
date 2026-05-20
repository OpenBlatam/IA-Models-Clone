"""
Configuration for AI Tutor Educacional with Open Router.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OpenRouterConfig:
    """Configuration for Open Router API integration."""
    api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "openai/gpt-4"
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 2000


@dataclass
class TutorConfig:
    """Main configuration for the AI Tutor system."""
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    
    # Educational settings
    subjects: List[str] = field(default_factory=lambda: [
        "matematicas", "ciencias", "historia", "literatura",
        "fisica", "quimica", "biologia", "programacion"
    ])
    
    difficulty_levels: List[str] = field(default_factory=lambda: [
        "basico", "intermedio", "avanzado"
    ])
    
    # Response settings
    response_style: str = "explicativo"
    use_examples: bool = True
    provide_exercises: bool = True
    adaptive_learning: bool = True
    
    # Storage settings
    conversation_history_path: str = "conversations"
    max_history_length: int = 50
    
    # Performance settings
    cache_enabled: bool = True
    cache_ttl: int = 3600
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.openrouter.api_key:
            raise ValueError("OPENROUTER_API_KEY must be set")
        return True






