"""
Perplexity Configuration - Configuration management
====================================================

Centralized configuration for Perplexity system.
"""

import os
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PerplexityConfig:
    """Configuration for Perplexity system."""
    
    # System prompt
    system_prompt_path: Optional[str] = None
    
    # Feature flags
    enable_validation: bool = True
    enable_cache: bool = True
    enable_metrics: bool = True
    
    # Cache settings
    cache_ttl: int = 3600  # 1 hour
    cache_max_size: int = 1000
    
    # Metrics settings
    metrics_retention_days: int = 30
    
    # Processing settings
    max_search_results: int = 50
    max_answer_length: int = 10000
    
    # LLM settings
    default_llm_model: str = "gpt-4"
    default_llm_temperature: float = 0.7
    llm_timeout_seconds: int = 60
    
    @classmethod
    def from_env(cls) -> 'PerplexityConfig':
        """
        Create configuration from environment variables.
        
        Environment variables:
        - PERPLEXITY_SYSTEM_PROMPT_PATH
        - PERPLEXITY_ENABLE_VALIDATION (true/false)
        - PERPLEXITY_ENABLE_CACHE (true/false)
        - PERPLEXITY_ENABLE_METRICS (true/false)
        - PERPLEXITY_CACHE_TTL (seconds)
        - PERPLEXITY_CACHE_MAX_SIZE
        - PERPLEXITY_MAX_SEARCH_RESULTS
        - PERPLEXITY_DEFAULT_LLM_MODEL
        """
        return cls(
            system_prompt_path=os.getenv('PERPLEXITY_SYSTEM_PROMPT_PATH'),
            enable_validation=os.getenv('PERPLEXITY_ENABLE_VALIDATION', 'true').lower() == 'true',
            enable_cache=os.getenv('PERPLEXITY_ENABLE_CACHE', 'true').lower() == 'true',
            enable_metrics=os.getenv('PERPLEXITY_ENABLE_METRICS', 'true').lower() == 'true',
            cache_ttl=int(os.getenv('PERPLEXITY_CACHE_TTL', '3600')),
            cache_max_size=int(os.getenv('PERPLEXITY_CACHE_MAX_SIZE', '1000')),
            max_search_results=int(os.getenv('PERPLEXITY_MAX_SEARCH_RESULTS', '50')),
            default_llm_model=os.getenv('PERPLEXITY_DEFAULT_LLM_MODEL', 'gpt-4'),
        )
    
    @classmethod
    def default(cls) -> 'PerplexityConfig':
        """Get default configuration."""
        # Try to find system prompt in default location
        default_prompt_path = Path(__file__).parent.parent.parent / "SYSTEM_PROMPT.md"
        system_prompt_path = str(default_prompt_path) if default_prompt_path.exists() else None
        
        return cls(system_prompt_path=system_prompt_path)




