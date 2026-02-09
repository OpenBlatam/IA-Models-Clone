"""
Configuration for Contabilidad Mexicana AI SAM3
===============================================
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class OpenRouterConfig:
    """OpenRouter configuration."""
    api_key: Optional[str] = None
    model: str = "anthropic/claude-3.5-sonnet"
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class TruthGPTConfig:
    """TruthGPT configuration."""
    enabled: bool = True
    endpoint: Optional[str] = None
    timeout: float = 60.0
    optimization_type: str = "standard"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "truthgpt_endpoint": self.endpoint,
            "timeout": self.timeout,
            "optimization_type": self.optimization_type,
        }


@dataclass
class ContadorSAM3Config:
    """Configuration for Contador SAM3 Agent."""
    
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    truthgpt: TruthGPTConfig = field(default_factory=TruthGPTConfig)
    max_parallel_tasks: int = 10
    output_dir: str = "contador_sam3_output"
    debug: bool = False
    
    def __post_init__(self):
        """Initialize configuration from environment variables."""
        # OpenRouter
        if not self.openrouter.api_key:
            self.openrouter.api_key = os.getenv("OPENROUTER_API_KEY")
        
        # TruthGPT
        if self.truthgpt.enabled:
            if not self.truthgpt.endpoint:
                self.truthgpt.endpoint = os.getenv("TRUTHGPT_ENDPOINT")
    
    def validate(self):
        """Validate configuration."""
        if not self.openrouter.api_key:
            raise ValueError("OpenRouter API key is required")
        
        if self.truthgpt.enabled and not self.truthgpt.endpoint:
            # TruthGPT is optional, just log a warning
            import logging
            logging.getLogger(__name__).warning(
                "TruthGPT endpoint not configured, running without TruthGPT optimization"
            )
