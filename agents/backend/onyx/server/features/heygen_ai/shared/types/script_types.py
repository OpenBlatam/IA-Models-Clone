"""
Script Types
============

Data types for script generation.
"""

from dataclasses import dataclass
from typing import Optional

from shared.enums import ScriptStyle


@dataclass
class ScriptGenerationConfig:
    """Configuration for script generation.
    
    Attributes:
        style: Script style
        language: Language code (ISO 639-1)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        duration: Target duration (e.g., "2 minutes")
        use_langchain: Use LangChain for generation
    """
    style: ScriptStyle = ScriptStyle.PROFESSIONAL
    language: str = "en"
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.9
    duration: str = "2 minutes"
    use_langchain: bool = False

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0.0 and 1.0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")



