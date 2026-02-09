"""
LLM Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Dict, Any, AsyncIterator
from dataclasses import dataclass


class LLMProvider(str, Enum):
    """LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    AZURE = "azure"
    GOOGLE = "google"


@dataclass
class LLMMessage:
    """LLM message"""
    role: str  # system, user, assistant
    content: str


@dataclass
class LLMResponse:
    """LLM response"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMBase(ABC):
    """Base interface for LLM"""
    
    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate text"""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        temperature: float = 0.7
    ) -> AsyncIterator[str]:
        """Generate text stream"""
        pass
    
    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        """Generate embedding"""
        pass

