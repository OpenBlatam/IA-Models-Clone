"""
ReAct LLM Adapter

Handles all LLM provider interactions, abstracting away provider-specific
details and providing a unified interface for the ReAct agent.
"""

import logging
from typing import Any, Dict, Optional

from .react_constants import Defaults, ErrorMessages

logger = logging.getLogger(__name__)


class LLMProvider:
    """Enum-like class for LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GENERIC = "generic"
    UNKNOWN = "unknown"


class LLMAdapter:
    """
    Adapter for multiple LLM providers.
    
    This class encapsulates all LLM provider-specific logic, making it easy
    to add new providers without modifying the main ReAct agent code.
    
    Supports:
    - OpenAI API
    - Anthropic API
    - Generic callable LLMs
    - Custom LLM interfaces
    """
    
    def __init__(self, llm: Any, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLM adapter.
        
        Args:
            llm: LLM instance (OpenAI, Anthropic, or callable)
            config: Configuration dictionary
        """
        self.llm = llm
        self.config = config or {}
        self.provider = self._detect_provider()
    
    def _detect_provider(self) -> str:
        """
        Detect LLM provider type.
        
        Returns:
            Provider type string
        """
        if not self.llm:
            return LLMProvider.UNKNOWN
        
        llm_type = type(self.llm).__name__.lower()
        
        # Check for OpenAI-style API
        if "openai" in llm_type or hasattr(self.llm, "chat"):
            return LLMProvider.OPENAI
        
        # Check for Anthropic-style API
        if "anthropic" in llm_type or hasattr(self.llm, "messages"):
            return LLMProvider.ANTHROPIC
        
        # Check if callable
        if callable(self.llm):
            return LLMProvider.GENERIC
        
        return LLMProvider.UNKNOWN
    
    def generate(self, prompt: str, stream: bool = False) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt: Input prompt
            stream: Whether to stream the response
            
        Returns:
            LLM response text
        """
        if not self.llm:
            logger.warning("No LLM provided, returning empty response")
            return ""
        
        try:
            if self.provider == LLMProvider.OPENAI:
                return self._call_openai(prompt, stream)
            elif self.provider == LLMProvider.ANTHROPIC:
                return self._call_anthropic(prompt)
            elif self.provider == LLMProvider.GENERIC:
                return self._call_generic(prompt)
            else:
                logger.warning(f"Unknown LLM provider: {self.provider}")
                return ""
        except Exception as e:
            logger.warning(ErrorMessages.LLM_CALL_FAILED.format(error=str(e)))
            return ""
    
    def _call_openai(self, prompt: str, stream: bool) -> str:
        """
        Call OpenAI-style API.
        
        Args:
            prompt: Input prompt
            stream: Whether to stream response
            
        Returns:
            Response text
        """
        model = self.config.get("model", Defaults.MODEL_OPENAI)
        
        if stream:
            # Streaming response
            response = ""
            for chunk in self.llm.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True
            ):
                if chunk.choices[0].delta.content:
                    response += chunk.choices[0].delta.content
            return response
        else:
            # Non-streaming response
            response = self.llm.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str) -> str:
        """
        Call Anthropic-style API.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Response text
        """
        model = self.config.get("model", Defaults.MODEL_ANTHROPIC)
        max_tokens = self.config.get("max_tokens", Defaults.MAX_TOKENS)
        
        response = self.llm.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def _call_generic(self, prompt: str) -> str:
        """
        Call generic callable LLM.
        
        Tries multiple common method names.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Response text
        """
        # Try as callable
        if callable(self.llm):
            response = self.llm(prompt)
            return response if isinstance(response, str) else str(response)
        
        # Try common method names
        if hasattr(self.llm, "generate"):
            response = self.llm.generate(prompt)
            return response if isinstance(response, str) else str(response)
        
        if hasattr(self.llm, "complete"):
            response = self.llm.complete(prompt)
            return response if isinstance(response, str) else str(response)
        
        logger.warning("Generic LLM has no callable interface")
        return ""



