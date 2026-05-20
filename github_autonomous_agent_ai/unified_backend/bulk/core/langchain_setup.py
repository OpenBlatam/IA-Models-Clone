"""
LangChain Setup - Shared LangChain configuration utilities
==========================================================

Provides shared utilities for setting up LangChain in processors.
"""

import logging
from typing import Dict, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from ..config.openrouter_config import OpenRouterConfig

logger = logging.getLogger(__name__)


class LangChainSetup:
    """Shared LangChain setup utilities."""
    
    @staticmethod
    def create_llm(
        openrouter_config: OpenRouterConfig,
        model_name: Optional[str] = None
    ) -> ChatOpenAI:
        """
        Create a ChatOpenAI instance with OpenRouter configuration.
        
        Args:
            openrouter_config: OpenRouter configuration
            model_name: Optional model name (defaults to config default)
            
        Returns:
            Configured ChatOpenAI instance
            
        Raises:
            ValueError: If OpenRouter is not configured
        """
        if not openrouter_config.is_configured():
            raise ValueError("OpenRouter API key not configured")
        
        model = model_name or openrouter_config.default_model
        
        return ChatOpenAI(
            model=model,
            openai_api_key=openrouter_config.api_key,
            openai_api_base=openrouter_config.base_url,
            temperature=0.7,
            max_tokens=4096,
            headers=openrouter_config.get_headers()
        )
    
    @staticmethod
    def create_multiple_llms(
        openrouter_config: OpenRouterConfig,
        model_names: List[str]
    ) -> Dict[str, ChatOpenAI]:
        """
        Create multiple ChatOpenAI instances for different models.
        
        Args:
            openrouter_config: OpenRouter configuration
            model_names: List of model names to initialize
            
        Returns:
            Dictionary mapping model names to ChatOpenAI instances
        """
        if not openrouter_config.is_configured():
            raise ValueError("OpenRouter API key not configured")
        
        models = {}
        for model_name in model_names:
            try:
                models[model_name] = ChatOpenAI(
                    model=model_name,
                    openai_api_key=openrouter_config.api_key,
                    openai_api_base=openrouter_config.base_url,
                    temperature=0.7,
                    max_tokens=4096,
                    headers=openrouter_config.get_headers()
                )
                logger.info(f"Model {model_name} initialized")
            except Exception as e:
                logger.error(f"Failed to initialize model {model_name}: {e}")
        
        if not models:
            raise ValueError("No models could be initialized")
        
        return models
    
    @staticmethod
    def create_output_parser():
        """Create a StrOutputParser instance."""
        return StrOutputParser()






