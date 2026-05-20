"""
Service Helpers for Contador AI
================================

Helper utilities for service methods in ContadorAI.
Centralizes common patterns: timing, response building, error handling.

Single Responsibility: Provide reusable utilities for service methods.
"""

import time
import logging
from typing import Dict, Any, Optional, Callable, Awaitable
from datetime import datetime

logger = logging.getLogger(__name__)


class ServiceResponseBuilder:
    """Builder for consistent service responses."""
    
    @staticmethod
    def build_success_response(
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        response_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Build a success response dictionary.
        
        Args:
            content: Main content/result
            metadata: Additional metadata
            response_time: Time taken for the operation
            
        Returns:
            Success response dictionary
        """
        response = {
            "success": True,
            "resultado": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if response_time is not None:
            response["tiempo_respuesta"] = response_time
        
        if metadata:
            response.update(metadata)
        
        return response
    
    @staticmethod
    def build_error_response(
        error: Exception,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build an error response dictionary.
        
        Args:
            error: Exception that occurred
            metadata: Additional metadata
            
        Returns:
            Error response dictionary
        """
        response = {
            "success": False,
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        }
        
        if metadata:
            response.update(metadata)
        
        return response


class ServiceExecutor:
    """Executor for service operations with timing and error handling."""
    
    def __init__(self, client, config):
        """
        Initialize ServiceExecutor.
        
        Args:
            client: OpenRouter client instance
            config: ContadorConfig instance
        """
        self.client = client
        self.config = config
    
    async def execute_with_timing(
        self,
        operation: Callable[[], Awaitable[Dict[str, Any]]],
        operation_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute an async operation with timing and error handling.
        
        Args:
            operation: Async callable to execute
            operation_name: Name of operation for logging
            metadata: Metadata to include in response
            
        Returns:
            Response dictionary with success/error and timing
        """
        start_time = time.time()
        
        try:
            response = await operation()
            response_time = time.time() - start_time
            
            content = self._extract_content(response)
            return ServiceResponseBuilder.build_success_response(
                content=content,
                metadata=metadata,
                response_time=response_time
            )
        except Exception as e:
            logger.error(f"Error in {operation_name}: {e}", exc_info=True)
            return ServiceResponseBuilder.build_error_response(
                error=e,
                metadata=metadata
            )
    
    async def call_openrouter(
        self,
        messages: list,
        system_prompt_key: str,
        metadata: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Call OpenRouter with standard configuration.
        
        Args:
            messages: List of message dictionaries
            system_prompt_key: Key for system prompt (not used here, but for consistency)
            metadata: Metadata to include in response
            temperature: Override temperature (defaults to config)
            max_tokens: Override max_tokens (defaults to config)
            
        Returns:
            Response dictionary
        """
        async def _call():
            return await self.client.generate_completion(
                messages=messages,
                model=self.config.openrouter.default_model,
                temperature=temperature or self.config.openrouter.temperature,
                max_tokens=max_tokens or self.config.openrouter.max_tokens
            )
        
        return await self.execute_with_timing(
            _call,
            "OpenRouter API call",
            metadata=metadata
        )
    
    def _extract_content(self, response: Dict[str, Any]) -> str:
        """Extract content from OpenRouter response."""
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            logger.error(f"Error extracting content from response: {e}")
            return "Error al procesar la respuesta"


class MessageBuilder:
    """Builder for OpenRouter messages."""
    
    @staticmethod
    def build_messages(
        system_prompt: str,
        user_prompt: str
    ) -> list:
        """
        Build messages list for OpenRouter.
        
        Args:
            system_prompt: System prompt content
            user_prompt: User prompt content
            
        Returns:
            List of message dictionaries
        """
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    @staticmethod
    def format_data(data: Dict[str, Any]) -> str:
        """
        Format data dictionary for prompts.
        
        Args:
            data: Dictionary to format
            
        Returns:
            Formatted string
        """
        formatted = []
        for key, value in data.items():
            formatted.append(f"- {key}: {value}")
        return "\n".join(formatted)
    
    @staticmethod
    def add_context(
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        context_label: str = "Contexto adicional"
    ) -> str:
        """
        Add context to a prompt.
        
        Args:
            prompt: Base prompt
            context: Context dictionary
            context_label: Label for context section
            
        Returns:
            Prompt with context appended
        """
        if not context:
            return prompt
        
        context_str = f"\n\n{context_label}:\n{MessageBuilder.format_data(context)}"
        return prompt + context_str

