"""
Service Method Helper for Contador AI
=====================================

Helper to reduce duplication in service methods.
Centralizes the common pattern: prompt → messages → API call → rename time field.

Single Responsibility: Provide a reusable pattern for service methods.
"""

from typing import Dict, Any, Optional, Callable
from .service_helpers import MessageBuilder
from .api_handler import APIHandler
from .response_formatter import ResponseFormatter


class ServiceMethodHelper:
    """
    Helper for executing service methods with consistent pattern.
    
    Responsibilities:
    - Build prompts and messages
    - Call API handler
    - Format responses
    - Rename time fields
    """
    
    @staticmethod
    async def execute_service(
        prompt: str,
        system_prompt: str,
        api_handler: APIHandler,
        service_name: str,
        service_data: Dict[str, Any],
        extract_key: str = "resultado",
        temperature: Optional[float] = None,
        time_field_rename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a service method following the standard pattern.
        
        Args:
            prompt: User prompt (already built)
            system_prompt: System prompt for the service
            api_handler: APIHandler instance
            service_name: Name of the service
            service_data: Service-specific data
            extract_key: Key to use for extracted content
            temperature: Optional temperature override
            time_field_rename: Optional new name for tiempo_respuesta field
            
        Returns:
            Service result dictionary
        """
        # Build messages
        messages = MessageBuilder.build_messages(
            system_prompt=system_prompt,
            user_prompt=prompt
        )
        
        # Call API
        result = await api_handler.call_with_metrics(
            messages=messages,
            service_name=service_name,
            service_data=service_data,
            temperature=temperature,
            extract_key=extract_key
        )
        
        # Rename time field if needed
        if time_field_rename:
            result = ResponseFormatter.rename_time_field(
                result, "tiempo_respuesta", time_field_rename
            )
        
        return result

