"""
Response formatting utilities for Contador AI.

Refactored to consolidate response field renaming logic.
"""

from typing import Dict, Any, Optional


class ResponseFormatter:
    """
    Formats responses with consistent field naming.
    
    Responsibilities:
    - Rename time fields for consistency
    - Format response dictionaries
    
    Single Responsibility: Handle all response formatting operations.
    """
    
    @staticmethod
    def rename_time_field(
        result: Dict[str, Any],
        old_key: str,
        new_key: str
    ) -> Dict[str, Any]:
        """
        Rename a time field in the result dictionary.
        
        Args:
            result: Result dictionary
            old_key: Old field name
            new_key: New field name
            
        Returns:
            Result dictionary with renamed field
        """
        if result.get(old_key):
            result[new_key] = result.pop(old_key)
        return result
    
    @staticmethod
    def format_calculation_response(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format calculation response with consistent field names.
        
        Args:
            result: Raw result from API handler
            
        Returns:
            Formatted result with tiempo_calculo
        """
        return ResponseFormatter.rename_time_field(
            result,
            "tiempo_respuesta",
            "tiempo_calculo"
        )
    
    @staticmethod
    def format_generation_response(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format generation response with consistent field names.
        
        Args:
            result: Raw result from API handler
            
        Returns:
            Formatted result with tiempo_generacion
        """
        return ResponseFormatter.rename_time_field(
            result,
            "tiempo_respuesta",
            "tiempo_generacion"
        )

