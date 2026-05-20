"""
Base Service Types
==================

Base types and constants for service handlers.
"""

from typing import Dict, Any
from enum import Enum

from ..service_handler import ServiceType, ServiceConfig


class BaseServiceHandlerMixin:
    """Mixin with common service handler functionality."""
    
    def build_base_prompt(
        self,
        file_path: str,
        service_name: str,
        parameters: Dict[str, Any]
    ) -> str:
        """
        Build base prompt structure.
        
        Args:
            file_path: Path to file
            service_name: Name of service
            parameters: Service parameters
            
        Returns:
            Base prompt string
        """
        prompt = f"Procesa el siguiente archivo para {service_name}: {file_path}\n"
        
        # Add parameters
        if parameters:
            prompt += "\nParámetros:\n"
            for key, value in parameters.items():
                if value is not None:
                    prompt += f"- {key}: {value}\n"
        
        return prompt
    
    def add_instructions(self, prompt: str, instructions: list[str]) -> str:
        """
        Add instructions to prompt.
        
        Args:
            prompt: Base prompt
            instructions: List of instructions
            
        Returns:
            Prompt with instructions
        """
        if instructions:
            prompt += "\nInstrucciones:\n"
            for i, instruction in enumerate(instructions, 1):
                prompt += f"{i}. {instruction}\n"
        return prompt




