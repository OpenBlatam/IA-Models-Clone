"""
Prompt formatting utilities for PromptBuilder.

Refactored to consolidate prompt formatting patterns.
"""

from typing import Dict, Any, Optional


class PromptFormatter:
    """
    Formats data and optional sections for prompts.
    
    Responsibilities:
    - Format data dictionaries for prompts
    - Format optional sections with labels
    
    Single Responsibility: Handle all prompt formatting operations.
    """
    
    @staticmethod
    def format_data(data: Dict[str, Any]) -> str:
        """
        Format data dictionary for prompts.
        
        Args:
            data: Data dictionary
            
        Returns:
            Formatted string with bullet points
        """
        formatted = []
        for key, value in data.items():
            formatted.append(f"- {key}: {value}")
        return "\n".join(formatted)
    
    @staticmethod
    def format_optional_section(
        data: Optional[Dict[str, Any]],
        label: str
    ) -> str:
        """
        Format optional data section with label.
        
        Args:
            data: Optional data dictionary
            label: Section label
            
        Returns:
            Formatted section string or empty string
        """
        if not data:
            return ""
        
        formatted_data = PromptFormatter.format_data(data)
        return f"\n\n{label}:\n{formatted_data}"

