"""
Filename Generator Utilities
============================

Utilities for generating safe and consistent filenames for safe tensors.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional


class FilenameGenerator:
    """Helper class for generating filenames for safe tensors."""
    
    @staticmethod
    def generate_embedding_filename(
        character_name: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> str:
        """
        Generate filename for character embedding safe tensor.
        
        Args:
            character_name: Optional character name
            timestamp: Optional timestamp string (defaults to current time)
            
        Returns:
            Generated filename
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        char_name_safe = FilenameGenerator._sanitize_name(character_name or "character")
        return f"{char_name_safe}_{timestamp}.safetensors"
    
    @staticmethod
    def generate_workflow_filename(timestamp: Optional[str] = None) -> str:
        """
        Generate filename for workflow safe tensor.
        
        Args:
            timestamp: Optional timestamp string (defaults to current time)
            
        Returns:
            Generated filename
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"workflow_{timestamp}.safetensors"
    
    @staticmethod
    def _sanitize_name(name: str) -> str:
        """
        Sanitize character name for use in filename.
        
        Args:
            name: Character name
            
        Returns:
            Sanitized name safe for filesystem
        """
        # Replace spaces and special characters
        sanitized = name.replace(" ", "_").lower()
        # Remove any remaining invalid characters
        sanitized = "".join(c for c in sanitized if c.isalnum() or c in "._-")
        return sanitized or "character"

