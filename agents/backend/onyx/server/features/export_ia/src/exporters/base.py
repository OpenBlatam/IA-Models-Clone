"""
Base exporter class for all export formats.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseExporter(ABC):
    """Base class for all export format handlers."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    async def export(
        self, 
        content: Dict[str, Any], 
        config: Any,  # ExportConfig
        output_path: str
    ) -> Dict[str, Any]:
        """
        Export content to the specific format.
        
        Args:
            content: Processed document content
            config: Export configuration
            output_path: Output file path
            
        Returns:
            Export result metadata
        """
        pass
    
    @abstractmethod
    def get_supported_features(self) -> list:
        """Get list of supported features for this format."""
        pass
    
    def validate_content(self, content: Dict[str, Any]) -> bool:
        """Validate content before export."""
        required_fields = ["title"]
        return all(field in content for field in required_fields)
    
    def preprocess_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess content before export."""
        processed = content.copy()
        
        # Ensure title exists
        if "title" not in processed:
            processed["title"] = "Untitled Document"
        
        # Ensure sections exist
        if "sections" not in processed:
            processed["sections"] = []
        
        return processed
    
    def postprocess_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess export result."""
        return result




