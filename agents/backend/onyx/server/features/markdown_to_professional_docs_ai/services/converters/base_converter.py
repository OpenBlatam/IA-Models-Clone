"""Base Converter - Abstract base class for all converters"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseConverter(ABC):
    """Abstract base class for all format converters"""
    
    @abstractmethod
    async def convert(
        self,
        parsed_content: Dict[str, Any],
        output_path: str,
        include_charts: bool = True,
        include_tables: bool = True,
        custom_styling: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Convert parsed Markdown content to target format
        
        Args:
            parsed_content: Parsed Markdown content
            output_path: Path where output file should be saved
            include_charts: Whether to include charts
            include_tables: Whether to include tables
            custom_styling: Custom styling options
        """
        pass

